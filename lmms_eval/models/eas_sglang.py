import base64
import gc
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

try:
    from decord import VideoReader, cpu
except ImportError:  # pragma: no cover - handled at runtime when video is used
    VideoReader = None
    cpu = None

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = 5


def _append_chat_completions(url: str) -> str:
    url = url.rstrip('/')
    if url.endswith('/v1/chat/completions'):
        return url
    if url.endswith('/v1'):
        return f'{url}/chat/completions'
    return f'{url}/v1/chat/completions'


@register_model('eas_sglang')
class EASSGLang(lmms):
    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        model_version: str = 'qwen36_27_infer_sglang',
        modality: str = 'video',
        max_frames_num: int = 32,
        timeout: int = 180,
        num_workers: int = 4,
        disable_thinking: bool = True,
        image_format: str = 'JPEG',
        image_quality: int = 90,
        max_image_size: int = 0,
        visual_cache_limit: int = 8,
        cache_dir: str = 'eas_cache',
        **kwargs,
    ) -> None:
        super().__init__()
        api_url = api_url or os.getenv('EAS_API_URL') or os.getenv('EAS_URL')
        api_key = api_key or os.getenv('EAS_API_KEY') or os.getenv('EAS_TOKEN')
        if not api_url:
            raise ValueError('EAS api_url is required. Set EAS_API_URL or pass api_url in model_args.')
        if not api_key:
            raise ValueError('EAS api_key is required. Set EAS_API_KEY/EAS_TOKEN or pass api_key in model_args.')

        self.api_url = _append_chat_completions(api_url)
        self.api_key = api_key
        self.model_version = model_version
        self.modality = modality
        self.max_frames_num = int(max_frames_num)
        self.timeout = int(timeout)
        self.num_workers = max(1, int(num_workers))
        self.disable_thinking = str(disable_thinking).lower() not in {'0', 'false', 'no'}
        self.image_format = image_format.upper()
        self.image_quality = int(image_quality)
        self.max_image_size = int(max_image_size or 0)
        self.visual_cache_limit = max(0, int(visual_cache_limit))
        self.image_token = '<image>'
        self.headers = {'Authorization': self.api_key, 'Content-Type': 'application/json'}

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], 'Unsupported distributed type provided.'
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        self.device = self.accelerator.device

        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        safe_name = ''.join(c if c.isalnum() or c in {'-', '_', '.'} else '_' for c in self.model_version)
        self.cache_file = cache_root / f'{safe_name}_responses.json'
        if self.cache_file.exists():
            self.response_cache = json.loads(self.cache_file.read_text())
        else:
            self.response_cache = {}
        self.cache_lock = threading.Lock()
        self.visual_cache = OrderedDict()
        self.visual_cache_lock = threading.Lock()

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        if self.max_image_size <= 0:
            return image
        if max(image.size) <= self.max_image_size:
            return image
        image = image.copy()
        resampling = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')
        image.thumbnail((self.max_image_size, self.max_image_size), resampling)
        return image

    def encode_image(self, image: Image.Image):
        image = self._resize_if_needed(image)
        output_buffer = BytesIO()
        if self.image_format in {'JPG', 'JPEG'}:
            image = image.convert('RGB')
            image.save(output_buffer, format='JPEG', quality=self.image_quality)
            mime = 'image/jpeg'
        else:
            image.save(output_buffer, format='PNG')
            mime = 'image/png'
        base64_str = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        return mime, base64_str

    def encode_video(self, video_path, for_get_frames_num):
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f'Could not open video: {video_path}')
            total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frame_num <= 0:
                raise RuntimeError(f'Could not read frame count for video: {video_path}')

            sampled_count = min(total_frame_num, int(for_get_frames_num))
            frame_idx = np.linspace(0, total_frame_num - 1, sampled_count, dtype=int).tolist()
            encoded_frames = []
            try:
                for idx in frame_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encoded_frames.append(self.encode_image(Image.fromarray(frame)))
            finally:
                cap.release()
            if encoded_frames:
                gc.collect()
                return encoded_frames
            raise RuntimeError(f'Could not decode sampled frames for video: {video_path}')
        except Exception as cv2_error:
            eval_logger.debug(f'OpenCV video decode failed for {video_path}: {cv2_error}; falling back to decord/pyav.')

        if VideoReader is None:
            raise ImportError('decord is required to decode videos for eas_sglang fallback.')
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frame_num = len(vr)
            sampled_count = min(total_frame_num, int(for_get_frames_num))
            frame_idx = np.linspace(0, total_frame_num - 1, sampled_count, dtype=int).tolist()
            frames = vr.get_batch(frame_idx).asnumpy()
        except Exception:
            import av

            decoded = []
            with av.open(video_path) as container:
                for frame in container.decode(video=0):
                    decoded.append(frame.to_ndarray(format='rgb24'))
            sampled_count = min(len(decoded), int(for_get_frames_num))
            frame_idx = np.linspace(0, len(decoded) - 1, sampled_count, dtype=int).tolist()
            frames = [decoded[i] for i in frame_idx]

        encoded_frames = []
        for frame in frames:
            encoded_frames.append(self.encode_image(Image.fromarray(frame)))
        del frames
        gc.collect()
        return encoded_frames

    def flatten(self, inputs):
        return [item for group in inputs for item in group]

    def _encode_visual(self, visual):
        if self.modality == 'image':
            return [self.encode_image(visual)]
        if self.modality != 'video':
            raise ValueError(f'Unsupported modality: {self.modality}')

        key = (str(visual), self.max_frames_num, self.image_format, self.image_quality, self.max_image_size)
        if self.visual_cache_limit > 0:
            with self.visual_cache_lock:
                cached = self.visual_cache.get(key)
                if cached is not None:
                    self.visual_cache.move_to_end(key)
                    return cached

        encoded = self.encode_video(visual, self.max_frames_num)
        if self.visual_cache_limit > 0:
            with self.visual_cache_lock:
                self.visual_cache[key] = encoded
                self.visual_cache.move_to_end(key)
                while len(self.visual_cache) > self.visual_cache_limit:
                    self.visual_cache.popitem(last=False)
        return encoded

    def _cache_key(self, contexts, visuals, gen_kwargs):
        metainfo = {
            'context': contexts,
            'visuals': visuals,
            'max_frames_num': self.max_frames_num,
            'gen_kwargs': gen_kwargs,
            'model_version': self.model_version,
            'api_url': self.api_url,
            'disable_thinking': self.disable_thinking,
            'image_format': self.image_format,
            'image_quality': self.image_quality,
            'max_image_size': self.max_image_size,
        }
        return json.dumps(metainfo, ensure_ascii=False, sort_keys=True, default=str)

    def _save_cache(self):
        tmp_file = self.cache_file.with_suffix('.tmp')
        tmp_file.write_text(json.dumps(self.response_cache, ensure_ascii=False, indent=2))
        tmp_file.replace(self.cache_file)

    def _build_payload(self, contexts, imgs, gen_kwargs):
        content = []
        for mime, img in imgs:
            content.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{img}'}})
        content.append({'type': 'text', 'text': contexts})

        max_tokens = int(gen_kwargs.get('max_new_tokens', 1024))
        payload = {
            'model': self.model_version,
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens,
            'temperature': gen_kwargs.get('temperature', 0),
        }
        if gen_kwargs.get('top_p') is not None:
            payload['top_p'] = gen_kwargs.get('top_p')
        if self.disable_thinking:
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        return payload

    def _generate_one(self, request):
        contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
        visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
        visuals = self.flatten(visuals)
        gen_kwargs = dict(gen_kwargs or {})
        gen_kwargs.setdefault('max_new_tokens', 1024)
        gen_kwargs.setdefault('temperature', 0)

        cache_key = self._cache_key(contexts, visuals, gen_kwargs)
        with self.cache_lock:
            cached = self.response_cache.get(cache_key)
        if cached is not None:
            return cached

        imgs = []
        for visual in visuals:
            imgs.extend(self._encode_visual(visual))

        payload = self._build_payload(contexts, imgs, gen_kwargs)
        response_text = ''
        for attempt in range(5):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                response_data = response.json()
                message = response_data['choices'][0]['message']
                response_text = (message.get('content') or message.get('reasoning_content') or '').strip()
                break
            except Exception as exc:
                detail = ''
                try:
                    detail = response.text[:1000]
                except Exception:
                    pass
                eval_logger.info(f'Attempt {attempt + 1} failed with error: {exc}. Response: {detail}')
                if attempt < 4:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:
                    eval_logger.error(f'All attempts failed. Last error: {exc}. Response: {detail}')

        del payload, imgs
        gc.collect()
        with self.cache_lock:
            self.response_cache[cache_key] = response_text
            self._save_cache()
        return response_text

    def generate_until(self, requests) -> List[str]:
        results = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc='Model Responding')
        if self.num_workers == 1:
            for idx, request in enumerate(requests):
                results[idx] = self._generate_one(request)
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_idx = {executor.submit(self._generate_one, request): idx for idx, request in enumerate(requests)}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        eval_logger.error(f'Request {idx} failed: {exc}')
                        results[idx] = ''
                    pbar.update(1)
        pbar.close()
        return results

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, 'eas_sglang does not support loglikelihood.'
