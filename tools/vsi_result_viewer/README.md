# VSI-Bench Result Viewer

Static viewer for `lmms-eval` VSI-Bench sample logs.

## Build Data

Run from the VSI-Bench repository root:

```bash
python3 tools/vsi_result_viewer/build_data.py --root .
```

By default it discovers `logs/**/vsibench.json` and includes runs with at least
1000 samples. To include smaller smoke runs:

```bash
python3 tools/vsi_result_viewer/build_data.py --root . --min-samples 1
```

## Serve

```bash
python3 -m http.server 8765 --directory tools/vsi_result_viewer
```

Then open `http://localhost:8765`.
