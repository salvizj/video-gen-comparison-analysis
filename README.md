# Video Generation Comparison – Analysis

Analysis code for bachelor's thesis comparing text-to-video diffusion models (Wan 2.2 5B, LTX-Video 2.0, HunyuanVideo 1.5).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/overall.py
python scripts/category_impact.py
python scripts/prompt_variation_impact.py
python scripts/video_length_impact.py
```

Results are saved to `tables/` and `figures/`.
