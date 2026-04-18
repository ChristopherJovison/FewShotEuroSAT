# FewShotEuroSAT

**Training Free Few Shot Adaptation for Remote Sensing Image Classification with CLIP and Gaussian Discriminant Analysis**

This repository contains the EuroSAT experiments for RemoteCLIP few shot adaptation. The main method is **RemoteCLIP + Gaussian Discriminant Analysis (GDA)**, which adapts frozen image features using a small labeled support set and no backbone fine tuning.

## Install Dependencies

Windows setup with Conda:

```powershell
conda env create -f environment.yml
conda activate fewshot_eurosat
```

Or with a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Recommended notebook kernel:

```powershell
python -m ipykernel install --user --name fewshot_eurosat --display-name "Python (fewshot_eurosat)"
```

Quick test:

```powershell
python -m unittest discover -s tests
```

## Main Files

The main workflow is in the notebooks:

```text
notebooks/1_data_loader.ipynb      prepare/check EuroSAT and dataset EDA
notebooks/2_model.ipynb            step-by-step RemoteCLIP + GDA model run
notebooks/3_visualization.ipynb    visualizations for the run from notebook 2
```

Important folders:

```text
configs/                 EuroSAT experiment configs
data/eurosat/            prepared EuroSAT ImageFolder split and metadata
outputs/eurosat/         reported run artifacts
paper_assets/            final paper figures, tables, and analytics
scripts/                 CLI runners and asset builders
src/                     reusable implementation code
tests/                   lightweight unit tests
```

The prepared dataset path is:

```text
data/eurosat/imagefolder/{train,val,test}/<class_name>/*.jpg
```

## Full Run Commands

Prepare EuroSAT from KaggleHub if the included split is missing:

```powershell
python scripts/prepare_eurosat.py
```

Run the main 16-shot RemoteCLIP + GDA result:

```powershell
python scripts/run_experiment.py --config configs/eurosat_remoteclip_gda_tta_supportviews.yaml --shots 16 --seed 42
```

Run the full benchmark sweep:

```powershell
python scripts/run_all.py --config configs/eurosat_benchmark_all.yaml --skip_existing
```

Regenerate paper assets:

```powershell
python scripts/make_paper_assets.py --input_dir outputs/eurosat --output_dir paper_assets
```

Generate analytics for a completed run:

```powershell
python scripts/run_analytics.py outputs/eurosat/latest --output_dir paper_assets/analytics
```

RemoteCLIP weights are downloaded from Hugging Face on first use into:

```text
checkpoints/remoteclip/
```

## Reported Results

Main training free result:

| Method | Labels | Top-1 | Macro F1 |
|---|---:|---:|---:|
| Zero-shot CLIP ViT-B/16 | 0-shot | 37.4% | 35.9% |
| Zero-shot RemoteCLIP ViT-B/32 | 0-shot | 37.6% | 34.3% |
| CLIP ViT-B/16 + GDA | 16-shot | 88.3% | 87.9% |
| RemoteCLIP ViT-B/32 + GDA | 16-shot | **90.2%** | **89.9%** |
| ConvNeXt-Tiny full fine-tune | full train | 98.6% | 98.6% |

ConvNeXt is upper bound, main architecture lies in RemoteCLIP ViT-B/32 + GDA
