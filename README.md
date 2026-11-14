# Flood Mapping Project

A prototype flood-segmentation project built with TensorFlow/Keras and Streamlit. It trains a U-Net model on paired pre-flood and post-flood images with binary flood masks, then provides web apps for visualizing predictions and estimating impacts.

> **Important:** The included dataset is synthetic and intended only for demonstrating the workflow. The damage, population, infrastructure, and area figures are heuristic estimates, not verified measurements. Do not use this project for emergency response or real-world decisions without suitable data, validation, and calibration.

## Features

- Train a binary flood-mask segmentation model using paired images and masks.
- Inspect predictions, flood coverage, severity, and image overlays.
- Run a Streamlit damage-assessment interface with configurable land-use and impact assumptions.
- Experiment with before/after change detection using the separate `06_before_after_app.py` interface.

## Project layout

```text
.
├── 02_train.py                  # Train and evaluate the U-Net model
├── 03_test.py                   # Run predictions on sample post-flood images
├── 04_app.py                    # Basic Streamlit flood-mapping app
├── 05_enhanced_app.py           # Streamlit app with heuristic impact estimates
├── 06_before_after_app.py       # Experimental before/after change analysis
├── config.yaml                  # Image size, model settings, and data paths
├── data/
│   └── raw/                     # pre_NNN.png, post_NNN.png, mask_NNN.png
├── inference/                   # Prediction and change-detection code
├── models/
│   ├── unet_model.py            # U-Net architecture
│   └── trained_models/
│       └── flood_model.h5       # Included trained model
└── preprocessing/               # Data loading and image preprocessing
```

`complete_setup.py` can create the directory structure and regenerate the small synthetic dataset. It also writes configuration and supporting source files, so run it only when you intend to recreate those generated files.

## Requirements

- Python with a TensorFlow version compatible with your operating system and Python version
- The packages imported by the scripts and apps: TensorFlow, NumPy, Matplotlib, Pillow, scikit-learn, PyYAML, Streamlit, OpenCV, pandas, and Plotly
- `albumentations` if using `preprocessing/image_preprocessor.py`

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow numpy matplotlib pillow scikit-learn pyyaml streamlit opencv-python pandas plotly
```

If PowerShell prevents activation, you can invoke the environment's Python directly as `.venv\Scripts\python.exe`.

## Run the included model

From the project root, start either Streamlit app:

```powershell
streamlit run 05_enhanced_app.py
```

The basic interface is also available:

```powershell
streamlit run 04_app.py
```

Open the local URL printed by Streamlit. Both apps expect `models/trained_models/flood_model.h5`. The uploaded-image predictor currently supplies the same image as both model inputs; it does not accept a pre-flood/post-flood pair through these two interfaces.

## Train the model

The expected training samples are triplets in `data/raw/` with matching numeric suffixes:

```text
pre_000.png     # Pre-event image
post_000.png    # Post-event image
mask_000.png    # Binary flood mask
```

The loader concatenates each pre/post image pair into a six-channel input. Update `paths.data_path` in `config.yaml` to the absolute path of this checkout's `data/raw` directory before training. The checked-in configuration contains machine-specific paths from a different location.

Then run:

```powershell
python 02_train.py
```

Training saves the model to `models/trained_models/flood_model.h5` and writes `training_history.png`. To evaluate the saved model against the sample post-event images:

```powershell
python 03_test.py
```

The model configuration uses 128 × 128 images, a six-channel input, a batch size of 4, and 15 training epochs. Actual run time depends on the machine and TensorFlow build.

## Before/after app

`06_before_after_app.py` accepts separate pre-event and post-event images and attempts to identify new water by comparing them. This app currently expects `models/trained_models/flood_model_200.h5`, which is not the model file included in this project. To try it, provide that compatible model or update the model path in the script to a compatible checkpoint:

```powershell
streamlit run 06_before_after_app.py
```

## Configuration and interpretation

`config.yaml` controls image dimensions, model input shape, training parameters, and dataset paths. Keep the model input shape consistent with the pre/post image channels and the data loader.

The enhanced app's area and impact outputs use fixed assumptions in the predictor (including a reference image area) and user-entered estimates such as cost per hectare and population density. They are illustrative calculations; supply geographically appropriate calibration and validate model predictions before interpreting them.
