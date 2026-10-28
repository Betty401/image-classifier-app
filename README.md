
# Image Classifier with ResNet-18 and Gradio

A computer vision application that classifies uploaded images
using a pretrained ResNet-18 model trained on ImageNet.

## Features

- Pretrained ResNet-18 image classification
- Image preprocessing using Torchvision
- Top-three predictions with confidence scores
- Interactive web interface built with Gradio

## Technologies

- Python
- PyTorch
- Torchvision
- Gradio
- Pillow

## How It Works

1. Load pretrained ResNet-18 weights.
2. Preprocess the input image.
3. Run model inference.
4. Convert output logits into probabilities.
5. Display the three most likely ImageNet categories.

## Installation

Create and activate a virtual environment, then install
the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Open the local URL displayed in the terminal.

The pretrained model weights are downloaded on first use.

## Limitations

This application uses ImageNet-1K categories. It is not
a specialized classifier for faces, medical images or
categories outside its training label set.

## Acknowledgments

This project uses pretrained ResNet-18 weights and
image preprocessing from Torchvision, with Gradio
for the user interface.
