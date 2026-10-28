
import torch
import gradio as gr
from torchvision.models import resnet18, ResNet18_Weights

# Step 1: Load a pretrained ResNet-18 model.
# The weights are downloaded automatically on first use.
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.eval()

# Use the preprocessing and class labels associated with these weights.
preprocess = weights.transforms()
labels = weights.meta["categories"]


# Step 2: Define the prediction function.
def predict(image):
    if image is None:
        return {}

    # Convert the input to RGB and prepare a batch of one image.
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    # Run inference without calculating gradients.
    with torch.inference_mode():
        logits = model(image_tensor)[0]
        probabilities = torch.softmax(logits, dim=0)

    # Return the three most likely classes and their probabilities.
    top_probabilities, top_indices = torch.topk(probabilities, k=3)

    predictions = {
        labels[index.item()]: float(probability.item())
        for probability, index in zip(top_probabilities, top_indices)
    }

    return predictions


# Step 3: Create the Gradio interface.
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(
        num_top_classes=3,
        label="Top predictions"
    ),
    title="Image Classifier",
    description=(
        "Upload an image to see the three most likely "
        "ImageNet categories predicted by ResNet-18."
    ),
)

if __name__ == "__main__":
    demo.launch()
