import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms

# Load the trained model and class names saved by train_flowers.py
checkpoint = torch.load("flower_model.pt", map_location="cpu")
classes = checkpoint["classes"]

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image):
    x = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, 5)
    return {classes[i]: float(p) for p, i in zip(top.values, top.indices)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a flower photo"),
    outputs=gr.Label(num_top_classes=5, label="Top guesses"),
    title="Flower Identifier",
    description="A ResNet18 fine-tuned on the Oxford 102 Flowers dataset.",
)

if __name__ == "__main__":
    demo.launch()
