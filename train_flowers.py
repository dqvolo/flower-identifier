import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---- Device ----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Class names for the Oxford 102 Flowers dataset (label index = list position)
CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster",
    "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
    "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula",
    "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen",
    "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
]
assert len(CLASSES) == 102

# ---- Data ----
# Pretrained models expect ImageNet-style normalization
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Augmentation for training: random crops and flips make the model more robust
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_data = datasets.Flowers102("data", split="train", transform=train_tf, download=True)
val_data = datasets.Flowers102("data", split="val", transform=eval_tf, download=True)
test_data = datasets.Flowers102("data", split="test", transform=eval_tf, download=True)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# ---- Model: pretrained ResNet18 with a new final layer ----
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 102)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()


def run_epoch(loader, optimizer=None):
    """Runs one pass over a loader. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    correct, total = 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if training:
                optimizer.zero_grad()
                loss_fn(outputs, labels).backward()
                optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


# ---- Phase 1: freeze the pretrained body, train only the new final layer ----
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True

opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
for epoch in range(1, 6):
    train_acc = run_epoch(train_loader, opt)
    val_acc = run_epoch(val_loader)
    print(f"[Phase 1] Epoch {epoch}: train {train_acc:.1f}%  val {val_acc:.1f}%")

# ---- Phase 2: unfreeze everything, fine-tune gently with a small learning rate ----
for p in model.parameters():
    p.requires_grad = True

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(1, 5):
    train_acc = run_epoch(train_loader, opt)
    val_acc = run_epoch(val_loader)
    print(f"[Phase 2] Epoch {epoch}: train {train_acc:.1f}%  val {val_acc:.1f}%")

# ---- Final score on images the model never saw ----
print(f"Final test accuracy: {run_epoch(test_loader):.2f}%")

torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, "flower_model.pt")
print("Saved model to flower_model.pt")
