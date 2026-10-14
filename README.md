# Flower Identifier

An image classifier that recognizes 102 flower species, built by fine-tuning a pretrained ResNet18 with PyTorch. It includes a small Gradio web app: upload a flower photo and get the top 5 predictions.

**Test accuracy: 88.62%** on the Oxford 102 Flowers test set.

## How it works

- **Transfer learning:** starts from ResNet18 pretrained on ImageNet and replaces the final layer with a new 102-class layer.
- **Two-phase training:**
  1. Phase 1 freezes the pretrained layers and trains only the new final layer.
  2. Phase 2 unfreezes the whole network and fine-tunes it with a smaller learning rate.
- **Data augmentation:** random crops and horizontal flips during training to reduce overfitting, since there are only 10 training images per class.
- **Proper evaluation:** separate train, validation, and test splits. The test set is used only once, at the end.

## Project structure

| File | Purpose |
|------|---------|
| `train_flowers.py` | Downloads the dataset, trains the model, saves `flower_model.pt` |
| `app.py` | Gradio web app that loads the trained model for predictions |
| `requirements.txt` | Python dependencies |

## Setup and usage

Requires Python 3.9 or newer.

```bash
git clone https://github.com/dqvolo/flower-identifier.git
cd flower-identifier
python -m venv venv
venv\Scripts\activate        # on Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Train the model (downloads about 330 MB of images on first run and takes roughly half an hour on a CPU):

```bash
python train_flowers.py
```

Launch the demo (open the local address it prints in your browser):

```bash
python app.py
```

## Results

| Split | Accuracy |
|-------|----------|
| Test  | 88.62%   |

## Findings and limitations

- **Domain shift:** accuracy is noticeably lower on stock photos than on the benchmark. The training set is small (10 images per species) and has a specific photo style, so the model generalizes less well to different backgrounds, lighting, and framing.
- **Closed set:** the model only knows these 102 species. For a flower outside the list, or for a non-flower image, it still picks the closest match with misleading confidence.
- **Ideas to improve:** train on train+val for the final run, stronger augmentation (colour jitter), a larger backbone such as ResNet50, and evaluating on a self-collected set of photos.

## Credits

- Dataset: [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) (Nilsback and Zisserman)
- Pretrained weights: torchvision ResNet18 (ImageNet)
- Web UI: [Gradio](https://www.gradio.org/)

## License

MIT, see `LICENSE`.
