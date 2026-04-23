"""
Debug script to test model loading and identify issues.
Run this to diagnose problems with model loading.
"""

import torch
import logging
from transformers import AutoModelForImageClassification, AutoImageProcessor
from huggingface_hub import list_repo_files, model_info

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 70)
print("INDIAN CURRENCY DETECTOR - MODEL LOADING DIAGNOSTICS")
print("=" * 70)

# 1. Check PyTorch
print("\n[1] PYTORCH CHECK")
print(f"    PyTorch version: {torch.__version__}")
print(f"    GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU name: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Using device: {device}")

# 2. Check HuggingFace connectivity
print("\n[2] HUGGINGFACE CONNECTIVITY CHECK")
MODEL_ID = "Shubhamm007/indian-currency-classifier"
print(f"    Model ID: {MODEL_ID}")

try:
    print(f"    Checking model repository access...")
    info = model_info(MODEL_ID)
    print(f"    ✓ Model found on HuggingFace!")
    print(f"    Model type: {info.model_name}")
    print(f"    Downloads: {info.downloads}")
    print(f"    Likes: {info.likes}")
except Exception as e:
    print(f"    ✗ Error accessing model: {e}")
    print(f"    Error type: {type(e).__name__}")
    print(f"\n    TROUBLESHOOTING:")
    print(f"    - Check your internet connection")
    print(f"    - Verify model ID is correct")
    print(f"    - Try: huggingface-cli login (if private model)")
    print(f"    - Check HuggingFace hub status at: https://status.huggingface.co/")

# 3. Try loading model
print("\n[3] MODEL LOADING TEST")
try:
    print(f"    Loading model: {MODEL_ID}")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    print(f"    ✓ Model loaded successfully!")
    print(f"    Model class: {model.__class__.__name__}")
    print(f"    Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Move to device
    model.to(device)
    model.eval()
    print(f"    ✓ Model moved to {device} and set to eval mode")
    
except Exception as e:
    print(f"    ✗ Error loading model: {e}")
    print(f"    Error type: {type(e).__name__}")
    print(f"\n    TROUBLESHOOTING:")
    print(f"    - Check model repository exists")
    print(f"    - Try clearing cache: rm -rf ~/.cache/huggingface/hub/")
    print(f"    - Try fallback model: google/vit-base-patch16-224")

# 4. Try loading image processor
print("\n[4] IMAGE PROCESSOR TEST")
try:
    print(f"    Loading image processor for: {MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    print(f"    ✓ Image processor loaded successfully!")
    print(f"    Processor class: {processor.__class__.__name__}")
except Exception as e:
    print(f"    ✗ Error loading processor: {e}")
    print(f"    Using fallback: Custom preprocessing")

# 5. Test inference
print("\n[5] INFERENCE TEST")
try:
    from PIL import Image
    import requests
    from io import BytesIO
    
    print(f"    Downloading sample image...")
    # Download a simple test image
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    response = requests.get(url, timeout=10)
    image = Image.open(BytesIO(response.content))
    
    print(f"    Image loaded: {image.size}")
    
    # Try prediction
    from torchvision import transforms
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs.logits
    
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item() * 100
    
    print(f"    ✓ Inference successful!")
    print(f"    Predicted class: {predicted_class}")
    print(f"    Confidence: {confidence:.2f}%")

except Exception as e:
    print(f"    ✗ Error during inference test: {e}")
    print(f"    Error type: {type(e).__name__}")

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)
print("\nIf you see ✓ marks above, your setup is working correctly!")
print("If you see ✗ marks, check the troubleshooting steps for that section.")
