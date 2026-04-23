import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformers import AutoModelForImageClassification, AutoImageProcessor
from huggingface_hub import HfApi, login
import os
import shutil
from pathlib import Path

# --- 1. RESEARCH-BACKED CONFIGURATION ---
MODEL_ID = "Shubhamm007/indian-currency-classifier"
DATASET_PATH = "feedback_dataset"
OUTPUT_DIR = "./temp_improved_model"
MIN_IMAGES_REQUIRED = 50 # Per your request for efficiency
LEARNING_RATE = 2e-6      # Ultra-stable for fine-tuning
WEIGHT_DECAY = 0.01      # Prevents the model from becoming too rigid
EPOCHS = 5               # More passes now that we have more data

def train_and_upload():
    print(f"--- Starting Professional Model Improvement (Threshold: {MIN_IMAGES_REQUIRED}) ---")
    
    # 2. Threshold Check
    total_images = sum([len(files) for r, d, files in os.walk(DATASET_PATH)])
    if total_images < MIN_IMAGES_REQUIRED:
        print(f"WAIT: Only {total_images} images collected. Need {MIN_IMAGES_REQUIRED} to proceed.")
        print("This prevents 'shocking' your computer and ensures the AI actually learns something useful.")
        return

    # 3. Data Augmentation (Research Parameter: Making the model robust)
    # This turns 50 images into hundreds of variations (angled, bright, dark, etc.)
    train_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),      # Handle angled photos
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Handle lighting
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Preparation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID, trust_remote_code=True)
        model.to(device).train()
        
        dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # 5. Training Loop
        print(f"Training on {total_images} images with Data Augmentation...")
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(loader):.4f}")

        # 6. Save locally for a moment
        model.save_pretrained(OUTPUT_DIR)
        
        # 7. AUTOMATIC UPLOAD TO HUGGING FACE
        token = os.getenv("HF_TOKEN")
        if not token:
            print("\n--- ACTION REQUIRED ---")
            token = input("Please enter your Hugging Face WRITE Token to upload: ")
        
        if token:
            print("Uploading improved model to Hugging Face...")
            api = HfApi()
            api.upload_folder(
                folder_path=OUTPUT_DIR,
                repo_id=MODEL_ID,
                token=token,
                commit_message=f"Auto-improvement from {total_images} user reports"
            )
            print("✓ SUCCESSFULLY REPLACED MODEL ON HUGGING FACE!")
            
            # 8. CLEANUP (Save your space)
            print("Cleaning up local files...")
            shutil.rmtree(OUTPUT_DIR)
            for label in ["Real", "Fake"]:
                for f in os.listdir(os.path.join(DATASET_PATH, label)):
                    os.remove(os.path.join(DATASET_PATH, label, f))
            print("✓ Local space cleared.")
            
        else:
            print("Upload skipped (No Token). Model is saved in ./temp_improved_model")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    train_and_upload()
