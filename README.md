# 🇮🇳 Indian Currency Detector: Self-Improving AI

This AI-powered app detects real vs. fake Indian currency using an EfficientNetV2-S model and a modern FastAPI web interface. It features an "Active Learning" loop where users report errors to build a feedback dataset. With SHA-256 duplicate detection and one-click local fine-tuning, the AI automatically evolves and improves over time.

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| **🤖 AI Detection** | Fine-tuned EfficientNetV2-S model for high accuracy authentication. |
| **💻 Modern UI** | Beautiful, dark-themed "Two-Page" web interface built with FastAPI & HTML. |
| **🔄 Dual Input** | Upload images directly from your device or paste web image URLs. |
| **🎯 Active Learning** | Built-in feedback buttons to report misclassifications and save them. |
| **🛡️ Smart Storage** | SHA-256 hashing prevents saving duplicate images in the feedback dataset. |
| **📈 Auto-Improvement** | One-click script to automatically retrain the AI on user feedback. |

---

## 📁 Project Structure

| File / Folder | Purpose |
| :--- | :--- |
| `app.py` | The main FastAPI application logic, endpoints, and model routing. |
| `templates/index.html` | The sleek user interface (frontend). |
| `improve_model.py` | The script used to retrain the AI using collected feedback images. |
| `requirements.txt` | List of all required Python libraries. |
| `feedback_dataset/` | *(Auto-created)* Stores images reported as incorrect for future learning. |
| `improved_model/` | *(Auto-created)* Stores the newly trained model after running the improver. |

---

## ⚙️ Installation

1. **Clone or Download** this repository.
2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows:
   venv\Scripts\activate
   
   # Mac/Linux:
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

Start the local server by running the following command in your terminal:

```bash
uvicorn app:app --reload
```

Then, open your web browser and go to: **http://127.0.0.1:8000**

---

## 🧠 The "Self-Learning" Workflow

This application doesn't just make predictions; it learns from its mistakes!

1. **Use the App**: Upload images and check the predictions.
2. **Report Errors**: If the AI makes a mistake, click the "It's Actually Real" or "It's Actually Fake" buttons. The image is saved locally (duplicates are ignored automatically).
3. **Retrain the AI**: Once you have collected **50+ images** in your `feedback_dataset`, stop the server and run:
   ```bash
   python improve_model.py
   ```
4. **Enjoy Better Accuracy**: The script will automatically retrain your model, update your files, and clean up the old dataset folder. Restart your app, and it will immediately use the smarter AI!

---

## ⚖️ Disclaimer
This application is for educational and initial screening purposes only. For official verification, always use legal banking channels.
