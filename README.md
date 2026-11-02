# Skin Condition Image Classifier

A Streamlit web application that runs the uploaded CustomCNN PyTorch model.

## Files required in the GitHub repository root

- `app.py` — Streamlit application and the exact CustomCNN architecture.
- `requirements.txt` — Python dependencies.
- `custom_skin_cnn.pth` — trained model weights (provided separately in this project).
- `README.md` — setup and deployment instructions.
- `.gitignore` — excludes local Python and environment files.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and `custom_skin_cnn.pth` to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
5. Deploy.

## Model classes

The class order follows the alphabetical `ImageFolder` order used by the training code:
`Herpes`, `Melanoma`, `Monkeypox`, `Sarampion`, `Varicela`.

## Important

This model is an educational image-classification tool, not a medical diagnostic device.
Its predictions may be incorrect and must not replace assessment by a qualified
healthcare professional. The training script used a small number of epochs, so
model performance should be independently evaluated before any real-world use.
