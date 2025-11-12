# facerec-backend

Backend package for YOLO + FER face emotion detection with MySQL storage.

## What is included
- app.py : Streamlit app + FastAPI `/predict` endpoint
- db.py : SQLAlchemy helper (insert/list/delete)
- requirements.txt : Python dependencies

## Before running
1. Do NOT commit secrets. Use environment variables.
2. Set these environment variables (in your deployment platform or local terminal):
   - DATABASE_URL (e.g. mysql+pymysql://root:PASSWORD@HOST:PORT/DATABASE)
   - CLOUDINARY_CLOUD_NAME
   - CLOUDINARY_API_KEY
   - CLOUDINARY_API_SECRET
   - (optional) MODEL_PATH - path to your YOLO model file inside the container or image

## Local test
1. Create a Python virtualenv and install deps:
   ```
   pip install -r requirements.txt
   ```
2. Set environment variables in your terminal (Windows CMD example):
   ```
   set DATABASE_URL=mysql+pymysql://root:PASSWORD@shuttle.proxy.rlwy.net:13169/railway
   set CLOUDINARY_CLOUD_NAME=...
   set CLOUDINARY_API_KEY=...
   set CLOUDINARY_API_SECRET=...
   ```
3. Run Streamlit:
   ```
   streamlit run app.py
   ```

## Deploy
- Push this repo to GitHub, then connect to Hugging Face Spaces (or Railway).  
- Add environment variables as repository secrets in Hugging Face / Railway.
- Rebuild the Space / Deploy the service.