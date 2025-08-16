Run:
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample.json
