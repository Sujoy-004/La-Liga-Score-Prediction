#!/bin/sh
# Start the FastAPI backend in the background
echo "Starting FastAPI Backend..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Wait for the backend to initialize (optional but safer)
sleep 5

# Start the Dash frontend (listening on port 7860 for Hugging Face Spaces)
echo "Starting Dash Frontend..."
python src/Dashboard.py --port 7860
