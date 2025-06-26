from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from model import VQADemo
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin for production
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "backend/temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize model once
model_path = 'models/best_vqa_model.pth'
vqa = VQADemo(model_path=model_path, answers_vocab_path='data/answer_vocab.json')

@app.post("/predict")
async def predict(image: UploadFile = File(...), question: str = Form(...)):
    try:
        # Save uploaded image to disk
        file_location = os.path.join(UPLOAD_FOLDER, image.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Get answer from model
        answer = vqa.get_answer(file_location, question)

        # Remove the uploaded file
        os.remove(file_location)

        return JSONResponse(content={"question": question, "answer": answer})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=True)
