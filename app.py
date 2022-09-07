from fastapi import FastAPI, File, UploadFile
from src.utils import MessageModel, VideoModel

app = FastAPI()


@app.get("/")
def root():
    return "api running!"


@app.post("/text/")
async def text(message:str):
    message_model=MessageModel()
    pred=message_model.predict(message=message)
    
    return pred


@app.post("/predict_vido")
async def predict_vido(video_file: UploadFile = File(...)):
    video_model=VideoModel()
    prediction=video_model.predict_using_video(video_file)
    print(prediction)
    return prediction