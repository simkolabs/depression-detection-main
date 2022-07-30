from src.utils import AudioModel

audio=AudioModel()

response, preds, abc =audio.predict_audio("data/v1_audio.wav")
print(f"response : {response}\n")
print(f"preds :\n{preds}\n")
print(f"abc :\n{abc}")