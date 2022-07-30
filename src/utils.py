import os
import re
from nltk.stem import WordNetLemmatizer
import pickle
import librosa
import numpy as np
import pandas as pd
from fer import FER
from fer import Video
import librosa.display
from tqdm import tqdm
from keras.utils import np_utils
import moviepy.editor as mp
from matplotlib.font_manager import json_load
from nltk.stem import WordNetLemmatizer
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder


# text classification
class MessageModel():
    class_name = os.path.basename(__file__)

    def __init__(self) -> None:
        self.wo = WordNetLemmatizer()
        self.mnb=pickle.load(open("models/message_model/prediction.pkl",'rb'))
        self.vectorizer=pickle.load(open("models/message_model/vectorizer.pkl",'rb'))


    def preprocess(self,data):
    #preprocess
        a = re.sub('[^a-zA-Z]',' ',data)
        a = a.lower()
        a = a.split()
        a = [self.wo.lemmatize(word) for word in a ]
        a = ' '.join(a)  
        return a


    def predict(self,message):
        a = self.preprocess(message)
        example_counts = self.vectorizer.transform([a])
        prediction = self.mnb.predict(example_counts)
        if prediction[0]==0:
            return "positive"
        elif prediction[0]==1:
            return "depressive"
        

# video classification
class VideoModel():
    class_name = os.path.basename(__file__)

    def __init__(self) -> None:
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


    def predict_using_video(self,video_path):
        # But the Face detection detector
        face_detector = FER(mtcnn=True)
        # Input the video for processing
        input_video = Video(video_path)
        processing_data = input_video.analyze(face_detector, display=False)
        vid_df = input_video.to_pandas(processing_data)
        vid_df = input_video.get_first_face(vid_df)
        vid_df = input_video.get_emotions(vid_df)
        # We will now work on the dataframe to extract which emotion was prominent in the video
        angry = sum(vid_df.angry)
        disgust = sum(vid_df.disgust)
        fear = sum(vid_df.fear)
        happy = sum(vid_df.happy)
        sad = sum(vid_df.sad)
        surprise = sum(vid_df.surprise)
        neutral = sum(vid_df.neutral)

        emotions_values = [angry, disgust, fear, happy, sad, surprise, neutral]

        score_comparisons = pd.DataFrame(self.emotions, columns = ['Human Emotions'])
        score_comparisons['Emotion Value from the Video'] = emotions_values

        return score_comparisons


# audio classification
class AudioModel():
    class_name = os.path.basename(__file__)

    def __init__(self) -> None:
        my_clip = mp.VideoFileClip("data/data.mp4")
        my_clip.audio.write_audiofile("data/v1_audio.wav")

        json_file = open('models/voice_model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        self.loaded_model.load_weights("models/voice_model/aug_noiseNshift_2class2_np.h5")
    

    def predict_audio(self, audio_path):
        model = self.loaded_model
        data_test = pd.DataFrame(columns=['feature'])

        X, sample_rate = librosa.load(audio_path , res_type='kaiser_fast',duration=3,sr=22050*2,offset=0.5)
        #     X = X[10000:90000]
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        feature = mfccs
        data_test.loc[0] = [feature]
            
        test_valid = pd.DataFrame(data_test['feature'].values.tolist())
        test_valid = np.array(test_valid)
        # test_valid_lb = np.array(data2_df.label)
        lb = LabelEncoder()
        # test_valid_lb = np_utils.to_categorical(lb.fit_transform(test_valid_lb))
        test_valid = np.expand_dims(test_valid, axis=2)

        preds = model.predict(test_valid, 
                                batch_size=16, 
                                verbose=1)
        preds1=preds.argmax(axis=1)
        abc = preds1.astype(int).flatten()
        # predictions = (lb.inverse_transform((abc)))
        if abc == 0:
            label = 'Depressive'
        elif abc == 1:
            label = 'Positive'
        return (label)

