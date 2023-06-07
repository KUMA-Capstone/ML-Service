from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import register_keras_serializable
from transformers import TFDistilBertForSequenceClassification

from preprocessing import preprocess_story

labels_dict = {
    0: "very bad",
    1: "bad",
    2: "neutral",
    3: "good",
    4: "very good",
}

register_keras_serializable(TFDistilBertForSequenceClassification)
# bikin fungsi buat ngeload semua lapisan2 dari bert yang bukan lapisan standar dari tf
def custom_objects():
    return {"TFDistilBertForSequenceClassification": TFDistilBertForSequenceClassification}

with keras.utils.custom_object_scope(custom_objects()):
    nlp_model = keras.models.load_model("emotion_detector_model.h5")

def predict_story(story):
    # Load the model
    story = preprocess_story(story)
    
    prediction = nlp_model.predict({
        'input_ids': story['input_ids'], 
        'attention_mask': story['attention_mask']
    })

    prediction = np.argmax(prediction,axis=1)
    return prediction[0]
