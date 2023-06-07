from flask import Flask, request

from ann import predict_survey
from nlp import predict_story


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def coba():
    data = request.json
    activities = data['activities']
    sub_mood = data['sub_mood']
    story = data['story']
    
    survey_prediction = predict_survey(activities, sub_mood)
    story_prediction = predict_story(story)

    result = ((survey_prediction * 0.3) + (story_prediction * 0.7))/2
    return {
        "status": 200,
        "message": "Predictions were successfully!",
        "prediction": result
    }