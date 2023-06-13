from tensorflow import keras

import pandas as pd
import numpy as np

from datetime import datetime as date

from preprocessing import preprocess_survey

model_ann = keras.models.load_model("model_ann.h5")

def predict_survey(activities, sub_mood):
    day_name = date.today().strftime("%A")
    lst_inp = [
        [""], 
        [""], 
        [day_name], 
        [""], 
        [sub_mood], 
        [activities],
    ]
    df_inp = pd.DataFrame(lst_inp).transpose()
    df_inp.columns=['full_date', 'date', 'weekday', 'time', 'sub_mood', 'activities']

    result = preprocess_survey(df_inp)[list(X_train.columns)]
    for col in result.columns:
        if result[col].dtype == bool:
            result[col] = result[col].astype(np.uint)

    result = result.to_numpy()
    pred = model_ann.predict(result)
    return np.argmax(pred, axis=1)[0]