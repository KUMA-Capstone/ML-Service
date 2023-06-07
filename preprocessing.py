import pandas as pd
import numpy as np

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("LawalAfeez/emotion_detection")

def split_activities(data, col_activities):
    data_new = data[col_activities].str.split(' \| ', expand=True)
    data_new.columns = ['activity_'+str(i+1) for i in range(data_new.shape[1])]

    return data_new

def combine_to_first_data_and_drop(first_data, second_data, drop_feature):
    df_combined = first_data.join(second_data)
    df_combined.drop(drop_feature, axis=1, inplace=True)
    return df_combined

def add_new_feature_is_weekend(data, new_feature):
    data[new_feature] = np.where((data['weekday'] == 'Saturday') | (data['weekday'] == 'Sunday'), 1, 0)
    return data

def encoding(data):
    data = pd.get_dummies(data)
    return data

def add_columns_activity_data(data, list_columns):
    for num in range(1,9):
        for item in list_columns:
            if 'activity_'+str(num)+'_'+item not in data.columns:
                data['activity_'+str(num)+'_'+item] = 0
            if num > 1:
                data['activity_'+str(num)+'_'+'0'] = 0
    return data

def add_columns_submood_data(data, list_columns):
    for item in list_columns:
        if 'sub_mood_'+item not in data.columns:
            data['sub_mood_'+item] = 0
    return data

def add_columns_weekday_data(data, list_columns):
    for item in list_columns:
        if 'weekday_'+item not in data.columns:
            data['weekday_'+item] = 0
    return data

def preprocess_survey(data):
    df_new = split_activities(data, 'activities')

    # gabung data
    list_drop = ['activities', 'full_date', 'date', 'time']
    df_new = combine_to_first_data_and_drop(data, df_new, list_drop)

    # tambah fitur is_weekend
    df_new = add_new_feature_is_weekend(df_new, 'is_weekend')

    # encode data
    df_new = encoding(df_new)

    # menambahkan kolom aktivitas
    list_of_activity = ['Reading and Learning', 'Spiritual','Social', 'Physical and Travel', 'Self-pleasure and Entertainment',
                        'Creative', 'Home', 'Other']
    df_new = add_columns_activity_data(df_new, list_of_activity)

    # menambahkan kolom submood
    list_of_submood = ['Yolo','Focused','Confused','Wondering','Angry','Blessed','Excited','Chill','Hungry','Happiest day',
                       'Weak','Meh','Awful','Cool','Worried','Over the moon','Triggered','Sad af','Scared','Good','Bad','Sick']
    df_new = add_columns_submood_data(df_new, list_of_submood)

    # menambahkan kolom weekday
    list_of_day = ['Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday', 'Sunday', 'Saturday']
    df_new = add_columns_weekday_data(df_new, list_of_day)
    
    return df_new

def preprocess_story(story):
    MAX_TEXT_LENGTH = 90
    story = tokenizer(text=story,
            add_special_tokens=True,
            padding="max_length",
            truncation=False,
            max_length=MAX_TEXT_LENGTH,
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True,
            verbose=1)
    
    return story