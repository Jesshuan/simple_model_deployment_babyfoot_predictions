
import boto3

import pandas as pd

import os

from io import StringIO

from tensorflow import keras

BUCKET_NAME = 'babyfoot'
KEY_FILE_DATA = "score_baby.csv"
KEY_FILE_MODEL = "model_baby.h5"
ACCESS_KEY_ID = os.environ['ACCESS_KEY_ID']
SECRET_ACCESS_KEY = os.environ['SECRET_ACCESS_KEY']

def import_data():

    s3_client = boto3.client(
        "s3",
        aws_access_key_id= ACCESS_KEY_ID ,
        aws_secret_access_key= SECRET_ACCESS_KEY
    )

    try:

        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=KEY_FILE_DATA)

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            data = pd.read_csv(response.get("Body"))

            try:
                data.drop(['Unnamed: 0'], axis=1, inplace=True)
            except:
                pass

            return data

        else:
            return pd.DataFrame()
        
    except:
        return pd.DataFrame()
    
def export_data(data):
    '''
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY
    )
    '''

    #Creating Session With Boto3.
    session = boto3.Session(
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY
    )
    #Creating S3 Resource From the Session.
    s3_res = session.resource('s3')

    csv_buffer = StringIO()
    data.to_csv(csv_buffer)
    response = s3_res.Object(BUCKET_NAME, KEY_FILE_DATA).put(Body=csv_buffer.getvalue())

    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    return status


def import_model():

    s3_client = boto3.client(
        "s3",
        aws_access_key_id= ACCESS_KEY_ID ,
        aws_secret_access_key= SECRET_ACCESS_KEY
    )

    try:

        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=KEY_FILE_MODEL)

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            model = keras.models.load_model(response.get("Body"))

            return model

        else:
            return None
        
    except:
        return None