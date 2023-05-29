
import boto3

import pandas as pd

import os

from io import StringIO

import tensorflow as tf

from tensorflow import keras


BUCKET_NAME = 'babyfoot'
KEY_FILE_DATA = "score_for_me.csv"
KEY_FILE_MODEL = "model_baby.h5"
ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")



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
            data = pd.read_csv(response.get("Body"), sep=',')

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

    #Creating Session With Boto3.
    session = boto3.Session(
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY
    )
    #Creating S3 Resource From the Session.
    s3_res = session.resource('s3')

    csv_buffer = StringIO()
    data.to_csv(csv_buffer, sep=',')
    response = s3_res.Object(BUCKET_NAME, KEY_FILE_DATA).put(Body=csv_buffer.getvalue())

    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    return status


def export_model(export_path, name_on_s3):


    s3_client = boto3.client(
        "s3",
        aws_access_key_id= ACCESS_KEY_ID ,
        aws_secret_access_key= SECRET_ACCESS_KEY
    )

    with open(export_path, 'rb') as file:
        s3_client.upload_fileobj(file, BUCKET_NAME, name_on_s3)

    print('Model uploaded to S3 successfully!')


def import_model(local_path, name_on_s3):

    s3_client = boto3.client(
        "s3",
        aws_access_key_id= ACCESS_KEY_ID ,
        aws_secret_access_key= SECRET_ACCESS_KEY
    )

    s3_client.download_file(BUCKET_NAME, name_on_s3, local_path)

# Load the model using TensorFlow
    return tf.keras.models.load_model(local_path)

    



