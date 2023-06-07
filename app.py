import streamlit as st

import numpy as np

import pandas as pd

from functions.import_export import import_data, export_data
from functions.preprocessing import preprocessing_vic_or_def
from functions.training_model import train_model

# -- FUNCTIONS -- 


# Pre-training


st.title('- DATA - ')

st.subheader('Utiliser l\'actuel jeu de données...')

data_load_state = st.text('Loading data...')

data = import_data()

data_load_state.text("Data loaded")

if st.button('-> Visualiser les données brutes'):
    st.subheader('Raw data')
    st.write(data)


st.subheader('OU importer de nouvelles données complètes pour un nouvel entrainement...')

if st.checkbox('Re-upload new data'):

    st.text('Le titre du fichier xlsx (obligatoire) : "score_baby" ')

    st.text('Nom des colonnes (dans cet ordre, obligatoire) : N_att - N_def - B_att - B_def - N_score - B_score')

    uploaded_file = st.file_uploader('Upload a file')

    if st.button('-> Envoyer le fichier'):

        if uploaded_file is not None:

            data = pd.read_excel(uploaded_file)

            try:
                data.drop(['Unnamed: 0'], axis=1, inplace=True)
            except:
                pass

            status_result = export_data(data)

            if status_result==200:

                st.write('Send OK !...')

            else:

                st.write(f'Problem... Request status : {status_result}')
            
        else:

            st.write("Oups ! .... Avez-vous chargé le fichier ?")

st.divider()

st.title('- TRAINING - ')

if st.button('Ré-entrainer le modèle'):

    st.write('Please wait....')

    X, Y, preprocessor = preprocessing_vic_or_def(data)

    st.write('start training....')

    model, train_accuracy, test_accuracy, total_accuracy = train_model(X, Y)

    st.write(f'train_data_accuracy : {np.round(train_accuracy*100, 4)} %')

    st.write(f'test_data_accuracy : {np.round(test_accuracy*100, 4)} %')

    st.write(f'total_data_training_accuracy : {np.round(total_accuracy*100, 4)} %')

st.divider()

st.title('- PREDICTIONS - ')

N_att_list = data['N_att'].unique()

N_def_list = data['N_def'].unique()

B_att_list = data['B_att'].unique()

B_def_list = data['N_def'].unique()

N_att = st.selectbox('Noir - attaquant', N_att_list)
N_def = st.selectbox('Noir - défenseur', N_def_list)
B_att = st.selectbox('Blanc - attaquant', B_att_list)
B_def = st.selectbox('Blanc - défenseur', B_def_list)


if st.button('Tenter une prédiction'):


    df = pd.DataFrame({'N_att':[N_att],
                           'N_def':[N_def],
                           'B_att':[B_att],
                           'B_def':[B_def]})

    _, _, preprocessor = preprocessing_vic_or_def(data)

    X_input = preprocessor.transform(df)

    result = model.predict(X_input)

    if result < 0.5:
        st.write('Victoire des Blancs !...')

    else:
        st.write('Victoire des noirs !...')






