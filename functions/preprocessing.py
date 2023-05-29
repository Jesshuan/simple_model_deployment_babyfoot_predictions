import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder



def remove_space(string):
    return string.strip()

def duplicates_row_without_position(df):
    df['N_att_2']=df['B_att']
    df['N_def_2']=df['B_def']
    df['B_att_2']=df['N_att']
    df['B_def_2']=df['N_def']
    df['Victoire_2']=df['Victoire'].map({'N':'B', 'B':'N'})

    df_1 = df[['N_att', 'N_def', 'B_att', 'B_def', 'Victoire']]
    df_2 = df[['N_att_2', 'N_def_2', 'B_att_2', 'B_def_2', 'Victoire_2']]

    df_2.rename(columns={'N_att_2':'N_att', 'N_def_2':'N_def', 'B_att_2':'B_att', 'B_def_2':'B_def', 'Victoire_2':'Victoire'}, inplace=True)

    return pd.concat([df_1, df_2], axis=0)




def preprocessing_vic_or_def(data):

    features = ['N_att', 'N_def', 'B_att', 'B_def']

    for col in features:
        data[col] = data[col].apply(remove_space)

    data['Victoire'] = data['N_score'].map(lambda s : 'N' if s==10 else 'B')

    data = duplicates_row_without_position(data)

    preprocessor_target = LabelEncoder()
    Y = preprocessor_target.fit_transform(data['Victoire'])

    preprocessor = OneHotEncoder(drop='first')
    X = preprocessor.fit_transform(data[features])

    

    return X, Y, preprocessor

