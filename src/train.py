import pandas as pd
from sklearn.preprocessing import StandardScaler

class train_model():

    def __init__(self, database):

        self.database = database
    
    def process_data(data):

        data['sex'].fillna(data['sex'].mode()[0], inplace=True)

        col_to_be_imputed = ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g']
        for item in col_to_be_imputed:
            data[item].fillna(data[item].mean(),inplace=True)
        
        data['species']=data['species'].map({'Adelie':0,'Gentoo':1,'Chinstrap':2})

        # creating dummy variables for categorical features
        dummies = pd.get_dummies(data[['island','sex']],drop_first=True)

        # we do not standardize dummy variables 
        df_to_be_scaled = data.drop(['island','sex'],axis=1)
        target = df_to_be_scaled.species
        df_feat= df_to_be_scaled.drop('species',axis=1)

        scaler = StandardScaler()
        scaler.fit(df_feat)
        df_scaled = scaler.transform(df_feat)
        df_scaled = pd.DataFrame(df_scaled,columns=df_feat.columns[:4])
        df_preprocessed = pd.concat([df_scaled,dummies,target],axis=1)
        df_preprocessed.head()

        return df_preprocessed

    def train_model()





data = pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')

data['sex'].fillna(data['sex'].mode()[0],inplace=True)
col_to_be_imputed = ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g']
for item in col_to_be_imputed:
    data[item].fillna(data[item].mean(),inplace=True)

data['species']=data['species'].map({'Adelie':0,'Gentoo':1,'Chinstrap':2})

# creating dummy variables for categorical features
dummies = pd.get_dummies(data[['island','sex']],drop_first=True)

# we do not standardize dummy variables 
df_to_be_scaled = data.drop(['island','sex'],axis=1)
target = df_to_be_scaled.species
df_feat= df_to_be_scaled.drop('species',axis=1)


scaler = StandardScaler()
scaler.fit(df_feat)
df_scaled = scaler.transform(df_feat)
df_scaled = pd.DataFrame(df_scaled,columns=df_feat.columns[:4])
df_preprocessed = pd.concat([df_scaled,dummies,target],axis=1)
df_preprocessed.head()