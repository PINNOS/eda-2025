# preprocess_data.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_titanic_data():
    # Загружаем данные
    df = pd.read_csv('titanic.csv')
    
    # Заполняем пропуски в возрасте медианой
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Заполняем пропуски в порту посадки
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # Создаем признак наличия каюты
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    
    # Создаем размер семьи
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Создаем признак "одинокий"
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Извлекаем титул из имени
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Группируем редкие титулы
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Кодируем категориальные признаки
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    # Кодируем титулы
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    
    # Выбираем финальные признаки
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                'Embarked', 'HasCabin', 'FamilySize', 'IsAlone', 'Title']
    
    # Сохраняем предобработанные данные
    df[features + ['Survived']].to_csv('preprocessed_data.csv', index=False)
    
    print("Данные предобработаны и сохранены в preprocessed_data.csv")

if __name__ == '__main__':
    preprocess_titanic_data()