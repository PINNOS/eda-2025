# app.py
from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier

app = Flask(__name__)

# Глобальные переменные
model = None
feature_names = []

def load_model():
    """Загрузка обученной модели"""
    global model, feature_names
    
    try:
        # Загружаем модель
        model = CatBoostClassifier()
        model.load_model('trained_model.cbm')
        
        # Загружаем названия признаков
        with open('feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
        
        print(f"Модель загружена. Признаки: {feature_names}")
        return True
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return False

def preprocess_input(data):
    """Предобработка входных данных для модели"""
    # Создаем DataFrame из входных данных
    df = pd.DataFrame([data])
    
    # Заполняем пропуски
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(29.7)  # медиана
    
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(14.45)  # медиана
    
    # Кодируем категориальные признаки
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Sex'] = df['Sex'].fillna(0)
    
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        df['Embarked'] = df['Embarked'].fillna(2)
    
    # Создаем дополнительные признаки как в обучении
    df['HasCabin'] = 0  # по умолчанию нет каюты
    if 'Cabin' in df.columns and pd.notna(df['Cabin'].iloc[0]) and df['Cabin'].iloc[0] != '':
        df['HasCabin'] = 1
    
    # Размер семьи
    sibsp = df.get('SibSp', pd.Series([0])).iloc[0]
    parch = df.get('Parch', pd.Series([0])).iloc[0]
    df['FamilySize'] = sibsp + parch + 1
    df['IsAlone'] = 1 if df['FamilySize'].iloc[0] == 1 else 0
    
    # Титул из имени
    title = 'Mr'  # по умолчанию
    if 'Name' in df.columns:
        name = df['Name'].iloc[0]
        if 'Mrs' in name or 'Lady' in name:
            title = 'Mrs'
        elif 'Miss' in name or 'Ms' in name:
            title = 'Miss'
        elif 'Master' in name:
            title = 'Master'
        elif 'Dr' in name:
            title = 'Rare'
    
    # Кодируем титул
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = title_mapping.get(title, 1)
    
    # Выбираем только нужные признаки в правильном порядке
    result_df = pd.DataFrame(columns=feature_names)
    for feature in feature_names:
        if feature in df.columns:
            result_df[feature] = df[feature]
        else:
            result_df[feature] = 0  # заполняем нулями если признака нет
    
    return result_df

# Загружаем модель при старте
load_model()

@app.route('/')
def home():
    """Главная страница"""
    return jsonify({
        "message": "Titanic Survival Prediction API",
        "status": "ready" if model else "model not loaded",
        "endpoints": ["/predict", "/health"]
    })

@app.route('/health')
def health():
    """Проверка работоспособности"""
    if model:
        return jsonify({"status": "healthy", "model_loaded": True})
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Предсказание выживаемости"""
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Получаем данные из запроса
        data = request.json
        
        # Предобрабатываем данные
        df = preprocess_input(data)
        
        # Делаем предсказание
        prediction = model.predict(df)
        probability = model.predict_proba(df)
        
        # Формируем ответ
        result = {
            "survived": bool(prediction[0]),
            "probability_survived": float(probability[0][1]),
            "probability_died": float(probability[0][0]),
            "passenger_id": data.get('PassengerId', 'unknown')
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Запускаем Flask сервер
    app.run(host='0.0.0.0', port=5000, debug=False)