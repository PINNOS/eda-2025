# model.py
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

def train_model():
    # Загружаем предобработанные данные
    data = pd.read_csv('preprocessed_data.csv')
    print(f"Загружено строк: {len(data)}")
    
    # Разделяем на признаки и целевую переменную
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    # Разделяем на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Тренировочная выборка: {X_train.shape}")
    print(f"Тестовая выборка: {X_test.shape}")
    
    # Определяем категориальные признаки (индексы столбцов)
    categorical_features = [1, 6]  # Sex и Embarked (индексы начиная с 0)
    
    # Создаем и обучаем модель
    print("\nОбучение модели CatBoost...")
    model = CatBoostClassifier(
        iterations=300,           # Количество итераций
        learning_rate=0.1,        # Скорость обучения
        depth=4,                  # Глубина деревьев
        random_seed=42,           # Для воспроизводимости
        verbose=50                # Вывод прогресса каждые 50 итераций
    )
    
    # Обучаем модель
    model.fit(
        X_train, y_train,
        cat_features=categorical_features,
        eval_set=(X_test, y_test),
        verbose=True
    )
    
    # Оцениваем модель
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nРЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
    print(f"Точность на тренировочных данных: {train_score:.4f}")
    print(f"Точность на тестовых данных: {test_score:.4f}")
    
    # Сохраняем модель
    model.save_model('trained_model.cbm')
    print(f"Модель сохранена в trained_model.cbm")
    
    # Сохраняем список признаков (нужно для предсказания)
    feature_names = list(X.columns)
    with open('feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"Список признаков сохранен в feature_names.txt")
    
    return model

if __name__ == '__main__':
    train_model()