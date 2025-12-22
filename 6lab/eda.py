# eda.py
import pandas as pd
import matplotlib.pyplot as plt

def perform_eda():
    # Загружаем данные
    df = pd.read_csv('titanic.csv')
    
    # Базовая информация
    print("=== ОСНОВНАЯ ИНФОРМАЦИЯ ===")
    print(f"Размер датасета: {df.shape}")
    print(f"\nТипы данных:")
    print(df.dtypes)
    
    # Пропущенные значения
    print(f"\nПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
    print(df.isnull().sum())
    
    # Статистика выживших
    print(f"\nСТАТИСТИКА ВЫЖИВШИХ:")
    survived_stats = df['Survived'].value_counts()
    print(f"Выжили: {survived_stats[1]} ({survived_stats[1]/len(df)*100:.1f}%)")
    print(f"Не выжили: {survived_stats[0]} ({survived_stats[0]/len(df)*100:.1f}%)")
    
    # Выживаемость по классу
    print(f"\nВЫЖИВАЕМОСТЬ ПО КЛАССУ:")
    for pclass in sorted(df['Pclass'].unique()):
        class_df = df[df['Pclass'] == pclass]
        survived = class_df['Survived'].sum()
        total = len(class_df)
        print(f"Класс {pclass}: {survived}/{total} ({survived/total*100:.1f}%)")
    
    # Выживаемость по полу
    print(f"\nВЫЖИВАЕМОСТЬ ПО ПОЛУ:")
    for sex in ['male', 'female']:
        sex_df = df[df['Sex'] == sex]
        survived = sex_df['Survived'].sum()
        total = len(sex_df)
        print(f"{sex}: {survived}/{total} ({survived/total*100:.1f}%)")
    
    # Создаем простую визуализацию
    plt.figure(figsize=(12, 4))
    
    # График 1: Выживаемость по классу
    plt.subplot(1, 3, 1)
    df.groupby('Pclass')['Survived'].mean().plot(kind='bar', color=['red', 'blue', 'green'])
    plt.title('Выживаемость по классу')
    plt.xlabel('Класс')
    plt.ylabel('Доля выживших')
    
    # График 2: Выживаемость по полу
    plt.subplot(1, 3, 2)
    df.groupby('Sex')['Survived'].mean().plot(kind='bar', color=['blue', 'pink'])
    plt.title('Выживаемость по полу')
    plt.xlabel('Пол')
    plt.ylabel('Доля выживших')
    
    # График 3: Распределение возраста
    plt.subplot(1, 3, 3)
    df['Age'].hist(bins=30, alpha=0.7, color='gray')
    plt.title('Распределение возраста')
    plt.xlabel('Возраст')
    plt.ylabel('Количество')
    
    plt.tight_layout()
    plt.savefig('eda_results.png')
    print(f"\nВизуализация сохранена в eda_results.png")
    
    # Предобрабатываем данные для модели
    from preprocess_data import preprocess_titanic_data
    preprocess_titanic_data()
    
    print("\n=== EDA ЗАВЕРШЕНА ===")

if __name__ == '__main__':
    perform_eda()