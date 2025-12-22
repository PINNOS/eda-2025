# Используем официальный образ Python 3.10
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /usr/src/app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы
COPY . .

# Открываем порт 5000 для Flask
EXPOSE 5000

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Команда для запуска приложения
CMD ["python", "app.py"]