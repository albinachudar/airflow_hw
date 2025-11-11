import os
from datetime import datetime

import dill
import json
import pandas as pd


# Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - текущая директория при локальном запуске
path = os.environ.get('PROJECT_PATH', '..')


def load_model():
    """Загружает обученную модель из папки models"""
    model_path = os.path.join(path, 'data/models')

    with open(
            os.path.join(
                model_path,
                os.listdir(model_path)[-1]),
            'rb'
    ) as file:
        model = dill.load(file)

    return model


def load_test_data() -> list:
    """Загружает все JSON файлы из папки test"""
    test_path = os.path.join(path, 'data/test')
    test_files = []

    for filename in os.listdir(test_path):
        file_path = os.path.join(test_path, filename)
        test_files.append(file_path)

    return test_files


def predict_single_file(json_file_path: str, model: object) -> dict:
    """Обрабатывает один JSON файл и возвращает предсказание"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Создаем DataFrame из данных
    df = pd.DataFrame([data])

    # Делаем предсказание
    prediction = model.predict(df)

    return {
        'id': df.id,
        'price_category': prediction[0],
        'price': df.price
    }


def predictions_to_csv(predictions: list) -> None:
    """Сохраняет предсказания в CSV файл"""
    df = pd.DataFrame(predictions)

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    output_file = os.path.join(
        path,
        'data/predictions',
        f'preds_{timestamp}.csv'
    )

    df.to_csv(output_file, index=False)


def predict() -> None:
    """Основная функция для выполнения предсказаний"""
    model = load_model()            # 1. Загружаем модель
    test_files = load_test_data()   # 2. Загружаем тестовые данные

    # 3. Обрабатываем каждый файл
    predictions = []
    for test_file in test_files:
        prediction = predict_single_file(test_file, model)
        predictions.append(prediction)

    # 4. Сохраняем результаты
    predictions_to_csv(predictions)


if __name__ == '__main__':
    predict()
