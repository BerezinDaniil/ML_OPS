# ML_OPS

Представленна модель, предсказывающая победу одной из двух сторонв в игре CS-GO(данные - https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification)

Конфигурации каждого этапа хранятся в папке configs.

# Инициализация проекта
`poetry install`
# Запуск севера MLFlow
`poetry run mlflow server --host 127.0.0.1 --port 8080`
# train
`poetry run python3 ML_OPS/train.py`
# test
`poetry run python3 ML_OPS/infer.py`
#Запуск сервера
`poetry run python3 ML_OPS/run_server.py`
