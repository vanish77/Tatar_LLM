# Быстрый старт за 5 минут

Краткая инструкция для быстрого запуска проекта.

## Шаг 1: Установка (1 мин)

```bash
# Установите зависимости
pip install -r requirements.txt
```

## Шаг 2: Подготовка данных (2-5 мин)

```bash
# Загрузка и подготовка татарского корпуса
python 01_prepare_data.py
```

Если загрузка не работает, скачайте вручную с:
https://wortschatz.uni-leipzig.de/en/download/Tatar

## Шаг 3: Обучение токенизатора (1 мин)

```bash
python 02_train_tokenizer.py
```

## Шаг 4: Обучение модели (30-60 мин)

```bash
# Стандартное обучение
python 03_train.py

# Или быстрое (для тестирования, ~10 мин)
# Измените max_iters на 2000 в 03_train.py
```

## Шаг 5: Тестирование (1 мин)

```bash
# Быстрый тест
python quick_test.py

# Или откройте Jupyter ноутбук
jupyter notebook demo_inference.ipynb

# Или интерактивный режим
python utils.py
```

## Альтернатива: Полный автоматический пайплайн

```bash
# Запустить всё одной командой (Mac/Linux)
chmod +x run_pipeline.sh
./run_pipeline.sh
```

## Если не хватает памяти

В файле `03_train.py` измените:

```python
batch_size = 16  # было 32
use_gradient_checkpointing = True  # было False
n_layer = 4  # было 6
n_embd = 256  # было 512
```

## Если хотите быстрее обучить

```python
max_iters = 3000  # было 10000
eval_interval = 200  # было 500
```

## Проверка результатов

После обучения у вас должны быть:

```
? data/processed/train.txt - обучающие данные
? data/processed/val.txt - валидационные данные
? tokenizer/tokenizer.json - обученный токенизатор
? models/best_model.pt - лучшая модель
? models/final_model.pt - финальная модель
```

## Тестовая генерация

```python
from utils import load_model, load_tokenizer, generate_text

model, _ = load_model("models/best_model.pt")
tokenizer, tok_config = load_tokenizer()

text = generate_text(model, tokenizer, tok_config, "Казан ш???ре")
print(text)
```

## Частые проблемы

### Ошибка "No module named ..."
```bash
pip install -r requirements.txt
```

### Ошибка "File not found: models/best_model.pt"
Сначала запустите обучение: `python 03_train.py`

### MPS backend out of memory
Уменьшите `batch_size` в `03_train.py`

### Слишком долго обучается
Уменьшите `max_iters` или используйте конфигурацию `tiny`

## Что дальше?

1. Прочитайте [README.md](README.md) для подробной информации
2. Посмотрите [EXAMPLES.md](EXAMPLES.md) для примеров работы
3. Экспериментируйте с гиперпараметрами в [config.py](config.py)
4. Пробуйте разные промпты в `demo_inference.ipynb`

Удачи! ??


