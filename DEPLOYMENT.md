# Развертывание модели

Инструкции по развертыванию татарской LLM на различных платформах.

## ?? Hugging Face Hub

### Загрузка модели

```bash
# Установите huggingface_hub
pip install huggingface_hub

# Авторизуйтесь
huggingface-cli login

# Создайте репозиторий (через веб-интерфейс или CLI)
# https://huggingface.co/new

# Загрузите модель
python -c "
from huggingface_hub import HfApi
api = HfApi()

# Загрузить модель
api.upload_file(
    path_or_fileobj='models/best_model.pt',
    path_in_repo='best_model.pt',
    repo_id='your-username/tatar-gpt',
    repo_type='model'
)

# Загрузить токенизатор
api.upload_folder(
    folder_path='tokenizer',
    repo_id='your-username/tatar-gpt',
    path_in_repo='tokenizer',
    repo_type='model'
)
"
```

### Создание Model Card

Создайте `README.md` в репозитории на HF:

```markdown
---
language: tt
license: mit
tags:
- text-generation
- gpt
- tatar
- low-resource
- decoder-only
widget:
- text: "Казан ш???ре"
---

# Tatar GPT

Small GPT model trained from scratch on Tatar language.

## Model Description

- **Language**: Tatar
- **Architecture**: GPT (decoder-only, 6 layers, 8 heads, 512 dim)
- **Parameters**: ~23M
- **Context**: 256 tokens
- **Tokenizer**: BPE (8192 vocab)

## Training Data

Leipzig Wortschatz Tatar corpus (~100MB)

## Usage

[См. README в репозитории GitHub]

## Limitations

- Small context window (256 tokens)
- May generate hallucinations
- Not fine-tuned for specific tasks
```

## ?? Hugging Face Spaces

### Создание демо

1. Создайте новый Space: https://huggingface.co/spaces
2. Выберите Gradio SDK
3. Загрузите файлы:

```
your-space/
??? app.py              # Gradio интерфейс
??? requirements.txt    # Зависимости
??? model.py           # Архитектура модели
??? utils.py           # Утилиты
??? README.md          # Описание
```

4. Скачайте модель в Space:

```python
# В app.py добавьте загрузку с HF Hub
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-username/tatar-gpt",
    filename="best_model.pt"
)
```

### requirements.txt для Spaces

```txt
torch
tokenizers
gradio
huggingface_hub
```

## ?? Google Colab

### Шаг 1: Создайте ноутбук

```python
# Установка зависимостей
!pip install torch tokenizers gradio

# Клонирование репозитория
!git clone https://github.com/your-username/tatar-llm.git
%cd tatar-llm

# Загрузка модели
!wget https://your-link-to-model/best_model.pt -P models/

# Запуск
from utils import load_model, load_tokenizer, generate_text

model, _ = load_model("models/best_model.pt")
tokenizer, tok_config = load_tokenizer()

text = generate_text(model, tokenizer, tok_config, "Казан ш???ре")
print(text)
```

### Шаг 2: Gradio интерфейс

```python
# В конце ноутбука
!python app.py --share
```

Получите публичную ссылку для демонстрации!

## ?? Docker

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Загрузка модели (или смонтировать volume)
# COPY models/best_model.pt models/
# COPY tokenizer/ tokenizer/

EXPOSE 7860

CMD ["python", "app.py"]
```

### docker-compose.yml

```yaml
version: '3'
services:
  tatar-gpt:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
      - ./tokenizer:/app/tokenizer
    environment:
      - GRADIO_SERVER_NAME=0.0.0.0
```

### Запуск

```bash
# Сборка
docker build -t tatar-gpt .

# Запуск
docker run -p 7860:7860 -v $(pwd)/models:/app/models tatar-gpt

# Или с docker-compose
docker-compose up
```

## ?? Streamlit Cloud

### app_streamlit.py

```python
import streamlit as st
import torch
from utils import load_model, load_tokenizer, generate_text

st.title("???? Татарская LLM")

# Кэширование загрузки модели
@st.cache_resource
def load():
    model, _ = load_model("models/best_model.pt")
    tokenizer, tok_config = load_tokenizer()
    return model, tokenizer, tok_config

model, tokenizer, tok_config = load()

# Интерфейс
prompt = st.text_area("Промпт на татарском:", "Казан ш???ре")
temperature = st.slider("Температура:", 0.1, 2.0, 0.8)
max_tokens = st.slider("Макс. токенов:", 10, 200, 50)

if st.button("Генерировать"):
    with st.spinner("Генерация..."):
        text = generate_text(
            model, tokenizer, tok_config, prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
    st.success("Готово!")
    st.write(text)
```

### Деплой

1. Создайте репозиторий на GitHub
2. Зайдите на https://streamlit.io/cloud
3. Подключите репозиторий
4. Деплой!

## ?? Android (TensorFlow Lite)

### Конвертация модели

```python
import torch
import torch.nn as nn

# Загрузить модель
model, config = load_model("models/best_model.pt")
model.eval()

# Экспорт в ONNX
dummy_input = torch.randint(0, 8192, (1, 64))
torch.onnx.export(
    model,
    dummy_input,
    "tatar_gpt.onnx",
    opset_version=14,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {1: 'seq_len'}}
)

# Конвертация ONNX -> TFLite
# Используйте onnx-tf и tensorflow
```

?? **Примечание**: Для мобильных устройств лучше использовать квантизацию!

## ?? REST API (FastAPI)

### api.py

```python
from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_model, load_tokenizer, generate_text

app = FastAPI()

# Загрузка модели при старте
model, _ = load_model("models/best_model.pt")
tokenizer, tok_config = load_tokenizer()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.8

@app.post("/generate")
async def generate(request: GenerateRequest):
    text = generate_text(
        model, tokenizer, tok_config,
        request.prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature
    )
    return {"generated_text": text}

# Запуск: uvicorn api:app --reload
```

### Использование

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Казан ш???ре", "max_tokens": 50}'
```

## ?? Квантизация для ускорения

### INT8 Quantization

```python
import torch
from model import GPT

# Загрузить модель
checkpoint = torch.load("models/best_model.pt")
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model'])
model.eval()

# Динамическая квантизация
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

# Сохранить
torch.save(quantized_model.state_dict(), "models/quantized_model.pt")

# Размер уменьшится в ~4 раза!
```

## ?? Мониторинг

### Добавление логирования

```python
import wandb

wandb.init(project="tatar-gpt-inference")

# В функции generate
@wandb.log_artifact
def generate_with_logging(prompt, **kwargs):
    start = time.time()
    result = generate_text(model, tokenizer, tok_config, prompt, **kwargs)
    duration = time.time() - start
    
    wandb.log({
        "prompt": prompt,
        "duration": duration,
        "temperature": kwargs.get('temperature', 0.8)
    })
    
    return result
```

## ?? Полезные ссылки

- [Hugging Face Docs](https://huggingface.co/docs)
- [Gradio Documentation](https://gradio.app/docs)
- [Streamlit Docs](https://docs.streamlit.io)
- [FastAPI](https://fastapi.tiangolo.com)
- [Docker](https://docs.docker.com)

---

## ?? Советы

1. **Hugging Face Spaces** - самый простой способ для демо
2. **Google Colab** - хорошо для быстрого тестирования
3. **Docker** - для production deployment
4. **FastAPI** - если нужен REST API
5. **Квантизация** - для мобильных устройств

Удачи с деплоем! ??


