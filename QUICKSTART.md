# Quick Start Guide

## 1. Clone and set up the environment

```bash
# clone the repository
git clone https://github.com/vanish77/Tatar_LLM.git
cd Tatar_LLM

# create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```


## 2. Using the model in your own code

```python
from utils import load_model, load_tokenizer, generate_text

model, _ = load_model("models/best_model.pt", device="cpu")
tokenizer, tok_config = load_tokenizer()

prompt = "Казан турында с?йл?."
response = generate_text(model, tokenizer, tok_config, prompt, max_new_tokens=60)
print(response)
```
