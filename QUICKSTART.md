# Quick Start Guide

Follow the steps below to reproduce the full Tatar GPT pipeline from scratch. All commands assume macOS or Linux; adapt paths for Windows if needed.

## 1. Clone and set up the environment

```bash
# clone the repository
git clone https://github.com/vanish77/Tatar_LLM.git
cd Tatar_LLM

# create and activate virtual environment
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Prepare the dataset

```bash
# download and clean the Leipzig Wortschatz Tatar corpus
python 01_prepare_data.py
```

Outputs:
- `data/processed/train.txt`
- `data/processed/val.txt`

## 3. Train the tokenizer

```bash
python 02_train_tokenizer.py --vocab_size 8192 --min_frequency 2
```

Outputs:
- `tokenizer/tokenizer.json`
- `tokenizer/config.json`

## 4. Train the GPT model

```bash
python 03_train.py
```

Key defaults:
- 6 layers, 8 heads, 512 hidden size (?23M parameters)
- Context length 256 tokens
- 10 000 iterations, cosine LR schedule with warmup
- Checkpoints saved to `models/`

You can tweak hyperparameters by editing the top section of `03_train.py` or using the dataclasses in `config.py`.

## 5. Smoke-test the model

```bash
python quick_test.py
```

This script loads `models/best_model.pt` and prints a few sample generations for sanity checking.

## 6. Run the inference notebook (optional)

Launch Jupyter or VS Code and open `demo_inference.ipynb` for richer experiments: perplexity checks, manual prompts, and sampling controls.

## 7. Upload artifacts (optional)

To publish the checkpoints and tokenizer on Hugging Face Hub:

```bash
huggingface-cli login
python -c "from huggingface_hub import HfApi; HfApi().create_repo('USERNAME/tatar-gpt', repo_type='model', exist_ok=True)"
python -c "from huggingface_hub import HfApi; HfApi().upload_folder('models', 'USERNAME/tatar-gpt', repo_type='model')"
python -c "from huggingface_hub import HfApi; HfApi().upload_folder('tokenizer', 'USERNAME/tatar-gpt', repo_type='model', path_in_repo='tokenizer')"
```

Replace `USERNAME` with your Hugging Face handle.

## 8. Using the model in your own code

```python
from utils import load_model, load_tokenizer, generate_text

model, _ = load_model("models/best_model.pt", device="cpu")
tokenizer, tok_config = load_tokenizer()

prompt = " ‡Á‡Ì ÚÛ˚Ì‰‡ Ò?ÈÎ?."
response = generate_text(model, tokenizer, tok_config, prompt, max_new_tokens=60)
print(response)
```

Thatís it! You now have a fully reproducible pipeline for training and testing a small GPT model on the Tatar language corpus. Feel free to iterate on the architecture, add instruction tuning, or extend the dataset for better fluency.
