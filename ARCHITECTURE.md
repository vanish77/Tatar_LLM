# Архитектура модели

Подробное описание архитектуры татарской LLM.

## Общая архитектура

```
Input Tokens (B, T)
    ?
Token Embeddings (B, T, C)
    ?
[Dropout]
    ?
???????????????????????????
?  Transformer Block 1    ?
?  ?? LayerNorm           ?
?  ?? Multi-Head Attention?
?  ?   ?? Rotary PE       ?
?  ?? Residual            ?
?  ?? LayerNorm           ?
?  ?? MLP (FFN)           ?
?  ?? Residual            ?
???????????????????????????
    ?
    ... (repeat N times)
    ?
???????????????????????????
?  Transformer Block N    ?
???????????????????????????
    ?
LayerNorm
    ?
LM Head (Linear)
    ?
Logits (B, T, vocab_size)
```

## Компоненты

### 1. Token Embeddings

```python
# Преобразование токенов в векторы
self.wte = nn.Embedding(vocab_size, n_embd)

# Input: (batch_size, seq_len) - индексы токенов
# Output: (batch_size, seq_len, n_embd) - эмбеддинги
```

**Особенности**:
- Размер словаря: 8192 токенов
- Размерность: 512
- Weight tying с output layer (экономия параметров)

### 2. Rotary Position Embeddings (RoPE)

Вместо классических абсолютных позиционных эмбеддингов используем RoPE:

```python
# Формула RoPE
q_embed = q * cos(?) + rotate_half(q) * sin(?)
k_embed = k * cos(?) + rotate_half(k) * sin(?)
```

**Преимущества RoPE**:
- ? Относительные позиции (лучше для длинных последовательностей)
- ? Экстраполяция на большие длины
- ? Более эффективное использование контекста
- ? Меньше параметров (не нужна отдельная таблица эмбеддингов)

### 3. Multi-Head Self-Attention

```
Input (B, T, C)
    ?
Split into Q, K, V (each B, T, C)
    ?
Reshape to (B, n_heads, T, head_dim)
    ?
Apply RoPE to Q and K
    ?
Attention(Q, K, V) = softmax(QK^T / ?d_k) V
    ?
Concat heads ? (B, T, C)
    ?
Output projection
```

**Параметры**:
- 8 attention heads
- head_dim = 512 / 8 = 64
- Scaled dot-product attention
- Causal masking (для autoregressive генерации)

**Опционально**: Flash Attention для ускорения

### 4. Feed-Forward Network (MLP)

```python
class MLP:
    def forward(x):
        x = Linear(n_embd ? 4*n_embd)(x)  # Expand
        x = GELU(x)                        # Activation
        x = Linear(4*n_embd ? n_embd)(x)  # Project
        x = Dropout(x)
        return x
```

**Особенности**:
- Expansion ratio: 4x (512 ? 2048 ? 512)
- GELU activation (smooth, лучше чем ReLU)
- Dropout для регуляризации

### 5. Transformer Block

```python
class Block:
    def forward(x):
        # Pre-normalization (современный подход)
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
        return x
```

**Pre-normalization vs Post-normalization**:
- ? Pre-norm: более стабильное обучение
- ? Pre-norm: не нужен learning rate warmup
- ? Post-norm: лучше для очень глубоких сетей (но нам не нужно)

### 6. Output Layer

```python
# Final normalization
x = LayerNorm(x)

# Linear projection to vocabulary
logits = Linear(n_embd ? vocab_size)(x)

# Loss computation
loss = CrossEntropy(logits, targets)
```

## Размеры и параметры

### Конфигурация по умолчанию:

```python
GPTConfig(
    vocab_size=8192,      # Размер словаря
    n_layer=6,            # Количество трансформерных блоков
    n_head=8,             # Количество attention heads
    n_embd=512,           # Размерность эмбеддингов
    block_size=256,       # Максимальная длина контекста
    dropout=0.1,          # Dropout rate
    use_rotary=True,      # Rotary Position Embeddings
)
```

### Подсчёт параметров:

```
Token Embeddings:     8,192 ? 512 = 4,194,304

Transformer Blocks (?6):
  - LayerNorm1:       512 ? 2 = 1,024
  - Attention QKV:    512 ? (3?512) = 786,432
  - Attention out:    512 ? 512 = 262,144
  - LayerNorm2:       512 ? 2 = 1,024
  - MLP fc:           512 ? 2048 = 1,048,576
  - MLP proj:         2048 ? 512 = 1,048,576
  
  Per block:          ? 3,147,776
  Total (?6):         ? 18,886,656

Final LayerNorm:      1,024

LM Head: (weight tied with embeddings, no extra params)

??????????????????????????????????
TOTAL:                ? 23M параметров
```

## Обучение

### Forward Pass:

1. **Embedding**: tokens ? embeddings
2. **Transformer blocks**: embeddings ? contextualized embeddings
3. **Normalization**: LayerNorm
4. **Prediction**: linear ? logits
5. **Loss**: CrossEntropy(logits, targets)

### Backward Pass:

1. **Loss gradient**: ?L/?logits
2. **Backprop через LM head**
3. **Backprop через transformer blocks**
4. **Gradient clipping**: clip_grad_norm(params, 1.0)
5. **Optimizer step**: AdamW

### Оптимизации:

- **Mixed Precision**: FP16/BF16 для ускорения (на CUDA)
- **Gradient Accumulation**: эффективный batch_size = batch ? accum_steps
- **Gradient Checkpointing**: trade-off memory ? compute
- **MPS backend**: для Apple Silicon
- **Weight Decay**: только для матриц весов, не для bias/LayerNorm

## Генерация

### Autoregressive Generation:

```python
def generate(prompt, max_tokens):
    tokens = tokenize(prompt)
    
    for _ in range(max_tokens):
        # Forward pass
        logits = model(tokens)[-1]  # Берём последний токен
        
        # Temperature scaling
        logits = logits / temperature
        
        # Top-k filtering
        logits = top_k_filter(logits, k=40)
        
        # Sample
        probs = softmax(logits)
        next_token = sample(probs)
        
        # Append
        tokens.append(next_token)
    
    return decode(tokens)
```

### Параметры генерации:

- **Temperature**: 
  - 0.3-0.5: консервативно, предсказуемо
  - 0.7-0.9: сбалансировано
  - 1.0-1.5: креативно, разнообразно

- **Top-k**: 
  - 10-20: сфокусированно
  - 40-50: сбалансировано
  - 100+: максимально разнообразно

## Сравнение с другими архитектурами

### vs GPT-2:
- ? Меньше параметров (23M vs 124M)
- ? Rotary embeddings (vs learned positional)
- ? Pre-normalization (vs post-normalization)
- ? Меньше слоёв (6 vs 12)

### vs GPT-3:
- ? Те же принципы, но масштаб меньше
- ? Decoder-only архитектура
- ? Значительно меньше параметров (23M vs 175B)

### vs BERT:
- ? Decoder-only (vs encoder-only)
- ? Autoregressive (vs masked LM)
- ? Генерация текста (vs только понимание)

## Возможные расширения

### 1. Mixture of Experts (MoE)
```python
class MoEBlock:
    def forward(x):
        # Router выбирает экспертов
        router_logits = Router(x)
        expert_weights = softmax(router_logits, top_k=2)
        
        # Только активные эксперты
        output = sum(weight * Expert_i(x) 
                    for i, weight in expert_weights)
        return output
```

### 2. Flash Attention 2
- Ещё более эффективная реализация attention
- До 3x ускорение на длинных последовательностях

### 3. Group Query Attention (GQA)
- Shared keys/values между головами
- Меньше параметров, быстрее инференс

### 4. Sliding Window Attention
- Attention только на локальном окне
- Позволяет обрабатывать более длинные последовательности

## Заключение

Архитектура сочетает:
- ? Проверенные принципы GPT
- ? Современные оптимизации (RoPE, pre-norm)
- ? Эффективность для ограниченных ресурсов
- ? Простоту реализации и понимания

Несмотря на небольшой размер, модель демонстрирует хорошие результаты на татарском языке!


