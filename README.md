# Toxic Comment Classifier

Проект по классификации токсичных комментариев на основе модели **RoBERTa**.  
Решение обучает модель на очищенных данных (`clean_train.csv`, `clean_test.csv`) и сохраняет готовые веса для последующего использования.

---

## Подготовка окружения

### Mac

```bash
conda create -n Ai_Lab_1 python=3.11
conda activate Ai_Lab_1
pip install -r requirements.txt
```

PyTorch автоматически будет использовать MPS (Metal) при наличии GPU Apple.

### Windows / Linux (GPU, NVIDIA):

```bash
conda create -n toxic_gpu python=3.11
conda activate toxic_gpu
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

PyTorch будет использовать cuda ядра для лучших вычислений

## Структура проекта
```bash 
data/
 └── preprocessed/
     ├── clean_train.csv
     └── clean_test.csv
models/
 └── roberta-toxic/
train.ipynb         # обучение и сохранение модели
evaluate.ipynb      # оценка модели и отчёт по метрикам
preprocess.py       # очистка исходных данных
requirements.txt
```

## Запуск

Открыть `train.ipynb` и выполнить блоки последовательно.
После завершения обучения модель сохраняется в `models/roberta-toxic` в формате `pytorch_model.bin`.

Для оценки точности и F1-метрики выполнить `evaluate.ipynb`.
Типичный результат на подготовленных данных:

```
accuracy: 0.78
f1-score: 0.75
```

Готовую модель можно загрузить отдельно:

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("models/roberta-toxic")
```

---

## Примечания

* На macOS используется Metal (`device = "mps"`).
* На Windows/Linux — CUDA (`device = "cuda"`).
* При сохранении модели используется параметр `safe_serialization=False`, чтобы избежать ошибок Windows.
* Папка `models/` создаётся автоматически при первом сохранении.
* Обучение проводилось 6 эпох, средняя потеря стабилизировалась около `0.49`.


