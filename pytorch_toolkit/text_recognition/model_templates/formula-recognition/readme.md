# Formula recognition

Models which are able to recognize latex formulas

| Model Name | Complexity (GFLOPs) | Size (M params) | Train dataset | Eval dataset | Accuracy (images in dataset) |
| ---------- | ------------------- | --------------- | ------------- | ------------ | ---------------------------- |
| medium-0001 | 1 | 1 | medium v2 1.5x | medium v2 1.5 val <br> medium scans | 95% <br> 80%  |
| polynomials-handwritten-0001 | - | - | polynomials handwritten | polynomials handwritten | 70% |

## Training pipeline

### 1. Change directory in your terminal
```bash
cd <training extensions>/pytorch_toolkit/text_recognition
```

### 1. Select a model template file and instantiate it in some directory.

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/horizontal-text-detection/horizontal-text-detection-0001/template.yaml`
export WORK_DIR=/tmp/my_model
python ../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
```

### 2. Download datasets

### 3. Convert datasets

### 4. Change current directory to directory where the model template has been instantiated.

```bash
cd ${WORK_DIR}
```

### 5. Training and Fine-tuning

### 6. Evaluation

### 7. Export PyTorch\* model to the OpenVINO™ format

### 8. Validation of IR