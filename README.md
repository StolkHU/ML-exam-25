# Eindopdracht Machine Learning - Hartslagdata Classificatie

## 📋 Project Overzicht

Dit project is ontwikkeld als eindopdracht voor Machine Learning aan de Hogeschool Utrecht. Het systeem classificeert hartslagdata met behulp van deep learning technieken.

**Student:** Adriaan Stolk  
**Studentnummer:** 1517495  
**Opleiding:** Hogeschool Utrecht  
**Vak:** Machine Learning (MADS)

## 🎯 Doel

Het hoofddoel van dit project is het bouwen van een machine learning model dat hartslagdata kan classificeren. Het project maakt gebruik van moderne deep learning frameworks en technieken voor data-analyse en model training.

## 📊 Project Structuur

```
MADS-exam-AdriaanS/
├── src/                    # Broncode
├── data/                  # Databestanden voor training en testing
├── hypertune.py           # Hoofdscript voor hyperparameter tuning
├── pyproject.toml         # Project configuratie en dependencies
├── README.md              # Dit bestand
└── [config.toml]          # Configuratiebestand voor model en trianing 
```

## 🛠️ Technische Vereisten

### Python Versie
- Python 3.12 (specifiek: >=3.12, <3.13)

### Belangrijkste Dependencies
- **PyTorch** (>=2.5.1) - Deep learning framework
- **scikit-learn** (>=1.6.1) - Machine learning tools
- **MLflow** (>=2.13.2) - Model tracking en management
- **Hyperopt** (>=0.2.7) - Hyperparameter optimalisatie
- **imbalanced-learn** (>=0.12.4) - Voor het omgaan met ongebalanceerde datasets
- **TensorBoard** (>=2.16.2) - Visualisatie van training
- **mads-datasets** (>=0.3.10) - Dataset library

Voor een complete lijst van dependencies, zie het `pyproject.toml` bestand.

## 📦 Installatie

### Stap 1: Clone het project
```bash
git clone [repository-url]
cd MADS-exam-AdriaanS
```

### Stap 2: Maak een virtuele omgeving aan
```bash
python -m venv venv
source venv/bin/activate  # Op Windows: venv\Scripts\activate
```

### Stap 3: Installeer dependencies met uv
Dit project gebruikt `uv` als package manager. Installeer eerst uv:
```bash
pip install uv
```

Installeer vervolgens alle dependencies:
```bash
uv sync 
```

## 🚀 Gebruik

### Configuratie
Voordat je het model gaat trainen, moet je eerst de configuratie aanpassen naar je wensen. De configuratie bepaalt:
- Model architectuur
- Hyperparameters
- Training instellingen
- Data preprocessing opties

### Model Training
Het hoofdscript voor het trainen van het model is `hypertune.py`. Dit script voert hyperparameter tuning uit op basis van het configuratiebestand dat je hebt ingesteld.

```bash
python hypertune.py
```

## 🏗️ Model Architectuur

Het model bestaat uit de volgende layers:

| Layer | Type | Output Shape | Kernel Size | Stride |
|-------|------|--------------|-------------|---------|
| Input | Input Layer | 187 x 1 | - | - |
| Layer 1 | Convolution | 187 x 32 | 11 | 1 |
| Layer 2 | Convolution | 187 x 64 | 7 | 1 |
| Layer 3 | Squeeze Excite Block | 187 x 64 | - | - |
| Layer 4 | MaxPool | 93 x 64 | 2 | 2 |
| Layer 5 | Residual Block | 93 x 64 | 5 | 1 |
| Layer 6 | Squeeze Excite Block | 93 x 64 | - | - |
| Layer 7 | Residual Block | 93 x 128 | 3 | 1 |
| Layer 8 | Squeeze Excite Block | 93 x 128 | - | - |
| Layer 9 | Convolution | 93 x 160 | 3 | 1 |
| Layer 10 | Squeeze Excite Block | 93 x 160 | - | - |
| Layer 11 | Average Pool | 8 x 160 | - | - |
| Layer 12 | Fully Connected | 384 | - | - |
| Layer 13 | Fully Connected | 256 | - | - |
| Layer 14 | Fully Connected | 96 | - | - |
| Output | Softmax Output Layer | 5 | - | - |

## 🔍 Hyperparameter Search Space

De volgende hyperparameters worden geoptimaliseerd tijdens de training:

| Parameter | Default Value | Search Space | Type |
|-----------|---------------|--------------|------|
| Aantal Conv Layers | 5 | [2, 3, 4, 5, 6] | Discrete |
| Kernel Sizes | [7,7,3,3] | [3, 5, 7, 9, 11] per layer | Discrete |
| Output Channels Layer 0 | 32 | [32, 64, 96, 128] | Discrete |
| Output Channels Layer 1 | 64 | [32, 64, 96, 128] | Discrete |
| Output Channels Layer 2 | 96 | [64, 96, 128, 160] | Discrete |
| Output Channels Layer 3 | 128 | [96, 128, 160, 192, 224] | Discrete |
| Output Channels Layer 4 | 160 | [128, 160, 192, 224, 256] | Discrete |
| Pooling Types | [max, max, avg, none] | ['max', 'avg', 'none'] per layer | Categorical |
| Batch Size | 32 | [16, 32, 64, 128] | Discrete |
| Sampling Method | sampling to 15,000 | Deleting 0-class, sampling all to [5,000 - 25,000] | Categorical |
| Dropout | 0.3 | [0.1, 0.5] | Continuous |
| Learning Rate | 0.001 | [1e-5, 1e-2] | Log-uniform |
| Weight Decay | 0.0001 | [1e-6, 1e-3] | Log-uniform |
| Dilation | 1 | [1, 2, 4, 8] | Discrete |
| SE-blocks | True | [true, false] | Boolean |
| Residual Blocks | True | [true, false] | Boolean |
| Attention Mechanism | - | [true, false] | Boolean |
| FC1 Size | 384 | [128, 256, 512, 1024] | Discrete |
| FC2 Size | 256 | [64, 128, 256, 512] | Discrete |
| FC3 Size | 96 | [32, 64, 96, 128] | Discrete |
| Optimizer | - | ['adam', 'adamw'] | Categorical |
| Early Stopping Patience | 10 | [5, 10, 15, 20] | Discrete |
| Epochs | 50 | [5 ... 50] | Discrete |
| Train_steps | - | [1 ... 10] | Discrete |

## 🔧 Development Tools

Dit project maakt gebruik van verschillende development tools voor code kwaliteit:
- **Black** - Code formatting
- **Ruff** - Linting
- **mypy** - Type checking
- **isort** - Import sorting

Run deze tools met:
```bash
black .
ruff check .
mypy .
isort .
```

## 📈 Resultaten Visualiseren

Training resultaten kunnen bekeken worden met:
- **MLflow UI**: `mlflow ui` (standaard op http://localhost:5000)
- **TensorBoard**: `tensorboard --logdir=./logs`

## ⚠️ Belangrijke Opmerkingen

- Zorg ervoor dat je Python 3.12 gebruikt (niet nieuwer of ouder)
- Het project gebruikt PyTorch CPU versie. Voor GPU support, pas de PyTorch installatie aan
- Configureer eerst je settings voordat je `hypertune.py` draait

## 📝 Licentie

Dit project is gelicenseerd onder de MIT licentie.
