# tammathon
Kaggle competition - Pet Facial Recognition

Dataset embedding and cleaning

/home/<user>/
├── venv/                         # ✅ Virtual environment (outside the project Git repo)
│
└── <project>/                         # ✅ Your GitHub project (run `git init` here)
    ├── main.py                   # 🔧 Main script to run cleaning
    ├── snippet.csv               # 📄 Small CSV for quick tests
    ├── train.csv                 # 📄 Full training CSV (optional)
    ├── requirements.txt          # 📦 All required libraries
    ├── .gitignore                # ❌ Ignore venv, logs, cache, etc.
    │
    ├── snippet/                  # 📁 Folder for actual images (test/train subset)
    │   └── train/
    │       └── 000000/
    │           ├── 00.png
    │           ├── 01.png
    │           └── ...
    │
    └── src/                      # 🧠 All your logic lives here
        ├── data_loader.py        # Load and prepare datasets
        ├── cleaner.py            # Cosine-sim cleaning logic
        ├── embeddings_resnet.py  # Embedding using ResNet50
        ├── embeddings_arcface.py # Embedding using ArcFace (InsightFace)
        └── embeddings_dinov2.py  # Embedding using DINOv2 (transformers)


1. Create the virtual environment (if you haven't already):
$ python3 -m venv ~/venv

2. Activate the virtual environment:
$ source ~/venv/bin/activate

3. Install dependencies:
$ (venv) user@host:$ pip install -r requirements.txt

4. Run script:
$ python3 main.py

To deactivate virtual enviroment:
(venv) user@host:$ deactivate
$
