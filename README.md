# Titanic – Passenger Survival Prediction


## Repo introduction
This repo contains **two** runnable pipelines for the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/code) challenge on **Kaggle**:

- **Python pipeline**: `src/python_code/main.py`
- **R pipeline**: `src/R_code/main.R`

Both do the same basic thing:

1. load `train.csv` and `test.csv`
2. do simple cleaning / feature engineering
3. fit a logistic model (Python: `sklearn`, R: `glm`)
4. predict survival on the Kaggle test set, the output will be predicted survival for each passenger, which contains two columns:
    - **PassengerId**: the unique ID for each passenger in the test set.
    - **Survived**: predicted survival for the corresponding passenger (1 for survived, 0 for not survived).
5. write predictions to `src/data/...csv`

You can run either pipeline **locally** or **in Docker**.

> **Important**: The dataset from Kaggle is **not** committed to this repo. You must download it yourself and place it in the right folder before running anything. Please read the following instructions carefully to ensure you run it smoothly.


## 1. Prerequisites

- Git
- Docker (Docker Desktop on macOS/Windows; `docker` CLI on Linux)
- A Kaggle account to download the Titanic data


## 2. Get the data from Kaggle

1. Go to the Kaggle competition page:  
   https://www.kaggle.com/competitions/titanic/code
2. Download:
   - `train.csv`
   - `test.csv`
3. In **your local clone** of this repo, create the `data` directory under `src` directory, you can either run following bash command in your project root, or manually create `data` folder within `src` folder:

   ```bash
   mkdir -p src/data
4. Put the two CSVs in `src/data` folder. 

    If `src/data/train.csv` or `src/data/test.csv` is missing, both scripts will fail, by design.


## 3. Run the Python pipeline with Docker

### 3.1 Build the image

From the project root, run the following bash command:
    
    docker build -f Dockerfile.python -t titanic-python .

### 3.2 Run it (with data mounted)

If you just run the container without mounting, the prediction file will be created inside the container and disappear when it exits. So mount the data folder:

**macOS / Linux**:

    docker run --rm \
    -v "$(pwd)/src/data:/app/src/data" \
    titanic-python


**Windows PowerShell**:

    docker run --rm `
    -v "${PWD}/src/data:/app/src/data" `
    titanic-python

**What this does**:

- mounts your local `src/data` → `container /app/src/data`
- script runs: `python src/python_code/main.py`
- output will be written to: `src/data/survival_predictions.csv` on your machine (not only inside the container).

**Expected output:**
- you will see the data loading, training, and prediction processes in your terminal
- you will find a new .csv file called `survival_predictions.csv` in your `src/data` folder

## 4. Run the R pipeline with Docker

### 4.1 Build the image

    docker build -f Dockerfile.Rscript -t titanic-r .

### 4.2 Run it (with data mounted)

**macOS / Linux**:

    docker run --rm \
    -v "$(pwd)/src/data:/app/src/data" \
    titanic-r


**Windows PowerShell**:

    docker run --rm `
    -v "${PWD}/src/data:/app/src/data" `
    titanic-r

The R script will write: `src/data/survival_predictions_r.csv`, and, because of the volume mount, you will see it locally.


## 5. What the scripts print

Both scripts are intentionally verbose:

- they log where they load data from
- they show head of the data (R)
- they print training accuracy
- they print predicted survival rate on the test set
- they show where the prediction file is saved

If you don’t see these messages, you’re probably:
- running in the wrong directory
- didn’t `mount src/data`
- or your `src/data` is empty / named wrong

## 6. Local (non-Docker) run [optional]
If you already have Python/R locally and just want to test quickly:

**Python**:

    pip install -r requirements.txt
    python src/python_code/main.py


**R**:

    Rscript install_packages.R
    Rscript src/R_code/main.R


Still make sure `src/data/train.csv` and `src/data/test.csv` exist.