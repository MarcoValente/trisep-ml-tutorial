# TRISEP Machine Learning tutorial
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e.svg)](https://scikit-learn.org/)
[![Status: Active](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)]()

Welcome to the Machine Learning Tutorial for the TRISEP HEP Summer School at TRIUMF, June 16–27, 2025!
https://www.trisep.ca

This is a **hands-on tutorial** focused on practical applications of machine learning in high energy physics (HEP). You will actively code, experiment, and solve real-world HEP problems using modern data science tools. The sessions are designed for students and researchers eager to learn by doing—no prior machine learning experience required, though familiarity with Python and basic statistics is recommended.

Get ready to dive in, collaborate, and discover how machine learning can accelerate research in particle physics!

## Installation

### Option 1: Using TRIUMF Remote Machines (`ml1`, `ml2`)

If you have access to the TRIUMF remote machines:

1. **Connect via SSH**  
    Replace `<username>` with your TRIUMF username:
    ```bash
    ssh <username>@ml1.triumf.ca
    ```
    or
    ```bash
    ssh <username>@ml2.triumf.ca
    ```

2. **Clone the Tutorial Repository**
    ```bash
    git clone https://github.com/MarcoValente/trisep-ml-tutorial.git
    cd trisep-ml-tutorial
    ```

3. **Start the Singularity Container**
    ```bash
    bash start_container.sh
    ```

You are ready to go!

---

### Option 2: Standalone Installation (Local Machine)

If you prefer to run the tutorial on your own computer:

1. **Clone the Repository**
    ```bash
    git clone https://github.com/MarcoValente/trisep-ml-tutorial.git
    cd trisep-ml-tutorial
    ```

2. **Create a Virtual Environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install Dependencies**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Exercise 1: Exploring the dataset
The data for the exercise are collected in side `data/dataWW_d1.root`. This is a dataset composed of signal and background events for a Higgs analysis decaying to two W bosons ($H\to W^+W^-$). Inside this dataset there are many different variables that we want to explore. For this task, have a look at [`scripts/explore.py`](scripts/explore.py) and look at the figures produced by the script. Do you see which variables provide the highest discriminating power between background and signal? 


## Exercise 2: Multi-layer perceptron (MLP) for Higgs classification

To run the exercise simply run
```bash
python scripts/train.py
```
This is done inside [`trisepmltutorial/dataset.py`](trisepmltutorial/dataset.py). For the first step, we will include in the network the following data features:
```python
features = [
    "met_et",
    "met_phi",
    "lep_pt_0",
    "lep_pt_1",
    "lep_eta_0",
    "lep_eta_1",
    "lep_phi_0",
    "lep_phi_1",
    "jet_n",
    "jet_pt_0",
    "jet_pt_1",
]
```
The training of the network is done through
```bash
mlp = mynn.train_mlp(X_train, X_val, y_train, y_val, output_dir=outdir_mlp)
```
This function is defined inside [`trisepmltutorial/mlp.py`](trisepmltutorial/mlp.py). Have a look at the file and read through the code. Do you understand how the network is defined?

 - Exercise: try to increase the number of nodes in the hidden layers. Is the network doing better for classification? Or worse?
 - Exercise: try to increase the learning rate. How is the learning time improving?
 - Exercise: try to prevent overtraining by adding dropout on neurons layers.

## Exercise 3: Boosted Decision Tree (BDT) for Higgs classification

## Exercise 4: Comparisons of ML and BDT performance