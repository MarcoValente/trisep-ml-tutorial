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

## Exercise 2: Multi-layer perceptron (MLP) for Higgs classification

## Exercise 3: Boosted Decision Tree (BDT) for Higgs classification

## Exercise 4: Comparisons of ML and BDT performance