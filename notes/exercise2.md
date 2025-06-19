
# Exercise 2: Multi-layer perceptron (MLP) for Higgs classification

Classification is one of the most common tasks in High Energy Physics. Its goal is to combine various inputs from data to obtain a numbers later used to classify data in different categories. A very standard task is the background vs signal classification, where each event is assigned assigned with a class number 0 (background) and 1 (signal).

In this tutorial we will try to classify $H\to W^+ W^-$ signals from background in 2-lepton final states.

In the first example we will use a multi-layer perceptron (MLP) model built with pytorch.

To run the training, simply run the command:
```bash
python scripts/train.py
```

Once this is finished, have a look at the  script to better understand what was done.

At first, the data are loaded and processed to create the training and testing samples. The testing sample is particularly important to evaluate the performance of the model after training, using some data that the network has never seen before. The dataset preparation is done inside [`trisepmltutorial/dataset.py`](trisepmltutorial/dataset.py). For the first training, we have simply included in the network the following data features:
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
In [Exercise 1: Dataset exploration](exercise1.md) we have seen that many of these quantities provide a good separation between signal and background. The classifier simply combines them together.

After the training and testing datasets are built, the training of the network is done by the following function:
```bash
mlp = mynn.train_mlp(X_train, X_val, y_train, y_val, output_dir=outdir_mlp)
```
This function is defined inside [`trisepmltutorial/mlp.py`](trisepmltutorial/mlp.py). Have a look at the file and read through the code. Do you understand where and how the network is defined? Try to modify it's parameters!

After the training is done, you should see the trained network inside `models/mlp`. To make plots of the output scores, learning curves and ROC curves, just run
```bash
python scripts/plot_mlp.py
```

A few questions:
  - Was the model overtrained? How do you know that?
  - Which features are found to be the most important after the training?

A few exercises:
1. Try to increase the number of nodes in the hidden layers to 128 inside [trisepmltutorial/mlp.py](trisepmltutorial/mlp.py). Is the network doing better for classification after training? Or worse?
2. Exercise: try to increase the learning rate. Is the learning time improving? Is the precision better?
3. Exercise: try to prevent overtraining by adding dropout layers (see [nn.Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)).
