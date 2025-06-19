
# Exercise 1: Exploring the dataset
<!-- The data for the exercise are collected in side `data/dataWW_d1.root`. This is a dataset composed of signal and background events for a Higgs analysis decaying to two W bosons ($H\to W^+W^-$). Inside this dataset there are many different variables that we want to explore before starting the classification task. For this task, have a look at [`scripts/explore.py`](scripts/explore.py) and execute it with -->
```bash
python scripts/explore.py
```
This will produce 2 figures inside the folder `figures/`. Have a look at them and try to answer to these:
 - Do you see which variables provide the highest discriminating power between background and signal?
 - How are variables correlated?
 - Are there variables that are probably not helping with the classification? If so, which ones?
 - Try to add more variables to the plots.