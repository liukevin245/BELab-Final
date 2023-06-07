# BELab-Final
All codes for the final project (Confusing / Not Confusing) of BELab

## File Structure
```
BELab-Final
|- collect_data.py
|- final.ipynb
|- serial_filter_no_abs.py
|- README.md
```
## Data Collection
Execute `collect_data.py` for collecting data from Arduino to `training_data.csv`. **(Remember to modify `COM_PORT` variable)**

You also have to open `final.ipynb` and execute the first cell to convert `training_data.csv` into useable csv format.
## Data Preprocess & Training
Please follow the instructions in `final.ipynb`. Note that there are 2 models (XGBoost & LSTM) available, just choose the one you want.
## Inference
Execute the *Inference* cell correspond to the model you trained.
