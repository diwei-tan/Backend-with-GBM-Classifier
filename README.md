# OCBC Assessment Application

## Overview

This folder/repository contains all the outcomes of the OCBC Assement for Full-Stack Data Scientist. There are two main folders, one is `backend` which contains all the necessary files to run the backend flask app which provides all the insights in graphs for the frontend, as well as a classification api that returns the prediction results and the row information for any given index in the test set.

training for the best model, stored in `backend/data/best_model.txt`, was done using the `train.py` script in `backend`.

## Getting started

**Setup Environment and Install the required packages**
to install required packages, cd to `backend` folder and run:

`pip install -r requirements.txt --user`

It is recommended to create a new environment and pip install. If using `conda` environment and pip is not installed when createing a new environment, simply run `conda install pip` before pip installing with requirement.txt.

## Run Backend

To run backend, cd to `backend` folder and run:

```
python app.py
```

This would setup the flask application as the backend. apis can be accessed through `localhost:5000`

For example, we can input `localhost:5000/plot/get_poverty_breakdown` into our browser to get the overall counts for each poverty level.

## Troubleshooting

Should the lgbm model not load as intended, you can train a model using `python train.py`. Doing so will run the training for the ten-fold lgbm classifier, and save iterations of models as `model_<modelaccuracy>_0.txt` in the `root` folder. Choose the model with highest accuracy and rename it `best_model.txt` and place it in `data` folder, replacing the current one. Delete the rest of the models, and rerun `python app.py`. This should solve issues with regards to the lgbm model.

## HTTP Calls

The following are the possible GET methods that can be used to return the insights or classification of poverty level for the test individuals:

- `localhost:5000/plot/get_poverty_breakdown`: This returns the overal counts of individuals in each of the 4 poverty levels as a graph.

- `localhost:5000/plot/get_household_level_corr`: This returns the correlation matrix (as an image) of useful household level features compared to the target (poverty level).

- `localhost:5000/plot/get_home_condition_to_target`: This returns a box graph (as an image) of home conditions against poverty level.

- `localhost:5000/plot/get_warning_to_target`: This returns a box graph (as an image) of the number of warning signs (no toilet, no electricity, no floor, no water service, no ceiling) against poverty level.

- `localhost:5000/plot/get_schooling_to_target`: This returns a box graph (as an image) of the years of schooling against poverty level.

- `localhost:5000/plot/get_overcrowding_to_target`: This returns a box graph (as an image) of the number of persons per room against poverty level.

- `localhost:5000/plot/get_female_head_education_to_target`: This returns a box graph (as an image) of years of eduction against poverty level, divided into whether the head of household was female.

- `localhost:5000/plot/get_female_head_education_to_target`: This returns a violin graph (as an image) of whether the head was female against poverty level. Shows how female heads could point towards higher poverty levels.

- `localhost:5000/data/get_row_info/<index>`: This returns a json response for individual <index> in the `test_set`. The json response is the following:
  - column_names: names of all the column values of the individual's data
  - row: values, matched with the column names, of all the columns of the individual's data
  - prediction: prediction of the poverty level, where 1: extreme, 2:moderate, 3: vulnerable, 4: non vulnerable. Value is the string description and not the value 1-4.
