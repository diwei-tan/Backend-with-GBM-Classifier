# OCBC Assessment Application

## Overview

This folder/repository contains all the outcomes of the OCBC Assement for Full-Stack Data Scientist. There are two main folders, one is `backend` which contains all the necessary files to run the backend flask app which provides all the insights in graphs for the frontend, as well as a classification api that returns the prediction results and the row information for any given index in the test set.

training for the best model, stored in `backend/data/best_model.txt`, was done using the `train.py` script in `backend`.

## Getting started

**Setup Environment and Install the required packages**
to install required packages, cd to `backend` folder and run:

`pip install -r requirements.txt --user`

## Run Backend

To run backend, cd to `backend` folder and run:

```
python app.py
```

This would setup the flask application as the backend. apis can be accessed through `localhost:5000`

For example, we can input `localhost:5000/plot/get_poverty_breakdown` into our browser to get the overall counts for each poverty level.
