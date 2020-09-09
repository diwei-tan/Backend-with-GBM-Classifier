import os

from flask import Flask, send_file, make_response, render_template, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
import math
import json

from data_reader import Data
from datapipeline import Datapipeline
from plot import Plotter
from gbm_model import Model

BASE_URL = "http://localhost:5000/"

print('Setting up application...')
app = Flask(__name__)

# required data and plotter
print('Retreiving and Preprocessing Data. This may take a few minutes...')
data = Data()
final, ind, _ = Datapipeline().preprocess_data(data.train, data.test)
del data
plotter = Plotter(final, ind)
print('Final features shape: ', final.shape)
print('Data processing completed. Loading model...')
model = Model(final)
print('Model loaded. Launching Application...')
del final, ind

@app.route('/')
def index():
    return render_template('index.html', base_url=BASE_URL)

@app.route('/plot/get_poverty_breakdown', methods=['GET'])
def get_poverty_breakdown():
    bytes_obj = plotter.plot_poverty_breakdown()
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/plot/get_household_level_corr', methods=['GET'])
def get_poverty_household_level_corr():
    bytes_obj = plotter.plot_household_level_data_corr()
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/plot/get_home_condition_to_target', methods=['GET'])
def get_home_condition_to_target():
    bytes_obj = plotter.plot_home_condition_to_target()
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/plot/get_warning_to_target', methods=['GET'])
def get_warning_to_target():
    bytes_obj = plotter.plot_warning_to_target()
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/plot/get_schooling_to_target', methods=['GET'])
def get_schooling_to_target():
    bytes_obj = plotter.plot_schooling_to_target()
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/plot/get_overcrowding_to_target', methods=['GET'])
def get_overcrowding_to_target():
    bytes_obj = plotter.plot_overcrowding_to_target()
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/plot/get_female_head_education_to_target', methods=['GET'])
def get_female_head_education_to_target():
    bytes_obj = plotter.plot_female_head_education_to_target()
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/plot/get_female_head_to_target', methods=['GET'])
def get_female_head_to_target():
    bytes_obj = plotter.plot_female_head_to_target()
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/data/get_row_info/<index>', methods=['GET'])
def get_row_info_and_prediction(index):
    # test_raw = model.test_raw.reset_index(drop=True)
    row = model.test_set.loc[int(index)]
    # row_raw = test_raw.loc[int(index)]
    result = model.predict(row)
    response = {}
    response['row'] = row.values.tolist()
    response['column_names'] = list(model.test_set.columns)
    response['prediction'] = result[0]
    return jsonify(response)

if __name__ == '__main__':
    app.run()