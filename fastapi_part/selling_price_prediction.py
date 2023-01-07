import pickle
import pandas as pd
import numpy as np
import sklearn

def data_processing(sample):
    sample['mileage'] = sample['mileage'].apply(lambda x: float(str(x).split(' ')[0]))
    sample['engine'] = sample['engine'].apply(lambda x: float(str(x).split(' ')[0]))
    sample['max_power'] = sample['max_power'].apply(
        lambda x: float(str(x).split(' ')[0]) if str(x).split(' ')[0] != '' else np.nan)

    cols = ['mileage', 'engine', 'max_power', 'seats']
    for col in cols:
        sample.loc[sample[col].isna(), col] = sample[col].median()

    sample['seats'] = sample['seats'].astype('str')
    sample['brand'] = sample['name'].apply(lambda x: x.split()[0])
    sample = sample.drop(columns=['torque', 'name'])

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats', 'brand']
    num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power']

    with open('model/poly.pkl', 'rb') as f:
        poly = pickle.load(f)

    with open('model/standard_scaler.pkl', 'rb') as f:
        standard_scaler = pickle.load(f)

    with open('model/one_hot_enc.pkl', 'rb') as f:
        one_hot_enc = pickle.load(f)

    sample_num_poly = pd.DataFrame(
        poly.transform(sample[num_cols]),
        columns=poly.get_feature_names_out())

    sample_num_poly_norm = pd.DataFrame(
        standard_scaler.transform(sample_num_poly),
        columns=standard_scaler.get_feature_names_out())

    sample_cat = pd.DataFrame(
        one_hot_enc.transform(sample[cat_cols]),
        columns=one_hot_enc.get_feature_names_out())

    sample = pd.concat([sample_num_poly_norm, sample_cat], axis=1)
    return sample


def prediction(sample):
    with open('model/ridge_model.pkl', 'rb') as f:
        ridge_model = pickle.load(f)

    sample = data_processing(sample)

    sample_pred = ridge_model.predict(sample)

    return sample_pred
