import os
import time
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from random import randrange
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import pandas as pd
# Remove model import since we're loading from .h5
from utils import train_dp, evaluate_dp, train_eo, evaluate_eo

def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_adult_ac1():
    """
    Load and preprocess adult dataset with proper encoding
    """
    train_path = './adult.data'
    test_path = './adult.test'
    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                    'education-num', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income-per-year']
    na_values = ['?']
    
    train = pd.read_csv(train_path, header=None, names=column_names,
                        skipinitialspace=True, na_values=na_values)
    test = pd.read_csv(test_path, header=0, names=column_names,
                       skipinitialspace=True, na_values=na_values)
    df = pd.concat([test, train], ignore_index=True)
    
    del_cols = ['fnlwgt']  # 'education-num'
    df.drop(labels=del_cols, axis=1, inplace=True)
    
    ##### Drop na values
    dropped = df.dropna()
    count = df.shape[0] - dropped.shape[0]
    print("Missing Data: {} rows removed.".format(count))
    df = dropped
    
    encoders = {}
    cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    
    ## Implement label encoder instead of one-hot encoder
    for feature in cat_feat:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        encoders[feature] = le
    
    ## Implement label encoder for race
    cat_feat = ['race']
    for feature in cat_feat:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        encoders[feature] = le
    
    bin_cols = ['capital-gain', 'capital-loss']
    for feature in bin_cols:
        bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
        df[feature] = bins.fit_transform(df[[feature]])
        encoders[feature] = bins
    
    label_name = 'income-per-year'
    favorable_label = 1
    unfavorable_label = 0
    favorable_classes = ['>50K', '>50K.']
    pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
    df.loc[pos, label_name] = favorable_label
    df.loc[~pos, label_name] = unfavorable_label
    
    X = df.drop(labels=[label_name], axis=1, inplace=False)
    y = df[label_name]
    seed = 42  # randrange(100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
    return (df, X_train.to_numpy(), y_train.to_numpy().astype('int'), X_test.to_numpy(), y_test.to_numpy().astype('int'), encoders)

def preprocess_adult_data(seed=0):
    """
    Preprocess adult data and create train/val/test splits
    Returns data with sensitive attribute (sex) extracted
    """
    df, X_train, y_train, X_test, y_test, encoders = load_adult_ac1()
    
    # Create validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed
    )
    
    # Extract sensitive attribute (sex) - assuming it's at a specific index
    # We need to find the index of 'sex' in the feature columns
    feature_names = ['age', 'workclass', 'education', 'education-num', 'marital-status', 
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 
                     'capital-loss', 'hours-per-week', 'native-country']
    
    sex_idx = feature_names.index('sex')
    
    A_train = X_train[:, sex_idx]
    A_val = X_val[:, sex_idx] 
    A_test = X_test[:, sex_idx]
    
    print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test

def load_and_inspect_model(model_path):
    """Load model and print its architecture"""
    model = tf.keras.models.load_model(model_path)
    
    print("=" * 50)
    print("LOADED MODEL ARCHITECTURE:")
    print("=" * 50)
    model.summary()
    print(f"Input shape: {model.input.shape}")
    print(f"Output shape: {model.output.shape}")
    print(f"Number of layers: {len(model.layers)}")
    print("=" * 50)
    
    return model

def run_experiments(method='mixup', mode='dp', lam=0.5, num_exp=10, neuron_ratio=1):
    # Hardcoded model path - change this to your model file
    model_path = "../model/AC-5.h5"  # Change this to your actual model path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    ap = []
    gap = []
    best_model = None
    best_gap = float('inf')
    
    for i in range(num_exp):
        seed_everything(i)
        print(f'On experiment {i}')
        
        X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = preprocess_adult_data(seed=i)
        
        # Convert to TF tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
        
        # Load the existing model
        model = load_and_inspect_model(model_path)
        
        # Verify input compatibility
        expected_input_shape = model.input.shape[1:]  # Exclude batch dimension
        actual_input_shape = X_train.shape[1:]
        
        if expected_input_shape != actual_input_shape:
            print(f"WARNING: Input shape mismatch!")
            print(f"Model expects: {expected_input_shape}")
            print(f"Data provides: {actual_input_shape}")
            
            # Optionally reshape or pad data if needed
            if len(expected_input_shape) == 1 and len(actual_input_shape) == 1:
                if expected_input_shape[0] > actual_input_shape[0]:
                    # Pad with zeros
                    pad_width = expected_input_shape[0] - actual_input_shape[0]
                    X_train = tf.pad(X_train, [[0, 0], [0, pad_width]])
                    X_val = tf.pad(X_val, [[0, 0], [0, pad_width]])
                    X_test = tf.pad(X_test, [[0, 0], [0, pad_width]])
                    print(f"Padded input data to match model input shape: {X_train.shape}")
                elif expected_input_shape[0] < actual_input_shape[0]:
                    # Truncate
                    X_train = X_train[:, :expected_input_shape[0]]
                    X_val = X_val[:, :expected_input_shape[0]]
                    X_test = X_test[:, :expected_input_shape[0]]
                    print(f"Truncated input data to match model input shape: {X_train.shape}")
        
        # Recompile the model with fresh optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        criterion = tf.keras.losses.BinaryCrossentropy()
        
        # If your model wasn't compiled or you want to change the compilation
        model.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])
        
        ap_val_epoch = []
        gap_val_epoch = []
        ap_test_epoch = []
        gap_test_epoch = []
        
        # Store model weights for each epoch
        model_weights_by_epoch = []
        
        start_time = time.time()
        
        for j in range(10):
            print(f'\nEpoch: {j}')
            
            if mode == 'dp':
                train_dp(model, X_train, A_train, y_train, method, lam, neuron_ratio)
                ap_val, gap_val = evaluate_dp(model, X_val, y_val, A_val)
                ap_test, gap_test = evaluate_dp(model, X_test, y_test, A_test)
                print(f'ap_test: {ap_test}')
                print(f'gap_test: {gap_test}')
            elif mode == 'eo':
                train_eo(model, X_train, A_train, y_train, method, lam, neuron_ratio)
                ap_val, gap_val = evaluate_eo(model, X_val, y_val, A_val)
                ap_test, gap_test = evaluate_eo(model, X_test, y_test, A_test)
                print(f'ap_test: {ap_test}')
                print(f'gap_test: {gap_test}')
            
            if j > 0:
                ap_val_epoch.append(ap_val)
                ap_test_epoch.append(ap_test)
                gap_val_epoch.append(gap_val)
                gap_test_epoch.append(gap_test)
                model_weights_by_epoch.append(model.get_weights())
        
        idx = gap_val_epoch.index(min(gap_val_epoch))
        gap.append(gap_test_epoch[idx])
        ap.append(ap_test_epoch[idx])
        
        # Check if this is the best model overall
        if gap_test_epoch[idx] < best_gap:
            best_gap = gap_test_epoch[idx]
            # Restore the best weights and save the model
            model.set_weights(model_weights_by_epoch[idx])
            best_model = tf.keras.models.clone_model(model)
            best_model.set_weights(model.get_weights())
        
        print('--------INDEX---------')
        print(f'idx: {idx + 1}')
        print(f'ap_test: {ap_test_epoch[idx]}')
        print(f'gap_test: {gap_test_epoch[idx]}')
        
        end_time = time.time()
        print(f'time costs: {end_time - start_time} s')
    
    # Save the best model
    if best_model is not None:
        best_model.save("../model/AC-5-Runner.h5")
        print(f'Best model saved to ../model/AC-1-Runner.h5 with gap: {best_gap:.6f}')
    
    print('--------AVG---------')
    print(f'Average Precision: {np.mean(ap):.6f}')
    print(f'{mode} gap: {np.mean(gap):.6f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adult Experiment with Existing Model')
    parser.add_argument('--method', default='mixup', type=str, help='mixup/GapReg/erm')
    parser.add_argument('--mode', default='dp', type=str, help='dp/eo')
    parser.add_argument('--lam', default=1, type=float, help='Lambda for regularization')
    parser.add_argument('--neuron_ratio', default=1, type=float, help='% of topk importance neuron')
    parser.add_argument('--ex_num', default=10, type=int, help='num of experiment')
    
    args = parser.parse_args()
    
    run_experiments(
        args.method, 
        args.mode, 
        args.lam, 
        num_exp=args.ex_num, 
        neuron_ratio=args.neuron_ratio
    )