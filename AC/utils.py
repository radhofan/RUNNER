import copy
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
import torch
import numpy as np
from numpy.random import beta
from sklearn.metrics import average_precision_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from numpy.random import beta

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np

def sample_batch_sen_id(X, A, y, batch_size):
    # Convert to numpy if they're tensors
    if isinstance(A, tf.Tensor):
        A = A.numpy()
    if isinstance(X, tf.Tensor):
        X_np = X.numpy()
    else:
        X_np = X
    if isinstance(y, tf.Tensor):
        y_np = y.numpy()
    else:
        y_np = y
        
    batch_idx = np.random.choice(len(A), size=batch_size, replace=False)
    batch_x = X_np[batch_idx]
    batch_y = y_np[batch_idx]
    
    batch_x = tf.constant(batch_x, dtype=tf.float32)
    batch_y = tf.constant(batch_y, dtype=tf.float32)
    return batch_x, batch_y

def sample_batch_sen_idx(X, A, y, batch_size, s):
    # Convert to numpy if they're tensors
    if isinstance(A, tf.Tensor):
        A_np = A.numpy()
    else:
        A_np = A
    if isinstance(X, tf.Tensor):
        X_np = X.numpy()
    else:
        X_np = X
    if isinstance(y, tf.Tensor):
        y_np = y.numpy()
    else:
        y_np = y
        
    valid_indices = np.where(A_np == s)[0]
    batch_idx = np.random.choice(valid_indices, size=batch_size, replace=False)
    batch_x = X_np[batch_idx]
    batch_y = y_np[batch_idx]
    
    batch_x = tf.constant(batch_x, dtype=tf.float32)
    batch_y = tf.constant(batch_y, dtype=tf.float32)
    return batch_x, batch_y

def sample_batch_sen_idx_y(X, A, y, batch_size, s):
    # Convert to numpy if they're tensors
    if isinstance(A, tf.Tensor):
        A_np = A.numpy()
    else:
        A_np = A
    if isinstance(X, tf.Tensor):
        X_np = X.numpy()
    else:
        X_np = X
    if isinstance(y, tf.Tensor):
        y_np = y.numpy()
    else:
        y_np = y
        
    batch_idx = []
    for i in range(2):
        # Find indices where both A == s and y == i
        condition_A = A_np == s
        condition_y = y_np == i
        valid_indices = np.where(condition_A & condition_y)[0]
        
        if len(valid_indices) >= batch_size:
            selected_idx = np.random.choice(valid_indices, size=batch_size, replace=False)
        else:
            # If not enough samples, use replacement
            selected_idx = np.random.choice(valid_indices, size=batch_size, replace=True)
        
        batch_idx.extend(selected_idx.tolist())
    
    batch_x = X_np[batch_idx]
    batch_y = y_np[batch_idx]
    
    batch_x = tf.constant(batch_x, dtype=tf.float32)
    batch_y = tf.constant(batch_y, dtype=tf.float32)
    return batch_x, batch_y

def all_sen_idx_y(X, A, y, s, i):
    # Convert to numpy if they're tensors
    if isinstance(A, tf.Tensor):
        A_np = A.numpy()
    else:
        A_np = A
    if isinstance(X, tf.Tensor):
        X_np = X.numpy()
    else:
        X_np = X
    if isinstance(y, tf.Tensor):
        y_np = y.numpy()
    else:
        y_np = y
        
    # Find indices where both A == s and y == i
    condition_A = A_np == s
    condition_y = y_np == i
    all_id = np.where(condition_A & condition_y)[0]
    
    batch_x = X_np[all_id]
    batch_y = y_np[all_id]
    
    batch_x = tf.constant(batch_x, dtype=tf.float32)
    batch_y = tf.constant(batch_y, dtype=tf.float32)
    return batch_x, batch_y

def train_dp(h5_model_path, X_train, A_train, y_train, method, lam, neuron_ratio, batch_size=500, niter=100):
    model = load_model(h5_model_path)
    optimizer = Adam(learning_rate=1e-3)
    criterion = BinaryCrossentropy()

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    A_train = tf.convert_to_tensor(A_train, dtype=tf.float32)

    dataset_size = X_train.shape[0]

    niter = int(niter)
    for it in range(niter):
        # Sample batch indices for sensitive attribute 0 and 1
        idx_0 = np.random.choice(np.where(A_train.numpy() == 0)[0], size=batch_size, replace=False)
        idx_1 = np.random.choice(np.where(A_train.numpy() == 1)[0], size=batch_size, replace=False)

        batch_x_0 = tf.gather(X_train, idx_0)
        batch_y_0 = tf.gather(y_train, idx_0)

        batch_x_1 = tf.gather(X_train, idx_1)
        batch_y_1 = tf.gather(y_train, idx_1)

        with tf.GradientTape() as tape:
            if method == 'mixup':
                alpha = 1
                gamma = beta(alpha, alpha)
                batch_x_mix = batch_x_0 * gamma + batch_x_1 * (1 - gamma)
                batch_x_mix = tf.Variable(batch_x_mix)

                output = model(batch_x_mix, training=True)
                output_sum = tf.reduce_sum(output)

                gradx = tape.gradient(output_sum, batch_x_mix)
                batch_x_d = batch_x_1 - batch_x_0
                grad_inn = tf.reduce_sum(gradx * batch_x_d, axis=1)
                loss_reg = tf.reduce_mean(tf.abs(tf.reduce_mean(grad_inn)))

            elif method == 'GapReg':
                output_0 = model(batch_x_0, training=True)
                output_1 = model(batch_x_1, training=True)
                loss_reg = tf.abs(tf.reduce_mean(output_0) - tf.reduce_mean(output_1))

            else:
                loss_reg = 0  # ERM or unsupported methods

            batch_x = tf.concat([batch_x_0, batch_x_1], axis=0)
            batch_y = tf.concat([batch_y_0, batch_y_1], axis=0)

            output = model(batch_x, training=True)
            loss_sup = criterion(batch_y, output)

            loss = loss_sup + lam * loss_reg

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if it % 10 == 0:
            print(f"Iteration {it}, Loss: {loss.numpy():.6f}, Reg Loss: {loss_reg.numpy():.6f}")

    # Save model back to H5
    model.save(h5_model_path)

    return model

import tensorflow as tf
import numpy as np

def train_eo(model, X_train, A_train, y_train, method, lam, neuron_ratio, batch_size=500, niter=100):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    for it in range(int(niter)):
        # Gender Split
        batch_x_0, batch_y_0 = sample_batch_sen_idx_y(X_train, A_train, y_train, batch_size, 0)
        batch_x_1, batch_y_1 = sample_batch_sen_idx_y(X_train, A_train, y_train, batch_size, 1)

        # separate class
        batch_x_0_ = [batch_x_0[:batch_size], batch_x_0[batch_size:]]
        batch_x_1_ = [batch_x_1[:batch_size], batch_x_1[batch_size:]]

        # ERM loss
        batch_x = tf.concat([batch_x_0, batch_x_1], axis=0)
        batch_y = tf.concat([batch_y_0, batch_y_1], axis=0)

        # Training step
        with tf.GradientTape() as tape:
            # Calculate fairness regularization
            loss_reg = tf.constant(0.0)
            
            if method == 'mixup':
                alpha = 1.0
                for i in range(2):
                    gamma = tf.constant(np.random.beta(alpha, alpha), dtype=tf.float32)
                    batch_x_0_i = batch_x_0_[i]
                    batch_x_1_i = batch_x_1_[i]

                    # Create mixed input
                    batch_x_mix = batch_x_0_i * gamma + batch_x_1_i * (1 - gamma)
                    batch_x_mix = tf.Variable(batch_x_mix)  # Make it a variable for gradient computation
                    
                    # Compute gradients with respect to mixed input
                    with tf.GradientTape() as inner_tape:
                        inner_tape.watch(batch_x_mix)
                        output = model(batch_x_mix, training=True)
                        output_sum = tf.reduce_sum(output)

                    # gradient regularization
                    gradx = inner_tape.gradient(output_sum, batch_x_mix)
                    if gradx is not None:
                        batch_x_d = batch_x_1_i - batch_x_0_i
                        grad_inn = tf.reduce_sum(gradx * batch_x_d, axis=1)
                        loss_reg += tf.abs(tf.reduce_mean(grad_inn))

            elif method == "GapReg":
                for i in range(2):
                    batch_x_0_i = batch_x_0_[i]
                    batch_x_1_i = batch_x_1_[i]

                    output_0 = model(batch_x_0_i, training=True)
                    output_1 = model(batch_x_1_i, training=True)
                    loss_reg += tf.abs(tf.reduce_mean(output_0) - tf.reduce_mean(output_1))
                    
            elif method == "NeuronImportance":
                important_index1, important_index2, _, _ = cal_importance(model, optimizer, batch_x, batch_y, neuron_ratio)
                
                # Create intermediate model to get layer outputs
                intermediate_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=[layer.output for layer in model.layers]
                )
                
                for i in range(2):
                    batch_x_0_i = batch_x_0_[i]
                    batch_x_1_i = batch_x_1_[i]

                    # Get all layer outputs
                    outputs_0 = intermediate_model(batch_x_0_i, training=True)
                    outputs_1 = intermediate_model(batch_x_1_i, training=True)
                    
                    # Flexible layer handling based on actual model architecture
                    if len(outputs_0) >= 4:  # input -> hidden1 -> hidden2 -> output
                        add1_0, add2_0, add3_0 = outputs_0[1], outputs_0[2], outputs_0[3]
                        add1_1, add2_1, add3_1 = outputs_1[1], outputs_1[2], outputs_1[3]
                        
                        # FIXED: Check bounds and clip indices
                        layer1_size = tf.shape(add1_0)[1]
                        layer2_size = tf.shape(add2_0)[1]
                        important_index1_clipped = tf.clip_by_value(important_index1, 0, layer1_size - 1)
                        important_index2_clipped = tf.clip_by_value(important_index2, 0, layer2_size - 1)
                        
                        # Apply constraints on important neurons
                        important_add1_0 = tf.gather(add1_0, important_index1_clipped, axis=1)
                        important_add1_1 = tf.gather(add1_1, important_index1_clipped, axis=1)
                        important_add2_0 = tf.gather(add2_0, important_index2_clipped, axis=1)
                        important_add2_1 = tf.gather(add2_1, important_index2_clipped, axis=1)
                        
                        loss_reg += tf.reduce_mean(tf.abs(important_add1_0 - important_add1_1)) + \
                                   tf.reduce_mean(tf.abs(important_add2_0 - important_add2_1)) + \
                                   tf.reduce_mean(tf.abs(add3_0 - add3_1))
                    
                    elif len(outputs_0) >= 3:  # input -> hidden1 -> output
                        add1_0, add2_0 = outputs_0[1], outputs_0[2]
                        add1_1, add2_1 = outputs_1[1], outputs_1[2]
                        
                        # FIXED: Check bounds and clip indices
                        layer1_size = tf.shape(add1_0)[1]
                        important_index1_clipped = tf.clip_by_value(important_index1, 0, layer1_size - 1)
                        
                        # Apply constraints on important neurons
                        important_add1_0 = tf.gather(add1_0, important_index1_clipped, axis=1)
                        important_add1_1 = tf.gather(add1_1, important_index1_clipped, axis=1)
                        
                        loss_reg += tf.reduce_mean(tf.abs(important_add1_0 - important_add1_1)) + \
                                   tf.reduce_mean(tf.abs(add2_0 - add2_1))
                    
                    elif len(outputs_0) >= 2:  # input -> output only
                        add1_0 = outputs_0[1]
                        add1_1 = outputs_1[1]
                        
                        loss_reg += tf.reduce_mean(tf.abs(add1_0 - add1_1))
                    
            elif method == "NeuronImportance_GapReg":
                loss_reg0 = tf.constant(0.0)
                important_index1, important_index2, _, _ = cal_importance_gapReg(model, optimizer, batch_x_0_, batch_x_1_, neuron_ratio, mode='eo')
                
                # Skip if importance calculation failed
                if important_index1 is None or important_index2 is None:
                    if it % 100 == 0:
                        print("Importance calculation failed, skipping regularization")
                    loss_reg = tf.constant(0.0)
                else:
                    # Create intermediate model to get layer outputs
                    intermediate_model = tf.keras.Model(
                        inputs=model.input,
                        outputs=[layer.output for layer in model.layers]
                    )
                    
                    for i in range(2):
                        batch_x_0_i = batch_x_0_[i]
                        batch_x_1_i = batch_x_1_[i]

                        # Get all layer outputs
                        outputs_0 = intermediate_model(batch_x_0_i, training=True)
                        outputs_1 = intermediate_model(batch_x_1_i, training=True)
                        
                        # Flexible layer handling based on actual model architecture
                        if len(outputs_0) >= 4:  # input -> hidden1 -> hidden2 -> output
                            add1_0, add2_0, add3_0 = outputs_0[1], outputs_0[2], outputs_0[3]
                            add1_1, add2_1, add3_1 = outputs_1[1], outputs_1[2], outputs_1[3]
                            
                            # FIXED: Check bounds and clip indices
                            layer1_size = tf.shape(add1_0)[1]
                            layer2_size = tf.shape(add2_0)[1]
                            important_index1_clipped = tf.clip_by_value(important_index1, 0, layer1_size - 1)
                            important_index2_clipped = tf.clip_by_value(important_index2, 0, layer2_size - 1)
                            
                            # Apply constraints on important neurons
                            important_add1_0 = tf.gather(add1_0, important_index1_clipped, axis=1)
                            important_add1_1 = tf.gather(add1_1, important_index1_clipped, axis=1)
                            important_add2_0 = tf.gather(add2_0, important_index2_clipped, axis=1)
                            important_add2_1 = tf.gather(add2_1, important_index2_clipped, axis=1)
                            
                            loss_reg += tf.reduce_mean(tf.abs(important_add1_0 - important_add1_1)) + \
                                       tf.reduce_mean(tf.abs(important_add2_0 - important_add2_1)) + \
                                       tf.reduce_mean(tf.abs(add3_0 - add3_1))

                            # Regular output constraint
                            output_0 = outputs_0[-1]  # Final output
                            output_1 = outputs_1[-1]
                            loss_reg0 += tf.abs(tf.reduce_mean(output_0) - tf.reduce_mean(output_1))
                        
                        elif len(outputs_0) >= 3:  # input -> hidden1 -> output
                            add1_0, add2_0 = outputs_0[1], outputs_0[2]
                            add1_1, add2_1 = outputs_1[1], outputs_1[2]
                            
                            # FIXED: Check bounds and clip indices
                            layer1_size = tf.shape(add1_0)[1]
                            important_index1_clipped = tf.clip_by_value(important_index1, 0, layer1_size - 1)
                            
                            # Apply constraints on important neurons
                            important_add1_0 = tf.gather(add1_0, important_index1_clipped, axis=1)
                            important_add1_1 = tf.gather(add1_1, important_index1_clipped, axis=1)
                            
                            loss_reg += tf.reduce_mean(tf.abs(important_add1_0 - important_add1_1)) + \
                                       tf.reduce_mean(tf.abs(add2_0 - add2_1))

                            # Regular output constraint
                            output_0 = outputs_0[-1]  # Final output
                            output_1 = outputs_1[-1]
                            loss_reg0 += tf.abs(tf.reduce_mean(output_0) - tf.reduce_mean(output_1))
                        
                        elif len(outputs_0) >= 2:  # input -> output only
                            add1_0 = outputs_0[1]
                            add1_1 = outputs_1[1]
                            
                            loss_reg += tf.reduce_mean(tf.abs(add1_0 - add1_1))

                            # Regular output constraint
                            output_0 = outputs_0[-1]  # Final output
                            output_1 = outputs_1[-1]
                            loss_reg0 += tf.abs(tf.reduce_mean(output_0) - tf.reduce_mean(output_1))
                        
                        else:
                            if it % 100 == 0:
                                print("Model has insufficient layers for NeuronImportance_GapReg")
                            loss_reg = tf.constant(0.0)
                            break
                        
                if it % 100 == 0:
                    print('loss_reg0:', loss_reg0.numpy())
                    print('loss_reg:', loss_reg.numpy())
                    
            elif method == 'fairlearn':
                # Calculate fairlearn metric outside gradient tape to avoid issues
                pass  # Will handle this after the gradient tape
            
            else:
                # ERM - no regularization
                loss_reg = tf.constant(0.0)

            if method == "van":
                batch_x, batch_y = sample_batch_sen_id(X_train, A_train, y_train, batch_size)
                
            # Calculate supervised loss
            output = model(batch_x, training=True)
            loss_sup = loss_fn(batch_y, output)
            
            if it % 100 == 0:
                print('loss_sup:', loss_sup.numpy())

            # Handle fairlearn method outside main gradient computation
            if method == 'fairlearn':
                batch_y_array = batch_y.numpy()
                batch_A = tf.concat([tf.zeros(tf.shape(batch_x_0)[0]), tf.ones(tf.shape(batch_x_1)[0])], axis=0).numpy()
                
                pred = tf.cast(output > 0.5, tf.int64).numpy()
                try:
                    from fairlearn.metrics import equalized_odds_difference
                    fairness_penalty = equalized_odds_difference(batch_y_array, pred, sensitive_features=batch_A)
                    loss_reg = tf.constant(fairness_penalty, dtype=tf.float32)
                except ImportError:
                    print("fairlearn not available, using zero regularization")
                    loss_reg = tf.constant(0.0)

            # final loss
            total_loss = loss_sup + lam * loss_reg

        # Apply gradients
        grads = tape.gradient(total_loss, model.trainable_variables)
        if grads is not None and any(g is not None for g in grads):
            # Filter out None gradients
            filtered_grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            if filtered_grads_and_vars:
                optimizer.apply_gradients(filtered_grads_and_vars)
            
            
def evaluate_dp(model, X_test, y_test, A_test):
    """
    TensorFlow/Keras version of evaluate_dp function
    """
    
    # Convert inputs to numpy if they're tensors
    if hasattr(X_test, 'numpy'):
        X_test = X_test.numpy()
    if hasattr(y_test, 'numpy'):
        y_test = y_test.numpy()
    if hasattr(A_test, 'numpy'):
        A_test = A_test.numpy()
    
    # Flatten arrays if needed
    y_test = y_test.flatten()
    A_test = A_test.flatten()
    
    # Calculate DP gap
    idx_0 = np.where(A_test == 0)[0]
    idx_1 = np.where(A_test == 1)[0]
    
    X_test_0 = X_test[idx_0]
    X_test_1 = X_test[idx_1]
    
    X_test_0 = tf.convert_to_tensor(X_test_0, dtype=tf.float32)
    X_test_1 = tf.convert_to_tensor(X_test_1, dtype=tf.float32)
    
    pred_0 = model(X_test_0, training=False)
    pred_1 = model(X_test_1, training=False)
    
    # Flatten predictions if needed
    pred_0 = tf.reshape(pred_0, [-1])
    pred_1 = tf.reshape(pred_1, [-1])
    
    gap = tf.abs(tf.reduce_mean(pred_0) - tf.reduce_mean(pred_1))
    gap = gap.numpy()
    
    # Calculate average precision
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_scores = model(X_test_tf, training=False)
    y_scores = tf.reshape(y_scores, [-1]).numpy()
    
    ap = average_precision_score(y_test, y_scores)
    
    return ap, gap


def evaluate_eo(model, X_test, y_test, A_test):
    """
    TensorFlow/Keras version of evaluate_eo function
    """
    
    # Convert inputs to numpy if they're tensors
    if hasattr(X_test, 'numpy'):
        X_test = X_test.numpy()
    if hasattr(y_test, 'numpy'):
        y_test = y_test.numpy()
    if hasattr(A_test, 'numpy'):
        A_test = A_test.numpy()
    
    # Flatten arrays if needed
    y_test = y_test.flatten()
    A_test = A_test.flatten()
    
    # Find indices for each group (A=0/1, y=0/1)
    idx_00 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 0)[0]))
    idx_01 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 1)[0]))
    idx_10 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 0)[0]))
    idx_11 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 1)[0]))

    # Extract subgroups
    X_test_00 = X_test[idx_00]
    X_test_01 = X_test[idx_01] 
    X_test_10 = X_test[idx_10]
    X_test_11 = X_test[idx_11]
    
    # Convert to TensorFlow tensors
    X_test_00 = tf.convert_to_tensor(X_test_00, dtype=tf.float32)
    X_test_01 = tf.convert_to_tensor(X_test_01, dtype=tf.float32)
    X_test_10 = tf.convert_to_tensor(X_test_10, dtype=tf.float32) 
    X_test_11 = tf.convert_to_tensor(X_test_11, dtype=tf.float32)

    # Get predictions for each subgroup
    pred_00 = model(X_test_00, training=False)
    pred_01 = model(X_test_01, training=False)
    pred_10 = model(X_test_10, training=False)
    pred_11 = model(X_test_11, training=False)
    
    # Handle different output shapes - flatten if needed
    pred_00 = tf.reshape(pred_00, [-1])
    pred_01 = tf.reshape(pred_01, [-1])
    pred_10 = tf.reshape(pred_10, [-1])
    pred_11 = tf.reshape(pred_11, [-1])

    # Calculate equalized odds gaps
    if len(pred_00) > 0 and len(pred_10) > 0:
        gap_0 = tf.abs(tf.reduce_mean(pred_00) - tf.reduce_mean(pred_10))
    else:
        gap_0 = tf.constant(0.0)
        
    if len(pred_01) > 0 and len(pred_11) > 0:
        gap_1 = tf.abs(tf.reduce_mean(pred_01) - tf.reduce_mean(pred_11))
    else:
        gap_1 = tf.constant(0.0)

    gap = gap_0 + gap_1
    gap = gap.numpy()

    # Calculate average precision for all test data
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_scores = model(X_test_tf, training=False)
    y_scores = tf.reshape(y_scores, [-1]).numpy()
    
    ap = average_precision_score(y_test, y_scores)

    return ap, gap


def cal_importance(model, optimizer, x_train, y_train, neuron_ratio):
    # Create temporary variables to hold model weights for gradient calculation
    temp_weights = [tf.Variable(w.numpy()) for w in model.trainable_variables]
    
    # Convert inputs to tensors
    x_train = tf.constant(x_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        # Watch the temporary weights
        tape.watch(temp_weights)
        
        # Manual forward pass using temp weights
        # Assuming 3-layer network: input -> fc1 -> fc2 -> output
        x = x_train
        
        # Layer 1 (fc1)
        x = tf.nn.relu(tf.matmul(x, temp_weights[0], transpose_b=True) + temp_weights[1])
        
        # Layer 2 (fc2)  
        x = tf.nn.relu(tf.matmul(x, temp_weights[2], transpose_b=True) + temp_weights[3])
        
        # Output layer
        output = tf.nn.sigmoid(tf.matmul(x, temp_weights[4], transpose_b=True) + temp_weights[5])
        
        # Calculate BCE loss
        loss = tf.keras.losses.binary_crossentropy(y_train, output)
        loss = tf.reduce_mean(loss)
    
    # Calculate gradients
    grads = tape.gradient(loss, temp_weights)
    
    # Extract fc1 and fc2 weights and gradients (assuming indices 0 and 2 for weights)
    layer1 = temp_weights[0]  # fc1.weight
    layer2 = temp_weights[2]  # fc2.weight
    grad1 = grads[0]         # fc1.weight gradient
    grad2 = grads[2]         # fc2.weight gradient
    
    # Calculate importance criteria: (weight * gradient)^2 summed across input dimension
    nunits1 = tf.shape(layer1)[0]
    nunits2 = tf.shape(layer2)[0]
    
    criteria_layer1 = tf.reduce_sum(tf.square(layer1 * grad1), axis=1)
    criteria_layer2 = tf.reduce_sum(tf.square(layer2 * grad2), axis=1)
    
    # Get top-k important neurons
    k1 = tf.cast(neuron_ratio * tf.cast(nunits1, tf.float32), tf.int32)
    k2 = tf.cast(neuron_ratio * tf.cast(nunits2, tf.float32), tf.int32)
    k1 = tf.maximum(k1, 1)  # At least 1 neuron
    k2 = tf.maximum(k2, 1)  # At least 1 neuron
    
    value1, index1 = tf.nn.top_k(criteria_layer1, k=k1, sorted=True)
    value2, index2 = tf.nn.top_k(criteria_layer2, k=k2, sorted=True)
    
    return index1, index2, value1, value2


def cal_importance_gapReg(model, optimizer, batch_x_0_, batch_x_1_, neuron_ratio, mode=None):
    # Get model weights
    weights = model.get_weights()
    
    # Create temporary variables to hold model weights for gradient calculation
    temp_weights = [tf.Variable(w, trainable=True) for w in weights]
    
    with tf.GradientTape() as tape:
        # Watch the temporary weights
        tape.watch(temp_weights)
        
        # Build temporary model with temp weights for forward pass
        loss_reg = tf.constant(0.0)
        
        if mode == 'eo':
            for i in range(2):
                batch_x_0_i = tf.constant(batch_x_0_[i], dtype=tf.float32)
                batch_x_1_i = tf.constant(batch_x_1_[i], dtype=tf.float32)
                
                # Manual forward pass using temp weights
                # Get the actual shapes to determine the correct matrix multiplication
                x0 = batch_x_0_i
                x1 = batch_x_1_i
                
                # Layer 1: Check if we need transpose based on weight shapes
                w1_shape = tf.shape(temp_weights[0])
                if len(temp_weights[0].shape) == 2:
                    # Standard dense layer: input_features x output_features
                    if w1_shape[0] == tf.shape(x0)[1]:  # weight rows match input features
                        x0 = tf.nn.relu(tf.matmul(x0, temp_weights[0]) + temp_weights[1])
                        x1 = tf.nn.relu(tf.matmul(x1, temp_weights[0]) + temp_weights[1])
                    else:  # need transpose
                        x0 = tf.nn.relu(tf.matmul(x0, temp_weights[0], transpose_b=True) + temp_weights[1])
                        x1 = tf.nn.relu(tf.matmul(x1, temp_weights[0], transpose_b=True) + temp_weights[1])
                
                # Layer 2
                if len(temp_weights) > 2:  # Check if there's a second layer
                    w2_shape = tf.shape(temp_weights[2])
                    if w2_shape[0] == tf.shape(x0)[1]:
                        x0 = tf.nn.relu(tf.matmul(x0, temp_weights[2]) + temp_weights[3])
                        x1 = tf.nn.relu(tf.matmul(x1, temp_weights[2]) + temp_weights[3])
                    else:
                        x0 = tf.nn.relu(tf.matmul(x0, temp_weights[2], transpose_b=True) + temp_weights[3])
                        x1 = tf.nn.relu(tf.matmul(x1, temp_weights[2], transpose_b=True) + temp_weights[3])
                
                # Output layer
                if len(temp_weights) > 4:  # Check if there's an output layer
                    w_out_shape = tf.shape(temp_weights[4])
                    if w_out_shape[0] == tf.shape(x0)[1]:
                        output_0 = tf.nn.sigmoid(tf.matmul(x0, temp_weights[4]) + temp_weights[5])
                        output_1 = tf.nn.sigmoid(tf.matmul(x1, temp_weights[4]) + temp_weights[5])
                    else:
                        output_0 = tf.nn.sigmoid(tf.matmul(x0, temp_weights[4], transpose_b=True) + temp_weights[5])
                        output_1 = tf.nn.sigmoid(tf.matmul(x1, temp_weights[4], transpose_b=True) + temp_weights[5])
                else:
                    # No separate output layer, x0/x1 are already outputs
                    output_0 = tf.nn.sigmoid(x0)
                    output_1 = tf.nn.sigmoid(x1)
                
                loss_reg += tf.abs(tf.reduce_mean(output_0) - tf.reduce_mean(output_1))
                
        elif mode == 'dp':
            batch_x_0 = tf.constant(batch_x_0_, dtype=tf.float32)
            batch_x_1 = tf.constant(batch_x_1_, dtype=tf.float32)
            
            # Manual forward pass
            x0 = batch_x_0
            x1 = batch_x_1
            
            # Layer 1
            w1_shape = tf.shape(temp_weights[0])
            if w1_shape[0] == tf.shape(x0)[1]:
                x0 = tf.nn.relu(tf.matmul(x0, temp_weights[0]) + temp_weights[1])
                x1 = tf.nn.relu(tf.matmul(x1, temp_weights[0]) + temp_weights[1])
            else:
                x0 = tf.nn.relu(tf.matmul(x0, temp_weights[0], transpose_b=True) + temp_weights[1])
                x1 = tf.nn.relu(tf.matmul(x1, temp_weights[0], transpose_b=True) + temp_weights[1])
            
            # Layer 2
            if len(temp_weights) > 2:
                w2_shape = tf.shape(temp_weights[2])
                if w2_shape[0] == tf.shape(x0)[1]:
                    x0 = tf.nn.relu(tf.matmul(x0, temp_weights[2]) + temp_weights[3])
                    x1 = tf.nn.relu(tf.matmul(x1, temp_weights[2]) + temp_weights[3])
                else:
                    x0 = tf.nn.relu(tf.matmul(x0, temp_weights[2], transpose_b=True) + temp_weights[3])
                    x1 = tf.nn.relu(tf.matmul(x1, temp_weights[2], transpose_b=True) + temp_weights[3])
            
            # Output layer
            if len(temp_weights) > 4:
                w_out_shape = tf.shape(temp_weights[4])
                if w_out_shape[0] == tf.shape(x0)[1]:
                    output_0 = tf.nn.sigmoid(tf.matmul(x0, temp_weights[4]) + temp_weights[5])
                    output_1 = tf.nn.sigmoid(tf.matmul(x1, temp_weights[4]) + temp_weights[5])
                else:
                    output_0 = tf.nn.sigmoid(tf.matmul(x0, temp_weights[4], transpose_b=True) + temp_weights[5])
                    output_1 = tf.nn.sigmoid(tf.matmul(x1, temp_weights[4], transpose_b=True) + temp_weights[5])
            else:
                output_0 = tf.nn.sigmoid(x0)
                output_1 = tf.nn.sigmoid(x1)
            
            loss_reg = tf.abs(tf.reduce_mean(output_0) - tf.reduce_mean(output_1))
        else:
            print("Error: No mode in cal_importance_gapReg()")
            return None, None, None, None
    
    # Calculate gradients
    grads = tape.gradient(loss_reg, temp_weights)
    
    if grads is None or len(grads) < 3:
        print("Error: Could not compute gradients or insufficient layers")
        return None, None, None, None
    
    # Extract fc1 and fc2 weights and gradients
    layer1 = temp_weights[0]  # fc1.weight
    grad1 = grads[0]          # fc1.weight gradient
    
    if len(temp_weights) > 2:
        layer2 = temp_weights[2]  # fc2.weight
        grad2 = grads[2]          # fc2.weight gradient
    else:
        # Only one hidden layer, duplicate for compatibility
        layer2 = layer1
        grad2 = grad1
    
    # Calculate importance criteria: (weight * gradient)^2 summed across appropriate dimension
    if len(layer1.shape) == 2:
        criteria_layer1 = tf.reduce_sum(tf.square(layer1 * grad1), axis=0)  # Sum across input features
    else:
        criteria_layer1 = tf.reduce_sum(tf.square(layer1 * grad1))
        
    if len(layer2.shape) == 2:
        criteria_layer2 = tf.reduce_sum(tf.square(layer2 * grad2), axis=0)  # Sum across input features
    else:
        criteria_layer2 = tf.reduce_sum(tf.square(layer2 * grad2))
    
    # Get top-k important neurons
    nunits1 = tf.shape(criteria_layer1)[0]
    nunits2 = tf.shape(criteria_layer2)[0]
    
    k1 = tf.cast(neuron_ratio * tf.cast(nunits1, tf.float32), tf.int32)
    k2 = tf.cast(neuron_ratio * tf.cast(nunits2, tf.float32), tf.int32)
    k1 = tf.maximum(k1, 1)  # At least 1 neuron
    k2 = tf.maximum(k2, 1)  # At least 1 neuron
    
    # Ensure k doesn't exceed available neurons
    k1 = tf.minimum(k1, nunits1)
    k2 = tf.minimum(k2, nunits2)
    
    value1, index1 = tf.nn.top_k(criteria_layer1, k=k1, sorted=True)
    value2, index2 = tf.nn.top_k(criteria_layer2, k=k2, sorted=True)
    
    return index1, index2, value1, value2


def evaluate_dp_new(model, X_test, y_test, A_test):
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    output = model(X_test, training=False)[0].numpy()  # take first element
    pred = (output > 0.5).astype(np.int64).flatten()
    y_scores = output.flatten()
    dp = demographic_parity_difference(y_test, pred, sensitive_features=A_test)
    ap = average_precision_score(y_test, y_scores)
    return ap, dp


def evaluate_eo_new(model, X_test, y_test, A_test):
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    output = model(X_test, training=False)[0].numpy()  # take first element
    pred = (output > 0.5).astype(np.int64).flatten()
    y_scores = output.flatten()
    eo = equalized_odds_difference(y_test, pred, sensitive_features=A_test)
    ap = average_precision_score(y_test, y_scores)
    return ap, eo


def evaluate_difference(model, X_test, y_test, A_test, optimizer=None, mode=None):
    model.eval()
    optimizer_cal = copy.deepcopy(optimizer)
    idx_00 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 0)[0]))
    idx_01 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 1)[0]))
    idx_10 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 0)[0]))
    idx_11 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 1)[0]))

    X_test_00 = X_test[idx_00]
    X_test_01 = X_test[idx_01]
    X_test_10 = X_test[idx_10]
    X_test_11 = X_test[idx_11]

    X_test_00 = torch.tensor(X_test_00).cuda().float()
    X_test_01 = torch.tensor(X_test_01).cuda().float()
    X_test_10 = torch.tensor(X_test_10).cuda().float()
    X_test_11 = torch.tensor(X_test_11).cuda().float()

    X_test = torch.tensor(X_test).cuda().float()
    y_test = torch.tensor(y_test).cuda().float()

    # diff
    X_test_0 = torch.cat((X_test_00, X_test_01), dim=0)
    X_test_1 = torch.cat((X_test_10, X_test_11), dim=0)
    outputs_0, sum1_0, sum2_0, sum3_0 = model(X_test_0)
    outputs_1, sum1_1, sum2_1, sum3_1 = model(X_test_1)

    outputs_00, sum1_00, sum2_00, sum3_00 = model(X_test_00)
    outputs_01, sum1_01, sum2_01, sum3_01 = model(X_test_01)
    outputs_10, sum1_10, sum2_10, sum3_10 = model(X_test_10)
    outputs_11, sum1_11, sum2_11, sum3_11 = model(X_test_11)

    # difference abs
    dif1_abs = abs(sum1_0 - sum1_1)
    dif2_abs = abs(sum2_0 - sum2_1)
    dif3_abs = abs(sum3_0 - sum3_1)

    dif1_0_abs = abs(sum1_00 - sum1_10)
    dif2_0_abs = abs(sum2_00 - sum2_10)
    dif3_0_abs = abs(sum3_00 - sum3_10)

    dif1_1_abs = abs(sum1_01 - sum1_11)
    dif2_1_abs = abs(sum2_01 - sum2_11)
    dif3_1_abs = abs(sum3_01 - sum3_11)

    torch.set_printoptions(precision=10, sci_mode=False)

    # importance

    # separate class
    batch_x_0_ = [X_test_00, X_test_01]
    batch_x_1_ = [X_test_10, X_test_11]

    if mode == 'eo':
        important_index1_Reg, important_index2_Reg, important_value1_Reg, important_value2_Reg = cal_importance_gapReg(
            model, optimizer_cal, batch_x_0_, batch_x_1_, 1, mode='eo')
        important_index1_CE, important_index2_CE, important_value1_CE, important_value2_CE = cal_importance(
            model, optimizer_cal, X_test, y_test, 1)
        dif1_abs = dif1_0_abs + dif1_1_abs
        dif2_abs = dif2_0_abs + dif2_1_abs
        dif3_abs = dif3_0_abs + dif3_1_abs
    elif mode == 'dp':
        important_index1_Reg, important_index2_Reg, important_value1_Reg, important_value2_Reg = cal_importance_gapReg(
            model, optimizer_cal, X_test_0, X_test_1, 1, mode='dp')
        important_index1_CE, important_index2_CE, important_value1_CE, important_value2_CE = cal_importance(
            model, optimizer_cal, X_test, y_test, 1)
    else:
        print("Error: no mode in evaluate_difference()")

    index1_Reg_CE = [list(important_index1_CE).index(i) for i in important_index1_Reg]
    index2_Reg_CE = [list(important_index2_CE).index(i) for i in important_index2_Reg]

    important_index1 = important_index1_Reg
    important_index2 = important_index2_Reg
    print("*" * 20)
    print('important_index1', important_index1_Reg)
    print('important_index1_CE', important_index1_CE)
    print('important_index2', important_index2_Reg)
    print('important_index2_CE', important_index2_CE)
    print("*" * 20)
    for i in range(10):
        print(f"index1_Reg_CE_{10*i}%-{10*i + 10}%:", index1_Reg_CE[20*i:20*i+20])
        print(f"index2_Reg_CE_{10*i}%-{10*i + 10}%:", index2_Reg_CE[20*i:20*i+20])

    print('\ndiff_value1')
    print('top10%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[0:20]].mean().item())
    print('top10%-20%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[20:40]].mean().item())
    print('top20%-30%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[40:60]].mean().item())
    print('top30%-40%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[60:80]].mean().item())
    print('top40%-50%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[80:100]].mean().item())
    print('top50%-60%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[100:120]].mean().item())
    print('top60%-70%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[120:140]].mean().item())
    print('top70%-80%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[140:160]].mean().item())
    print('top80%-90%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[160:180]].mean().item())
    print('top90%-100%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[180:200]].mean().item())

    print('\ndiff_value2')
    print('top10%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[0:20]].mean().item())
    print('top10%-20%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[20:40]].mean().item())
    print('top20%-30%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[40:60]].mean().item())
    print('top30%-40%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[60:80]].mean().item())
    print('top40%-50%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[80:100]].mean().item())
    print('top50%-60%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[100:120]].mean().item())
    print('top60%-70%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[120:140]].mean().item())
    print('top70%-80%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[140:160]].mean().item())
    print('top80%-90%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[160:180]].mean().item())
    print('top90%-100%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[180:200]].mean().item())

    print('\ndiff_value3')
    print('important_avg_diff_value3: ', dif3_abs.mean().item())


def mask_neuron_test(model, X_test, y_test, A_test, criterion, optimizer=None, mode=None):
    model.eval()
    optimizer_cal = copy.deepcopy(optimizer)
    idx_00 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 0)[0]))
    idx_01 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 1)[0]))
    idx_10 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 0)[0]))
    idx_11 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 1)[0]))

    X_test_00 = X_test[idx_00]
    X_test_01 = X_test[idx_01]
    X_test_10 = X_test[idx_10]
    X_test_11 = X_test[idx_11]

    X_test_00 = torch.tensor(X_test_00).cuda().float()
    X_test_01 = torch.tensor(X_test_01).cuda().float()
    X_test_10 = torch.tensor(X_test_10).cuda().float()
    X_test_11 = torch.tensor(X_test_11).cuda().float()

    X_test = torch.tensor(X_test).cuda().float()
    y_test = torch.tensor(y_test).cuda().float()

    # diff
    X_test_0 = torch.cat((X_test_00, X_test_01), dim=0)
    X_test_1 = torch.cat((X_test_10, X_test_11), dim=0)

    torch.set_printoptions(precision=10, sci_mode=False)

    # importance

    # separate class
    batch_x_0_ = [X_test_00, X_test_01]
    batch_x_1_ = [X_test_10, X_test_11]

    if mode == 'eo':
        important_index1, important_index2, important_value1, important_value2 = cal_importance_gapReg(
            model, optimizer_cal, batch_x_0_, batch_x_1_, 1, mode='eo')
    elif mode == 'dp':
        important_index1, important_index2, important_value1, important_value2 = cal_importance_gapReg(
            model, optimizer_cal, X_test_0, X_test_1, 1, mode='dp')
    else:
        print("Error: no mode in evaluate_difference()")

    interval1 = len(important_index1) // 5
    interval2 = len(important_index2) // 5

    # no mask
    output = model.mask_forward(X_test)

    y_scores = output[:, 0].data.cpu().numpy()
    ap_no_mask = average_precision_score(y_test.cpu(), y_scores)
    loss_no_mask = criterion(output, y_test).item()
    print(f'no mask')
    print("loss:", loss_no_mask)
    print('ap:', ap_no_mask)

    # mask
    ap = []
    loss = []
    for j in range(5):
        output = model.mask_forward(X_test, important_index1[interval1 * j:interval1 * j + interval1],
                                         important_index2[interval2 * j:interval2 * j + interval2])

        y_scores = output[:, 0].data.cpu().numpy()
        ap_j = average_precision_score(y_test.cpu(), y_scores)
        ap.append(ap_j)

        loss_sup = criterion(output, y_test)
        loss.append(loss_sup.item())

    for i in range(5):
        print(f'{20 * i}%-{20 * i + 20}%:')
        print("loss:", loss[i])
        print('ap:', ap[i])
        print("loss_dif:", loss[i]-loss_no_mask)
        print('ap_dif:', ap[i]-ap_no_mask)

