import numpy as np
import sklearn.metrics as skm
import tensorflow as tf
from tensorflow.keras.losses import (
    BinaryCrossentropy,
    # CategoricalCrossentropy,
    # SparseCategoricalCrossentropy,
)
from multiprocessing import Pool
# ------------------------------------------------------------------------
tf.get_logger().setLevel(3) # Log errors only
_BC_ = BinaryCrossentropy()
# ------------------------------------------------------------------------
def f1_over_bin_cross(dataframe_y_true, dataframe_y_pred):
    f1 = skm.f1_score(dataframe_y_true, dataframe_y_pred, average="weighted")
    bc = _BC_(dataframe_y_true, dataframe_y_pred)
    return f1/bc
def f1_score(dataframe_y_true, dataframe_y_pred):
    return skm.f1_score(dataframe_y_true, dataframe_y_pred, average="weighted")
def matthews_corr(dataframe_y_true, dataframe_y_pred):
    return skm.matthews_corrcoef(dataframe_y_true, dataframe_y_pred)
def bin_cross(dataframe_y_true, dataframe_y_pred):
    bc = _BC_(dataframe_y_true, dataframe_y_pred)
    return 1/bc
# ====================== THRESOLDING METHODS =============================
def fixed_thresholding(input_data, labels, thresholds):

    output_data = input_data[:][labels.columns[:]].copy()

    for col in output_data.columns:
        output_data[col].values[:] = 0
    
    for i,j in enumerate(labels.columns[:]):
        output_data[j] = [v > thresholds[i] for v in output_data[j]]
        print(output_data)
        
    return output_data
# ------------------------------------------------------------------------
def threshold_worker(input_column, y_true_col, output_format, metric, precision, verbose):
    
    # index = 0 --> [sti, sti+1, sti+2] : sti max
    # --> dic[index] = ti : best threshold
    # new range from sti-1 to sti+1
    # [x, x, x, x, x] --> [x, [x, x, x, x, x], x, x, x] --> [x, [x, x, [x, x, x, x, x], x, x], x, x, x]

    unit = 0.1; best_t = 0
    dic = np.arange(1, 10) * unit        

    # do for each precision level
    for p in range(precision):
        thresh_scores = [] # thresholds list

        for t in dic:
            # ---
            index = input_column.index[input_column >= t]
            column = output_format.copy()
            column.loc[index] = 1
            score = metric(y_true_col, column) # get F1 score of threshold t
            # ---
            thresh_scores.append(score) # append threshold scores to list

        best_t = dic[np.argmax(thresh_scores)] # takes first of best thresholds from score
        
        if (p+1 >= precision): break
        
        dic = np.arange(best_t - unit, best_t + unit, unit/10)
        dic = dic[dic > 0]

        unit /= 10

    if (verbose): print("%s : %s -- DONE" % (input_column.name , best_t))
    return best_t
# ------------------------------------------------------------------------
def compute_multithread(input_data, y_true, labels, metric=f1_score, precision=3, workers=32, verbose=True):
    
    genres = labels.columns[:]

    # ---
    blank_data = input_data[:][genres].copy()
    for col in blank_data.columns:
        blank_data[col].values[:] = 0
    # ---

    with Pool(processes=workers) as pool:
        
        results = {
            genre:pool.apply_async(
                threshold_worker, 
                (
                    input_data[genre][:], 
                    y_true[genre][:],
                    blank_data[genre][:], 
                    metric,
                    precision,
                    verbose
                )
            ) for genre in genres
        }

        thresholds = {
            genre:res.get(timeout=None)
            for (genre, res) in results.items()
        }

    # ---
    return thresholds
# ------------------------------------------------------------------------
def compute_singlethread(input_data, y_true, labels, metric=f1_score, precision=3):
    
    genres = labels.columns[:]
    thresholds = {lab:0 for lab in genres} # final threshold list

    # ---
    blank_data = input_data[:][labels.columns[:]].copy()
    for col in blank_data.columns:
        blank_data[col].values[:] = 0
    # ---

    for genre in genres:
        unit = 0.1; best_t = 0
        dic = np.arange(1, 10) * unit        

        # do for each precision level
        for p in range(precision):
            thresh_scores = [] # thresholds list

            for t in dic:
                # ---
                index = input_data[genre].index[input_data[genre] > t]
                column = blank_data[genre].copy()
                column.loc[index] = 1
                score = metric(y_true[genre], column) # get F1 score of threshold t
                # ---
                thresh_scores.append(score) # append threshold scores to list

            best_t = dic[np.argmax(thresh_scores)] # takes first of best thresholds from score
            
            if (p+1 >= precision): break
            
            dic = np.arange(best_t - unit, best_t + unit, unit/10)
            dic = dic[dic > 0]

            unit /= 10
        
        thresholds[genre] = best_t

    # ---
    return thresholds
# ========================================================================