import numpy as np
import pandas as pd
import dynamic_thresholding as dt
import sklearn.metrics as skm
import sys, json
# --------------------------------------------------------------------
_METRICS_ = {
    'f1_bc' : dt.f1_over_bin_cross,
    'f1'    : dt.f1_score,
    'bc'    : dt.bin_cross,
    'mcc'   : dt.matthews_corr
}

_DEFAULT_METRIC_ = _METRICS_['f1']
_DEFAULT_PRECISION_ = 3
_DEFAULT_MODE_ = 3 # testing

DRIVE_FOLDER_PATH = './'

CSV_FILE_Y_TRUE = DRIVE_FOLDER_PATH + 'train_y_YZCqwbD.csv'
CSV_FILE_Y_PRED = DRIVE_FOLDER_PATH + 'train_X_beAW6y8.csv'
CSV_FILE_X_TEST = DRIVE_FOLDER_PATH + 'test_X_nPKMRWK.csv'
CSV_FILE_LABELS = DRIVE_FOLDER_PATH + 'annex/mewo-labels.csv'
CSV_TEST_OUTPUT_FOLD = DRIVE_FOLDER_PATH + 'results/'
CSV_TRAIN_OUTPUT_FOLD = DRIVE_FOLDER_PATH + '.bench/'

JSON_TRAIN_SCORE_FILE = CSV_TRAIN_OUTPUT_FOLD + 'mp_scores.json'
CSV_OUTPUT_TRAIN = CSV_TRAIN_OUTPUT_FOLD + 'train_y_dt_%s_p%s.csv'
CSV_OUTPUT_TEST = CSV_TEST_OUTPUT_FOLD + 'test_y_dt_%s_p%s.csv'
CSV_OUTPUT_THRESHOLDS = CSV_TEST_OUTPUT_FOLD + 'thresh_dt_%s_p%s.csv'
# ------------------------------------------------------------------------
def generate_configs(*arrays):
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))
# ------------------------------------------------------------------------
def evaluate(df_y_true, df_y_pred, avg="weighted"):
    return skm.f1_score(
        df_y_true, 
        df_y_pred, 
        average=avg
    )
# ------------------------------------------------------------------------
def apply_thresholds(thresholds, input_data):
    
    output_data = input_data[:][thresholds.keys()].copy()

    for (genre, t) in thresholds.items():
        output_data[genre].values[:] = 0
        indices = input_data.index[input_data[genre] > t]
        output_data[genre].loc[indices] = 1

    return output_data.astype('int32')
# ------------------------------------------------------------------------
def get_scores(combs, df_y_pred, df_y_true, labels):

    score_dict = {i:{} for i in range(len(combs))}
            
    for i, (m, p) in enumerate(combinations) :

        verb = "[DEBUG] Combination %i of %i : \n\tMetric: %s\n\tPrecision: %s"
        print(verb % (i+1, len(combinations), m.__name__, p))

        thresholds = dt.compute_multithread(
            input_data  = df_y_pred,
            y_true      = df_y_true,  
            labels      = labels, 
            metric      = m,
            precision   = p,
            workers     = 24,
            verbose     = False
        )

        y_train = apply_thresholds(thresholds, df_y_pred)

        score = evaluate(
            df_y_true=df_y_true,
            df_y_pred=y_train
        )

        score_dict[i] = {
            "metric" : m.__name__,
            "precision" : p,
            "score" : score
        }

    with open('./.opt/mp_scores.json', 'w') as score_file :
        json.dump(score_dict, score_file)

    # score_file.write("\n%s | %i : %f" % (m.__name__, p, score))
# ------------------------------------------------------------------------
def inspect_scores(metric='all', precision='all'):
    with open(JSON_TRAIN_SCORE_FILE, 'r') as score_file :
        scores = json.load(score_file)
    
# ------------------------------------------------------------------------
param_string = "[LAUNCH] Using %s parameters (mode: %s, metric: \"%s\", precision: %s)"
warn_string = "[WARN] launch parameter %s wasn't satisfied, using default (%S)"
param, mode, precision, metric = "default", _DEFAULT_MODE_, _DEFAULT_PRECISION_, _DEFAULT_METRIC_

if (len(sys.argv) > 1):
    params = "launch"
    try: mode = int(sys.argv[1]) if sys.argv else _DEFAULT_MODE_
    except (IndexError): print(warn_string % ("\nmode\n", _DEFAULT_MODE_))
    try: precision = int(sys.argv[2]) if sys.argv else _DEFAULT_PRECISION_
    except (IndexError): print(warn_string % ("\nprecision\n", _DEFAULT_PRECISION_))
    try: metric = _METRICS_[sys.argv[3]] if sys.argv[3] in _METRICS_.keys() else _DEFAULT_METRIC_
    except (IndexError): print(warn_string % ("\nmetric\n", _DEFAULT_METRIC_.__name__))

print(param_string % (param, mode, metric.__name__, precision))
# ------------------------------------------------------------------------
if __name__ == '__main__' :
        
    if (mode):

        df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
        df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
        df_x_test = pd.read_csv(CSV_FILE_X_TEST, index_col=0, sep=',')
        labels = pd.read_csv(CSV_FILE_LABELS, index_col=0, sep=',')

        if (mode in [1, 2]):

            with dt.timing():
                # thresholds = dyn_thresh_single(df_y_pred, df_y_true, labels, precision=3)
                thresholds = dt.compute_multithread(
                    input_data  = df_y_pred,
                    y_true      = df_y_true, 
                    labels      = labels, 
                    metric      = metric, 
                    precision   = precision, 
                    workers     = 12
                )

            if (mode == 1):
                # Mode 1: Compute train
                df_y_train = dt.apply_thresholds(thresholds, df_y_pred)
                print("F1 score: %s" % dt.evaluate(df_y_true, df_y_train))
                # df_y_train.to_csv(CSV_OUTPUT_TRAIN % (metric.__name__, precision))

            elif (mode == 2):
                # Mode 2: Compute test
                df_y_test = dt.apply_thresholds(thresholds, df_x_test)
                df_y_test.to_csv(CSV_OUTPUT_TEST % (metric.__name__, precision))

                df_thresh = pd.DataFrame.from_dict(thresholds, orient='index')
                df_thresh.to_csv(CSV_OUTPUT_THRESHOLDS % (metric.__name__, precision))

        elif (mode == 3):
            # Mode 3: TESTING

            combinations = generate_configs(
                list(_METRICS_.values()),   # metric functions
                list(range(1, 11))          # precision levels
            )

            get_scores(combinations, df_y_pred, df_y_true, labels)

    else :
        # Mode 0: DEBUG
        inspect_scores()