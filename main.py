import numpy as np
import pandas as pd
import dynamic_thresholding as dt
import sklearn.metrics as skm
import sys
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
_METRICS_ = {
    # 'f1_bc' : dt.f1_over_bin_cross,
    # 'f1'    : dt.f1_score,
    # 'bc'    : dt.bin_cross,
    'mcc'   : dt.matthews_corr
}
# ------------------------------------------------------------------------
_DEFAULT_METRIC_ = 'f1'
_DEFAULT_PRECISION_ = 3
_DEFAULT_MODE_ = 3
# ------------------------------------------------------------------------
try:
    mode = int(sys.argv[1]) if sys.argv else _DEFAULT_MODE_
    precision = int(sys.argv[2]) if sys.argv else _DEFAULT_PRECISION_
    metric = sys.argv[3] if sys.argv[3] in _METRICS_.keys() else _DEFAULT_METRIC_
    print("[DEBUG] using launch parameters...")
    
except (IndexError):
    mode = _DEFAULT_MODE_
    precision = _DEFAULT_PRECISION_
    metric = _DEFAULT_METRIC_
    print("[DEBUG] using default parameters...")
# ------------------------------------------------------------------------
DRIVE_FOLDER_PATH = './'

CSV_FILE_Y_TRUE = DRIVE_FOLDER_PATH + 'train_y_YZCqwbD.csv'
CSV_FILE_Y_PRED = DRIVE_FOLDER_PATH + 'train_X_beAW6y8.csv'
CSV_FILE_X_TEST = DRIVE_FOLDER_PATH + 'test_X_nPKMRWK.csv'
CSV_FILE_LABELS = DRIVE_FOLDER_PATH + 'annex/mewo-labels.csv'
CSV_TEST_OUTPUT_FOLD = DRIVE_FOLDER_PATH + 'results/'
CSV_TRAIN_OUTPUT_FOLD = DRIVE_FOLDER_PATH + 'benchmarks/'

CSV_OUTPUT_TRAIN = CSV_TRAIN_OUTPUT_FOLD + 'train_y_dt_%s_p%s.csv'
CSV_OUTPUT_TEST = CSV_TEST_OUTPUT_FOLD + 'test_y_dt_%s_p%s.csv'
CSV_OUTPUT_THRESHOLDS = CSV_TEST_OUTPUT_FOLD + 'thresh_dt_%s_p%s.csv'
# --------------------------------------------------------------------
if __name__ == '__main__' :
        
    if (mode):

        df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
        df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
        df_x_test = pd.read_csv(CSV_FILE_X_TEST, index_col=0, sep=',')
        labels = pd.read_csv(CSV_FILE_LABELS, index_col=0, sep=',')

        # with dt.timing():
        #     # thresholds = dyn_thresh_single(df_y_pred, df_y_true, labels, precision=3)
        #     thresholds = dt.compute_multithread(
        #         input_data  = df_y_pred,
        #         y_true      = df_y_true, 
        #         labels      = labels, 
        #         metric      = _METRICS_[metric], 
        #         precision   = precision, 
        #         workers     = 12
        #     )

        # if (mode == 1):
        #     # Compute train
        #     df_y_train = dt.apply_thresholds(thresholds, df_y_pred)
        #     print("F1 score: %s" % dt.evaluate(df_y_true, df_y_train))
        #     # df_y_train.to_csv(CSV_OUTPUT_TRAIN % (metric, precision))

        # elif (mode == 2):
        #     # Compute test
        #     df_y_test = dt.apply_thresholds(thresholds, df_x_test)
        #     df_y_test.to_csv(CSV_OUTPUT_TEST % (metric, precision))

        #     df_thresh = pd.DataFrame.from_dict(thresholds, orient='index')
        #     df_thresh.to_csv(CSV_OUTPUT_THRESHOLDS % (metric, precision))

        # elif (mode == 3):

        combinations = dt.generate_configs(
            list(_METRICS_.values()),   # metric functions
            list(range(1, 6))          # precision levels
        )

        with open('./.opt/mp_scores.txt', 'a') as score_file :
                
            for i, (m, p) in enumerate(combinations) :

                verb = "[DEBUG] Combination %i of %i : \n\tMetric: %s\n\tPrecision: %s"
                print(verb % (i, len(combinations), m.__name__, p))

                thresholds = dt.compute_multithread(
                    input_data  = df_y_pred,
                    y_true      = df_y_true,  
                    labels      = labels, 
                    metric      = m,
                    precision   = p,
                    workers     = 12,
                    verbose     = False
                )

                y_train = dt.apply_thresholds(thresholds, df_y_pred)

                score = dt.evaluate(
                    df_y_true=df_y_true,
                    df_y_pred=y_train
                )

                score_file.write("\n%s | %i : %f" % (m.__name__, p, score))

    else :
        # Debug
        combinations = generate_configs(
            list(_METRICS_.values()),   # metric functions
            list(range(0, 11))          # precision levels
        )

        print("[DEBUG] configs:", combinations)