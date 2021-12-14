import json
from logging import error
import sys
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import dynamic_thresholding as DT
import typing as T
from time import time
# ------------------------------------------------------------------------
@contextmanager
def timing() -> None:
    start = time()
    print("[TIMER] Started timer")
    yield
    ellapsed_time = time() - start
    m = ellapsed_time/60
    print("[TIMER] Process ended - Lasted: {:.2f} minutes".format(m))
# ------------------------------------------------------------------------
_METRICS_ = {func.__name__: func for func in [
    DT.f1_over_bin_cross,
    DT.f1_score,
    DT.bin_cross,
    DT.matthews_corr,
    DT.pr_auc
    # dt.cat_cross,
    # dt.sparse_cat_cross
]}
# ------------------------------------------------------------------------
_DEFAULT_METRIC_ = _METRICS_[DT.f1_score.__name__]
_DEFAULT_PRECISION_ = 3
_DEFAULT_MODE_ = 0 # 

_METRIC_DOMAIN_ = _METRICS_.keys()
_PRECISION_DOMAIN_ = range(1, 11)
_MODE_DOMAIN_ = range(0, 4)

def str_range(r: range): return "[%i:%i]" % (r[0], r[-1])
# ------------------------------------------------------------------------
MAIN_FOLDER_PATH = './' # Interoperability with Google Collab

CSV_FILE_Y_TRUE = MAIN_FOLDER_PATH + 'train_y_YZCqwbD.csv'
CSV_FILE_Y_PRED = MAIN_FOLDER_PATH + 'train_X_beAW6y8.csv'
CSV_FILE_X_TEST = MAIN_FOLDER_PATH + 'test_X_nPKMRWK.csv'
CSV_FILE_LABELS = MAIN_FOLDER_PATH + 'annex/mewo-labels.csv'
CSV_TEST_OUTPUT_FOLD = MAIN_FOLDER_PATH + 'results/'
CSV_TRAIN_OUTPUT_FOLD = MAIN_FOLDER_PATH + '.bench/'

JSON_TRAIN_SCORE_FILE = CSV_TRAIN_OUTPUT_FOLD + 'mp_scores.json'
CSV_OUTPUT_TRAIN = CSV_TRAIN_OUTPUT_FOLD + 'train_y_dt_%s_p%s.csv'
CSV_OUTPUT_TEST = CSV_TEST_OUTPUT_FOLD + 'test_y_dt_%s_p%s.csv'
CSV_OUTPUT_THRESHOLDS = CSV_TEST_OUTPUT_FOLD + 'thresh_dt_%s_p%s.csv'
# ------------------------------------------------------------------------
def config_hash(metric, precision):
    return '%s_%i' % (metric.__name__, precision)
# ------------------------------------------------------------------------
def generate_configs(*arrays):
    """Built each possible combination between available parameters"""
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
def dump_scores(combs, df_y_pred, df_y_true, labels, append=True):

    score_dict = {i: {} for i in range(len(combs))}
    if append:
        with open(JSON_TRAIN_SCORE_FILE, 'r') as score_file:
            score_dict = json.load(score_file)

    for i, (m, p) in enumerate(combs):

        _hash = config_hash(m, p)
        if _hash in score_dict: 
            print("[DEBUG] Skipping (%s, %i): a value already exists"%(m.__name__, p))
            continue

        verb = "[DEBUG] Combination %i of %i : \n\tMetric: %s\n\tPrecision: %s"
        print(verb % (i+1, len(combs), m.__name__, p))

        thresholds = DT.compute_multithread(
            input_data=df_y_pred,
            y_true=df_y_true,
            labels=labels,
            metric=m,
            precision=p,
            workers=24,
            verbose=False
        )

        y_train = apply_thresholds(thresholds, df_y_pred)
        score = evaluate(df_y_true, y_train)
        score_dict[_hash] = {
            "metric": m.__name__,
            "precision": p,
            "score": score
        }

    with open(JSON_TRAIN_SCORE_FILE, 'w') as score_file:
        json.dump(score_dict, score_file, separators=(", ", ": "), indent=4)

    return score_dict
# ------------------------------------------------------------------------
def extract_scores(metric=[], precision=[]):
    with open(JSON_TRAIN_SCORE_FILE, 'r') as score_file:
        scores = pd.DataFrame(json.load(score_file)).T
    return scores.sort_values(by='score', ascending=False)
# ------------------------------------------------------------------------
def get_score(df_x_train, df_y_true, labels, metric, precision):
    with timing():
        # thresholds = dt.dyn_thresh_single(df_y_pred, df_y_true, labels, precision=3)
        thresholds = DT.compute_multithread(
            input_data=df_x_train,
            y_true=df_y_true,
            labels=labels,
            metric=metric,
            precision=precision,
            workers=12,
        )

    df_y_train = apply_thresholds(thresholds, df_x_train)
    return thresholds, evaluate(df_y_true, df_y_train)
# ------------------------------------------------------------------------
def export_to_csv(dataset, thresholds, metric, precision):
    df_y = apply_thresholds(thresholds, dataset)
    df_y.to_csv(CSV_OUTPUT_TEST % (metric.__name__, precision))
    df_thresh = pd.DataFrame.from_dict(thresholds, orient='index')
    df_thresh.to_csv(CSV_OUTPUT_THRESHOLDS % (metric.__name__, precision))
# ------------------------------------------------------------------------
def compute_all_and_export_best(df_x_train, df_x_test, df_y_true, labels):
    
    combinations = generate_configs(
        list(_METRICS_.values()),   # metric functions
        list(range(1, 11))          # precision levels
    )

    dump_scores(combinations, df_x_train, df_y_true, labels)

    scores = extract_scores()
    best_metric, best_precision, _ = scores.iloc[0].values
    print('# Best score - metric: %s precision: %i'%(best_metric, best_precision))

    thresholds = DT.compute_multithread(
        input_data  = df_x_train,
        y_true      = df_y_true,
        labels      = labels,
        metric      = _METRICS_[best_metric],
        precision   = best_precision,
        workers     = 12,
        verbose     = True
    )

    export_to_csv(df_x_test, thresholds, _METRICS_[best_metric], best_precision)
#
if __name__ == '__main__':
    
    # -----
    param_string = "[DEBUG] using %s parameters (mode: %s, metric: \"%s\", precision: %s)"
    warn_string = "[WARN] launch parameter \"%s\" wasn't satisfied (must be in %s), using default (%s)"
    param, mode, precision, metric = "default", _DEFAULT_MODE_, _DEFAULT_PRECISION_, _DEFAULT_METRIC_

    if (len(sys.argv) > 1):
        param = "launch"

        try:
            _val = int(sys.argv[1])
            assert(int(sys.argv[1]) in _MODE_DOMAIN_)
            mode = _val
        except:
            print(warn_string % ("mode", str_range(_MODE_DOMAIN_), _DEFAULT_MODE_))

        try:
            _val = sys.argv[2]
            assert(_val in _METRIC_DOMAIN_)
            metric = _METRICS_[_val]
        except:
            print(warn_string % ("metric", '|'.join(
                _METRIC_DOMAIN_), _DEFAULT_METRIC_.__name__))

        try:
            _val = int(sys.argv[3])
            assert(_val in _PRECISION_DOMAIN_)
            precision = _val
        except:
            print(warn_string % ("precision", str_range(
                _PRECISION_DOMAIN_), _DEFAULT_PRECISION_))

    print(param_string % (param, mode, metric.__name__, precision))

    # -----
    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',') # binary classification (labels)
    df_x_train = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',') # fuzzy classification to apply thresholding to
    df_x_test = pd.read_csv(CSV_FILE_X_TEST, index_col=0, sep=',') # data to predict for grading over at the challenge page
    labels = pd.read_csv(CSV_FILE_LABELS, index_col=0, sep=',')

    # -----
    if mode:

        if (mode in [1, 2]):

            thresholds, f1score = get_score(
                df_x_train, 
                df_y_true, 
                labels, 
                metric, 
                precision
            )

            print("[INFO] Config: (%s, %i)" % (metric.__name__, precision))
            print("[INFO] Global score using the F1 metric: {0:.10f}".format(f1score))

            if (mode == 2):
                export_to_csv(df_x_test, thresholds, metric, precision)

        elif (mode == 3):
            compute_all_and_export_best(df_x_train, df_x_test, df_y_true, labels)

    else:
        
        # Mode 0: Test section
        df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',') # binary classification (labels)
        df_x_train = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',') # fuzzy classification to apply thresholding to
        # df_x_test = pd.read_csv(CSV_FILE_X_TEST, index_col=0, sep=',') # data to predict for grading over at the challenge page
        labels = pd.read_csv(CSV_FILE_LABELS, index_col=0, sep=',')

        # Are there any columns empty?
        # zero_cols = df_y_true.any(axis='index')
        # print(zero_cols.all()) # Yes!

        # Are there any lines empty?
        # zero_rows = df_y_true.any(axis='columns')
        # print(zero_rows.all()) # Yes!

        tags = labels.columns.to_list()
        classes = labels.T['class']
        categories = labels.T['category']
        # --- 
        genres = classes[classes == 'Genres'].index.to_list()
        instruments = classes[classes == 'Instruments'].index.to_list()
        moods = classes[classes == 'Moods'].index.to_list()
        # ---
        # genres_corr = df_y_true[genres].corr() # corrélation entre les colonnes associées à la classe 'Genres'
        # instruments_corr = df_y_true[instruments].corr() # corrélation entre les colonnes associées à la classe 'Instruments'
        # moods_corr = df_y_true[moods].corr() # corrélation entre les colonnes associées à la classe 'Moods'
        # --- 
        # col_to_cat = df_y_true.rename(columns={col:categories[col] for col in tags})
        # genres_cat_corr = col_to_cat.corr()

        # genres_cat_corr_avg = pd.DataFrame(columns=categories)
        # for cat in categories:
        #     _cols = genres_cat_corr[cat]
        #     if _cols.shape[1] > 1: pass


        #_t = col_to_cat_corr['genres-jazz']
        # np.mean(_t[_t != 1].T)


    print("[DEBUG] Done.")