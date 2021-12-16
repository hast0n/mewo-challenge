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
import argparse
# ------------------------------------------------------------------------
_METRICS_ = {func.__name__: func for func in [
    # DT.f1_over_bin_cross,
    # DT.accuracy,
    # DT.log_loss,
    # dt.cat_cross,
    # dt.sparse_cat_cross
    # DT.bin_cross,
    DT.f1_score,
    DT.matthews_corr,
    DT.pr_auc
]}
# ------------------------------------------------------------------------
_DEFAULT_METRIC_ = DT.f1_score.__name__
_DEFAULT_PRECISION_ = 3
_DEFAULT_MODE_ = 2

_METRIC_DOMAIN_ = _METRICS_.keys()
_PRECISION_DOMAIN_ = range(1, 11)
_MODE_DOMAIN_ = range(0, 4)

def str_range(r: range): return "[%i:%i]" % (r[0], r[-1])
# ------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("-e", "--execution-mode",
                    help="Execution mode.",
                    dest='mode',
                    type=int,
                    metavar="MODE",
                    choices=_MODE_DOMAIN_,
                    default=_DEFAULT_MODE_,
                    required=True
                    )
parser.add_argument("-m", "--metric",
                    help="Metric used to compute classification score.",
                    default=_DEFAULT_METRIC_,
                    metavar="METRIC",
                    choices=_METRIC_DOMAIN_
                    )
parser.add_argument("-p", "--precision",
                    help="Number of decimals to compute for thresholds.",
                    type=int, 
                    default=_DEFAULT_PRECISION_,
                    metavar="PRECISION",
                    choices=_PRECISION_DOMAIN_
                    )
parser.add_argument("-v", "--verbosity",
                    help="Verbosity.",
                    action="store_false"
                    )
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
@contextmanager
def timing() -> None:
    start = time()
    print("[TIMER] Started timer")
    yield
    ellapsed_time = time() - start
    m = ellapsed_time/60
    print("[TIMER] Process ended - Lasted: {:.2f} minutes".format(m))
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

    df_y = apply_thresholds(thresholds, df_x_train)
    f1score = evaluate(df_y_true, df_y)

    return thresholds, f1score
# ------------------------------------------------------------------------
def export_to_csv(dataset, metric=None, precision=0, thresholds=None):

    metric_name = metric.__name__ if callable(metric) else str(metric)
    df_y_filename = CSV_OUTPUT_TEST % (metric_name, precision)

    df_y = apply_thresholds(thresholds, dataset)
    df_y.to_csv(df_y_filename)

    print("[INFO] Exported classification to %s" % df_y_filename)

    if thresholds is not None:
        thresh_filename = CSV_OUTPUT_THRESHOLDS % (metric_name, precision)

        df_thresh = pd.DataFrame.from_dict(thresholds, orient='index')
        df_thresh.to_csv(thresh_filename)
        
        print("[INFO] Exported thresholds to %s" % thresh_filename)
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

    export_to_csv(df_x_test, _METRICS_[best_metric], best_precision, thresholds)
# ------------------------------------------------------------------------
if __name__ == '__main__':
    
    args = parser.parse_args()
    mode = args.mode
    metric = _METRICS_[args.metric]
    precision = args.precision
    verbosity = bool(args.verbosity)

    # -----
    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',') # binary classification (labels)
    df_x_train = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',') # fuzzy classification to apply thresholding to
    df_x_test = pd.read_csv(CSV_FILE_X_TEST, index_col=0, sep=',') # data to predict for grading over at the challenge page
    labels = pd.read_csv(CSV_FILE_LABELS, index_col=0, sep=',')

    # -----
    if (mode in [1, 2]):

        thresholds, f1score = get_score(
            df_x_train, 
            df_y_true, 
            labels, 
            metric, 
            precision,
        )

        print("[INFO] Config: (%s, %i)" % (metric.__name__, precision))
        print("[INFO] Global score using the F1 metric: {0:.10f}".format(f1score))

        if (mode == 2):
            export_to_csv(df_x_test, metric, precision, thresholds)

    elif (mode == 3):
        compute_all_and_export_best(df_x_train, df_x_test, df_y_true, labels)

    else:
        
        # for t in np.arange(0.2, 0.3, 0.01):
        # for t in np.arange(0.23, 0.25, 0.001):
        for t in np.arange(0.243, 0.244, 0.0001): # best t at 0.2432
            thresholds = {g:t for g in labels.columns} 

            df_y_train = apply_thresholds(thresholds, df_x_train)
            score = evaluate(df_y_true, df_y_train)
            print("{:<6.4f}{:>10.6f}".format(t, score))


    print("[DEBUG] Done.")