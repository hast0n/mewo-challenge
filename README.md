# Challenge Data - Mewo

Amandine Vallée & Martin Devreese (IA - promo 2022)

# Features

This repo contains the source code for the data classification problem proposed by Mewo on [Challenge Data](https://challengedata.ens.fr/participants/challenges/43/). It primirily relies on the dynamic thresholding method.

Leaderboard presence:
> **avallee002 & hskll**

# Prerequisites
Clone the repo.

Download x_train (input data of the training set), y_train (output data of the training set) and x_test from the challenge web page and copy them at the project root.

The project directory should look like this:

```
.
├── .bench
│   └── mp_scores.json
├── annex
│   ├── mewo-labels.csv
│   └── train_y.category.csv
├── README.md
├── main.py
├── dynamic_thresholding.py
├── test_X_nPKMRWK.csv
├── train_X_beAW6y8.csv
└── train_y_YZCqwbD.csv
```

# Usage

Them `main.py` file allows to you compute thresholds by columns. You can choose a metric and a precision level (how many decimals) to determine the best threshold possible.

## Metrics

There are several metrics availables:
- F1 Score over Binary Crossentropy (`f1_over_bin_cross`)
- Standard F1 Score (`f1_score`)
- Matthews Correlation Coefficient (`matthews_corr`)
- Binary Crossentropy (`bin_cross`)
- Average Precision Score (`pr_auc`)

## Precision

Precision level can be computed up to the 10th decimal.

## Modes

The program has 3 execution modes:
1. Computes the overall F1 Score on the train data for one `(metric, precision)` pair.
2. Computes thresholds for one `(metric, precision)` pair and applies them to the test set for an export. Can then be submitted for grading
3. Computes every `(metric, precision)` pair and returns the one with the best F1 Score. Also applies & exports the thresholds for the test set. The operation may take a while.
4. (Debug)

## Run

To run the program, use the following scheme:
```
python main.py <MODE> <METRIC> <PRECISION>
```

For example, to export the `(matthews_corr, 3)` pair, run the following:

```bash
python main.py 2 matthews_corr 3
```

To run the 3rd execution mode, use the following:

```bash
python main.py 3
```

In case of an invalid arguments passed, the program will default to the following run profile

```bash
python main.py 1 f1_score 3
```