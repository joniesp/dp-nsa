# Datasets

The datasets are not redistributed. Download them with

```bash
pip install openml ucimlrepo
python scripts/download_data.py
python scripts/check_data.py
```

and place `creditcard.csv` by hand (it requires a Kaggle account). Each `X.csv` holds the features and each `y.csv` the **original** labels; `dpnsa/datasets.py` applies the anomaly conventions of ODDS (Section 4.1.1 of the paper), including the seeded subsampling, so the files must not be pre-processed.

| Folder | Dataset | Source |
|---|---|---|
| `annthyroid/` | Annthyroid | OpenML [40886](https://www.openml.org/d/40886) |
| `arrhythmia/` | Arrhythmia | OpenML [1017](https://www.openml.org/d/1017) |
| `cardiotocography/` | Cardiotocography | OpenML [1466](https://www.openml.org/d/1466) |
| `ecg5000/` | ECG5000 | OpenML [1471](https://www.openml.org/d/1471) |
| `mammography/` | Mammography | OpenML [310](https://www.openml.org/d/310) |
| `musk/` | Musk (Version 2) | OpenML [300](https://www.openml.org/d/300) |
| `ionosphere/` | Ionosphere | UCI [52](https://archive.ics.uci.edu/dataset/52/ionosphere) |
| `satimage/` | Statlog (Landsat Satellite) | UCI [146](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite) |
| `shuttle/` | Statlog (Shuttle) | UCI [148](https://archive.ics.uci.edu/dataset/148/statlog+shuttle) |
| `creditcard.csv` | Credit Card Fraud Detection (ULB) | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

## Checking the download

`scripts/check_data.py` checks that every file is byte-identical to the one used for the paper (`data/SHA256SUMS`) and that, after the label conventions, every dataset has the size and contamination rate of Table 1. With the data in place, `pytest` also checks Table 1 and the dataset numbers quoted in the text.

If a source has changed its file format since, the checksum fails but the second check may still pass; the second check is the one that matters for reproducing the results.
