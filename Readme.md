# Correlation tool between BTC transactions and purchases from black market


## Setup

To install required libraries, simply do:
```bash
pip install -r requirements.txt
```
Other requirements are to have `PostgoreSQL` database set up locally, or remotely and have user/password access (can be set in `config.py` file).

## Run

To run this program use script `load.py` accordingly:

```bash
usage: load.py [-h] [--time_threads TIME_THREADS]
               [--corr_threads CORR_THREADS] [--start START] [--end END]
               [--plot] [--sequential] [--dims DIMS] [--process_outliers]
               [--model_list MODEL_LIST [MODEL_LIST ...]]
               [--k_neighbors K_NEIGHBORS]

optional arguments:
  -h, --help            show this help message and exit
  --time_threads TIME_THREADS, -t TIME_THREADS
    'Number of time threads'
  --corr_threads CORR_THREADS, -c CORR_THREADS
    'Number of correlation threads'
  --start START, -s START
    'Beginning of the query from product__stats ordered by date'
  --end END, -e END
    'End of the query from product__stats ordered by date'
  --plot, -p
    'Plots K-Nearest neighbors for some data points to plots folder'
  --sequential
    'Sequential correlation run'
  --dims DIMS, -d DIMS
    '2D or 1D correlations'
  --process_outliers
    'If used, outliers will be processed sequentially'
  --model_list MODEL_LIST [MODEL_LIST ...]
    'List of distane metrics used for corellation - for full list refer to either documentation or' Technicals.model_list 'variable'
  --k_neighbors K_NEIGHBORS, -k K_NEIGHBORS
    'K parameter for KNN'
```