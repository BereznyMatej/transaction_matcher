import argparse
import psycopg
from sklearn.cluster import KMeans
import pandas as pd
from runner import RunnerThread
from technicals import Technicals
from utils import plot_neighs
from utils import distance_from_center
from config import DBNAME
from config import USER
from config import PASSWORD

__author__ = "xberez03"


parser = argparse.ArgumentParser()
parser.add_argument('--time_threads', '-t', default=1, type=int)
parser.add_argument('--corr_threads', '-c', default=1, type=int)
parser.add_argument('--start', '-s', default=0, type=int)
parser.add_argument('--end', '-e', default=0, type=int)
parser.add_argument('--plot', '-p', action='store_true')
parser.add_argument('--sequential', action='store_true')
parser.add_argument('--dims', '-d', default=2, type=int)
parser.add_argument('--process_outliers', action='store_true')
parser.add_argument("--model_list", nargs="+", default=Technicals.model_list)
parser.add_argument('--k_neighbors', '-k', default=1, type=int)
args = parser.parse_args()


conn = psycopg.connect(f"dbname={DBNAME} user={USER} password={PASSWORD}")
table_data = [(lambda x: x[0].timestamp())(x) for x in conn.execute("SELECT created_at FROM product__stats WHERE sales_delta != 0;").fetchall()]
df = pd.DataFrame({'time': table_data})


if args.model_list != Technicals.model_list:
    Technicals.model_list = args.model_list

if args.end > len(df) or args.end <= 0:
    args.end = len(df)

if args.start < 0:
    args.start = 0

if args.sequential:
    args.process_outliers = True


df = df.iloc[args.start:args.end]
if args.time_threads > 1:
    kmeans = KMeans(n_clusters=args.time_threads).fit(df)
    df['label'] = kmeans.labels_
    df['distance'] = distance_from_center(df['time'], kmeans, df.label)
    df.loc[df['distance'] > df['distance'].quantile(0.75), ['label']]= args.time_threads
else:
    df['label'] = 0

threads = []
results_to_plot = [] if args.plot else None
out_df = []

if args.process_outliers:
    outliers = len(df[df['label'] == args.time_threads]) > 0
else:
    outliers = 0

for i in range(args.time_threads+outliers):

    if not len(df[df['label'] == i]):
        continue
    if not args.sequential:
        sequential = (i == args.time_threads)
    else:
        sequential = args.sequential
    technicals_args={'dims': args.dims,
                    'interval': df[df['label'] == i],
                    'sequential': sequential,
                    'k':args.k_neighbors}
    t = RunnerThread(interval=df[df['label'] == i],
                     conn=conn,
                     technicals_args=technicals_args,
                     out=out_df,
                     index=i,
                     corr_threads=args.corr_threads,
                     sequential=sequential,
                     results_to_plot=results_to_plot,
                     query_len=len(df))
    threads.append(t)
    t.start()


for t in threads:
    t.join()

if args.plot:
    plot_neighs(results_to_plot)
out_df = pd.concat(out_df, ignore_index=True)
out_df = out_df[RunnerThread.cols]

for metric in Technicals.model_list:
    avg_acc = out_df.loc[out_df['Metric'] == metric]['Accuracy'].mean()
    print(f"Average {metric} accuracy accross all transactions is {avg_acc}")
out_df.to_csv('out_corr.csv', index=False)
