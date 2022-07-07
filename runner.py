import requests
import pandas as pd
import numpy as np
from threading import Thread
from datetime import datetime
from random import choice
from random import seed
from config import API_STRING
from config import BLOCK_OFFSET
from config import PRICE_BOUNDARY
from config import SATOSHI
from config import TIME_BOUNDARY
from config import NGRAPHS
from technicals import Technicals
from utils import query_wrapper
from utils import get_block_height

seed(69)
__author__ = "xberez03"


class RunnerThread(Thread):


    cols = ['Product_ID', 'Variant_ID', 'Name', 'Description', 'Transaction_ID',
            'Target_Address', 'Price_(BTC)', 'Price_diff_(BTC)','Accuracy',
            'Metric', 'Distance']


    def __init__(self, conn, sequential, query_len,
                 out, index, results_to_plot,
                 technicals_args, interval,
                 corr_threads):
        """
        Args:
            conn (_type_): _description_
            sequential (bool): If True, sequential correlation
                               approach is chosen.
            query_len (int): length of the initial product__stats query
            out (list): output list
            index (int): thread index
            results_to_plot (list): list used for storing datapoints to plot
            technicals_args (list): arguments for Technicals class
            interval (pd.Dataframe): time interval for querying blocks and
                                     fetching data from database.
            corr_threads (int): number of correlation threads to be set up
        """
        Thread.__init__(self)
        self.conn = conn
        self.results_to_plot = results_to_plot
        self.sequential = sequential
        self.blocks = []
        self.index = index
        self.corr_threads=corr_threads
        self.interval = interval
        self.out = out
        self.technicals_args = technicals_args
        req = requests.get(f"{API_STRING}").json()
        self.lblock_height = req['blockbook']['bestHeight']
        self.tstamp_current = datetime.now().timestamp()
        self.idx_to_plot = [choice(range(0, query_len)) for _ in range(NGRAPHS)]


    def __get_query_len(self):
        """Calculates total length of the query for single RunnerThread
        """
        start = datetime.fromtimestamp(self.interval.time.min())
        end = datetime.fromtimestamp(self.interval.time.max())
        return self.conn.execute(f"""
            SELECT COUNT(*)
            FROM product__stats
            INNER JOIN product__variant__items ON 
                product__stats.product_id = product__variant__items.product_id
            INNER JOIN product__variant__prices ON
                product__variant__items.id = product__variant__prices.variant_id
            INNER JOIN product__items ON
                product__items.id = product__stats.product_id
            WHERE product__stats.sales_delta != 0 AND
                  product__stats.created_at >= '{start}' AND
                  product__stats.created_at <= '{end}';
            """).fetchone()[0]



    def __query_data(self, offset, length):
        """
        """
        
        start = datetime.fromtimestamp(self.interval.time.min())
        end = datetime.fromtimestamp(self.interval.time.max())
        cursor = self.conn.execute(f"""
            SELECT
                 product__stats.product_id as product_id,
                 product__stats.created_at as created_at,
                 product__variant__prices.variant_id as variant_id,
                 product__items.name as name,
                 product__variant__prices.btc as btc,
                 product__variant__items.name as desc
            FROM product__stats
            INNER JOIN product__variant__items ON 
                product__stats.product_id = product__variant__items.product_id
            INNER JOIN product__variant__prices ON
                product__variant__items.id = product__variant__prices.variant_id
            INNER JOIN product__items ON
                product__items.id = product__stats.product_id
            WHERE product__stats.sales_delta != 0 AND
                  product__stats.created_at >= '{start}' AND
                  product__stats.created_at <= '{end}'
            ORDER BY created_at
            OFFSET {offset} ROWS
            FETCH NEXT {length} ROWS ONLY;
            """)

        return cursor if self.sequential else cursor.fetchall()

    def __acc_sequential(self, actual_price, row, actual_time=None):
        """Sequential accuracy and price difference computation

        Args:
            actual_price (int): price of the purchase in satoshis
            row (pd.Series): one row from the dataframe
            actual_time (int, optional): timestamp of the time
                                          when purchase happened.
                                          Defaults to None.

        Returns:
            tuple(int, int): accuracy, difference between purchase's and transaction's price
        """
        
        price_boundary = actual_price*PRICE_BOUNDARY
        diff = abs(actual_price - row['price'].values[0])
        res = diff / price_boundary
        if actual_time is not None:
            time_boundary = actual_time*TIME_BOUNDARY
            time_diff = abs(actual_time - row['time'].values[0])
            res = (res + time_diff / time_boundary) / 2
        res = round(res, 3)
        res = 0 if res >= 1 else 1-res
        return res, diff


    
    def __acc(self, sample, row):
        """Parallel accuracy and price difference computation

        Args:
            sample (pd.Dataframe:
            row (pd.Series): one row from the dataframe
        """
        diff = abs(sample['btc'] - row['price'])
        sample['Price_diff_(BTC)'] = diff / SATOSHI
        sample['Price_diff_(BTC)'] = sample['Price_diff_(BTC)'].round(4)
        time_diff = abs(sample['created_at'] - row['time'])
        sample['Accuracy'] = 1 - (diff / (sample['btc']*PRICE_BOUNDARY) + 
                                  time_diff / TIME_BOUNDARY)/2
        sample['Accuracy'] = sample['Accuracy'].round(4)
        sample.loc[(sample['Accuracy'] < 0.0001), ['Accuracy']] = 0


    def add_to_plot(self, point, neighs, metric):
        """Add points to the plot

        Args:
            point (list): purchase
            neighs (list): list of neighbor indexes
            metric (str): distance metric
        """
        neighss = self.blocks.iloc[neighs]
        lower_bound = min(point[1], neighss['price'].min())
        upper_bound = max(point[1], neighss['price'].max())
        df = self.blocks
        df = df[(df['price'] >= lower_bound) &
                (df['price'] <= upper_bound)]
        self.results_to_plot.append([df,
                                     neighss,
                                     pd.DataFrame({'time': [point[0]],
                                                   'price': [point[1]]}),
                                     metric])


    def __get_block_range(self, record):
        tstamp = record[1].timestamp()
        block_height = get_block_height(last_block_height=self.lblock_height,
                                        tstamp_current=self.tstamp_current,
                                        tstamp=tstamp)
        return query_wrapper(start=int(block_height-BLOCK_OFFSET),
                             end=int(block_height+BLOCK_OFFSET))


    def __correlate_sequential(self, record, technicals):
        """_summary_

        Args:
            record (list): one row from database query
            technicals (Technicals):

        Returns:
            list: list of records
        """
        t = []

        tstamp = record[1].timestamp()
        price = np.int64(float(record[4])*SATOSHI)
        
        sample = pd.DataFrame({'btc': [price], 'created_at': [tstamp]})

        for metric in Technicals.model_list:
            dist, neighs = technicals.get_knn(sample=sample,
                                              metric=metric,
                                              samples=self.blocks)
            for idx, t_idx in enumerate(neighs):
                row = self.blocks.iloc[t_idx]
                acc, diff = self.__acc_sequential(price, row, actual_time=tstamp)
                if acc > 0.000:
                    t.append({'Product_ID': record[0],
                              'Variant_ID': record[2],
                              'Transaction_ID': row['tid'].values[0],
                              'Target_Address': row['outaddr'].values[0],
                              'Name': record[3],
                              'Description': record[5],
                              'Price_(BTC)': row['price'].values[0]/SATOSHI,
                              'Distance': round(dist[idx][0], 4),
                              'Accuracy': acc,
                              'Metric': metric,
                              'Price_diff_(BTC)': diff/SATOSHI})
        return t


    def __correlate(self, table, technicals):
        """Main correlation function

        Args:
            table (pd.Dataframe): dataframe containing results from database query
            technicals (Technicals):
        """
        table['created_at'] = table['created_at'].values.astype(np.int64) // 10 ** 9
        table['created_at'] =  table['created_at'].astype(np.int64)
        table['btc'] = table['btc'] * SATOSHI
        table['btc'] = table['btc'].astype(np.int64)
        res_list = []

        for metric in Technicals.model_list:
            
            sample = table
            sample['Metric'] = metric
            dist, neighs = technicals.get_knn(sample=table,
                                              metric=metric,
                                              samples=self.blocks)
            for idx in range(neighs.shape[1]):
                row = self.blocks.iloc[neighs[:, idx]].reset_index()
                neigh_sample = sample.copy(deep=True).reset_index()
                self.__acc(sample=neigh_sample, row=row)
                neigh_sample['Distance'] = dist[:, idx]
                neigh_sample['Distance'] = neigh_sample['Distance'].round(4)
                neigh_sample['Transaction_ID'] = row['tid']
                neigh_sample['Target_Address'] = row['outaddr']
                neigh_sample['Price_(BTC)'] = row['price']/SATOSHI
                res_list.append(neigh_sample)
        

            if self.results_to_plot is not None:
                for i in self.idx_to_plot:
                    self.add_to_plot(point=table.iloc[i][['created_at', 'btc']],
                                     neighs=neighs[i, :],
                                     metric=metric)

        if len(res_list) > 1:
            res_list = pd.concat(res_list, ignore_index=True)
        else:
            res_list = res_list[0]
        self.out.append(res_list)


    def __run_core_thread_sequential(self, offset, length):
        """Run function of sequential correlation thread

        Args:
            offset (int): thread's offset in queru
            length (int): thread's query length
        """
        table = self.__query_data(offset, length)
        technicals = Technicals(**self.technicals_args, blocks=self.blocks)
        out = []
        
        for idx, record in enumerate(table):
            if self.sequential:
                self.blocks = self.__get_block_range(record)
            out.extend(self.__correlate_sequential(record, technicals))
        
        self.out.append(pd.DataFrame(out))      


    def __run_core_thread(self, offset, length):
        """Run function of correlation thread

        Args:
            offset (int): thread's offset in queru
            length (int): thread's query length
        """
        table = self.__query_data(offset, length)
        table = pd.DataFrame(table, columns=['Product_ID',
                                             'created_at',
                                             'Variant_ID',
                                             'Name',
                                             'btc',
                                             'Description'])
        technicals = Technicals(**self.technicals_args, blocks=self.blocks)
        self.__correlate(table, technicals)
        

    def run(self):
        """Main time thread loop
        """
        self.technicals = Technicals(**self.technicals_args)
        runner_func = self.__run_core_thread_sequential
        if not self.sequential:
            self.blocks = self.technicals.blocks
            runner_func = self.__run_core_thread
        threads = []
        length = self.__get_query_len() // self.corr_threads
        offset=0
        
        
        for i in range(self.corr_threads):
            threads.append(Thread(target=runner_func,
                                  args=[offset, length]))
            threads[i].start()
            offset+=length
        
        for t in threads:
            t.join()

        print(f"Thread: {self.index} - finished working")
