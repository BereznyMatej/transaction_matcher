import enum
import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from utils import query_wrapper
from utils import get_block_height
from config import API_STRING
from config import BLOCK_OFFSET


__author__ = "xberez03"

class Technicals:
    blocks = []
    model_list = ['canberra', 'euclidean', 'manhattan',
                  'chebyshev', 'braycurtis', 'hamming']

    def __init__(self, interval, sequential, dims,  k=5,
                 blocks=None):
        """
        Args:
            interval (pd.Dataframe): time interval for querying
                                     blocks.
            sequential (bool): If True, correlations
            will be calculated sequentially
            dims (int): dimensions for KNN
            k (int, optional): K parameter for KNN. Defaults to 5.
            blocks (pd.Dataframe, optional): Dataframe with blocks.
                                             If passed, block interval
                                             wont be queried from blockchain.
                                              Defaults to None.
        """
        self.k = k
        self.dims = dims
        self.sequential = sequential
        if self.sequential:
            self.blocks = blocks
        else:
            if blocks is None:
                self.load_btc_blocks(interval=interval)
            else:
                self.blocks = blocks
            self.load_models()


    def load_btc_blocks(self, interval):
        """Loads the interval of BTC blocks to memory.

        Args:
            interval (pd.Dataframe): time interval for querying blocks.
        """
        current_time = datetime.now().timestamp()
        req = requests.get(f"{API_STRING}").json()
        last_block_height = req['blockbook']['bestHeight']
        start, end = self.__block_interval(last_block_height=last_block_height,
                                          tstamp_current=current_time,
                                          interval=interval)
        self.blocks = query_wrapper(start, end)       


    
    def __block_interval(self, last_block_height, tstamp_current, 
                         interval):
        """Calculates the block height interval from given parameters and
           applies offset.

        Args:
            last_block_height (int): height of the newest block in blockchain
            tstamp_current (int): timestamp of present time
            interval (pd.Dataframe): time interval for querying blocks.

        Returns:
            tuple(int, int): interval with applied offset
        """
        tstamp_first = interval.time.min()
        tstamp_last = interval.time.max()
        fblock = get_block_height(last_block_height=last_block_height,
                                  tstamp_current=tstamp_current,
                                  tstamp=tstamp_first)
        lblock = get_block_height(last_block_height=last_block_height, 
                                  tstamp_current=tstamp_current,
                                  tstamp=tstamp_last)

        offset = BLOCK_OFFSET//4 if self.sequential else BLOCK_OFFSET

        return int(fblock-offset), int(lblock+offset)
    

    def setup_model(self, model_name):                
        """Setups the KNN model

        Args:
            model_name (str): model name

        Returns:
            NearestNeighbors: KNN model
        """
        model = getattr(self, f"setup_{self.dims}d_knn")(self.blocks,
                                                         self.k,
                                                         model_name)
        return model


    def setup_1d_knn(self, samples, k, metric):
        """Setups one dimensional KNN model with
           only prices as points.

        Args:
            samples (pd.Dataframe): dataframe with blocks
            k (int): K parameter for KNN
            metric (str): metric type

        Returns:
            NearestNeighbors: KNN model
        """
        knn = NearestNeighbors(n_neighbors=k,
                               radius=0.4,
                               metric=metric)
        samples = [[value] for value in samples['price'].values]
        knn.fit(samples)
        return knn


    def setup_2d_knn(self, samples, k, metric):
        """Setups two dimensional KNN model with
           prices and timestamps as points.

        Args:
            samples (pd.Dataframe): dataframe with blocks
            k (int): K parameter for KNN
            metric (str): metric type

        Returns:
            NearestNeighbors: KNN model
        """
        knn = NearestNeighbors(n_neighbors=k,
                               radius=0.4,
                               metric=metric)
        knn.fit(samples[['time', 'price']].values)
        return knn


    def get_1d_knn(self, sample, knn):
        """Finds the K nearest neighbors
           for given samples with specified metric
           in 1D space

        Args:
            sample (pd.Dataframe): samples to be predicted
            knn  (NearestNeighbors): KNN model

        Returns:
            tuple(int, int): neighbors and their respective distances
        """
        samples = [[value] for value in sample['btc'].values]
        neighs = knn.kneighbors(samples, self.k, return_distance=True)
        dist = neighs[0]
        neighs = neighs[1]
        return np.array(dist), np.array(neighs)


    def get_2d_knn(self, sample, knn):
        """Finds the K nearest neighbors
           for given samples with specified metric
           in 2D space

        Args:
            sample (pd.Dataframe): samples to be predicted
            knn  (NearestNeighbors): KNN model

        Returns:
            tuple(int, int): neighbors and their respective distances
        """
        neighs = knn.kneighbors(sample[['created_at', 'btc']].values, self.k, return_distance=True)
        dist = neighs[0]
        neighs = neighs[1]
        return np.array(dist), np.array(neighs)

    def load_models(self):        
        """Initiates all models currently
           in Technicals.model_list
        """
        for model in self.model_list:
            key = f"{model}_{self.dims}d_knn_obj"
            self.__dict__[key] = self.setup_model(model_name=model)
    

    def get_knn(self, sample, metric, samples=None):
        """Wrapper for calling get_1/2d_knn functions

        Args:
            sample (pd.Dataframe): samples to be predicted
            metric (str): desired distance metric
            samples (pd.Dataframe, optional): dataframe with blocks,
                                              used only in sequential
                                              approach. Defaults to None.

        Returns:
            tuple(int, int): neighbors and their respective distances
        """
        if self.sequential:
            knn = getattr(self, f"setup_{self.dims}d_knn")(samples,
                                                           k=self.k,
                                                           metric=metric)
        else:
            knn = self.__dict__[f'{metric}_{self.dims}d_knn_obj']
        return getattr(self, f"get_{self.dims}d_knn")(sample, knn)
       