    
from datetime import datetime
import os
import asyncio
import aiohttp
import tqdm
import pandas as pd
import numpy as np
from config import API_STRING, SATOSHI
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from config import API_STRING
from config import BLOCK_TIME


__author__ = "xberez03"


def flatten(data):
    """Flattens out the list of lists to list

    Args:
        data (list): list of lists

    Returns:
        list: flattened list
    """
    return [item for sublist in data for item in sublist]


def get_or_create_eventloop():
    """Creates thread-safe asyncio loop

    Returns:
        asyncio.loop: asyncio event loop
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


def query_wrapper(start, end):
    """Wrapper for btc blockchain async query functions

    Args:
        start (int): starting block height
        end (int): end block height

    Returns:
        pd.Dataframe: block interval 
    """
    blocks = []
    loop = get_or_create_eventloop()
    future = asyncio.ensure_future(__query_blocks(start=start,
                                                  end=end,
                                                  blocks=blocks))
    loop.run_until_complete(future)
           
    blocks = pd.DataFrame(flatten(blocks))
    blocks.time = blocks.time.astype(np.uint64)
    blocks.price = blocks.price.astype(np.int64)
    return blocks


async def __bound_fetch(sem, session, i, blocks):
    """Wrapper for async GET request

    Args:
        sem (asyncio.Semaphore):
        session:
        i (int): block height of the requested block
        blocks (list): block list
    """
    async with sem:
        await __query_block(session, i, blocks)


async def __query_blocks(start, end, blocks):
    """Aynchronous function, that queries desired 
       block interval from btc blockchain.

    Args:
        start (int): starting block height
        end (int): end block height
        blocks (list): block list
    """
    tasks = []

    sem = asyncio.Semaphore(1000)
    async with aiohttp.ClientSession() as session:
        for i in range(start, end):
            task = asyncio.ensure_future(__bound_fetch(sem, session, i, blocks))
            tasks.append(task)
        await asyncio.gather(*tasks)


async def __query_block(session, i, blocks):
    """Async. function querying one block
       from BTC blockchain

    Args:
        session:
        i (int): block height of the requested block
        blocks (list): block list
    """
    try:
        async with session.get(f'{API_STRING}/block/{int(i)}') as response:
            block_tlist = []
            req = await response.json()
            for t in req['txs']:
                addr_list = []
                for in_tr in t['vin']:
                    if 'addresses' in in_tr.keys():
                        addr_list.extend(in_tr['addresses'])
                for out_tr in t['vout']:
                    if out_tr['addresses'] and \
                       not out_tr['addresses'][0] in addr_list and \
                       int(out_tr['value']) > 0:
                        block_tlist.append({'tid': t['txid'],
                                            'outaddr': out_tr['addresses'][0],
                                            'price': np.int64(out_tr['value']),
                                            'time': t['blockTime'],
                                            'block_height': i})
            blocks.append(block_tlist)
    except:
        pass


def get_block_height(last_block_height, 
                     tstamp_current, tstamp):
    """Calculates block height of the desired block
    from given parameters

    Args:
        last_block_height (int): block height of the last mined block
        tstamp_current (int): timestamp of the present time
        tstamp (int): timestamp of the desired time

    Returns:
        int: block heihgt
    """
    return last_block_height - (tstamp_current - tstamp) // BLOCK_TIME


def distance_from_center(income, kmeans, label):
    '''
    Calculate the Euclidean distance between a data point and the center of its cluster.
    :param float income: the standardized income of the data point 
    :param float age: the standardized age of the data point 
    :param int label: the label of the cluster
    :rtype: float
    :return: The resulting Euclidean distance  
    '''
    center_income =  kmeans.cluster_centers_[label,0]
    distance = np.sqrt((income - center_income) ** 2)
    return np.round(distance, 3)


def plot_neighs(results_to_plot):
    """Plots purchase's data points, 
       their nearest neighbors and
       surrounding transactions

    Args:
        results_to_plot (list): list of datapoints to plost
    """
    settings = {'marker_size': [50, 200, 200],
                'colors': ['black', 'red', 'blue']}
    
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    
    for plt_idx, result in enumerate(tqdm.tqdm(results_to_plot,
                                               desc="Plotting results")):
        result = results_to_plot[plt_idx]
        metric = result.pop()
        f, ax = plt.subplots(figsize=(10, 10))
        for i, item in enumerate(result):
            sns.scatterplot(x=item['time'], y=item['price']/SATOSHI,
                            marker='o', color=settings['colors'][i],
                            s=settings['marker_size'][i], ax=ax)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Price (BTC)')
        ax.grid(True)
        plt.tight_layout()
        f.savefig(f'plots/{metric}_{plt_idx}.pdf')
        plt.close(f)