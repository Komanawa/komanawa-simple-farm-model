"""
created matt_dumont 
on: 8/24/24
"""
"""
created matt_dumont 
on: 25/11/22
"""
import warnings

import psutil
import multiprocessing
import logging
import os
import sys
from functools import partial
from concurrent.futures import ProcessPoolExecutor


def _start_process():
    """
    function to run at the start of each multiprocess sets the priority lower

    :return:
    """
    logger = multiprocessing.get_logger()
    if logger.level <= logging.INFO:
        print('Starting', multiprocessing.current_process().name)
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    if sys.platform == "linux":
        p.nice(19)
        # linux
    elif sys.platform == "darwin":
        # OS X
        p.nice(19)
        raise NotImplementedError
    elif sys.platform == "win32":
        # Windows...
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        raise ValueError(f'unexpected platform: {sys.platform}')


def run_multiprocess(func, runs, logical=True, num_cores=None, logging_level=logging.INFO, constant_kwargs=None,
                     subprocess_cores=1):
    """
    count the number of processors and then instiute the runs of a function to

    :param func: function with one argument kwargs.
    :param runs: a list of runs to pass to the function the function is called via func(kwargs)
    :param num_cores: int or None, if None then use all cores (+-logical) if int, set pool size to number of cores
    :param logical: bool if True then add the logical processors to the count
    :param logging_level: logging level to use one of: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL more info https://docs.python.org/3/howto/logging.html default is logging.INFO
    :param constant_kwargs: dict of kwargs to pass to func each time it is called, these kwargs are constant across all runs.
    :param subprocess_cores: int a number of cores needed for a subprocess (e.g. if the subprocess needs 5 cores to run then only create ncores//5 processes
    :return:
    """
    assert isinstance(num_cores, int) or num_cores is None
    multiprocessing.log_to_stderr(logging_level)
    if num_cores is None:
        pool_size = psutil.cpu_count(logical=logical)
    else:
        pool_size = num_cores

    pool_size = int(pool_size // subprocess_cores)
    if subprocess_cores > 1:
        warnings.warn('allowing pool processes to spawn other children... use with care')
        with ProcessPoolExecutor(pool_size) as pool:
            if constant_kwargs is not None:
                func = partial(func, **constant_kwargs)
            results = pool.map(func, runs)
            pool_outputs = results

    else:
        with multiprocessing.Pool(processes=pool_size,
                                  initializer=_start_process,
                                  ) as pool:

            if constant_kwargs is not None:
                func = partial(func, **constant_kwargs)
            results = pool.map_async(func, runs)
            pool_outputs = results.get()
    return pool_outputs
