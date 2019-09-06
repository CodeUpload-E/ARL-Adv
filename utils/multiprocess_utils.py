import multiprocessing as mp
import time

import utils.array_utils as au


def split_multi(array, process_num):
    from math import ceil
    return au.split_slices(array, int(ceil(len(array) / process_num)))


def multi_process(func, args_list=None, kwargs_list=None, while_wait=None):
    res_list = list()
    process_num = len(args_list) if args_list is not None else len(kwargs_list)
    pool = mp.Pool(processes=process_num)
    for i in range(process_num):
        args = args_list[i] if args_list else ()
        kwds = kwargs_list[i] if kwargs_list else {}
        res_list.append(pool.apply_async(func=func, args=args, kwds=kwds))
    pool.close()
    if while_wait is not None:
        while_wait()
    pool.join()
    results = [res.get() for res in res_list]
    return results


def multi_process_batch(func, batch_size, args_list, kwargs_list=None):
    n_args = len(args_list)
    if kwargs_list is None:
        kwargs_list = [{}] * n_args
    results = list()
    for s, u in au.split_since_until(n_args, batch_size):
        print('going to process {} / {}'.format(u, n_args))
        results.extend(multi_process(func, args_list[s: u], kwargs_list[s: u]))
    return results


def span_and_key_event(func, *args, **kwargs):
    from queue import Queue
    from threading import Thread

    msg_quit = 'q'
    span = 80
    t_last = 0
    t_wait = 600
    span_bound = 1800

    def read_key_input():
        nonlocal t_last
        while True:
            value = input()
            t_last = 0
            q_exec.put(value)
            q_time.put('')

    def read_time_event():
        nonlocal span
        nonlocal t_last
        double_flag = False
        while True:
            q_wait.get()
            for i in range(span):
                print('\r{}\r[{:>3} / {:>3}]  {:<80}'.format(
                    ' ' * 80, span - i, span, '■' * (min(span - i, 80))), end='', flush=True)
                time.sleep(1)
                t_last += 1
                if t_last > 0 and t_last % t_wait == 0 and span < span_bound:
                    double_flag = True
                if not q_time.empty():
                    break
            if double_flag:
                double_flag = False
                span = min(span * 2, span_bound)
            if not q_time.empty():
                q_time.get()
                continue
            q_exec.put('')

    q_exec, q_time, q_wait = Queue(), Queue(), Queue()
    Thread(target=read_key_input, daemon=True).start()
    Thread(target=read_time_event, daemon=True).start()
    while True:
        func(*args, **kwargs)
        q_wait.put(object)
        msg = str(q_exec.get()).strip()
        if msg == msg_quit:
            exit()
        try:
            span = int(msg)
        except:
            pass
