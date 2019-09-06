import time

default_name = 'main'
check_points = dict()


def time_format(secs):
    if secs <= 60:
        return '{:.1f} s'.format(secs)
    elif secs <= 3600:
        return '{:.1f} m'.format(secs / 60)
    else:
        return '{:.1f} h'.format(secs / 3600)


def format_date(str_format='%Y%m%d'):
    return time.strftime(str_format, time.localtime(time.time()))


def check_time(name=default_name, print_func='default'):
    if name not in check_points:
        check_points[name] = time.time()
        return None
    else:
        cur_time = time.time()
        delta_t = cur_time - check_points[name]
        check_points[name] = cur_time
        if print_func == 'default':
            print('{}: {:.2f}s since last check'.format(name, delta_t))
        elif callable(print_func):
            print_func(delta_t)
        return delta_t


def stat_time_elapse(func):
    def f(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        delta = time.time() - t0
        print('{} time elapse: {:.2f}'.format(func.__name__, delta))
        return res
    
    return f


class Timer:
    def __init__(self, total):
        self.init_t = time.time()
        self.step = 0
        self.total = total
    
    def progress(self, can_print):
        cur_t = time.time()
        self.step += 1
        if self.step > 0 and can_print:
            sum_t = time_format(cur_t - self.init_t)
            more_t = time_format((cur_t - self.init_t) * (self.total / self.step - 1))
            print('{}/{}, sum {}, remain {}'.format(self.step, self.total, sum_t, more_t))
