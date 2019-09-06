import json
import os
import pickle
import re
import shutil
from pathlib import Path

dump = json.dump
load = json.load
dumps = json.dumps
loads = json.loads


def load_json(file):
    return json.load(open(file, mode='r'))


def dump_json(file, obj):
    json.dump(obj, open(file, mode='w'))


def read_lines(file, mode='r', newline='\n'):
    with open(file, mode=mode, newline=newline, errors='ignore') as fp:
        return [line.rstrip(newline) for line in fp.readlines()]


def write_lines(file, lines, mode='w', newline='\n'):
    with open(file, mode=mode, encoding='utf8') as fp:
        fp.writelines([line + newline for line in lines])


def load_array(file, mode='r', **kwargs):
    lines = read_lines(file, mode=mode)
    return [loads(line, **kwargs) for line in lines]


def dump_array(file, array, mode='w', **kwargs):
    lines = [dumps(obj, **kwargs) for obj in array]
    write_lines(file, lines=lines, mode=mode)


def load_pickle(file):
    return pickle.load(open(file, 'rb'))


def dump_pickle(file, obj, parents=True):
    if parents:
        mkprnts(file)
    pickle.dump(obj, open(file, mode='wb'), protocol=4)


def get_cwd(): return os.getcwd()


def get_name(path): return Path(path).name


def rename(path, target): Path(path).rename(target)


def base_name(abs_path): return os.path.basename(abs_path)


def parent_name(path): return os.path.dirname(path)


def is_dir(path): return os.path.isdir(path)


def is_file(file): return os.path.isfile(file)


def exists(path): return os.path.exists(path)


def join(*args): return os.path.join(*list(map(str, args)))


def rmtree(path):
    if os.path.exists(path):
        print('remove path {}'.format(path))
        shutil.rmtree(path)


def concat_files(input_files, output_file):
    os.popen('cat {} > {}'.format(' '.join(input_files), output_file)).close()


def copy(source, target):
    target_parent = Path(target).parent
    if not target_parent.exists():
        print('create', target_parent)
        target_parent.mkdir(parents=True)
    shutil.copy(source, target)


def mkdir(path, rm_prev=False):
    if rm_prev:
        rmtree(path)
    if not exists(path):
        os.makedirs(path)


def mkprnts(path):
    p = Path(path).parent
    if not p.exists():
        print('going to make dir', str(p))
        p.mkdir(parents=True)


ALL = 0
DIR = 1
FILE = 2


def list_children(path, ctype=FILE, pattern=None, full_path=False):
    children = list()
    for c in Path(path).iterdir():
        if ctype == ALL or (c.is_file() and ctype == FILE) or \
                (c.is_dir() and ctype == DIR):
            children.append(c)
    if pattern is not None and type(pattern) is str:
        children = [c for c in children if re.search(pattern, c.name) is not None]
    children = [str(c.absolute()) if full_path else c.name for c in children]
    return sorted(children)


def most_recent(files, full_path=True):
    assert len(files) >= 1
    c = sorted([Path(f) for f in files], key=lambda p: p.stat().st_mtime)[-1]
    return str(c.absolute()) if full_path else c.name


def choose_from(files, full_path=True):
    file_objs = sorted([Path(f) for f in files], key=lambda p: p.stat().st_mtime)[::-1]
    for idx, c in enumerate(file_objs):
        print('*' if idx == 0 else ' ', '{} - {}'.format(idx, c.name))
    while True:
        try:
            choice = input('select idx (default 0): ').strip()
            c = file_objs[0 if choice == '' else int(choice)]
            break
        except KeyboardInterrupt:
            exit('ctrl + c')
        except:
            print('invalid index, please re-input')
    return str(c.absolute()) if full_path else c.name
