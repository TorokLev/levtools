import base64
import datetime
import importlib
import itertools
import json_tricks as json
import os
import subprocess
import sys
import re
import tempfile
import string
import random
import numpy as np
import pandas as pd
from collections.abc import MutableMapping


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def exec(cli, cwd="."):
    process = subprocess.Popen(
        cli, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
    )
    out, err = process.communicate()
    return {
        "out": out.decode("utf-8"),
        "err": err.decode("utf-8"),
        "return_code": process.returncode,
    }


def dict_from_module(module):
    context = {}
    for setting in dir(module):
        # you can write your filter here
        if setting.islower() and setting.isalpha():
            context[setting] = getattr(module, setting)

    return context


def get_recursive_modif_dates(dir):
    file_dates = dict()
    for dirpath, dirs, files in os.walk(dir):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            file_dates[fname] = os.path.getmtime(fname)
    return file_dates


def base64_decode(base64_message):
    base64_bytes = base64_message.encode("ascii")
    message_bytes = base64.b64decode(base64_bytes)
    return message_bytes.decode("ascii")


def none_resilient_max(arr):
    return np.max(arr[arr != None])


def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


def flatten_dict(d: MutableMapping, sep: str = ".") -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    return flat_dict


def replace_in_keys(input_dict, from_what, to_what):
    return dict(
        [(key.replace(from_what, to_what), value) for key, value in input_dict.items()]
    )


def json_load_from_file(json_filename):
    with open(json_filename, "rt") as f:
        return json.loads(f.read())


def json_write_to_file(obj, json_filename, **kwargs):
    with open(json_filename, "wt") as f:
        f.write(json.dumps(obj, **kwargs))


def get_size(obj, seen=None):  # TODO: bugfix to be able to measure MABs size
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def to_json(obj):
    def default_serializer(obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if type(obj) == pd.core.frame.DataFrame:
            return obj.to_json()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    return json.dumps(obj, default=default_serializer, sort_keys=True, indent=4)


def pprint(obj):
    print(to_json(obj))


def grep_cut(
    input_string, match, field=None, field_from=None, field_to=None, field_separator=" "
):
    for line in input_string.splitlines():
        if match in line:
            parts = line.split(field_separator)
            if field:
                return parts[field].strip()
            return field_separator.join(parts[field_from:field_to]).strip()


def get_pip_version(pkg_name):
    ret = exec("pip show " + pkg_name)
    if ret["return_code"] != 0:
        return

    return grep_cut(ret["out"], match="Version:", field=1)


def get_pkgroot(pkg):
    module = importlib.import_module(pkg)
    return getattr(module, "__path__")[0]


def get_module_version_if_loadable(pkg):
    try:
        module = importlib.import_module(pkg)
        model_id = getattr(module, "VERSION")
    except Exception:
        model_id = ""

    try:
        module = importlib.import_module(pkg)
        commit_id = getattr(module, "COMMIT_ID")
    except Exception:
        commit_id = ""

    return {"commit_id": commit_id, "model_id": model_id}


def path_resolver(in_path, create_dirs=False, create_as_dir=False, rebase_dir="."):
    """
    If path is <...> string then package root is substituted.
    If create_dirs is set then all parent dirs created.
    If create_as_dir then this and parent directories created
    If package name is 'tmpdir' then temp directory created to systems temp directory
    """
    if in_path.startswith("<") and in_path.find(">") > 0:
        pkg_name = in_path[1 : in_path.find(">")]
        if pkg_name == "tmpdir":
            in_path = in_path.replace(
                "<" + pkg_name + ">", tempfile.TemporaryDirectory().name
            )
        else:
            pkgroot = get_pkgroot(pkg_name)
            in_path = in_path.replace("<" + pkg_name + ">", pkgroot)

    path = rebase_path_if_relative(rebase_dir, in_path)

    if path and (create_dirs or create_as_dir):
        dirs = os.path.dirname(path) if create_dirs else path
        os.makedirs(dirs, exist_ok=True)

    return path


def assert_for_unknown_params(kwargs, known_params):
    n_of_unknown_params = len(set(kwargs.keys()).difference(known_params)) == 0
    assert n_of_unknown_params, "Unkown parameters passed: " + str(kwargs)


def _windows_to_linux(input_path):
    path = input_path.replace("\\", "/")
    return path[2:] if path[1] == ":" else path


def _linux_to_windows(input_path):
    path = os.path.expanduser(input_path) if input_path.startswith("~") else input_path
    path = os.path.normpath(path)  # replaces /  with \\ on Windows
    return path


def is_rel_path(input_path):
    # os.path.isabs() is incorrect on Windows as it doesn't use drive letter

    if is_windows():
        path = _linux_to_windows(input_path)
        return len(re.findall(r"[a-zA-Z]+:\\", path)) == 0  # startwith c:\\

    if is_linux():
        return os.path.abspath(input_path) != input_path


def rebase_path_if_relative(base_path, input_path):
    if not input_path:
        return None
    return os.path.normpath(
        os.path.join(path(base_path), input_path)
        if is_rel_path(input_path)
        else path(input_path)
    )


def is_windows():
    return hasattr(sys, "getwindowsversion")


def is_linux():
    return os.name == "posix"


def is_windows_path(input_path):
    return "\\" in input_path


def is_linux_path(input_path):
    return "/" in input_path


def path(input_path):
    """
    Translate path if written not the same platform as the current one.
    I.e Linux -> Win or Win -> Linux
    """

    if is_linux():  # running on Linux of Mac platform
        if is_windows_path(input_path):
            return _windows_to_linux(input_path)

    elif is_windows():  # running on Windows platform
        if is_linux_path(input_path):
            return _linux_to_windows(input_path)
    else:
        raise Exception("Unkown operating system type")

    return input_path


def to_datetime(date_str, to_naive=True):
    dt = pd.to_datetime(date_str)

    if to_naive:
        if dt.tz is not None:
            return dt.tz_convert(None)

    return dt


def get_random_string(len):
    letters = string.ascii_lowercase
    random_string = "".join(random.choice(letters) for i in range(len))
    return random_string


def skip_chars_from_start(text, chars_to_skip):
    for inx, char in enumerate(text):
        if char not in chars_to_skip:
            return text[inx:]


def get_chars_from_start(text, stop_chars):
    for inx, char in enumerate(text):
        if char in stop_chars:
            return text[:inx]


def firstn(iterator, n):
    return itertools.islice(iterator, n)


def keep_by_set(source, keep_set):
    return [entry for entry in source if entry in keep_set]


def drop_by_set(source, drop_set):
    return [entry for entry in source if entry not in drop_set]


def days_from_epoch(input_datetime):
    return (input_datetime - datetime.datetime(1970, 1, 1)).days


def has_any(strings, text):
    for filter in strings:
        if filter in text:
            return True
    return False


from functools import wraps


# for demonstration purposes
def mydecorator(decorator_params=""):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            print("decorator params: ", str(decorator_params))
            print("wrapped func name:" + str(func.__name__))
            print("wrapped func arguments:", str(args), str(kwargs))
            ret = await func(*args, **kwargs)
            print("wrapped func returned object: ", str(ret))
            return ret

        return wrapper

    return decorator


class SingletonClass(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SingletonClass, cls).__new__(cls)
        return cls.instance
