import functools
import hashlib
import json
import logging
import time

import redis
import rediscluster

from . import utils as u

# alternative to https://github.com/comeuplater/fastapi_cache
# or RedisCacheBackend https://pythonrepo.com/repo/comeuplater-fastapi_cache-python-fastapi-utilities
# or apshceduler at https://gist.github.com/ivanleoncz/21293b00d0ea54db8ee3b57fb1170ddf
# or async event loops for signal handling
# Beware that callers are from different processes


# Singleton since modules are singleton in Python
# Only server_up and server_down are meant to be used for manipulating these variables
# and server_alive for checking

server_alive = False
conn = None
key_expire = 1200
key_prefix = ""

check_deadline = -1
_alive_check_timeout = 5


def server_down():
    """
    sets internal state variable to down
    """
    global server_alive
    global check_deadline
    global _alive_check_timeout

    logging.warning("Redis server down")
    server_alive = False
    check_deadline = time.time() + _alive_check_timeout


def server_up():
    """
    sets internal state variable to up
    """
    global server_alive
    global check_deadline

    logging.warning("Redis server up")
    server_alive = True
    check_deadline = -1


def check_server_alive(enforce_check=False) -> bool:
    global conn
    global server_alive
    global check_deadline

    if server_alive:
        return True

    if check_deadline < time.time() or enforce_check:
        try:
            conn.ping()
            server_up()
            return True
        except:
            server_down()

    return False



def gen_hash(input_str):
    return str(hashlib.sha256(input_str.encode('utf-8')).hexdigest())


def generate_cache_key(input_dict):
    cache_key = key_prefix + ":" + gen_hash(json.dumps(input_dict))
    return cache_key


def cache_get(key):
    global conn

    try:
        cache_key = generate_cache_key(key)
        response = conn.get(cache_key)
        if response:
            decoded_response = json.loads(response)
            return decoded_response

    except redis.ConnectionError as e:
        server_down()
        logging.error("Redis server down: cache disabled in cache_get!")

    except json.JSONDecodeError:
        logging.error("Redis server corrupted response!")

    except Exception as e:
        logging.error("Redis server error: " + str(e))


def cache_get_protected(key):
    global server_alive
    if server_alive:
        return cache_get(key)
    else:
        return None


def cache_set_protected(key, value):
    global server_alive
    if server_alive:
        cache_set(key, value)


def cache_set(key, value):
    global conn
    global key_expire

    try:
        cache_key = generate_cache_key(key)
        value_json = u.to_json(value)

        conn.set(cache_key, value_json)
        conn.expire(cache_key, key_expire)

    except redis.ConnectionError as e:
        server_down()
        logging.error("Redis server down: cache disabled in cache_get!")

    except json.JSONDecodeError:
        logging.error("Redis key cannot be Json encoded! " + str(key))

    except Exception as e:
        server_down()
        logging.error("Redis cache disabled in cache_set! + Exception: " + str(e))


# decorator
def cache(**decorator_kwargs):
    global server_alive
    global key_prefix

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            if not check_server_alive():
                return func(*func_args, **func_kwargs)

            cache_key = {'func': func.__globals__['__file__'] + '::' + str(func),
                         'func_args': str(func_args),
                         'func_kwargs': str(func_kwargs),
                         'decorator_kwargs': str(decorator_kwargs)
                         }

            response_from_cache = cache_get(cache_key)

            if response_from_cache:
                logging.debug(f"Redis returned object from cache for key ({cache_key}):" + str(response_from_cache))
                return response_from_cache
            else:
                response = func(*func_args, **func_kwargs)
                logging.debug(f"Redis returned object from wrapped function for key ({cache_key}): " + str(response))
                cache_set(cache_key, response)
            return response

        return wrapper

    return decorator


def setup(host='127.0.0.1', port=6379, startup_nodes=None, prefix="", expire=1200, alive_check_timeout=5, **kwargs):
    global key_prefix
    global key_expire
    global _alive_check_timeout
    global conn

    key_prefix = prefix
    key_expire = expire
    _alive_check_timeout = alive_check_timeout

    if startup_nodes is None:
        conn = redis.Redis(host=host, port=port, **kwargs)
        logging.debug("Redis client started")
    else:
        conn = rediscluster.RedisCluster(startup_nodes=startup_nodes, decode_responses=True, **kwargs)
        logging.debug("RedisCluster client started")

    check_server_alive(enforce_check=True)

def close():
    global conn
    global server_alive

    server_alive = False

    if conn:
        conn.close()
        conn = None
