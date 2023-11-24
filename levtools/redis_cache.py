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
_shared_across_processes = True


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
    return str(hashlib.sha256(input_str.encode("utf-8")).hexdigest())


def generate_hash_with_prefix(input_dict):
    return key_prefix + ":" + gen_hash(json.dumps(input_dict))


def cache_get(key):
    global conn

    try:
        cache_key = generate_hash_with_prefix(key)
        response = conn.get(cache_key)
        if response:
            decoded_response = json.loads(response)["value"]
            return decoded_response

    except redis.ConnectionError as e:
        server_down()
        logging.error("Redis server down: cache disabled in cache_get!")

    except json.JSONDecodeError:
        logging.error("Redis server corrupted response!")

    except Exception as e:
        server_down()
        logging.error("Redis server error: " + str(e))


def cache_delete(key):
    global conn

    try:
        cache_key = generate_hash_with_prefix(key)
        response = conn.delete(cache_key)

        logging.debug(f"Redis delete cache for key ({key}):" + str(response))

        return response

    except redis.ConnectionError as e:
        server_down()
        logging.error("Redis server down: cache disabled in cache_get!")

    except json.JSONDecodeError:
        pass

    except Exception as e:
        server_down()
        logging.error("Redis server error: " + str(e))


def cache_set(key, value):
    global conn
    global key_expire

    try:
        cache_key = generate_hash_with_prefix(key)
        value_json = u.to_json({"type": "pure", "value": value})
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


def _convert_func_call_attributes_to_str(
    func, func_args, func_kwargs, decorator_kwargs
):
    func_name = str(func)

    if _shared_across_processes:
        func_name = func_name[: func_name.find("at")] + ">"

    module = func.__globals__["__file__"]

    cache_key = {
        "func": module + "::" + func_name,
        "func_args": str(func_args),
        "func_kwargs": str(func_kwargs),
        "decorator_kwargs": str(decorator_kwargs),
    }

    return cache_key


def checkpoint_caching(key, func, func_args=(), **func_kwargs):
    if not check_server_alive():
        return func(*func_args, **func_kwargs)

    response_from_service = cache_get(key)
    if response_from_service:
        logging.debug(
            f"Redis returned object from cache for key ({key}):"
            + str(response_from_service)
        )
        return response_from_service
    else:
        response_from_func = func(*func_args, **func_kwargs)
        logging.debug(
            f"Redis returned object from wrapped function for key ({key}): "
            + str(response_from_func)
        )
        cache_set(key, response_from_func)
        return response_from_func


def cache(**decorator_kwargs):
    """
    Decorator
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            if not check_server_alive():
                return func(*func_args, **func_kwargs)

            call_str = _convert_func_call_attributes_to_str(
                func, func_args, func_kwargs, decorator_kwargs
            )
            response_from_service = cache_get(call_str)

            if response_from_service:
                logging.debug(
                    f"Redis returned object from cache for key ({call_str}):"
                    + str(response_from_service)
                )
                return response_from_service
            else:
                response = func(*func_args, **func_kwargs)
                logging.debug(
                    f"Redis returned object from wrapped function for key ({call_str}): "
                    + str(response)
                )
                cache_set(call_str, response)
            return response

        return wrapper

    return decorator


def delete_all_keys():
    global conn
    global key_prefix
    for key in conn.scan_iter(key_prefix + "*"):
        conn.delete(key)


def setup(
    host="127.0.0.1",
    port=6379,
    startup_nodes=None,
    prefix="",
    expire=1200,
    alive_check_timeout=5,
    shared_across_processes=True,
    **kwargs,
):
    global key_prefix
    global key_expire
    global _alive_check_timeout
    global _shared_across_processes
    global conn

    key_prefix = prefix
    key_expire = expire
    _alive_check_timeout = alive_check_timeout
    _shared_across_processes = shared_across_processes

    if startup_nodes is None:
        conn = redis.Redis(host=host, port=port, **kwargs)
        logging.debug("Redis client started")
    else:
        conn = rediscluster.RedisCluster(
            startup_nodes=startup_nodes, decode_responses=True, **kwargs
        )
        logging.debug("RedisCluster client started")

    check_server_alive(enforce_check=True)
    if check_server_alive():
        conn.flushall()
    else:
        logging.warning("No Redis server found")


def close():
    global conn
    global server_alive
    global check_deadline

    server_alive = False

    if conn:
        conn.close()
        conn = None

    server_alive = False
    check_deadline = -1
