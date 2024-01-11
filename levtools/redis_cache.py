import functools
import hashlib
import json
import logging
import time
import pandas as pd
import numpy as np

import redis
import rediscluster

from . import utils as u

# alternative to https://github.com/comeuplater/fastapi_cache
# or RedisCacheBackend https://pythonrepo.com/repo/comeuplater-fastapi_cache-python-fastapi-utilities
# or apscheduler at https://gist.github.com/ivanleoncz/21293b00d0ea54db8ee3b57fb1170ddf
# or async event loops for signal handling
# Beware that callers are from different processes

# Singleton since modules are singleton in Python
# Only server_up and server_down are meant to be used for manipulating these variables
# and server_alive for checking

# state variables:
logging.basicConfig()
logger = logging.getLogger("root")


class Redis(u.SingletonClass):
    server_alive = False
    conn = None

    _next_check_deadline = -1
    _alive_check_timeout = 5
    _shared_across_processes = True

    def setup(
        self,
        host="127.0.0.1",
        port=6379,
        startup_nodes=None,
        key_expire=1200,
        alive_check_timeout=5,
        shared_across_processes=True,
        **kwargs,
    ):
        self._alive_check_timeout = alive_check_timeout
        self._shared_across_processes = shared_across_processes

        if startup_nodes is None:
            self.conn = redis.Redis(host=host, port=port, **kwargs)
            logger.debug("Redis client started")
        else:
            self.conn = rediscluster.RedisCluster(
                startup_nodes=startup_nodes, decode_responses=True, **kwargs
            )
            logger.debug("RedisCluster client started")

        self._low_level_check()
        return self

    def close(self):
        self.server_alive = False
        self._next_check_deadline = -1

        if self.conn:
            self.conn.close()
            self.conn = None

    def _server_down(self):
        """
        sets internal state variable to down
        """
        logger.warning("Redis server down")
        self.server_alive = False
        self._next_check_deadline = time.time() + self._alive_check_timeout

    def _server_up(self):
        """
        sets internal state variable to up
        """
        logger.warning("Redis server up")
        self.server_alive = True
        self._next_check_deadline = -1

    def _low_level_check(self):
        try:
            self.conn.ping()
            self._server_up()
            return True
        except:
            self._server_down()
            return False

    def check_server_alive(self):
        if self.server_alive:
            return True

        if self._next_check_deadline < time.time():
            return self._low_level_check()

        return False

    @staticmethod
    def hash_it(input_str):
        return str(hashlib.sha256(str(input_str).encode("utf-8")).hexdigest())

    def _cache_set(self, key, value, prefix=""):
        cache_key = (prefix + ":" if len(prefix) > 0 else "") + self.hash_it(key)
        if type(value) == pd.DataFrame:
            value_json = u.to_json({"type": str(type(value)), "value": value.to_json()})
        else:
            value_json = u.to_json({"type": str(type(value)), "value": str(value)})

        self.conn.set(cache_key, value_json)

    def _cache_get(self, key, prefix=""):
        cache_key = (prefix + ":" if len(prefix) > 0 else "") + self.hash_it(key)
        response = self.conn.get(cache_key)
        if response:
            decoded_response = json.loads(response)
            if decoded_response["type"] == "<class 'pandas.core.frame.DataFrame'>":
                return pd.DataFrame(json.loads(decoded_response["value"]))
            elif decoded_response["type"] == "<class 'float'>":
                return float(decoded_response["value"])
            elif decoded_response["type"] == "<class 'int'>":
                return int(decoded_response["value"])
            elif decoded_response["type"] == "<class 'numpy.ndarray'>":
                return np.array(decoded_response["value"])
            return decoded_response["value"]  # native types

    def _cache_delete(self, key, prefix=""):
        cache_key = (prefix + ":" if len(prefix) > 0 else "") + self.hash_it(key)
        response = self.conn.delete(cache_key)
        logger.debug(f"Redis delete cache for key ({key}):" + str(response))
        return response

    def _delete_all_keys(self, prefix=""):
        for key in self.conn.scan_iter(prefix + ":*" if len(prefix) > 0 else "*"):
            self.conn.delete(key)

    # def _error_handler_wrapper(self, func, *args, **kwargs):
    #    return func(*args, **kwargs)

    def _error_handler_wrapper(self, func, *args, **kwargs):
        try:
            if not self.check_server_alive():
                return

            return func(*args, **kwargs)

        except redis.ConnectionError as e:
            self._server_down()
            logging.error("Redis server down: cache disabled!")

        except json.JSONDecodeError as e:
            logging.error("Json decoding error:" + str(e))

        except Exception as e:
            self._server_down()
            logging.error("Redis server error: " + str(e))

    def get(self, key, prefix=""):
        return self._error_handler_wrapper(self._cache_get, key, prefix)

    def set(self, key, value, prefix=""):
        return self._error_handler_wrapper(self._cache_set, key, value, prefix)

    def delete(self, key, prefix):
        return self._error_handler_wrapper(self._cache_delete, key, prefix)

    def delete_all_keys(self, filtering_prefix=""):
        return self._error_handler_wrapper(self._delete_all_keys, filtering_prefix)


_redis_obj = Redis()


def _convert_func_call_attributes_to_str(
    func, func_args, func_kwargs, decorator_kwargs
):
    func_name = str(func)

    if _redis_obj._shared_across_processes:
        func_name = func_name[: func_name.find("at")] + ">"

    module = func.__globals__["__file__"]

    cache_key = {
        "func": module + "::" + func_name,
        "func_args": str(func_args),
        "func_kwargs": str(func_kwargs),
        "decorator_kwargs": str(decorator_kwargs),
    }

    return cache_key


def checkpoint_caching(key, func, prefix="", func_args=(), **func_kwargs):
    args_str = "args:" + str(func_args) + "kwargs: " + str(func_kwargs)
    key = key + ":" + str(func) + ":" + Redis().hash_it(args_str)
    if not _redis_obj.check_server_alive():
        logger.debug("Redis not alive: func exec")
        return func(*func_args, **func_kwargs)

    response_from_service = _redis_obj.get(key, prefix)
    if response_from_service is not None:
        logger.debug(
            f"Redis returned object from cache for key ({key}):"
            + str(response_from_service)
        )
        return response_from_service
    else:
        response_from_func = func(*func_args, **func_kwargs)
        logger.debug(
            f"Redis returned object from wrapped function for key ({key}): "
            + str(response_from_func)
        )
        _redis_obj.set(key, response_from_func, prefix)
        return response_from_func


def cache(**decorator_kwargs):
    """
    Decorator
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            if not _redis_obj.check_server_alive():
                return func(*func_args, **func_kwargs)

            call_str = _convert_func_call_attributes_to_str(
                func, func_args, func_kwargs, decorator_kwargs
            )
            response_from_service = _redis_obj.get(call_str)

            if response_from_service is not None:
                logger.debug(
                    f"Redis returned object from cache for key ({call_str}):"
                    + str(response_from_service)
                )
                return response_from_service
            else:
                response = func(*func_args, **func_kwargs)
                logger.debug(
                    f"Redis returned object from wrapped function for key ({call_str}): "
                    + str(response)
                )
                _redis_obj.set(call_str, response)
            return response

        return wrapper

    return decorator
