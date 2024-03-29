"""
Copyright 2024 Levente Torok TorokLev@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation the

rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import functools
import hashlib
import json
import logging
import time
import json_tricks as json

import redis
import rediscluster

from levtools import utils


# alternative to https://github.com/comeuplater/fastapi_cache
# or RedisCacheBackend https://pythonrepo.com/repo/comeuplater-fastapi_cache-python-fastapi-utilities
# or apscheduler at https://gist.github.com/ivanleoncz/21293b00d0ea54db8ee3b57fb1170ddf
# or async event loops for signal handling
# Beware that callers are from different processes

# Singleton since modules are singleton in Python
# Only server_up and server_down are meant to be used for manipulating these variables
# and server_alive for checking

# state variables:

class Redis(utils.SingletonClass):
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
            logging.debug("Redis client started")
        else:
            self.conn = rediscluster.RedisCluster(
                startup_nodes=startup_nodes, decode_responses=True, **kwargs
            )
            logging.debug("RedisCluster client started")

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
        logging.warning("Redis server down")
        self.server_alive = False
        self._next_check_deadline = time.time() + self._alive_check_timeout

    def _server_up(self):
        """
        sets internal state variable to up
        """
        logging.warning("Redis server up")
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
        self.conn.set(cache_key, str(json.dumps(value)))

    def _cache_get(self, key, prefix=""):
        cache_key = (prefix + ":" if len(prefix) > 0 else "") + self.hash_it(key)
        response = self.conn.get(cache_key)
        if response:
            return json.loads(response.decode("utf-8"))

    def _cache_delete(self, key, prefix=""):
        cache_key = (prefix + ":" if len(prefix) > 0 else "") + self.hash_it(key)
        response = self.conn.delete(cache_key)
        logging.debug(f"Redis delete cache for key ({key}):" + str(response))
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
        return self._error_handler_wrapper(self._cache_get, key=key, prefix=prefix)

    def set(self, key, value, prefix=""):
        return self._error_handler_wrapper(self._cache_set, key=key, value=value, prefix=prefix)

    def delete(self, key, prefix):
        return self._error_handler_wrapper(self._cache_delete, key=key, prefix=prefix)

    def delete_all_keys(self, filtering_prefix=""):
        return self._error_handler_wrapper(self._delete_all_keys, filtering_prefix)


_redis_obj = Redis()


def _convert_func_call_attributes_to_str(
        func, func_args, func_kwargs, decorator_kwargs
):
    func_name = func.__name__

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


def checkpoint_caching(func,
                       prefix="",
                       key="",
                       to_json_converter=json.dumps,
                       from_json_converter=json.loads,
                       func_args=(), **func_kwargs):
    """
    func:  function to be cached, func name added to prefix
    func_args, func_kwargs:  parameters of the function (encoded in hash)
    prefix:  redis_cache_prefix
    key:     encoded in hash
    to_json_converter:  if json.dumps is not sufficient supply your own
    from_json_converter:  if json.dumps is not sufficient supply your own
    """

    redis_key = f"key: {key} :func: {func.__name__} :args: {str(func_args)} :kwargs: {str(func_kwargs)}"

    if not _redis_obj.check_server_alive():
        logging.debug("Redis: Not alive! Function {func.__name__} called")
        return func(*func_args, **func_kwargs)

    response_from_service = _redis_obj.get(key=redis_key, prefix=prefix)
    if response_from_service is not None:
        logging.debug(
            f"Redis: Returned response prefix={prefix}, key={redis_key}, value={response_from_service[:10]}..."
        )
        return from_json_converter(response_from_service)
    else:
        logging.debug(f"Redis: No response prefix={prefix} key={redis_key}")

        response_from_func = func(*func_args, **func_kwargs)

        json_response_from_func = to_json_converter(response_from_func)
        logging.debug(
            f"Redis: Save to cache prefix={prefix} key={redis_key} value={json_response_from_func[:10]}..."
        )
        _redis_obj.set(key=redis_key, prefix=prefix, value=json_response_from_func)
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
                _redis_obj.set(call_str, response)
            return response

        return wrapper

    return decorator
