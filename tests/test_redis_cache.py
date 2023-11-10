import logging

from levtools import redis_cache
logging.basicConfig(level="DEBUG", force=True)

def test_get_set():
    # lower layer test
    redis_cache.setup()
    if not redis_cache.check_server_alive():

        logging.error("Redis server cannot be connected to.")
        return

    redis_cache.cache_set("1", "random_value")
    assert redis_cache.cache_get("1") == "random_value"

    redis_cache.cache_set("2", "random_value2")
    assert redis_cache.cache_get("2") == "random_value2"

    redis_cache.close()


def test_redis_func_decorator_without_redis_server():
    redis_cache.setup()
    if not redis_cache.check_server_alive():
        logging.error("Redis server cannot be connected to.")
        return

    not_from_cache = None

    @redis_cache.cache()
    def dummy(variable):
        nonlocal not_from_cache
        not_from_cache = True
        return variable + 3

    # missing setup to prevent redis client connection
    assert dummy(1) == 1 + 3
    assert not_from_cache
    redis_cache.close()

def test_redis_func_decorator_with_redis_server():

    redis_cache.setup()
    if not redis_cache.check_server_alive():
        logging.error("Redis server cannot be connected to.")
        return

    calculated_in_function = False

    @redis_cache.cache()
    def dummy(variable):
        nonlocal calculated_in_function
        calculated_in_function = True
        return variable + 3

    assert dummy(2) == 2 + 3
    assert calculated_in_function


    calculated_in_function = False
    assert dummy(2) == 2 + 3
    assert not calculated_in_function
    redis_cache.close()


def test_redis_func_decorator_without_redis_server():

    redis_cache.close()
    assert not redis_cache.check_server_alive()

    calculated_in_function = False

    @redis_cache.cache()
    def dummy(variable):
        nonlocal calculated_in_function
        calculated_in_function = True
        return variable + 3

    assert dummy(2) == 2 + 3
    assert calculated_in_function


    calculated_in_function = False
    assert dummy(2) == 2 + 3
    assert calculated_in_function


def test_redis_class_member_decorator_with_redis_server():

    redis_cache.setup()
    if not redis_cache.check_server_alive():
        logging.error("Redis server cannot be connected to.")
        return

    calculated_in_func = None
    class AnyClass:
        @redis_cache.cache()
        def member_func(self, variable):
            nonlocal calculated_in_func
            calculated_in_func = True
            return variable + 3

    dummy = AnyClass()

    # missing setup to prevent redis client connection
    calculated_in_func = False
    assert dummy.member_func(1) == 1 + 3
    assert calculated_in_func

    calculated_in_func = False
    assert dummy.member_func(1) == 1 + 3
    assert not calculated_in_func

    redis_cache.close()


def test_redis_subfunc_decorator_with_redis_server():

    redis_cache.setup()
    if not redis_cache.check_server_alive():
        logging.error("Redis server cannot be connected to.")
        return

    calculated_in_func = None
    def outer_func(arg):
        @redis_cache.cache()
        def subfunc(variable):
            nonlocal calculated_in_func
            calculated_in_func = True
            return variable + 3

        return subfunc(arg)

    # missing setup to prevent redis client connection
    calculated_in_func = False
    assert outer_func(1) == 1 + 3
    assert calculated_in_func

    calculated_in_func = False
    assert outer_func(1) == 1 + 3
    assert not calculated_in_func

    redis_cache.close()
