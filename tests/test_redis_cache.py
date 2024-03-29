import logging
import multiprocessing as mp

from levtools import redis_cache

logging.basicConfig(level="DEBUG", force=True)


def test_set_get_delete():
    # lower layer test
    cache = redis_cache.Redis().setup()
    if not cache.check_server_alive():
        logging.error("Redis server cannot be connected to.")
        return
    cache.delete_all_keys()

    cache.set("1", "random_value", prefix="pytest")
    assert cache.get("1", prefix="pytest") == "random_value"

    cache.set("2", "random_value2", prefix="pytest")
    assert cache.get("2", prefix="pytest") == "random_value2"

    cache.delete("1", prefix="pytest")
    cache.delete("2", prefix="pytest")

    cache.close()


def test_redis_func_decorator_with_redis_server():
    cache = redis_cache.Redis().setup()
    if not cache.check_server_alive():
        logging.error("Redis server cannot be connected to.")
        return
    cache.delete_all_keys()

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
    cache.delete_all_keys()
    cache.close()


def test_redis_func_decorator_without_redis_server():
    cache = redis_cache.Redis().setup()
    cache.close()

    not_from_cache = None

    # missing setup to prevent redis client connection
    @redis_cache.cache()
    def dummy(variable):
        nonlocal not_from_cache
        not_from_cache = True
        return variable + 3

    assert dummy(1) == 1 + 3
    assert not_from_cache

    not_from_cache = False
    assert dummy(1) == 1 + 3
    assert not_from_cache

    cache.close()


def test_redis_func_decorator_without_redis_server():
    redis_cache.Redis().close()

    not_from_cache = None

    # missing setup to prevent redis client connection
    @redis_cache.cache()
    def dummy(variable):
        nonlocal not_from_cache
        not_from_cache = True
        return variable + 3

    assert dummy(1) == 1 + 3
    assert not_from_cache

    not_from_cache = False
    assert dummy(1) == 1 + 3
    assert not_from_cache

    redis_cache.Redis().close()


def test_redis_class_member_decorator_with_redis_server():
    cache = redis_cache.Redis().setup()
    if not cache.check_server_alive():
        logging.error("Redis server cannot be connected to.")
        return
    cache.delete_all_keys()

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

    cache.delete_all_keys()
    cache.close()


def test_redis_subfunc_decorator_with_redis_server():
    cache = redis_cache.Redis().setup()
    if not cache.check_server_alive():
        logging.error("Redis server cannot be connected to.")
        return
    cache.delete_all_keys()

    calculated_in_func = None

    def outer_func(arg):
        @redis_cache.cache()
        def subfunc(variable):
            nonlocal calculated_in_func
            calculated_in_func = True
            return variable + 3

        return subfunc(arg)

    calculated_in_func = False
    assert outer_func(1) == 1 + 3
    assert calculated_in_func

    calculated_in_func = False
    assert outer_func(1) == 1 + 3
    assert not calculated_in_func

    cache.delete_all_keys()
    cache.close()


def test_redis_cache_shared_across_processes():
    cache = redis_cache.Redis().setup(shared_across_processes=True)
    if not cache.check_server_alive():
        logging.error("Redis server cannot be connected to.")
        return

    calculated_in_func = False

    @redis_cache.cache()
    def calculator_func(variable):
        nonlocal calculated_in_func
        calculated_in_func = True
        return variable + 3

    def checker_write_to_cache():
        # write into redis
        nonlocal calculated_in_func
        calculated_in_func = False
        assert calculator_func(10) == 10 + 3
        assert calculated_in_func

    def checker_read_from_cache(return_dict):
        # write into redis
        nonlocal calculated_in_func
        calculated_in_func = False
        return_dict["calculated"] = calculator_func(10)
        return_dict["calculated_in_func"] = calculated_in_func
        assert not calculated_in_func

    cache.delete_all_keys()

    checker_write_to_cache()

    manager = mp.Manager()
    return_dict = manager.dict()
    p = mp.Process(target=checker_read_from_cache, args=(return_dict,))
    p.start()
    p.join()

    assert return_dict["calculated"] == 13
    assert not return_dict["calculated_in_func"]
    cache.delete_all_keys()
    cache.close()


def test_checkpoint_caching():
    cache = redis_cache.Redis().setup(shared_across_processes=True)
    if not cache.check_server_alive():
        logging.error("Redis server cannot be connected to.")
        return
    cache.delete_all_keys()

    calculated_in_func = False

    def calculator_func(variable):
        nonlocal calculated_in_func
        calculated_in_func = True
        return variable + 3

    value = redis_cache.checkpoint_caching(
        key="pytest_checkpoint_caching", func=calculator_func, func_args=[3]
    )
    assert value == 6
    assert calculated_in_func == True

    calculated_in_func = False

    value = redis_cache.checkpoint_caching(
        key="pytest_checkpoint_caching", func=calculator_func, func_args=[3]
    )
    assert value == 6
    assert calculated_in_func == False

    cache.delete_all_keys()
    cache.close()
