import logging

from levtools import redis_cache


def test_redis_decorator_without_redis_server():
    logging.basicConfig(level="DEBUG", force=True)

    not_from_cache = None

    # missing setup to prevent redis client connection
    @redis_cache.cache()
    def dummy(variable):
        nonlocal not_from_cache
        not_from_cache = True
        return variable + 3

    assert dummy(12345) == 12345 + 3
    assert not_from_cache


def test_redis_decorator_with_redis_server():
    logging.basicConfig(level="DEBUG", force=True)

    not_from_cache = None

    redis_cache.setup()
    assert redis_cache.conn.ping()

    @redis_cache.cache()
    def dummy(variable):
        nonlocal not_from_cache
        not_from_cache = True
        return variable + 3

    assert dummy(12345) == 12345 + 3
    assert not_from_cache
    not_from_cache = False
    assert dummy(12345) == 12345 + 3
    assert not not_from_cache

    redis_cache.close()
     

def test_get_set():

    redis_cache.setup()
    if not redis_cache.check_server_alive():
        print("Redis server down")
        return

    redis_cache.cache_set("random_key", "random_value")
    assert redis_cache.cache_get("random_key") == "random_value" # write into cache
    assert redis_cache.cache_get("random_key") == "random_value" # from cache

    redis_cache.cache_set("random_key", "random_value2")
    assert redis_cache.cache_get("random_key") == "random_value2"
    redis_cache.close()

