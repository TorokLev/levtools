import logging

from levtools import redis_cache


def test_redis_decorator_with_or_without_redis_server():
    logging.basicConfig(level="DEBUG", force=True)

    not_from_cache = None

    @redis_cache.cache()
    def dummy(variable):
        nonlocal not_from_cache
        not_from_cache = True
        return variable + 3

    # missing setup to prevent redis client connection
    assert dummy(12345) == 12345 + 3
    assert not_from_cache

def test_get_set():

    redis_cache.setup()
    if not redis_cache.check_server_alive():
        return

    redis_cache.cache_set("random_key", "random_value")
    assert redis_cache.cache_get("random_key") == "random_value"

    redis_cache.cache_set("random_key", "random_value2")
    assert redis_cache.cache_get("random_key") == "random_value2"
