from datetime import datetime
import os

from levtools import utils as u


def test_to_json():
    # given
    class A:
        dt: datetime  # this is a non serializable object

    a = A()
    a.dt = datetime(2021, 1, 1)
    expected_json_str = '{\n    "dt": "2021-01-01T00:00:00"\n}'

    # when
    actual_json_str = u.to_json(a)

    # then
    assert expected_json_str == actual_json_str


def test_path_automatic_coversion():
    if u.is_linux():
        assert u.path("a/b") == 'a/b'
        assert u.path("a\\b") == "a/b"

    if u.is_windows():
        assert u.path("a/b") == "a\\b"
        assert u.path("a\\b") == "a\\b"


def test_is_rel_path():
    assert u.is_rel_path(u.path("a/b"))

    if u.is_linux():
        assert not u.is_rel_path(u.path("/a/b"))

    if u.is_windows():
        assert not u.is_rel_path(u.path("c:\\a\\b"))


def test_create_dirs(tmp_path):

    path = u.path_resolver('any_path')

    path = str(tmp_path / "subdir")
    path = u.path_resolver(path, create_as_dir=True)
    assert os.path.exists(path)

    path = str(tmp_path / "subdir2" / "file")
    _ = u.path_resolver(path, create_dirs=True)
    assert os.path.exists(os.path.dirname(path))

    path = u.path_resolver('<tmpdir>', create_as_dir=True)
    assert os.path.exists(path)

    path = u.path_resolver('<json>')
    assert os.path.exists(path)


def test_pip_version():
    assert '.' in u.get_pip_version("pip")


def test_get_pkgroot():
    assert len(u.get_pkgroot('numpy')) > len('numpy')


def test_to_datetime():
    assert u.to_datetime("2022-03-17T00:00:00+01:00", to_naive=True).tz is None
    assert u.to_datetime("2022-03-17T00:00:00+01:00", to_naive=False).tz is not None


def test_get_recursive_modif_date():
    assert max(u.get_recursive_modif_dates(".").values()) > 0


def test_keep_by_set():

    assert u.keep_by_set(["first", "second", "third"], keep_set=["third", "first"]) == ['first', 'third']


def test_drop_by_set():

    assert u.drop_by_set(["first", "second", "third"], drop_set=["third", "first"]) == ['second']


def test_skip_chars_from_start():

    assert u.skip_chars_from_start("SKIP-from text", chars_to_skip="-PISK") == "from text"


def test_get_chars_from_start():

    assert  u.get_chars_from_start("some text here!.\"", stop_chars="!") == "some text here"


def test_firstn():

    assert list(u.firstn(['a', 'b', 'c', 'd'], n=2)) == ['a', 'b']


def test_has_any():

    assert u.has_any(strings=["aa", "bb"], text="something aa")
    assert u.has_any(strings=["aa", "bb"], text="something bb")
    assert not u.has_any(strings=["aa", "bb"], text="something cc")


def test_replace_in_keys():
    assert u.replace_in_keys({"aa__bb": 3}, from_what="__", to_what=".") == {"aa.bb": 3}


def test_get_days_from_epoch():
    assert u.days_from_epoch(datetime.now()) > 0
    assert type(u.days_from_epoch(datetime.now())) == int
