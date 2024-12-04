from fedn import APIClient


def test_api():
    api = APIClient("localhost", 8092)
    res = api.get_active_clients()
    print(res)

    assert isinstance(res["count"], int)
