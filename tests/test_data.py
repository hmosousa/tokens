from src.data import read_data


def test_read_data():
    data = read_data()
    assert len(data) == 7_802_675
