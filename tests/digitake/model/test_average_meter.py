from src.digitake.model import AverageMeter


def test_average_meter():
    loss = AverageMeter('loss')
    v1 = loss(2, 4)
    v2 = loss(2, 8)
    test_val = (2*4+2*8)/12
    assert v2 == test_val, f"{v2} != {test_val}"
