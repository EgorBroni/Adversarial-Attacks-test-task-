from imagenette_stage0.data import stage0_class_split


def test_stage0_split_keeps_two_classes_held_out():
    split = stage0_class_split(training_class_count=8)

    assert len(split.training) == 8
    assert len(split.held_out) == 2
    assert set(split.training).isdisjoint(split.held_out)
