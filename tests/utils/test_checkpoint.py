import json

from neurostream.utils.checkpoint import CheckpointManager


def test_new_checkpoint_is_empty(tmp_path):
    cp = CheckpointManager(tmp_path / "checkpoint.json")
    assert cp.completed_count == 0


def test_is_done_false_for_unknown(tmp_path):
    cp = CheckpointManager(tmp_path / "checkpoint.json")
    assert not cp.is_done("PhysionetMI", 1)


def test_mark_done_makes_is_done_true(tmp_path):
    cp = CheckpointManager(tmp_path / "checkpoint.json")
    cp.mark_done("PhysionetMI", 1)
    assert cp.is_done("PhysionetMI", 1)


def test_mark_done_does_not_affect_other_subjects(tmp_path):
    cp = CheckpointManager(tmp_path / "checkpoint.json")
    cp.mark_done("PhysionetMI", 1)
    assert not cp.is_done("PhysionetMI", 2)
    assert not cp.is_done("Cho2017", 1)


def test_mark_done_persists_to_disk(tmp_path):
    path = tmp_path / "checkpoint.json"
    cp = CheckpointManager(path)
    cp.mark_done("PhysionetMI", 3)
    assert path.exists()
    data = json.loads(path.read_text())
    assert ["PhysionetMI", 3] in data["completed"]


def test_reload_from_disk_restores_state(tmp_path):
    path = tmp_path / "checkpoint.json"
    cp1 = CheckpointManager(path)
    cp1.mark_done("Cho2017", 7)
    cp1.mark_done("Stieger2021", 42)

    cp2 = CheckpointManager(path)
    assert cp2.is_done("Cho2017", 7)
    assert cp2.is_done("Stieger2021", 42)
    assert not cp2.is_done("Cho2017", 8)
    assert cp2.completed_count == 2


def test_mark_done_idempotent(tmp_path):
    cp = CheckpointManager(tmp_path / "checkpoint.json")
    cp.mark_done("PhysionetMI", 1)
    cp.mark_done("PhysionetMI", 1)
    assert cp.completed_count == 1


def test_checkpoint_file_not_created_until_mark_done(tmp_path):
    path = tmp_path / "checkpoint.json"
    CheckpointManager(path)
    assert not path.exists()


def test_missing_checkpoint_file_is_not_an_error(tmp_path):
    cp = CheckpointManager(tmp_path / "nonexistent" / "checkpoint.json")
    assert not cp.is_done("PhysionetMI", 1)


def test_rejected_entries_persisted_and_reloaded(tmp_path):
    path = tmp_path / "checkpoint.json"
    entry = {
        "source": "Cho2017",
        "subject": 5,
        "session": "0",
        "run": "0",
        "reason": "TOO_SHORT",
    }
    cp1 = CheckpointManager(path)
    cp1.mark_done("Cho2017", 5, rejected=[entry])

    cp2 = CheckpointManager(path)
    assert cp2.rejected == [entry]


def test_rejected_accumulates_across_subjects(tmp_path):
    path = tmp_path / "checkpoint.json"
    cp = CheckpointManager(path)
    cp.mark_done("PhysionetMI", 1, rejected=[{"reason": "TOO_SHORT"}])
    cp.mark_done(
        "PhysionetMI", 2, rejected=[{"reason": "NAN_HEAVY"}, {"reason": "TOO_SHORT"}]
    )
    assert len(cp.rejected) == 3


def test_rejected_empty_by_default(tmp_path):
    cp = CheckpointManager(tmp_path / "checkpoint.json")
    assert cp.rejected == []
