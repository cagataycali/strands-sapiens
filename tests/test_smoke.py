"""Smoke tests - do NOT require a GPU or checkpoints to import."""


import pytest


def test_imports():
    import strands_sapiens as ss
    assert len(ss.TOOLS) == 8
    # All tools must be @tool-decorated callables
    for t in ss.TOOLS:
        assert callable(t)


def test_arch_name():
    from strands_sapiens._common import arch_name
    # Dot must be stripped from short sizes.
    assert arch_name("0.1b") == "sapiens2_01b"
    assert arch_name("0.4b") == "sapiens2_04b"
    assert arch_name("0.8b") == "sapiens2_08b"
    # Round sizes pass through.
    assert arch_name("1b") == "sapiens2_1b"
    assert arch_name("5b") == "sapiens2_5b"
    # 4k variant shares the 1b arch.
    assert arch_name("1b_4k") == "sapiens2_1b"


def test_validate_size():
    from strands_sapiens._common import validate_size
    assert validate_size("pretrain", "0.1b") == "0.1b"
    assert validate_size("seg", "0.4B") == "0.4b"  # normalizes case
    with pytest.raises(ValueError):
        validate_size("seg", "0.1b")  # 0.1b not valid for seg
    with pytest.raises(ValueError):
        validate_size("seg", "bogus")


def test_checkpoint_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SAPIENS_CHECKPOINT_ROOT", str(tmp_path))
    from strands_sapiens._common import checkpoint_path, checkpoint_root
    assert checkpoint_root() == tmp_path
    p = checkpoint_path("seg", "0.4b")
    assert p == tmp_path / "seg" / "sapiens2_0.4b_seg.safetensors"
    p = checkpoint_path("pretrain", "1b")
    assert p == tmp_path / "pretrain" / "sapiens2_1b_pretrain.safetensors"


def test_resolve_input_file(tmp_path):
    from strands_sapiens._common import resolve_input
    f = tmp_path / "a.jpg"
    f.write_bytes(b"\xff\xd8\xff")
    in_dir, images = resolve_input(str(f))
    assert in_dir == tmp_path
    assert images == [f]


def test_resolve_input_dir(tmp_path):
    from strands_sapiens._common import resolve_input
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "ignore.txt").write_bytes(b"")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.jpg").write_bytes(b"")
    in_dir, images = resolve_input(str(tmp_path))
    names = sorted(i.name for i in images)
    assert names == ["a.jpg", "b.png"]  # non-recursive default
    # Recursive mode picks up the nested one.
    _, images_r = resolve_input(str(tmp_path), recursive=True)
    assert sorted(i.name for i in images_r) == ["a.jpg", "b.png", "c.jpg"]


def test_resolve_input_missing(tmp_path):
    from strands_sapiens._common import resolve_input
    with pytest.raises(FileNotFoundError):
        resolve_input(str(tmp_path / "nope.jpg"))


def test_resolve_input_unsupported(tmp_path):
    from strands_sapiens._common import resolve_input
    f = tmp_path / "doc.txt"
    f.write_text("hi")
    with pytest.raises(ValueError):
        resolve_input(str(f))


def test_sapiens_info_runs(monkeypatch, tmp_path):
    """sapiens_info should return a structured dict even without checkpoints."""
    monkeypatch.setenv("SAPIENS_CHECKPOINT_ROOT", str(tmp_path / "nonexistent"))
    from strands_sapiens import sapiens_info
    # @tool-wrapped; still directly callable.
    result = sapiens_info()
    assert result["status"] == "success"
    assert len(result["content"]) >= 1
    assert result["content"][0]["text"] == "sapiens info"
    # Structured data in JSON content block
    json_block = result["content"][1]["json"]
    assert json_block["checkpoint_root_exists"] is False
    assert json_block["available"] == {}
    assert json_block["detector_present"] is False
    assert "cuda" in json_block
    assert "available" in json_block["cuda"]
    assert "device_count" in json_block["cuda"]


def test_ok_err_format():
    from strands_sapiens._common import err, ok
    r = ok("hi", foo=1)
    assert r == {"status": "success", "content": [{"text": "hi"}, {"json": {"foo": 1}}]}
    r = err("bad", code=2)
    assert r == {"status": "error", "content": [{"text": "bad"}, {"json": {"code": 2}}]}
    # No extra kwargs -> no json block
    r = ok("simple")
    assert r == {"status": "success", "content": [{"text": "simple"}]}
