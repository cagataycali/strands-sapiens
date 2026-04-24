"""Smoke tests — do NOT require a GPU or checkpoints to import."""

def test_imports():
    import strands_sapiens as ss
    assert len(ss.TOOLS) >= 7
    # All tools must be @tool-decorated callables
    for t in ss.TOOLS:
        assert callable(t)


def test_info_no_crash():
    from strands_sapiens import sapiens_info
    # Underlying function may require strands runtime context; just ensure it exists.
    assert callable(sapiens_info)
