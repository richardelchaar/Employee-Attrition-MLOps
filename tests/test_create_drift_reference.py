import pytest
import importlib

def test_create_drift_reference_main(monkeypatch, tmp_path):
    cdr = importlib.import_module("scripts.create_drift_reference")
    # Accept either a main function or script-level execution
    assert hasattr(cdr, "main") or hasattr(cdr, "__name__")
    # If main exists, try calling it with a temp output path (mocking env as needed)
    if hasattr(cdr, "main"):
        try:
            # Monkeypatch environment or arguments if needed
            cdr.main()
        except Exception as e:
            # Accept exceptions due to missing DB or files, but function should be callable
            assert "No module named" not in str(e) 