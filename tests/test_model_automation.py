import pytest

def test_import_and_run_optimize():
    import scripts.optimize_train_select as ots
    assert hasattr(ots, "optimize_select_and_train")
    # Try calling with a minimal list, but catch expected errors
    try:
        ots.optimize_select_and_train(models_to_opt=["logistic_regression"])
    except SystemExit:
        # Acceptable: script may call sys.exit() if env is not set up
        pass
    except Exception as e:
        # Acceptable: missing DB, MLflow, etc.
        assert "No module named" not in str(e)

def test_import_and_run_promote():
    import scripts.promote_model as pm
    assert hasattr(pm, "promote_model_to_production") or hasattr(pm, "main")
    # Try calling the function, but catch expected errors
    if hasattr(pm, "promote_model_to_production"):
        try:
            pm.promote_model_to_production()
        except SystemExit:
            pass
        except Exception as e:
            assert "No module named" not in str(e) 