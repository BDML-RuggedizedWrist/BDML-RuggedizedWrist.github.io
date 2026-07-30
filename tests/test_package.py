def test_public_package_version_exists():
    import rizon_osc

    assert rizon_osc.__version__
