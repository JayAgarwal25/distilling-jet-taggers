from jetkd.benchmark import random_momenta, settings, summarize


def test_settings():
    assert settings("transformer_200k") == ["packed,eager,xformers", "packed,compiled,xformers",
                                            "padded,eager,xformers", "padded,compiled,xformers"]
    assert settings("lorentznet_200k") == ["packed,eager,builtin", "packed,compiled,builtin"]
    assert settings("part") == ["padded,eager,builtin", "padded,compiled,builtin"]


def test_random_jets_are_massive():
    p = random_momenta(4)
    assert p.shape == (4, 50, 4) and (p[..., 0] ** 2 - p[..., 1:].square().sum(-1) > 0).all()


def test_fastest_setting_by_the_median_of_its_runs():
    def m(setting, ms, repeat):
        return {"model": "x", "device": "cuda", "regime": "real", "setting": setting, "median_ms": ms,
                "peak_mb": 10.0, "repeat": repeat}

    runs = [m("a", 1.0, 0), m("a", 5.0, 1), m("a", 5.0, 2), m("b", 2.0, 0), m("b", 2.0, 1), m("b", 9.0, 2),
            {"model": "x", "device": "cuda", "regime": "real", "setting": "c", "error": "no kernel"}]
    (row,) = summarize(runs)
    assert row["selected"] == "b" and row["median_ms"] == 2.0 and row["runs_ms"] == [2.0, 2.0, 9.0]
    assert row["settings"]["c"] == {"error": "no kernel"} and row["batch_size"] == 512
