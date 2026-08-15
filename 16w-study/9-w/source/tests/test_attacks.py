from attack_cases import cases, run_suite


def test_attack_catalog_has_at_least_fifteen_distinct_cases():
    catalog = cases()
    assert len(catalog) >= 15
    assert len({item.id for item in catalog}) == len(catalog)
    required = {"prompt-injection", "authorization", "cross-tenant", "exfiltration", "sandbox", "privacy"}
    assert required <= {item.category for item in catalog}


def test_every_deterministic_attack_meets_its_expected_defense():
    results = run_suite()
    assert len(results) == len(cases())
    assert {item.status for item in results} == {"PASS"}
