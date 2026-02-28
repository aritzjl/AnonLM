from anonlm import deanonymize


def test_deanonymize_reconstructs_text() -> None:
    text = "Hello [[PERSON_1]], email [[EMAIL_1]]."
    mapping_reverse = {
        "[[PERSON_1]]": "Maria Garcia",
        "[[EMAIL_1]]": "maria@example.com",
    }

    restored = deanonymize(text, mapping_reverse)
    assert restored == "Hello Maria Garcia, email maria@example.com."
