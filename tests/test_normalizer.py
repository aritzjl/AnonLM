from anonlm.normalizer import normalize
from anonlm.schema import PIIType


def test_email_lowercased() -> None:
    assert normalize("  User@Example.COM  ", PIIType.EMAIL) == "user@example.com"


def test_email_already_lowercase() -> None:
    assert normalize("user@example.com", PIIType.EMAIL) == "user@example.com"


def test_phone_digits_only() -> None:
    assert normalize("612-345-678", PIIType.PHONE) == "612345678"


def test_phone_preserves_leading_plus() -> None:
    assert normalize("+34 612 345 678", PIIType.PHONE) == "+34612345678"


def test_phone_strips_spaces_and_dashes() -> None:
    assert normalize("(91) 234-56-78", PIIType.PHONE) == "912345678"


def test_id_number_removes_spaces_dashes_dots() -> None:
    assert normalize("12 345 678-A", PIIType.ID_NUMBER) == "12345678A"


def test_id_number_uppercased() -> None:
    assert normalize("xyz-987-654", PIIType.ID_NUMBER) == "XYZ987654"


def test_id_number_removes_dots() -> None:
    assert normalize("B.87.654.321", PIIType.ID_NUMBER) == "B87654321"


def test_person_collapses_spaces() -> None:
    assert normalize("Maria   Garcia   Lopez", PIIType.PERSON) == "Maria Garcia Lopez"


def test_person_strips() -> None:
    assert normalize("  Juan Perez  ", PIIType.PERSON) == "Juan Perez"


def test_org_collapses_spaces() -> None:
    assert normalize("Acme   Corporation  ", PIIType.ORG) == "Acme Corporation"
