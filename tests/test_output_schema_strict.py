from __future__ import annotations

from jsonschema import Draft202012Validator


def test_production_output_schema_rejects_unknown_fields() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"srd_version": {"const": "8.7"}},
        "required": ["srd_version"],
    }
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors({"srd_version": "8.7", "extra": True}))
    assert errors
