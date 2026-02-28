"""Schema definitions used by the anonymization agent."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PIIType(str, Enum):
    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ID_NUMBER = "ID_NUMBER"
    ORG = "ORG"


class PIIEntity(BaseModel):
    type: PIIType
    text: str


class PIIResponse(BaseModel):
    entities: list[PIIEntity]


class PIILink(BaseModel):
    type: PIIType
    representative: str
    aliases: list[str] = Field(default_factory=list)


class PIILinkingResponse(BaseModel):
    links: list[PIILink]
