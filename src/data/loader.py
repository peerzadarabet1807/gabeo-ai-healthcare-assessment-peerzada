from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from src.models.claim import Claim835, Claim837, JoinedClaim


class ClaimLoader:
    def load_file(self, path: Union[str, Path]) -> list[JoinedClaim]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Claims file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        claims_data = raw if isinstance(raw, list) else raw.get("claims", [])
        return [self._parse_claim(c) for c in claims_data]

    def load_from_dict(self, data: dict) -> JoinedClaim:
        return self._parse_claim(data)

    def load_from_dicts(self, data: list[dict]) -> list[JoinedClaim]:
        return [self._parse_claim(c) for c in data]

    def _parse_claim(self, raw: dict) -> JoinedClaim:
        claim_id = raw.get("claim_id", "")
        label = raw.get("label")
        raw_835 = raw.get("claim_835", {})
        raw_837 = raw.get("claim_837", {})

        # Ensure pc_ClaimID is populated
        if "pc_ClaimID" not in raw_835 or not raw_835["pc_ClaimID"]:
            raw_835["pc_ClaimID"] = claim_id

        # Ensure ec_ClaimNo is populated
        if "ec_ClaimNo" not in raw_837 or not raw_837["ec_ClaimNo"]:
            raw_837["ec_ClaimNo"] = claim_id

        return JoinedClaim(
            claim_id=claim_id,
            claim_835=Claim835(**raw_835),
            claim_837=Claim837(**raw_837),
            label=label,
        )

    def filter_denied(self, claims: list[JoinedClaim]) -> list[JoinedClaim]:
        return [c for c in claims if c.is_denied]

    def filter_paid(self, claims: list[JoinedClaim]) -> list[JoinedClaim]:
        return [c for c in claims if not c.is_denied]
