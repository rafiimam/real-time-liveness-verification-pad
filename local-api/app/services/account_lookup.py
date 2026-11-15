"""Account-number resolution via Jamuna Bank core middleware APIs.

Flow:
1. Enquire customer → get NID, name, CIF
2. Login to NID verification service → get access token
3. Call NID demographic verification → get NID photo URL
4. Fetch customer sign (account photo) from core middleware
5. Return both reference photos + metadata for face matching
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Core middleware
CORE_MW_URL = os.getenv(
    "CORE_MIDDLEWARE_URL",
    "http://localhost:4018/core-middleware",
).rstrip("/")
CORE_MW_TIMEOUT = float(os.getenv("CORE_MIDDLEWARE_TIMEOUT", "30"))

# NID verification service
NID_SERVICE_URL = os.getenv(
    "NID_SERVICE_URL",
    "http://localhost:8012/nid-verification",
).rstrip("/")
NID_USERNAME = os.getenv("NID_USERNAME", "")
NID_PASSWORD = os.getenv("NID_PASSWORD", "")
NID_TIMEOUT = float(os.getenv("NID_TIMEOUT", "30"))

# Cached NID auth token
_nid_token_cache: dict = {"token": None, "expires_at": 0.0}


async def resolve_account_number(account_number: str) -> dict:
    """Resolve account_number to reference photos and customer info.

    Returns dict with keys:
      - account_number, subject_id, resolved (bool)
      - customer_name, nid_number
      - nid_photo_b64 (base64 JPEG from NID service, or None)
      - account_photo_b64 (base64 JPEG customer sign, or None)
      - nid_verified (bool)
      - errors (list of error messages for partial failures)
    """
    acct = (account_number or "").strip()
    result: dict = {
        "account_number": acct,
        "subject_id": acct,
        "resolved": False,
        "customer_name": None,
        "nid_number": None,
        "nid_photo_b64": None,
        "account_photo_b64": None,
        "nid_verified": False,
        "errors": [],
    }

    if not acct:
        result["errors"].append("Empty account number")
        return result

    # Step 1: Customer enquiry
    customer = await _enquire_customer(acct)
    if customer is None:
        result["errors"].append("Customer enquiry failed")
        return result

    result["customer_name"] = _customer_display_name(customer)
    result["subject_id"] = customer.get("cif_number") or acct

    # Extract NID: type 01=10-digit, 02=17-digit, 51=13-digit (+ birth year → 17)
    nid_info = _extract_nid(customer)
    if not nid_info:
        result["errors"].append("No NID found for customer")
    else:
        result["nid_number"] = nid_info["nid_for_api"]
        name_candidates = _nid_name_candidates(customer)
        print(
            f"[ACCOUNT] NID type={nid_info['identity_type']} "
            f"raw={nid_info['nid_raw']} api={nid_info['nid_for_api']} "
            f"field={nid_info['api_field']} name_candidates={name_candidates!r}",
            flush=True,
        )

        nid_photo = None
        for name_en in name_candidates:
            nid_photo = await _get_nid_photo(
                api_field=nid_info["api_field"],
                nid_value=nid_info["nid_for_api"],
                customer_name=name_en,
            )
            if nid_photo:
                print(f"[NID] photo OK with nameEn={name_en!r}", flush=True)
                break
            print(f"[NID] no photo for nameEn={name_en!r} — trying next", flush=True)

        if nid_photo:
            result["nid_photo_b64"] = nid_photo["photo_b64"]
            result["nid_verified"] = nid_photo["verified"]
        else:
            result["errors"].append(
                f"NID photo retrieval failed (tried nameEn={name_candidates})"
            )

    # Step 4: Customer sign (account photo)
    account_photo = await _get_customer_sign(acct)
    if account_photo:
        result["account_photo_b64"] = account_photo
    else:
        result["errors"].append("Customer photo not found for this account")

    result["resolved"] = True
    return result


def _customer_display_name(customer: dict) -> str:
    """Prefer customer_name; fall back to first_name + last_name."""
    candidates = _nid_name_candidates(customer)
    return candidates[0] if candidates else ""


def _nid_name_candidates(customer: dict) -> list[str]:
    """Name variants for NID nameEn — CBS customer_name often omits last_name.

    Order: customer_name alone, then first_name + last_name, then first_name.
    Deduped case-insensitively while preserving first-seen casing.
    """
    names: list[str] = []
    seen: set[str] = set()

    def add(raw: str | None) -> None:
        cleaned = " ".join((raw or "").split())
        if not cleaned:
            return
        key = cleaned.casefold()
        if key in seen:
            return
        seen.add(key)
        names.append(cleaned)

    add(customer.get("customer_name"))
    first = (customer.get("first_name") or "").strip()
    last = (customer.get("last_name") or "").strip()
    add(f"{first} {last}".strip())
    add(first)
    return names


def _birth_year(dob: Optional[str]) -> Optional[str]:
    """Parse DOB (DD-MM-YYYY or YYYY-MM-DD) → 4-digit year."""
    if not dob:
        return None
    parts = dob.strip().split("-")
    if len(parts) != 3:
        return None
    if len(parts[2]) == 4:
        return parts[2]  # DD-MM-YYYY
    if len(parts[0]) == 4:
        return parts[0]  # YYYY-MM-DD
    return None


def _extract_nid(customer: dict) -> Optional[dict]:
    """Extract NID from identities.

    identity_type:
      01 → 10-digit NID  → EC field nid10Digit
      02 → 17-digit NID  → EC field nid17Digit
      51 → 13-digit NID  → prepend birth year → 17-digit → nid17Digit

    Preference order when multiple present: 02, then 01, then 51.
    """
    identities = customer.get("identities") or []
    by_type: dict[str, str] = {}
    for ident in identities:
        itype = str(ident.get("identity_type") or "").strip()
        inum = (ident.get("identity_number") or "").strip()
        if itype in ("01", "02", "51") and inum:
            by_type[itype] = inum

    dob = customer.get("date_of_birth")

    if "02" in by_type:
        nid = by_type["02"]
        # If stored as 13 by mistake, still try to expand
        if len(nid) == 13:
            year = _birth_year(dob)
            if year:
                nid = year + nid
        return {
            "identity_type": "02",
            "nid_raw": by_type["02"],
            "nid_for_api": nid,
            "api_field": "nid17Digit" if len(nid) == 17 else "nid10Digit",
        }

    if "01" in by_type:
        nid = by_type["01"]
        return {
            "identity_type": "01",
            "nid_raw": nid,
            "nid_for_api": nid,
            "api_field": "nid10Digit",
        }

    if "51" in by_type:
        nid = by_type["51"]
        year = _birth_year(dob)
        if len(nid) == 13 and year:
            nid17 = year + nid
            return {
                "identity_type": "51",
                "nid_raw": nid,
                "nid_for_api": nid17,
                "api_field": "nid17Digit",
            }
        if len(nid) == 17:
            return {
                "identity_type": "51",
                "nid_raw": nid,
                "nid_for_api": nid,
                "api_field": "nid17Digit",
            }
        logger.warning(
            "identity_type 51 present but cannot build 17-digit NID "
            "(len=%s dob=%s)",
            len(nid),
            dob,
        )
        return None

    return None


async def _enquire_customer(account_number: str) -> Optional[dict]:
    """Call core-middleware customer enquiry."""
    url = f"{CORE_MW_URL}/customer/enquire-customer"
    headers = {
        "Content-Type": "application/json",
        "conversation_id": f"liveness-{account_number}-{int(time.time())}",
    }
    payload = {"account_number": account_number}

    try:
        async with httpx.AsyncClient(timeout=CORE_MW_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            data = resp.json()
            if data.get("success") and data.get("data"):
                logger.info("Customer enquiry OK for %s", account_number)
                return data["data"]
            else:
                logger.warning("Customer enquiry failed: %s", data.get("message"))
                return None
    except Exception as e:
        logger.exception("Customer enquiry error for %s", account_number)
        return None


async def _get_nid_auth_token() -> Optional[str]:
    """Login to NID verification service and cache the token."""
    now = time.time()
    if _nid_token_cache["token"] and _nid_token_cache["expires_at"] > now + 60:
        return _nid_token_cache["token"]

    url = f"{NID_SERVICE_URL}/auth/login"
    payload = {"username": NID_USERNAME, "password": NID_PASSWORD}

    try:
        async with httpx.AsyncClient(timeout=NID_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()
            if data.get("statusCode") == "SUCCESS":
                token = data["success"]["data"]["access_token"]
                _nid_token_cache["token"] = token
                _nid_token_cache["expires_at"] = now + 11 * 3600  # ~11 hrs
                return token
            else:
                logger.warning("NID auth failed: %s", data)
                return None
    except Exception as e:
        logger.exception("NID auth error")
        return None


async def _get_nid_photo(
    *,
    api_field: str,
    nid_value: str,
    customer_name: str,
) -> Optional[dict]:
    """Call NID demographic verification and fetch the photo.

    api_field: 'nid10Digit' or 'nid17Digit'
    Returns {"photo_b64": ..., "verified": bool} or None.
    """
    token = await _get_nid_auth_token()
    if not token:
        return None

    url = f"{NID_SERVICE_URL}/voter/demographic/verification"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    name_en = (customer_name or "").strip()
    payload = {
        "identify": {api_field: nid_value},
        "verify": {"nameEn": name_en},
    }
    print(f"[NID] verify field={api_field} value={nid_value} nameEn={name_en!r}", flush=True)

    try:
        async with httpx.AsyncClient(timeout=NID_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            data = resp.json()

            verified = data.get("verified", False)
            photo_url = None

            if data.get("statusCode") == "SUCCESS" or data.get("status") == "OK":
                success_data = data.get("success", {}).get("data", {})
                photo_url = success_data.get("photo")

            if not photo_url:
                logger.warning(
                    "NID verification response: status=%s message=%s verified=%s",
                    data.get("status"),
                    data.get("message"),
                    verified,
                )
                success_data = data.get("success", {}).get("data", {}) if data.get("success") else {}
                photo_url = success_data.get("photo") if success_data else None

            if photo_url:
                photo_b64 = await _download_image_as_b64(photo_url)
                if photo_b64:
                    return {"photo_b64": photo_b64, "verified": verified}

            logger.warning("NID photo URL not available")
            return None
    except Exception as e:
        logger.exception("NID verification error")
        return None


async def _download_image_as_b64(url: str) -> Optional[str]:
    """Download an image URL and return as base64 string."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return base64.b64encode(resp.content).decode()
            logger.warning("Image download failed status=%s url=%s", resp.status_code, url[:80])
            return None
    except Exception as e:
        logger.exception("Image download error")
        return None


async def _get_customer_sign(account_number: str) -> Optional[str]:
    """Fetch customer sign (photo) from core middleware.

    Returns base64-encoded image or None.
    """
    url = f"{CORE_MW_URL}/customer/view-customer-details"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
    }
    payload = {"account_number": account_number}

    try:
        async with httpx.AsyncClient(timeout=CORE_MW_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            data = resp.json()
            if data.get("success") and data.get("data"):
                sign_b64 = data["data"].get("customer_sign")
                if sign_b64:
                    return sign_b64
                logger.warning("No customer_sign in response for %s", account_number)
                return None
            else:
                logger.warning(
                    "Customer details failed for %s: %s",
                    account_number,
                    data.get("message") or data.get("errorMsg"),
                )
                return None
    except Exception as e:
        logger.exception("Customer details error for %s", account_number)
        return None
