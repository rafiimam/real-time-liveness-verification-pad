# Pilot metrics playbook

Track FAR/FRR on a **fixed labeled pack** before raising traffic.

## Labels

In `training_sessions`:

- `real` — genuine live face on that device
- `spoof` — screen replay, print, or other presentation attack

Correct labels via:

```http
POST /training/sessions/{session_id}/label
Authorization: Bearer <ADMIN_API_KEY>
{"label": "spoof"}
```

## Run

```powershell
cd F:\Project\LivenessVerification\local-api
.\.venv\Scripts\python.exe ..\scripts\pilot_metrics.py
# or a known pack:
.\.venv\Scripts\python.exe ..\scripts\pilot_metrics.py --ids 4,5,7,10,11,12,14,24,25,26,27,28
```

Output: console table + `docs/pilot_metrics_latest.json`.

## Targets (pilot)

| Metric | Meaning | Pilot goal |
|--------|---------|------------|
| FAR | Spoof approved | Prefer near 0 on known PC/screen pack |
| FRR | Live rejected | Keep low enough for UX; investigate BADs |

Freeze thresholds only after the pack is stable week-over-week.
