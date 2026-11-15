# Android Liveness SDK (v1.0.0)

Native Android client for Jamuna Liveness Verification. Talks to the same Decision API as the web microsite (`WS /ws/live`).

Public identifier is **account number** (not account id). Backend KYC lookup for that number can be wired later — today the number is stored/passed through as the session subject.

## Give this to Android developers

### Option A — open as a project (easiest for testing)

1. Open folder `android-sdk/` in **Android Studio**.
2. Let Gradle sync.
3. Run **sample-app** on a phone (same Wi‑Fi / hotspot as the Decision API).
4. Set:
   - **API base URL** → prefer `https://your-liveness-api.example.com` (proxies to PC `:8001`)
     - LAN fallback: `http://<your-pc-lan-ip>:8001` (same network as PC; cleartext OK in sample)
   - **API key** → `liveness-pilot-key` (from `local-api/.env`)
   - **Account number** → any test account number

### Option B — use the library module in the bank app

In the bank app `settings.gradle.kts`:

```kotlin
include(":liveness-sdk")
project(":liveness-sdk").projectDir = File("path/to/LivenessVerification/android-sdk/liveness-sdk")
```

In `app/build.gradle.kts`:

```kotlin
dependencies {
    implementation(project(":liveness-sdk"))
}
```

Manifest permissions (library already declares them; merge is automatic):

- `CAMERA`
- `INTERNET`

Cleartext HTTP for local UAT (if not on HTTPS):

```xml
android:usesCleartextTraffic="true"
```

### Kotlin usage

```kotlin
import com.jamuna.liveness.LivenessSdk
import com.jamuna.liveness.model.LivenessRequest
import com.jamuna.liveness.model.LivenessResult

private val livenessLauncher = registerForActivityResult(LivenessSdk.Contract()) { result ->
    when (result) {
        is LivenessResult.Approved -> { /* allow transaction */ }
        is LivenessResult.Rejected -> { /* show result.reasonCodes */ }
        is LivenessResult.Error -> { /* network / camera */ }
        is LivenessResult.Cancelled -> { /* user closed */ }
    }
}

fun startLiveness(accountNumber: String) {
    livenessLauncher.launch(
        LivenessRequest(
            baseUrl = "https://your-liveness-host",   // no trailing slash
            apiKey = "liveness-pilot-key",
            accountNumber = accountNumber,
            // transactionAmount = 150000.0,  // optional
            // transactionId = "TXN-1",
        )
    )
}
```

### Build an AAR to share

```bash
cd android-sdk
./gradlew :liveness-sdk:assembleRelease
# output: liveness-sdk/build/outputs/aar/liveness-sdk-release.aar
```

Host app:

```kotlin
implementation(files("libs/liveness-sdk-release.aar"))
// plus transitive deps: CameraX, OkHttp, AppCompat, Material (see liveness-sdk/build.gradle.kts)
```

Prefer module dependency over raw AAR when possible.

## UI notes (v1.0.1+)

- Camera preview uses **fitCenter** (no fill crop / digital zoom).
- Face oval is sized to ~86% width (device-adaptive) so forehead → chin fits.
- Guidance copy: “Position your full face in the oval”.

## Flow

1. User taps Start → SDK opens camera UI  
2. **Start Test** (reference)  
3. Live frames stream to `wss://…/ws/live` with `account_number` + session `nonce`  
4. API returns approve/reject → SDK returns `LivenessResult`

## Wire protocol (for reference)

`start` payload includes:

```json
{
  "type": "start",
  "account_number": "1234567890",
  "account_id": "1234567890",
  "photo_base64": "<jpeg>",
  "access_token": "<apiKey>",
  "sdk_version": "1.0.0",
  "platform": "android"
}
```

`account_id` is duplicated for older API builds. Prefer `account_number` going forward.

## Backend note

Account-number → customer face / KYC resolution will be added later on the Decision API. Until then, the number is used as the faceprint key (same as previous account id).

## Requirements

- minSdk 24  
- Decision API reachable from the phone  
- Valid API key (`AUTH_REQUIRED=true`)  
- Camera permission granted at runtime  
