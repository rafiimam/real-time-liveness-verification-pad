# iOS Liveness SDK (v1.0.0)

Native iOS client for Jamuna Liveness Verification. Same Decision API protocol and UI as the Android SDK (`WS /ws/live`).

Public identifier is **account number**. UI: large face oval, aspect-fit camera (no digital zoom crop), SecureCheck chrome.

> XCFramework binaries must be built on a **Mac with Xcode**. Sources live in this folder so you can produce `LivenessSDK.xcframework` or consume via Swift Package Manager.

## Give this to iOS developers

### Option A — Swift Package Manager (easiest)

In the bank app Xcode project:

1. **File → Add Package Dependencies…**
2. Add local package path: `…/LivenessVerification/ios-sdk`
3. Product: **LivenessSDK**

Or in `Package.swift` of an app:

```swift
.package(path: "../LivenessVerification/ios-sdk")
```

### Option B — XCFramework

On a Mac:

```bash
cd ios-sdk
brew install xcodegen   # once
chmod +x Scripts/build_xcframework.sh
./Scripts/build_xcframework.sh
# output: build/LivenessSDK.xcframework
```

In the bank app:

1. Drag `LivenessSDK.xcframework` into the project
2. Target → **General → Frameworks, Libraries, and Embedded Content** → Embed & Sign
3. Add `NSCameraUsageDescription` to Info.plist

### Info.plist

```xml
<key>NSCameraUsageDescription</key>
<string>Camera is required for face liveness verification.</string>
```

App Transport Security: for HTTP UAT hosts only, allow cleartext / exception domains as needed. Prefer HTTPS (`https://your-liveness-api.example.com`).

### Swift usage

```swift
import LivenessSDK
import UIKit

func startLiveness(from vc: UIViewController, accountNumber: String) {
    let request = LivenessRequest(
        baseUrl: "https://your-liveness-api.example.com",  // no trailing slash
        apiKey: "liveness-pilot-key",
        accountNumber: accountNumber
        // transactionAmount: 150000,
        // transactionId: "TXN-1"
    )
    LivenessSDK.present(from: vc, request: request) { result in
        switch result {
        case .approved(_, _, _, let codes, _, _, _, _, _, _, _, _):
            print("APPROVED", codes)
        case .rejected(_, _, _, let codes, _, let message, _, _, _, _, _, _):
            print("REJECTED", message ?? "", codes)
        case .error(let message):
            print("ERROR", message)
        case .cancelled:
            print("Cancelled")
        }
    }
}
```

## Flow

1. Host calls `LivenessSDK.present`
2. User aligns full face in the oval → **Start Test**
3. Reference JPEG + live frames stream to `wss://…/ws/live` with `account_number` + session `nonce`
4. API returns approve/reject → completion handler receives `LivenessResult`

## Wire protocol

`start` payload:

```json
{
  "type": "start",
  "account_number": "1234567890",
  "account_id": "1234567890",
  "photo_base64": "<jpeg>",
  "access_token": "<apiKey>",
  "sdk_version": "1.0.0",
  "platform": "ios"
}
```

## Requirements

- iOS 15+
- Decision API reachable from the device
- Valid API key
- Camera permission
- Xcode 15+ to build XCFramework
