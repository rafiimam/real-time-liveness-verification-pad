#!/usr/bin/env bash
# Build LivenessSDK.xcframework on macOS with Xcode 15+.
# Run from ios-sdk/:  ./Scripts/build_xcframework.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SCHEME="LivenessSDK"
OUT_DIR="$ROOT/build"
ARCHIVE_IOS="$OUT_DIR/ios.xcarchive"
ARCHIVE_SIM="$OUT_DIR/sim.xcarchive"
XCFRAMEWORK="$OUT_DIR/LivenessSDK.xcframework"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Generate an Xcode project from the Swift Package (requires Xcode).
if ! command -v xcodebuild >/dev/null; then
  echo "xcodebuild not found — run this script on a Mac with Xcode installed."
  exit 1
fi

# Resolve package into a workspace via swift package generate-xcodeproj is deprecated;
# use xcodebuild -scheme after opening Package.swift once, or create framework via:
swift package resolve

# Create a temporary Xcode project wrapping the package target as a dynamic framework.
PROJECT="$OUT_DIR/LivenessSDK.xcodeproj"
rm -rf "$PROJECT"

cat > "$OUT_DIR/project.yml" <<'EOF'
name: LivenessSDK
options:
  bundleIdPrefix: com.jamuna
  deploymentTarget:
    iOS: "15.0"
settings:
  SWIFT_VERSION: "5.9"
  BUILD_LIBRARY_FOR_DISTRIBUTION: YES
  SKIP_INSTALL: NO
targets:
  LivenessSDK:
    type: framework
    platform: iOS
    sources:
      - path: ../Sources/LivenessSDK
    settings:
      PRODUCT_BUNDLE_IDENTIFIER: com.jamuna.liveness
      PRODUCT_NAME: LivenessSDK
      DEFINES_MODULE: YES
      GENERATE_INFOPLIST_FILE: YES
      MARKETING_VERSION: "1.0.0"
      CURRENT_PROJECT_VERSION: 1
EOF

if command -v xcodegen >/dev/null; then
  (cd "$OUT_DIR" && xcodegen generate --spec project.yml)
else
  # Fallback: use swift build + lipo is not enough for XCFramework of UIKit code.
  # Instruct user to install xcodegen OR open Package.swift in Xcode and archive.
  echo "xcodegen not found."
  echo "Install: brew install xcodegen"
  echo "Or open Package.swift in Xcode, create an Archive for Any iOS Device + Simulator, then:"
  echo "  xcodebuild -create-xcframework \\"
  echo "    -framework ios.xcarchive/Products/Library/Frameworks/LivenessSDK.framework \\"
  echo "    -framework sim.xcarchive/Products/Library/Frameworks/LivenessSDK.framework \\"
  echo "    -output LivenessSDK.xcframework"
  exit 1
fi

xcodebuild archive \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -destination "generic/platform=iOS" \
  -archivePath "$ARCHIVE_IOS" \
  SKIP_INSTALL=NO \
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
  ONLY_ACTIVE_ARCH=NO

xcodebuild archive \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -destination "generic/platform=iOS Simulator" \
  -archivePath "$ARCHIVE_SIM" \
  SKIP_INSTALL=NO \
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
  ONLY_ACTIVE_ARCH=NO

rm -rf "$XCFRAMEWORK"
xcodebuild -create-xcframework \
  -framework "$ARCHIVE_IOS/Products/Library/Frameworks/LivenessSDK.framework" \
  -framework "$ARCHIVE_SIM/Products/Library/Frameworks/LivenessSDK.framework" \
  -output "$XCFRAMEWORK"

echo ""
echo "Built: $XCFRAMEWORK"
echo "Drag into the bank app or: File → Add Package / Embed & Sign the XCFramework."
