// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LivenessSDK",
    platforms: [
        .iOS(.v15),
    ],
    products: [
        .library(name: "LivenessSDK", targets: ["LivenessSDK"]),
    ],
    targets: [
        .target(
            name: "LivenessSDK",
            path: "Sources/LivenessSDK"
        ),
    ]
)
