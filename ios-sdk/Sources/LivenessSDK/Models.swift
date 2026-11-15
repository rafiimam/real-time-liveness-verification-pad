import Foundation

/// Request to start a liveness session (mirrors Android `LivenessRequest`).
public struct LivenessRequest: Sendable {
    /// Decision API origin, e.g. `https://host` (no trailing slash).
    public var baseUrl: String
    /// Bearer / API key configured on the Decision API.
    public var apiKey: String
    /// Customer account number — backend resolves NID + customer photo.
    public var accountNumber: String
    public var transactionAmount: Double?
    public var transactionId: String?

    public init(
        baseUrl: String,
        apiKey: String,
        accountNumber: String,
        transactionAmount: Double? = nil,
        transactionId: String? = nil
    ) {
        self.baseUrl = baseUrl
        self.apiKey = apiKey
        self.accountNumber = accountNumber
        self.transactionAmount = transactionAmount
        self.transactionId = transactionId
    }
}

/// Result returned to the host app (mirrors Android `LivenessResult`).
public enum LivenessResult: Sendable {
    case approved(
        sessionId: String?,
        faceMatch: Bool?,
        liveness: Bool?,
        reasonCodes: [String],
        captureTrust: String?,
        rawJson: String?,
        nidPhotoMatch: Bool?,
        accountPhotoMatch: Bool?,
        nidVerified: Bool?,
        customerName: String?,
        faceMatchProvider: String?,
        cbsFacesDetected: Int?
    )
    case rejected(
        sessionId: String?,
        faceMatch: Bool?,
        liveness: Bool?,
        reasonCodes: [String],
        captureTrust: String?,
        message: String?,
        rawJson: String?,
        nidPhotoMatch: Bool?,
        accountPhotoMatch: Bool?,
        nidVerified: Bool?,
        customerName: String?,
        faceMatchProvider: String?,
        cbsFacesDetected: Int?
    )
    case error(message: String)
    case cancelled
}
