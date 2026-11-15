import Foundation

/// Client for Decision API `WS /ws/live` protocol (mirrors Android `LiveWsClient`).
final class LiveWsClient: NSObject {
    protocol Listener: AnyObject {
        func onReady(sessionId: String, nonce: String, earlySeconds: Double, maxSeconds: Double)
        func onProgress(frames: Int, canFinish: Bool, elapsedMs: Int64)
        func onStatus(message: String)
        func onResult(json: [String: Any])
        func onError(message: String)
    }

    private let baseUrl: String
    private let apiKey: String
    private weak var listener: Listener?

    private var task: URLSessionWebSocketTask?
    private var session: URLSession?
    private var closed = false
    private var nonce: String?

    init(baseUrl: String, apiKey: String, listener: Listener) {
        self.baseUrl = baseUrl
        self.apiKey = apiKey
        self.listener = listener
        super.init()
    }

    func connectAndStart(
        accountNumber: String,
        photoJpegBase64: String,
        transactionAmount: Double? = nil,
        transactionId: String? = nil
    ) {
        closed = false
        nonce = nil

        guard let url = URL(string: Self.toWsUrl(baseUrl) + "/ws/live?access_token=\(apiKey)") else {
            listener?.onError(message: "Invalid WebSocket URL")
            return
        }

        var request = URLRequest(url: url)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        let urlSession = URLSession(configuration: config, delegate: nil, delegateQueue: nil)
        session = urlSession
        let ws = urlSession.webSocketTask(with: request)
        task = ws
        ws.resume()

        var start: [String: Any] = [
            "type": "start",
            "account_number": accountNumber,
            "account_id": accountNumber,
            "photo_base64": photoJpegBase64,
            "access_token": apiKey,
            "sdk_version": LivenessSDK.version,
            "platform": "ios",
        ]
        if let transactionAmount { start["transaction_amount"] = transactionAmount }
        if let transactionId, !transactionId.isEmpty { start["transaction_id"] = transactionId }

        sendJSON(start)
        receiveLoop()
    }

    func sendFrame(jpegBase64: String, ts: Int64 = Int64(Date().timeIntervalSince1970 * 1000)) {
        guard let nonce else { return }
        sendJSON([
            "type": "frame",
            "image_base64": jpegBase64,
            "ts": ts,
            "nonce": nonce,
        ])
    }

    func sendEnd() {
        guard let nonce else { return }
        sendJSON(["type": "end", "nonce": nonce])
    }

    func closeQuietly() {
        closed = true
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        session?.invalidateAndCancel()
        session = nil
    }

    private func sendJSON(_ dict: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let text = String(data: data, encoding: .utf8) else { return }
        task?.send(.string(text)) { [weak self] error in
            if let error, !(self?.closed ?? true) {
                self?.failOnce(error.localizedDescription)
            }
        }
    }

    private func receiveLoop() {
        task?.receive { [weak self] result in
            guard let self, !self.closed else { return }
            switch result {
            case .failure(let error):
                self.failOnce(error.localizedDescription)
            case .success(let message):
                self.handle(message)
                if !self.closed {
                    self.receiveLoop()
                }
            }
        }
    }

    private func handle(_ message: URLSessionWebSocketTask.Message) {
        let text: String?
        switch message {
        case .string(let s): text = s
        case .data(let d): text = String(data: d, encoding: .utf8)
        @unknown default: text = nil
        }
        guard let text,
              let data = text.data(using: .utf8),
              let msg = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }

        let type = msg["type"] as? String ?? ""
        switch type {
        case "ready":
            nonce = msg["nonce"] as? String
            listener?.onReady(
                sessionId: msg["session_id"] as? String ?? "",
                nonce: nonce ?? "",
                earlySeconds: (msg["early_seconds"] as? Double) ?? 4,
                maxSeconds: (msg["max_seconds"] as? Double) ?? 10
            )
        case "progress":
            listener?.onProgress(
                frames: msg["frames"] as? Int ?? 0,
                canFinish: msg["can_finish"] as? Bool ?? false,
                elapsedMs: (msg["elapsed_ms"] as? Int64) ?? Int64(msg["elapsed_ms"] as? Int ?? 0)
            )
        case "status":
            listener?.onStatus(message: msg["message"] as? String ?? "Checking…")
        case "result":
            closed = true
            listener?.onResult(json: msg)
            closeQuietly()
        case "error":
            failOnce(msg["message"] as? String ?? "Unknown error")
        case "pong":
            break
        default:
            break
        }
    }

    private func failOnce(_ message: String) {
        guard !closed else { return }
        closed = true
        listener?.onError(message: message)
        closeQuietly()
    }

    static func toWsUrl(_ httpBase: String) -> String {
        let trimmed = httpBase.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if trimmed.hasPrefix("https://") {
            return "wss://" + trimmed.dropFirst("https://".count)
        }
        if trimmed.hasPrefix("http://") {
            return "ws://" + trimmed.dropFirst("http://".count)
        }
        return trimmed
    }
}
