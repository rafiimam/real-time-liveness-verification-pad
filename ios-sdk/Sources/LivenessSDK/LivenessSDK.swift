import UIKit

/// Entry point for host iOS apps (mirrors Android `LivenessSdk`).
public enum LivenessSDK {
    public static let version = "1.0.0"

    /// Presents the liveness camera UI modally and returns the result on the main queue.
    @MainActor
    public static func present(
        from presenter: UIViewController,
        request: LivenessRequest,
        completion: @escaping (LivenessResult) -> Void
    ) {
        let trimmedBase = request.baseUrl.trimmingCharacters(in: .whitespacesAndNewlines)
        let account = request.accountNumber.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedBase.isEmpty, !account.isEmpty else {
            completion(.error(message: "Invalid LivenessRequest (baseUrl / accountNumber required)"))
            return
        }

        var req = request
        req.baseUrl = trimmedBase.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        req.accountNumber = account

        let vc = LivenessViewController(request: req) { result in
            completion(result)
        }
        vc.modalPresentationStyle = .fullScreen
        presenter.present(vc, animated: true)
    }
}
