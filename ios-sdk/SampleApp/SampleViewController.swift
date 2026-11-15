import UIKit
import LivenessSDK

/// Minimal host harness — open this folder as a package dependency from an Xcode app target,
/// or copy this view controller into a new iOS App project that depends on LivenessSDK.
final class SampleViewController: UIViewController {
    private let resultLabel = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground

        let button = UIButton(type: .system)
        button.setTitle("Start Liveness", for: .normal)
        button.titleLabel?.font = .systemFont(ofSize: 18, weight: .semibold)
        button.addTarget(self, action: #selector(start), for: .touchUpInside)
        button.translatesAutoresizingMaskIntoConstraints = false

        resultLabel.numberOfLines = 0
        resultLabel.textAlignment = .center
        resultLabel.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(button)
        view.addSubview(resultLabel)
        NSLayoutConstraint.activate([
            button.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            button.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: -40),
            resultLabel.topAnchor.constraint(equalTo: button.bottomAnchor, constant: 24),
            resultLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            resultLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),
        ])
    }

    @objc private func start() {
        let request = LivenessRequest(
            baseUrl: "https://your-liveness-api.example.com",
            apiKey: "liveness-pilot-key",
            accountNumber: "1234567890"
        )
        LivenessSDK.present(from: self, request: request) { [weak self] result in
            switch result {
            case .approved:
                self?.resultLabel.text = "APPROVED"
            case .rejected(_, _, _, let codes, _, let message, _, _, _, _, _, _):
                self?.resultLabel.text = "REJECTED\n\(message ?? "")\n\(codes)"
            case .error(let message):
                self?.resultLabel.text = "ERROR\n\(message)"
            case .cancelled:
                self?.resultLabel.text = "Cancelled"
            }
        }
    }
}
