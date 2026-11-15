import AVFoundation
import UIKit

/// Full-screen liveness UI matching Android SDK (oval overlay + Start Test + WS stream).
final class LivenessViewController: UIViewController, LiveWsClient.Listener {
    private enum Phase { case photo, streaming, done }

    private let request: LivenessRequest
    private let completion: (LivenessResult) -> Void

    private let previewView = PreviewContainerView()
    private let ovalOverlay = FaceOvalOverlayView()
    private let brandLabel = UILabel()
    private let accountLabel = UILabel()
    private let closeButton = UIButton(type: .system)
    private let guideLabel = UILabel()
    private let statusLabel = UILabel()
    private let progressView = UIProgressView(progressViewStyle: .bar)
    private let primaryButton = UIButton(type: .system)
    private let panel = UIStackView()

    private var captureSession: AVCaptureSession?
    private var photoOutput: AVCapturePhotoOutput?
    private var videoOutput: AVCaptureVideoDataOutput?
    private let sessionQueue = DispatchQueue(label: "com.jamuna.liveness.camera")
    private let frameQueue = DispatchQueue(label: "com.jamuna.liveness.frames")

    private var wsClient: LiveWsClient?
    private var photoBase64: String?
    private var phase: Phase = .photo
    private var streaming = false
    private var lastFrameSentMs: Int64 = 0
    private let frameIntervalMs: Int64 = 125
    private var maxFinishWorkItem: DispatchWorkItem?

    private var lastGuidance: FaceGuideChecker.Guidance = .noFace
    private var guidanceSuppressUntil: TimeInterval = 0
    private var latestSampleBuffer: CMSampleBuffer?
    private var finished = false

    init(request: LivenessRequest, completion: @escaping (LivenessResult) -> Void) {
        self.request = request
        self.completion = completion
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = UIColor(red: 15 / 255, green: 20 / 255, blue: 25 / 255, alpha: 1)
        buildUI()
        accountLabel.text = "A/C \(request.accountNumber)"
        setStatus("Position your full face in the oval")
        requestCameraAccess()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        ovalOverlay.bottomReserve = panel.bounds.height + view.safeAreaInsets.bottom
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        stopCamera()
    }

    // MARK: - UI

    private func buildUI() {
        previewView.translatesAutoresizingMaskIntoConstraints = false
        ovalOverlay.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(previewView)
        view.addSubview(ovalOverlay)

        brandLabel.text = "SecureCheck"
        brandLabel.textColor = UIColor(red: 232 / 255, green: 238 / 255, blue: 246 / 255, alpha: 1)
        brandLabel.font = .systemFont(ofSize: 18, weight: .bold)

        accountLabel.textColor = UIColor(red: 139 / 255, green: 155 / 255, blue: 180 / 255, alpha: 1)
        accountLabel.font = .systemFont(ofSize: 12)

        closeButton.setTitle("✕", for: .normal)
        closeButton.setTitleColor(UIColor(red: 232 / 255, green: 238 / 255, blue: 246 / 255, alpha: 1), for: .normal)
        closeButton.titleLabel?.font = .systemFont(ofSize: 20, weight: .medium)
        closeButton.addTarget(self, action: #selector(closeTapped), for: .touchUpInside)

        let topBar = UIStackView(arrangedSubviews: [brandLabel, accountLabel, closeButton])
        topBar.axis = .horizontal
        topBar.alignment = .center
        topBar.spacing = 12
        brandLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        topBar.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(topBar)

        guideLabel.textColor = UIColor(red: 245 / 255, green: 158 / 255, blue: 11 / 255, alpha: 1)
        guideLabel.font = .systemFont(ofSize: 16, weight: .bold)
        guideLabel.textAlignment = .center
        guideLabel.numberOfLines = 2
        guideLabel.isHidden = true
        guideLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(guideLabel)

        statusLabel.textColor = UIColor(red: 232 / 255, green: 238 / 255, blue: 246 / 255, alpha: 1)
        statusLabel.font = .systemFont(ofSize: 15)
        statusLabel.textAlignment = .center
        statusLabel.numberOfLines = 2

        progressView.progressTintColor = UIColor(red: 20 / 255, green: 184 / 255, blue: 166 / 255, alpha: 1)
        progressView.trackTintColor = UIColor.white.withAlphaComponent(0.15)
        progressView.isHidden = true
        progressView.heightAnchor.constraint(equalToConstant: 6).isActive = true

        primaryButton.setTitle("Start Test", for: .normal)
        primaryButton.setTitleColor(.white, for: .normal)
        primaryButton.titleLabel?.font = .systemFont(ofSize: 16, weight: .semibold)
        primaryButton.backgroundColor = UIColor(red: 20 / 255, green: 184 / 255, blue: 166 / 255, alpha: 1)
        primaryButton.layer.cornerRadius = 28
        primaryButton.heightAnchor.constraint(equalToConstant: 56).isActive = true
        primaryButton.addTarget(self, action: #selector(primaryTapped), for: .touchUpInside)

        panel.axis = .vertical
        panel.spacing = 14
        panel.translatesAutoresizingMaskIntoConstraints = false
        panel.isLayoutMarginsRelativeArrangement = true
        panel.layoutMargins = UIEdgeInsets(top: 16, left: 24, bottom: 24, right: 24)
        panel.backgroundColor = UIColor(red: 15 / 255, green: 20 / 255, blue: 25 / 255, alpha: 0.8)
        panel.addArrangedSubview(statusLabel)
        panel.addArrangedSubview(progressView)
        panel.addArrangedSubview(primaryButton)
        view.addSubview(panel)

        NSLayoutConstraint.activate([
            previewView.topAnchor.constraint(equalTo: view.topAnchor),
            previewView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            previewView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            previewView.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            ovalOverlay.topAnchor.constraint(equalTo: view.topAnchor),
            ovalOverlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            ovalOverlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            ovalOverlay.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            topBar.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 8),
            topBar.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            topBar.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),

            guideLabel.topAnchor.constraint(equalTo: topBar.bottomAnchor, constant: 24),
            guideLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 32),
            guideLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -32),

            panel.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            panel.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            panel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
        ])
    }

    // MARK: - Camera

    private func requestCameraAccess() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            startCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted { self?.startCamera() }
                    else { self?.finishWith(.error(message: "Camera permission denied")) }
                }
            }
        default:
            finishWith(.error(message: "Camera permission denied"))
        }
    }

    private func startCamera() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            let session = AVCaptureSession()
            session.beginConfiguration()
            session.sessionPreset = .high

            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
                ?? AVCaptureDevice.default(for: .video)
            guard let device,
                  let input = try? AVCaptureDeviceInput(device: device),
                  session.canAddInput(input) else {
                DispatchQueue.main.async {
                    self.finishWith(.error(message: "Cannot open camera"))
                }
                return
            }
            session.addInput(input)

            let photo = AVCapturePhotoOutput()
            if session.canAddOutput(photo) { session.addOutput(photo) }
            self.photoOutput = photo

            let video = AVCaptureVideoDataOutput()
            video.alwaysDiscardsLateVideoFrames = true
            video.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            video.setSampleBufferDelegate(self, queue: self.frameQueue)
            if session.canAddOutput(video) { session.addOutput(video) }
            if let conn = video.connection(with: .video) {
                if conn.isVideoMirroringSupported {
                    conn.isVideoMirrored = true
                }
                if conn.isVideoOrientationSupported {
                    conn.videoOrientation = .portrait
                }
            }
            self.videoOutput = video

            session.commitConfiguration()
            self.captureSession = session

            DispatchQueue.main.async {
                self.previewView.attach(session: session)
            }
            session.startRunning()
        }
    }

    private func stopCamera() {
        sessionQueue.async { [weak self] in
            self?.captureSession?.stopRunning()
        }
    }

    // MARK: - Actions

    @objc private func closeTapped() {
        finishWith(.cancelled)
    }

    @objc private func primaryTapped() {
        switch phase {
        case .photo: takeReferencePhoto()
        case .streaming: break
        case .done: dismiss(animated: true)
        }
    }

    private func takeReferencePhoto() {
        guard lastGuidance == .ok else {
            setStatus("Align your face first")
            return
        }
        primaryButton.isEnabled = false
        setStatus("Capturing…")

        // Prefer latest preview frame so capture matches what user sees in the oval.
        if let buffer = latestSampleBuffer,
           let b64 = FrameEncoder.jpegBase64(from: buffer, quality: 0.85, maxWidth: 960) {
            photoBase64 = b64
            beginLiveStream(photoB64: b64)
            return
        }

        guard let photoOutput else {
            primaryButton.isEnabled = true
            setStatus("Capture failed — try again")
            return
        }
        let settings = AVCapturePhotoSettings()
        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    private func beginLiveStream(photoB64: String) {
        phase = .streaming
        primaryButton.isHidden = true
        progressView.isHidden = false
        progressView.progress = 0.05
        ovalOverlay.borderState = .active
        showGuide(nil)
        setStatus("Connecting…")

        wsClient?.closeQuietly()
        let client = LiveWsClient(baseUrl: request.baseUrl, apiKey: request.apiKey, listener: self)
        wsClient = client
        client.connectAndStart(
            accountNumber: request.accountNumber,
            photoJpegBase64: photoB64,
            transactionAmount: request.transactionAmount,
            transactionId: request.transactionId
        )
    }

    // MARK: - WS Listener

    func onReady(sessionId: String, nonce: String, earlySeconds: Double, maxSeconds: Double) {
        DispatchQueue.main.async {
            self.setStatus("Hold still…")
            self.streaming = true
            self.maxFinishWorkItem?.cancel()
            let work = DispatchWorkItem { [weak self] in
                guard let self, self.phase == .streaming else { return }
                self.setStatus("Wrapping up…")
                self.wsClient?.sendEnd()
            }
            self.maxFinishWorkItem = work
            DispatchQueue.main.asyncAfter(deadline: .now() + max(4, maxSeconds), execute: work)
        }
    }

    func onProgress(frames: Int, canFinish: Bool, elapsedMs: Int64) {
        DispatchQueue.main.async {
            let pct: Float = canFinish ? 0.92 : min(0.90, 0.05 + Float(min(frames, 40)) * 0.02)
            self.progressView.progress = pct
            self.setStatus(canFinish ? "Almost done…" : "Recording · \(frames)")
        }
    }

    func onStatus(message: String) {
        DispatchQueue.main.async {
            self.setStatus(message)
            self.progressView.progress = 0.96
        }
    }

    func onResult(json: [String: Any]) {
        DispatchQueue.main.async {
            self.streaming = false
            self.phase = .done
            self.progressView.progress = 1
            self.clearTimers()
            self.finishWith(self.parseResult(json))
        }
    }

    func onError(message: String) {
        DispatchQueue.main.async {
            self.streaming = false
            self.phase = .done
            self.clearTimers()
            self.finishWith(.error(message: message))
        }
    }

    // MARK: - Helpers

    private func parseResult(_ json: [String: Any]) -> LivenessResult {
        let approved = json["is_approved"] as? Bool ?? false
        let sessionId = json["session_id"] as? String
        let faceMatch = json["face_match"] as? Bool
        let liveness = json["liveness"] as? Bool
        let nidPhotoMatch = json["nid_photo_match"] as? Bool
        let accountPhotoMatch = json["account_photo_match"] as? Bool
        let nidVerified = json["nid_verified"] as? Bool
        let customerName = json["customer_name"] as? String
        let faceMatchProvider = json["face_match_provider"] as? String
        let cbsFacesDetected = json["cbs_faces_detected"] as? Int
        let reasons = (json["reason_codes"] as? [String]) ?? []
        let trust = json["capture_trust"] as? String
        let raw: String?
        if let data = try? JSONSerialization.data(withJSONObject: json),
           let s = String(data: data, encoding: .utf8) {
            raw = s
        } else {
            raw = nil
        }

        if approved {
            return .approved(
                sessionId: sessionId,
                faceMatch: faceMatch,
                liveness: liveness,
                reasonCodes: reasons,
                captureTrust: trust,
                rawJson: raw,
                nidPhotoMatch: nidPhotoMatch,
                accountPhotoMatch: accountPhotoMatch,
                nidVerified: nidVerified,
                customerName: customerName,
                faceMatchProvider: faceMatchProvider,
                cbsFacesDetected: cbsFacesDetected
            )
        }
        let message = reasons.joined(separator: ", ").isEmpty ? "Rejected" : reasons.joined(separator: ", ")
        return .rejected(
            sessionId: sessionId,
            faceMatch: faceMatch,
            liveness: liveness,
            reasonCodes: reasons,
            captureTrust: trust,
            message: message,
            rawJson: raw,
            nidPhotoMatch: nidPhotoMatch,
            accountPhotoMatch: accountPhotoMatch,
            nidVerified: nidVerified,
            customerName: customerName,
            faceMatchProvider: faceMatchProvider,
            cbsFacesDetected: cbsFacesDetected
        )
    }

    private func finishWith(_ result: LivenessResult) {
        guard !finished else { return }
        finished = true
        streaming = false
        clearTimers()
        wsClient?.closeQuietly()
        wsClient = nil
        stopCamera()
        dismiss(animated: true) { [completion] in
            completion(result)
        }
    }

    private func clearTimers() {
        maxFinishWorkItem?.cancel()
        maxFinishWorkItem = nil
    }

    private func showGuide(_ text: String?) {
        if let text, !text.isEmpty {
            guideLabel.text = text
            guideLabel.isHidden = false
        } else {
            guideLabel.text = nil
            guideLabel.isHidden = true
        }
    }

    private func setStatus(_ text: String) {
        statusLabel.text = text
    }

    private func updateGuidanceUi(_ guidance: FaceGuideChecker.Guidance) {
        if let msg = FaceGuideChecker.message(guidance) {
            showGuide(msg)
            ovalOverlay.borderState = .warning
        } else {
            showGuide(nil)
            ovalOverlay.borderState = phase == .streaming ? .active : .idle
        }
    }

    /// Lightweight luminance heuristic (same idea as Android FaceGuideChecker).
    private func detectFaceBox(from sampleBuffer: CMSampleBuffer) -> FaceGuideChecker.FaceBox? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        guard w > 0, h > 0,
              let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let ptr = base.assumingMemoryBound(to: UInt8.self)

        let gridCols = 8
        let gridRows = 10
        let cellW = w / gridCols
        let cellH = h / gridRows
        var totalLum: Int64 = 0
        var sumX: Int64 = 0
        var sumY: Int64 = 0
        var brightCells = 0

        for r in 0..<gridRows {
            for c in 0..<gridCols {
                let px = c * cellW + cellW / 2
                let py = r * cellH + cellH / 2
                let offset = py * bytesPerRow + px * 4
                guard offset + 2 < bytesPerRow * h else { continue }
                // BGRA
                let b = Int64(ptr[offset])
                let g = Int64(ptr[offset + 1])
                let rCh = Int64(ptr[offset + 2])
                let lum = (rCh + g + b) / 3
                if lum > 90 {
                    totalLum += lum
                    sumX += lum * Int64(px)
                    sumY += lum * Int64(py)
                    brightCells += 1
                }
            }
        }

        guard brightCells >= 5, totalLum >= 500 else { return nil }

        let cx = CGFloat(sumX) / CGFloat(totalLum) / CGFloat(w)
        let cy = CGFloat(sumY) / CGFloat(totalLum) / CGFloat(h)
        let approxFaceW = (CGFloat(brightCells) / CGFloat(gridCols * gridRows)) * 1.8
        let approxFaceH = approxFaceW * 1.3

        return FaceGuideChecker.FaceBox(
            centerX: cx,
            centerY: cy,
            width: min(max(approxFaceW, 0.05), 0.95),
            height: min(max(approxFaceH, 0.05), 0.95)
        )
    }
}

// MARK: - Video frames

extension LivenessViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        latestSampleBuffer = sampleBuffer
        let nowMs = Int64(Date().timeIntervalSince1970 * 1000)
        let now = Date().timeIntervalSince1970

        if phase == .photo || phase == .streaming {
            let faceBox = detectFaceBox(from: sampleBuffer)
            let oval = ovalOverlay.ovalBounds
            let vw = ovalOverlay.bounds.width
            let vh = ovalOverlay.bounds.height
            if vw > 0, vh > 0 {
                let guidance = FaceGuideChecker.check(
                    faceBox: faceBox,
                    ovalBounds: oval,
                    viewWidth: vw,
                    viewHeight: vh
                )
                if guidance != lastGuidance, now > guidanceSuppressUntil {
                    lastGuidance = guidance
                    guidanceSuppressUntil = now + 0.6
                    DispatchQueue.main.async { self.updateGuidanceUi(guidance) }
                }
            }
        }

        if streaming, phase == .streaming, nowMs - lastFrameSentMs >= frameIntervalMs {
            lastFrameSentMs = nowMs
            if let b64 = FrameEncoder.jpegBase64(from: sampleBuffer) {
                wsClient?.sendFrame(jpegBase64: b64, ts: nowMs)
            }
        }
    }
}

// MARK: - Photo capture fallback

extension LivenessViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        if let error {
            DispatchQueue.main.async {
                self.primaryButton.isEnabled = true
                self.setStatus("Capture error: \(error.localizedDescription)")
            }
            return
        }
        guard let data = photo.fileDataRepresentation(),
              let image = UIImage(data: data),
              let b64 = FrameEncoder.jpegBase64(from: image, quality: 0.85, maxWidth: 960) else {
            DispatchQueue.main.async {
                self.primaryButton.isEnabled = true
                self.setStatus("Capture failed — try again")
            }
            return
        }
        DispatchQueue.main.async {
            self.photoBase64 = b64
            self.beginLiveStream(photoB64: b64)
        }
    }
}

// MARK: - Preview (aspect-fit, no digital zoom crop)

final class PreviewContainerView: UIView {
    private var previewLayer: AVCaptureVideoPreviewLayer?

    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    private var videoLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    func attach(session: AVCaptureSession) {
        videoLayer.session = session
        videoLayer.videoGravity = .resizeAspect // fitCenter equivalent — no crop zoom
        previewLayer = videoLayer
    }
}
