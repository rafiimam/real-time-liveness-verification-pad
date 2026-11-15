package com.jamuna.liveness.ui

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.PointF
import android.graphics.RectF
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Size
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.jamuna.liveness.LivenessSdk
import com.jamuna.liveness.camera.FrameEncoder
import com.jamuna.liveness.databinding.ActivityLivenessBinding
import com.jamuna.liveness.model.LivenessRequest
import com.jamuna.liveness.model.LivenessResult
import com.jamuna.liveness.net.LiveWsClient
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class LivenessActivity : AppCompatActivity(), LiveWsClient.Listener {

    private lateinit var binding: ActivityLivenessBinding
    private lateinit var request: LivenessRequest

    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var wsClient: LiveWsClient? = null

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var photoBase64: String? = null
    private var phase: Phase = Phase.PHOTO
    private val streaming = AtomicBoolean(false)
    private val lastFrameSentMs = AtomicLong(0)
    private val frameIntervalMs = 125L // ~8 fps
    private var maxFinishRunnable: Runnable? = null

    private var lastGuidance: FaceGuideChecker.Guidance = FaceGuideChecker.Guidance.NO_FACE
    private var guidanceSuppressUntil = 0L

    private enum class Phase { PHOTO, STREAMING, DONE }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else finishWith(LivenessResult.Error("Camera permission denied"))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLivenessBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val req = readRequest()
        if (req == null || req.accountNumber.isBlank() || req.baseUrl.isBlank()) {
            finishWith(LivenessResult.Error("Invalid LivenessRequest (baseUrl / accountNumber required)"))
            return
        }
        request = req

        binding.txtAccount.text = "A/C ${request.accountNumber}"
        binding.btnClose.setOnClickListener { finishWith(LivenessResult.Cancelled) }
        binding.btnPrimary.setOnClickListener {
            when (phase) {
                Phase.PHOTO -> takeReferencePhoto()
                Phase.STREAMING -> Unit
                Phase.DONE -> finish()
            }
        }

        showGuide(null)
        setStatus("Position your full face in the oval")
        binding.btnPrimary.text = "Start Test"

        binding.panel.post {
            binding.ovalOverlay.bottomReservePx = binding.panel.height.toFloat()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    @Suppress("DEPRECATION")
    private fun readRequest(): LivenessRequest? {
        return if (Build.VERSION.SDK_INT >= 33) {
            intent.getParcelableExtra(LivenessSdk.EXTRA_REQUEST, LivenessRequest::class.java)
        } else {
            intent.getParcelableExtra(LivenessSdk.EXTRA_REQUEST)
        }
    }

    // =============== Camera ===============

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider
            bindUseCases(provider)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindUseCases(provider: ProcessCameraProvider) {
        provider.unbindAll()
        val preview = Preview.Builder().build().also {
            it.surfaceProvider = binding.previewView.surfaceProvider
        }
        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
        imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(cameraExecutor) { image ->
                    processFrame(image)
                }
            }

        val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
        try {
            provider.bindToLifecycle(this, cameraSelector, preview, imageCapture, imageAnalysis)
        } catch (_: Exception) {
            try {
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture, imageAnalysis
                )
            } catch (e2: Exception) {
                finishWith(LivenessResult.Error("Cannot open camera: ${e2.message}"))
            }
        }
    }

    // =============== Frame processing (guidance + streaming) ===============

    private fun processFrame(image: ImageProxy) {
        val now = System.currentTimeMillis()

        // Face guidance (runs during PHOTO and STREAMING phases)
        if (phase == Phase.PHOTO || phase == Phase.STREAMING) {
            val faceBox = detectFaceBox(image)
            val ovalBounds = binding.ovalOverlay.ovalBounds
            val vw = binding.ovalOverlay.width
            val vh = binding.ovalOverlay.height
            if (vw > 0 && vh > 0) {
                val guidance = FaceGuideChecker.check(faceBox, ovalBounds, vw, vh)
                if (guidance != lastGuidance && now > guidanceSuppressUntil) {
                    lastGuidance = guidance
                    guidanceSuppressUntil = now + 600
                    mainHandler.post { updateGuidanceUi(guidance) }
                }
            }
        }

        // Live frame streaming
        if (streaming.get() && phase == Phase.STREAMING) {
            val last = lastFrameSentMs.get()
            if (now - last >= frameIntervalMs && lastFrameSentMs.compareAndSet(last, now)) {
                val b64 = FrameEncoder.imageProxyToJpegBase64(image)
                image.close()
                if (!b64.isNullOrBlank()) {
                    wsClient?.sendFrame(b64, now)
                }
                return
            }
        }
        image.close()
    }

    private fun detectFaceBox(image: ImageProxy): FaceGuideChecker.FaceBox? {
        // Heuristic face-box: look at luminance center-of-mass in the upper portion.
        // For production with ML Kit, replace this with FaceDetector.
        val w = image.width
        val h = image.height
        if (w == 0 || h == 0) return null

        val plane = image.planes.getOrNull(0) ?: return null
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        if (buffer.remaining() < h * rowStride) return null

        // Sample a grid to find brightest region (proxy for face presence/position)
        val gridCols = 8
        val gridRows = 10
        val cellW = w / gridCols
        val cellH = h / gridRows
        var totalLum = 0L
        var sumX = 0L
        var sumY = 0L
        var brightCells = 0

        for (r in 0 until gridRows) {
            for (c in 0 until gridCols) {
                val px = c * cellW + cellW / 2
                val py = r * cellH + cellH / 2
                val idx = py * rowStride + px
                if (idx >= buffer.limit()) continue
                val lum = (buffer.get(idx).toInt() and 0xFF).toLong()
                if (lum > 90) {
                    totalLum += lum
                    sumX += lum * px
                    sumY += lum * py
                    brightCells++
                }
            }
        }

        if (brightCells < 5 || totalLum < 500) return null

        val cx = (sumX.toFloat() / totalLum) / w
        val cy = (sumY.toFloat() / totalLum) / h
        val approxFaceW = (brightCells.toFloat() / (gridCols * gridRows)) * 1.8f
        val approxFaceH = approxFaceW * 1.3f

        // Mirror for front camera
        val mirroredCx = 1f - cx

        return FaceGuideChecker.FaceBox(
            centerX = mirroredCx,
            centerY = cy,
            width = approxFaceW.coerceIn(0.05f, 0.95f),
            height = approxFaceH.coerceIn(0.05f, 0.95f),
        )
    }

    private fun updateGuidanceUi(guidance: FaceGuideChecker.Guidance) {
        val msg = FaceGuideChecker.guidanceMessage(guidance)
        if (msg != null) {
            showGuide(msg)
            binding.ovalOverlay.borderState = FaceOvalOverlayView.BorderState.WARNING
        } else {
            showGuide(null)
            binding.ovalOverlay.borderState =
                if (phase == Phase.STREAMING) FaceOvalOverlayView.BorderState.ACTIVE
                else FaceOvalOverlayView.BorderState.IDLE
        }
    }

    // =============== Photo & stream ===============

    private fun takeReferencePhoto() {
        if (lastGuidance != FaceGuideChecker.Guidance.OK) {
            setStatus("Align your face first")
            return
        }
        val capture = imageCapture ?: return
        binding.btnPrimary.isEnabled = false
        setStatus("Capturing…")
        capture.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val b64 = FrameEncoder.imageProxyToJpegBase64(image, quality = 85, maxWidth = 960)
                image.close()
                mainHandler.post {
                    if (b64.isNullOrBlank()) {
                        binding.btnPrimary.isEnabled = true
                        setStatus("Capture failed — try again")
                        return@post
                    }
                    photoBase64 = b64
                    beginLiveStream(b64)
                }
            }

            override fun onError(exception: ImageCaptureException) {
                mainHandler.post {
                    binding.btnPrimary.isEnabled = true
                    setStatus("Capture error: ${exception.message}")
                }
            }
        })
    }

    private fun beginLiveStream(photoB64: String) {
        phase = Phase.STREAMING
        binding.btnPrimary.visibility = View.GONE
        binding.progress.visibility = View.VISIBLE
        binding.progress.progress = 5
        binding.ovalOverlay.borderState = FaceOvalOverlayView.BorderState.ACTIVE
        showGuide(null)
        setStatus("Connecting…")

        wsClient?.closeQuietly()
        val client = LiveWsClient(
            baseUrl = request.baseUrl,
            apiKey = request.apiKey,
            listener = this,
        )
        wsClient = client
        client.connectAndStart(
            accountNumber = request.accountNumber.trim(),
            photoJpegBase64 = photoB64,
            transactionAmount = request.transactionAmount,
            transactionId = request.transactionId,
        )
    }

    // =============== WS callbacks ===============

    override fun onReady(sessionId: String, nonce: String, earlySeconds: Double, maxSeconds: Double) {
        mainHandler.post {
            setStatus("Hold still…")
            streaming.set(true)
            maxFinishRunnable?.let { mainHandler.removeCallbacks(it) }
            maxFinishRunnable = Runnable {
                if (phase == Phase.STREAMING) {
                    setStatus("Wrapping up…")
                    wsClient?.sendEnd()
                }
            }
            mainHandler.postDelayed(maxFinishRunnable!!, (maxSeconds * 1000).toLong().coerceAtLeast(4000))
        }
    }

    override fun onProgress(frames: Int, canFinish: Boolean, elapsedMs: Long) {
        mainHandler.post {
            val pct = if (canFinish) 92 else (5 + (frames.coerceAtMost(40) * 2)).coerceAtMost(90)
            binding.progress.progress = pct
            setStatus(if (canFinish) "Almost done…" else "Recording · $frames")
        }
    }

    override fun onStatus(message: String) {
        mainHandler.post {
            setStatus(message)
            binding.progress.progress = 96
        }
    }

    override fun onResult(json: JSONObject) {
        mainHandler.post {
            streaming.set(false)
            phase = Phase.DONE
            binding.progress.progress = 100
            clearTimers()
            finishWith(parseResult(json))
        }
    }

    override fun onError(message: String) {
        mainHandler.post {
            streaming.set(false)
            phase = Phase.DONE
            clearTimers()
            finishWith(LivenessResult.Error(message))
        }
    }

    // =============== Helpers ===============

    private fun parseResult(json: JSONObject): LivenessResult {
        val approved = json.optBoolean("is_approved", false)
        val sessionId = json.optString("session_id").ifBlank { null }
        val faceMatch = if (json.has("face_match")) json.optBoolean("face_match") else null
        val liveness = if (json.has("liveness")) json.optBoolean("liveness") else null
        val nidPhotoMatch = if (json.has("nid_photo_match")) json.optBoolean("nid_photo_match") else null
        val accountPhotoMatch = if (json.has("account_photo_match")) json.optBoolean("account_photo_match") else null
        val nidVerified = if (json.has("nid_verified")) json.optBoolean("nid_verified") else null
        val customerName = json.optString("customer_name").ifBlank { null }
        val faceMatchProvider = json.optString("face_match_provider").ifBlank { null }
        val cbsFacesDetected = if (json.has("cbs_faces_detected") && !json.isNull("cbs_faces_detected")) {
            json.optInt("cbs_faces_detected")
        } else null
        val reasons = mutableListOf<String>()
        json.optJSONArray("reason_codes")?.let { arr ->
            for (i in 0 until arr.length()) reasons += arr.optString(i)
        }
        val trust = json.optString("capture_trust").ifBlank { null }
        val raw = json.toString()
        return if (approved) {
            LivenessResult.Approved(
                sessionId, faceMatch, liveness, reasons, trust, raw,
                nidPhotoMatch, accountPhotoMatch, nidVerified, customerName,
                faceMatchProvider, cbsFacesDetected,
            )
        } else {
            LivenessResult.Rejected(
                sessionId, faceMatch, liveness, reasons, trust,
                message = reasons.joinToString(", ").ifBlank { "Rejected" },
                rawJson = raw,
                nidPhotoMatch = nidPhotoMatch,
                accountPhotoMatch = accountPhotoMatch,
                nidVerified = nidVerified,
                customerName = customerName,
                faceMatchProvider = faceMatchProvider,
                cbsFacesDetected = cbsFacesDetected,
            )
        }
    }

    private fun finishWith(result: LivenessResult) {
        streaming.set(false)
        clearTimers()
        wsClient?.closeQuietly()
        wsClient = null
        val data = Intent().putExtra(LivenessSdk.EXTRA_RESULT, result)
        val code = when (result) {
            is LivenessResult.Cancelled -> Activity.RESULT_CANCELED
            else -> Activity.RESULT_OK
        }
        setResult(code, data)
        finish()
    }

    private fun clearTimers() {
        maxFinishRunnable?.let { mainHandler.removeCallbacks(it) }
        maxFinishRunnable = null
    }

    private fun showGuide(text: String?) {
        binding.txtGuide.apply {
            if (text.isNullOrBlank()) {
                visibility = View.GONE
                this.text = ""
            } else {
                visibility = View.VISIBLE
                this.text = text
            }
        }
    }

    private fun setStatus(text: String) {
        binding.txtStatus.text = text
    }

    override fun onDestroy() {
        clearTimers()
        streaming.set(false)
        wsClient?.closeQuietly()
        cameraExecutor.shutdown()
        cameraProvider?.unbindAll()
        super.onDestroy()
    }
}
