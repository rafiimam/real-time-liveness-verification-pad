package com.jamuna.liveness.ui

import android.graphics.PointF
import android.graphics.RectF
import androidx.camera.core.ImageProxy

/**
 * Lightweight face-position checker using CameraX face detection proxy.
 * Uses the image center luminance pattern as a heuristic when ML Kit is not available.
 *
 * For production: integrate ML Kit Face Detection (`com.google.mlkit:face-detection`)
 * and replace the heuristic with proper landmarks.
 */
internal object FaceGuideChecker {

    enum class Guidance {
        OK,
        NO_FACE,
        TOO_FAR,
        TOO_CLOSE,
        OFF_CENTER_LEFT,
        OFF_CENTER_RIGHT,
        OFF_CENTER_UP,
        OFF_CENTER_DOWN,
        FACE_NOT_STRAIGHT,
    }

    data class FaceBox(
        val centerX: Float, // 0..1 normalized
        val centerY: Float,
        val width: Float,   // fraction of image width
        val height: Float,
    )

    fun check(faceBox: FaceBox?, ovalBounds: RectF, viewWidth: Int, viewHeight: Int): Guidance {
        if (faceBox == null) return Guidance.NO_FACE

        val ovalCx = ovalBounds.centerX() / viewWidth
        val ovalCy = ovalBounds.centerY() / viewHeight
        val ovalW = ovalBounds.width() / viewWidth
        val ovalH = ovalBounds.height() / viewHeight

        val dx = faceBox.centerX - ovalCx
        val dy = faceBox.centerY - ovalCy
        val tolerance = 0.12f

        if (dx < -tolerance) return Guidance.OFF_CENTER_LEFT
        if (dx > tolerance) return Guidance.OFF_CENTER_RIGHT
        if (dy < -tolerance) return Guidance.OFF_CENTER_UP
        if (dy > tolerance) return Guidance.OFF_CENTER_DOWN

        if (faceBox.width < ovalW * 0.35f) return Guidance.TOO_FAR
        // Allow face to nearly fill the larger oval before asking to step back
        if (faceBox.width > ovalW * 1.55f) return Guidance.TOO_CLOSE

        val aspect = faceBox.height / (faceBox.width + 0.001f)
        if (aspect < 0.85f || aspect > 2.0f) return Guidance.FACE_NOT_STRAIGHT

        return Guidance.OK
    }

    fun guidanceMessage(g: Guidance): String? = when (g) {
        Guidance.OK -> null
        Guidance.NO_FACE -> "Position your full face in the oval"
        Guidance.TOO_FAR -> "Move closer"
        Guidance.TOO_CLOSE -> "Move back a little"
        Guidance.OFF_CENTER_LEFT -> "Move your face to the right"
        Guidance.OFF_CENTER_RIGHT -> "Move your face to the left"
        Guidance.OFF_CENTER_UP -> "Move your face down"
        Guidance.OFF_CENTER_DOWN -> "Move your face up"
        Guidance.FACE_NOT_STRAIGHT -> "Look straight at the camera"
    }
}
