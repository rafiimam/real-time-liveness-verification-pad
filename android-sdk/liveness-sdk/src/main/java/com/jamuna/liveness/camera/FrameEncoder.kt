package com.jamuna.liveness.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

internal object FrameEncoder {
    /** JPEG bytes → raw base64 (no data-URL prefix). */
    fun bitmapToJpegBase64(bitmap: Bitmap, quality: Int = 70, maxWidth: Int = 480): String {
        val scaled = scaleDown(bitmap, maxWidth)
        val out = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, quality, out)
        if (scaled !== bitmap) scaled.recycle()
        return android.util.Base64.encodeToString(out.toByteArray(), android.util.Base64.NO_WRAP)
    }

    fun imageProxyToJpegBase64(image: ImageProxy, quality: Int = 68, maxWidth: Int = 480): String? {
        val bitmap = imageProxyToBitmap(image) ?: return null
        return try {
            bitmapToJpegBase64(bitmap, quality, maxWidth)
        } finally {
            bitmap.recycle()
        }
    }

    private fun scaleDown(src: Bitmap, maxWidth: Int): Bitmap {
        if (src.width <= maxWidth) return src
        val ratio = maxWidth.toFloat() / src.width
        val w = maxWidth
        val h = (src.height * ratio).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(src, w, h, true)
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        return try {
            when (image.format) {
                ImageFormat.JPEG -> {
                    val buffer = image.planes[0].buffer
                    val bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    var bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
                    val rotation = image.imageInfo.rotationDegrees
                    if (rotation != 0) {
                        val m = Matrix().apply { postRotate(rotation.toFloat()) }
                        val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
                        if (rotated !== bmp) bmp.recycle()
                        bmp = rotated
                    }
                    bmp
                }
                ImageFormat.YUV_420_888 -> yuv420ToBitmap(image)
                else -> null
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun yuv420ToBitmap(image: ImageProxy): Bitmap? {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        var bmp = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size()) ?: return null
        val rotation = image.imageInfo.rotationDegrees
        if (rotation != 0) {
            val m = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            if (rotated !== bmp) bmp.recycle()
            bmp = rotated
        }
        return bmp
    }
}
