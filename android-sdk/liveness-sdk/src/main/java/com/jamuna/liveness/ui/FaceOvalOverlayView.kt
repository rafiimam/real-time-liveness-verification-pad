package com.jamuna.liveness.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * Custom overlay that draws a translucent dark mask with a clear oval cutout.
 * Oval sizes flexibly to the device so a full face (forehead → chin) fits
 * without digital zoom cropping from the camera preview.
 */
class FaceOvalOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : View(context, attrs, defStyleAttr) {

    private val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#B30F1419")
        style = Paint.Style.FILL
    }

    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#5514B8A6")
        style = Paint.Style.STROKE
        strokeWidth = 3f * resources.displayMetrics.density
    }

    private val activeBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF14B8A6")
        style = Paint.Style.STROKE
        strokeWidth = 3.5f * resources.displayMetrics.density
    }

    private val warningBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFF59E0B")
        style = Paint.Style.STROKE
        strokeWidth = 3.5f * resources.displayMetrics.density
    }

    enum class BorderState { IDLE, ACTIVE, WARNING }

    var borderState: BorderState = BorderState.IDLE
        set(value) {
            field = value
            invalidate()
        }

    /** Extra bottom inset (px) reserved for the action panel so the oval stays clear of it. */
    var bottomReservePx: Float = 0f
        set(value) {
            field = value
            if (width > 0 && height > 0) {
                computeOval(width, height)
                invalidate()
            }
        }

    private val ovalRect = RectF()
    private val maskPath = Path()
    private val ovalPath = Path()

    val ovalBounds: RectF get() = RectF(ovalRect)

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        computeOval(w, h)
    }

    private fun computeOval(w: Int, h: Int) {
        val density = resources.displayMetrics.density
        val topSafe = 72f * density
        val bottomSafe = maxOf(bottomReservePx, 120f * density)
        val availH = (h - topSafe - bottomSafe).coerceAtLeast(h * 0.45f)

        // Large portrait oval — prioritize fitting a full face on any phone size
        var ovalW = w * 0.86f
        var ovalH = ovalW * (4f / 3f) // taller than wide (face-shaped)
        if (ovalH > availH * 0.92f) {
            val scale = (availH * 0.92f) / ovalH
            ovalW *= scale
            ovalH *= scale
        }
        // Never exceed ~90% width on short devices
        if (ovalW > w * 0.92f) {
            val scale = (w * 0.92f) / ovalW
            ovalW *= scale
            ovalH *= scale
        }

        val cx = w / 2f
        val cy = topSafe + availH * 0.48f

        ovalRect.set(cx - ovalW / 2f, cy - ovalH / 2f, cx + ovalW / 2f, cy + ovalH / 2f)

        maskPath.reset()
        maskPath.addRect(0f, 0f, w.toFloat(), h.toFloat(), Path.Direction.CW)

        ovalPath.reset()
        ovalPath.addOval(ovalRect, Path.Direction.CCW)
        maskPath.addPath(ovalPath)
        maskPath.fillType = Path.FillType.EVEN_ODD
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawPath(maskPath, maskPaint)
        val bp = when (borderState) {
            BorderState.IDLE -> borderPaint
            BorderState.ACTIVE -> activeBorderPaint
            BorderState.WARNING -> warningBorderPaint
        }
        canvas.drawOval(ovalRect, bp)
    }
}
