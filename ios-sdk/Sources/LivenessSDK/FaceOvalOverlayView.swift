import UIKit

/// Translucent mask with a clear oval cutout — sized so a full face fits (matches Android).
final class FaceOvalOverlayView: UIView {
    enum BorderState {
        case idle
        case active
        case warning
    }

    var borderState: BorderState = .idle {
        didSet { setNeedsDisplay() }
    }

    /// Bottom inset reserved for the action panel.
    var bottomReserve: CGFloat = 0 {
        didSet { setNeedsDisplay() }
    }

    private(set) var ovalBounds: CGRect = .zero

    override init(frame: CGRect) {
        super.init(frame: frame)
        isOpaque = false
        backgroundColor = .clear
        contentMode = .redraw
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        isOpaque = false
        backgroundColor = .clear
        contentMode = .redraw
    }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        let w = bounds.width
        let h = bounds.height
        guard w > 0, h > 0 else { return }

        let topSafe: CGFloat = 72
        let bottomSafe = max(bottomReserve, 120)
        let availH = max(h - topSafe - bottomSafe, h * 0.45)

        var ovalW = w * 0.86
        var ovalH = ovalW * (4.0 / 3.0)
        if ovalH > availH * 0.92 {
            let scale = (availH * 0.92) / ovalH
            ovalW *= scale
            ovalH *= scale
        }
        if ovalW > w * 0.92 {
            let scale = (w * 0.92) / ovalW
            ovalW *= scale
            ovalH *= scale
        }

        let cx = w / 2
        let cy = topSafe + availH * 0.48
        ovalBounds = CGRect(x: cx - ovalW / 2, y: cy - ovalH / 2, width: ovalW, height: ovalH)

        ctx.saveGState()
        ctx.setFillColor(UIColor(red: 15 / 255, green: 20 / 255, blue: 25 / 255, alpha: 0.70).cgColor)
        ctx.fill(bounds)

        ctx.setBlendMode(.clear)
        ctx.fillEllipse(in: ovalBounds)
        ctx.restoreGState()

        let border: UIColor
        let lineWidth: CGFloat
        switch borderState {
        case .idle:
            border = UIColor(red: 20 / 255, green: 184 / 255, blue: 166 / 255, alpha: 0.33)
            lineWidth = 3
        case .active:
            border = UIColor(red: 20 / 255, green: 184 / 255, blue: 166 / 255, alpha: 1)
            lineWidth = 3.5
        case .warning:
            border = UIColor(red: 245 / 255, green: 158 / 255, blue: 11 / 255, alpha: 1)
            lineWidth = 3.5
        }
        ctx.setStrokeColor(border.cgColor)
        ctx.setLineWidth(lineWidth)
        ctx.strokeEllipse(in: ovalBounds)
    }
}
