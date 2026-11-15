import CoreGraphics
import Foundation

enum FaceGuideChecker {
    enum Guidance {
        case ok
        case noFace
        case tooFar
        case tooClose
        case offCenterLeft
        case offCenterRight
        case offCenterUp
        case offCenterDown
        case faceNotStraight
    }

    struct FaceBox {
        var centerX: CGFloat // 0..1
        var centerY: CGFloat
        var width: CGFloat
        var height: CGFloat
    }

    static func check(
        faceBox: FaceBox?,
        ovalBounds: CGRect,
        viewWidth: CGFloat,
        viewHeight: CGFloat
    ) -> Guidance {
        guard let faceBox else { return .noFace }
        guard viewWidth > 0, viewHeight > 0 else { return .noFace }

        let ovalCx = ovalBounds.midX / viewWidth
        let ovalCy = ovalBounds.midY / viewHeight
        let ovalW = ovalBounds.width / viewWidth

        let dx = faceBox.centerX - ovalCx
        let dy = faceBox.centerY - ovalCy
        let tolerance: CGFloat = 0.12

        if dx < -tolerance { return .offCenterLeft }
        if dx > tolerance { return .offCenterRight }
        if dy < -tolerance { return .offCenterUp }
        if dy > tolerance { return .offCenterDown }

        if faceBox.width < ovalW * 0.35 { return .tooFar }
        if faceBox.width > ovalW * 1.55 { return .tooClose }

        let aspect = faceBox.height / (faceBox.width + 0.001)
        if aspect < 0.85 || aspect > 2.0 { return .faceNotStraight }

        return .ok
    }

    static func message(_ g: Guidance) -> String? {
        switch g {
        case .ok: return nil
        case .noFace: return "Position your full face in the oval"
        case .tooFar: return "Move closer"
        case .tooClose: return "Move back a little"
        case .offCenterLeft: return "Move your face to the right"
        case .offCenterRight: return "Move your face to the left"
        case .offCenterUp: return "Move your face down"
        case .offCenterDown: return "Move your face up"
        case .faceNotStraight: return "Look straight at the camera"
        }
    }
}
