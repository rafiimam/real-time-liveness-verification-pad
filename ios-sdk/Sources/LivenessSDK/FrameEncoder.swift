import UIKit
import AVFoundation

enum FrameEncoder {
    /// JPEG → raw base64 (no data-URL prefix).
    static func jpegBase64(from image: UIImage, quality: CGFloat = 0.7, maxWidth: CGFloat = 480) -> String? {
        let scaled = scaleDown(image, maxWidth: maxWidth)
        guard let data = scaled.jpegData(compressionQuality: quality) else { return nil }
        return data.base64EncodedString(options: [])
    }

    static func jpegBase64(from sampleBuffer: CMSampleBuffer, quality: CGFloat = 0.68, maxWidth: CGFloat = 480) -> String? {
        guard let image = uiImage(from: sampleBuffer) else { return nil }
        return jpegBase64(from: image, quality: quality, maxWidth: maxWidth)
    }

    static func uiImage(from sampleBuffer: CMSampleBuffer) -> UIImage? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        // Front camera buffer is landscape; rotate for portrait UI + mirror.
        return UIImage(cgImage: cgImage, scale: 1, orientation: .leftMirrored)
    }

    private static func scaleDown(_ image: UIImage, maxWidth: CGFloat) -> UIImage {
        guard image.size.width > maxWidth else { return image }
        let ratio = maxWidth / image.size.width
        let size = CGSize(width: maxWidth, height: max(1, image.size.height * ratio))
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
    }
}
