using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using OpenCvSharp;

namespace EyeTracking.Desktop.Extensions;

internal static class BitmapExtensions
{
    public static WriteableBitmap ToWriteableBitmap(this Mat mat)
    {
        var tmp = new Mat();

        // 检查通道数，决定如何转换
        if (mat.Channels() == 1) // 灰度图 → RGBA
        {
            Cv2.CvtColor(mat, tmp, ColorConversionCodes.GRAY2RGBA);
        }
        else if (mat.Channels() == 3) // BGR → RGBA
        {
            Cv2.CvtColor(mat, tmp, ColorConversionCodes.BGR2RGBA);
        }
        else // 其他情况（如已经是 RGBA）
        {
            tmp = mat.Clone();
        }
        return new WriteableBitmap(PixelFormat.Rgb32, AlphaFormat.Opaque, tmp.DataStart,
            new PixelSize(mat.Width, mat.Height), new Vector(96, 96), mat.Width * 4);
    }
}