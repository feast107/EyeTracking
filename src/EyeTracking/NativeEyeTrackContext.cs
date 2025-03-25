using OpenCvSharp;
using EyeTracking.Windows.CLR.Detect;
namespace EyeTracking;

public class NativeEyeTrackContext : EyeTrackContext<EyeDetectResult>
{

    public override unsafe void DetectSight(Mat thisMat, out EyeDetectResult? result)
    {
        var r = new DetectResult();
        Detector.Detect((void*)thisMat.CvPtr, r);
        result = new EyeDetectResult
        {
            Left  = new Point(r.left_x, r.left_y),
            Right = new Point(r.right_x, r.right_y),
        };
    }
}