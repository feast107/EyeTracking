using OpenCvSharp;

namespace EyeTracking;

public class EyeDetectResult
{
    public Point Left  { get; set; }
    public Point Right { get; set; }
    
    public Point? LeftEyeCenter { get; set; }
    public Point? RightEyeCenter { get; set; }
}