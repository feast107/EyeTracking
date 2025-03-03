using OpenCvSharp;

namespace EyeTracking;

public abstract class EyeTrackContext : IDisposable
{
    protected         Mat?                LastMat    { get; set; }
    protected virtual Point?              LeftEyeVector  { get; }
    protected virtual Point?              RightEyeVector { get; }
    public            EyeDetectParameters Parameters { get; set; } = new();

    public abstract void DetectLights(Mat thisMat, out Point? leftEyeVector, out Point? rightEyeVector);

    protected void DisplayResult(Mat mat, Point? left, Point? right)
    {
        if (left != null)
        {
            mat.PutText($"X:{left.Value.X},{left.Value.Y}",
                left.Value,
                HersheyFonts.HersheySimplex,
                1,
                Scalar.White);
            mat.Circle(left.Value, Parameters.DesiredEyeRadius, Scalar.Black, 3);
            mat.Circle(left.Value, 2, Scalar.Black, 4);
        }

        if (right != null)
        {
            mat.PutText($"X:{right.Value.X},{right.Value.Y}",
                right.Value,
                HersheyFonts.HersheySimplex,
                1,
                Scalar.White);
            mat.Circle(right.Value, Parameters.DesiredEyeRadius, Scalar.Black, 3);
            mat.Circle(right.Value, 2, Scalar.Black, 4);
        }

        OnDebug?.Invoke(DebugHint.Output, mat);
    }

    public event DebugHandler? OnDebug;

    public delegate void DebugHandler(DebugHint hint, params object[] args);

    protected void Debug(DebugHint hint, params object[] args)
    {
#if DEBUG
        OnDebug?.Invoke(hint, args);
#endif
    }

    public enum DebugHint
    {
        Origin,
        Output,
        Candidate,
        Subtraction,
        Bin_Subtraction
    }

    public void Dispose()
    {
        OnDebug = null;
        LastMat?.Dispose();
        LastMat = null;
    }
}