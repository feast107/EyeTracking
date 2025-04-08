using OpenCvSharp;
using System.Collections.ObjectModel;

namespace EyeTracking;

public abstract class EyeTrackContext<TResult> : IDisposable
{
    public ObservableCollection<EyeDetectTrace> Traces { get; } = [];

    protected         Mat?                LastMat    { get; set; }
    public            EyeDetectParameters Parameters { get; set; } = new();

    public abstract void DetectSight(Mat thisMat, out TResult? result);

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
        Bin_Subtraction,
        Debug_right_light,
        Debug_right_dark
    }

    public void Dispose()
    {
        OnDebug = null;
        LastMat?.Dispose();
        LastMat = null;
    }
}