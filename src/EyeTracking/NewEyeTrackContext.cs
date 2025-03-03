using EyeTracking.Extensions;
using OpenCvSharp;

namespace EyeTracking;

public class NewEyeTrackContext : EyeTrackContext<EyeDetectResult>
{
    private readonly DetectedLights detected = new();

    protected  Point? LeftEyeVector  => detected.Left.Current;
    protected  Point? RightEyeVector => detected.Right.Current;

    public override void DetectLights(Mat thisMat, out EyeDetectResult? result)
    {
        /*leftEyeVector  = null;
        rightEyeVector = null;
        Debug(DebugHint.Origin, thisMat);
        DetectLightsInternal(thisMat, ref leftEyeVector, ref rightEyeVector);
        detected.Left.Current  = leftEyeVector;
        detected.Right.Current = rightEyeVector;
        using var clone = thisMat.Clone();
        DisplayResult(clone, leftEyeVector, rightEyeVector);
        LastMat?.Dispose();
        LastMat = thisMat;*/
        result = null;
    }

    private void DetectLightsInternal(Mat thisMat, ref Point? leftLightPos, ref Point? rightLightPos)
    {
        if (LastMat == null) return;
        using var subMat = Subtract(thisMat, LastMat, out _, out var darker);
        Debug(DebugHint.Subtraction, subMat);
        using var binMat = Parameters.Threshold(subMat);
        Debug(DebugHint.Bin_Subtraction, binMat);

        if (LeftEyeVector != null && CheckLight(thisMat, binMat, LeftEyeVector.Value, false).DisposeThen() is
            {
                Certain: true,
                Point  : var left
            })
            leftLightPos = left;
        if (RightEyeVector != null && CheckLight(thisMat, binMat, RightEyeVector.Value, false).DisposeThen() is
            {
                Certain: true,
                Point  : var right
            })
            rightLightPos = right;

        if (leftLightPos != null && rightLightPos != null) return;
        using var       tmp        = binMat.Clone();
        List<Candidate> candidates = [];
        while (true)
        {
            tmp.MinMaxLoc(out _, out var maxVal, out _, out var maxPoint);
            if (maxVal <= 0) break;
            tmp.Circle(maxPoint, Parameters.DesiredEyeRadius, Scalar.Black, 2 * Parameters.DesiredEyeRadius);
            candidates.Add(CheckLight(thisMat, binMat, maxPoint));
        }

        var motionRadius = Parameters.MotionRadius;
        foreach (var candidate in candidates.OrderByDescending(x => x.Sum))
        {
            var (_, point, _, _, _) = candidate;
            if (leftLightPos != null && rightLightPos != null) break;
            if (leftLightPos != null)
            {
                if (leftLightPos.Value.DistanceTo(point) < motionRadius) continue; //符合左侧
            }
            else if (LeftEyeVector != null && LeftEyeVector.Value.DistanceTo(point) < motionRadius) //有上一个参考
            {
                leftLightPos = point;
                candidate.Debug(true);
                continue;
            }

            if (rightLightPos != null)
            {
                if (rightLightPos.Value.DistanceTo(point) < motionRadius) continue; //符合右侧
            }
            else if (RightEyeVector != null && RightEyeVector.Value.DistanceTo(point) < motionRadius) //有上一个参考
            {
                rightLightPos = point;
                candidate.Debug(true);
                continue;
            }

            if (leftLightPos == null && rightLightPos == null)
            {
                if (point.X <= thisMat.Width / 2)
                    leftLightPos = point;
                else
                    rightLightPos = point;
                candidate.Debug(true);
                continue;
            }

            var horizon = (leftLightPos ?? rightLightPos)!.Value.Y;
            if (Math.Abs(horizon - point.Y) < Parameters.MaxVerticalDistance)
            {
                leftLightPos  ??= point;
                rightLightPos ??= point;
            }
            
            candidate.Debug(false);
        }
        foreach (var candidate in candidates) candidate.Dispose();
    }

    private Candidate CheckLight(Mat origin, Mat binMat, Point lightPos, bool isPointDetected = true)
    {
        var       rect   = Parameters.GetDesiredEyeRect(lightPos, binMat.Size());
        var originSub    = origin.SubMat(rect);
        using var sub    = binMat.SubMat(rect);
        var       weight = Parameters.Weighted(binMat);
        var       x      = sub.Width;
        var       last   = 0d;
        var       state  = 0;
        while (x-- > 0)
        {
            using var tmp   = sub.Col(x);
            var       value = tmp.Sum().Val0;
            switch (state)
            {
                case 0: //initial
                    state = 1;
                    break;
                case 1: //decreasing
                    if (value < last) state = 2;
                    break;
                case 2: //increasing
                    if (value > last) state = 3;
                    break;
                case 3: // to zero
                    if (value == 0) state = 4;
                    break;
            }

            if (state == 4)
            {
                if (isPointDetected)
                {
                    return new(originSub, lightPos, true, weight, Debug);
                }

                using var t = origin.SubMat(rect);
                t.MinMaxLoc(out _, out Point max);
                var ret = new Candidate(originSub, max.Add(rect.TopLeft), true, weight, Debug);
                ret.Debug(true);
                return ret;
            }

            last = value;

        }

        return new(originSub, lightPos, false, weight, Debug);
    }

    private static Mat Subtract(Mat one, Mat another, out Mat brighter, out Mat darker)
    {
        var ret = new Mat();
        Cv2.Absdiff(one, another, ret);
        if (one.Sum().Val0 > another.Sum().Val0)
        {
            brighter = one;
            darker   = another;
        }
        else
        {
            brighter = another;
            darker   = one;
        }

        return ret;
    }

    private record Candidate(Mat Gray, Point Point, bool Certain, double Sum, DebugHandler Handler) : IDisposable
    {
        private bool debugged;

        public void Debug(bool result)
        {
            if (debugged) return;
            Gray.ThrowIfDisposed();
            debugged = true;
            Handler.Invoke(DebugHint.Candidate, Gray, result, Point);
        }

        public void Dispose()
        {
            if (!Gray.IsDisposed) Gray.Dispose();
        }
    }


    private class DetectedLights
    {
        public Points Left { get; } = new();
        public Points Right { get; } = new();

        public int Confidence
        {
            get => confidence;
            set
            {
                confidence = value <= 0 ? 0 : value;
                if (confidence > 0) return;
                Left.Last  = null;
                Right.Last = null;
            }
        }

        private int confidence;
        
        public class Points
        {
            public Point? Last { get; internal set; }

            public Point? Current
            {
                get => current;
                set
                {
                    current = value;
                    if (current is not null)
                    {
                        Last = value;
                    }
                }
            }

            private Point? current;
        }

    }
}