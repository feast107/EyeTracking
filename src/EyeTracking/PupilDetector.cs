using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;

public class FastPupilDetector
{
    // 历史位置缓存用于稳定结果
    private Queue<Point> _positionBuffer = new Queue<Point>();
    private const int BUFFER_SIZE = 5;

    public Point Detect(Mat eyeImage)
    {
        // 1. 高效预处理
        using var processed = PreprocessImage(eyeImage);

        // 2. 候选区域检测
        var candidates = FindPupilCandidates(processed);

        // 3. 选择最佳候选
        var bestCandidate = SelectBestCandidate(processed, candidates);

        // 4. 时间域滤波稳定结果
        return StabilizePosition(bestCandidate);
    }

    private Mat PreprocessImage(Mat input)
    {
        // 转为灰度图
        var gray = input.Channels() == 3 ?
            input.CvtColor(ColorConversionCodes.BGR2GRAY) :
            input.Clone();

        // 自适应直方图均衡化
        var clahe = CLAHE.Create();
        clahe.Apply(gray, gray);

        // 自适应阈值二值化
        var binary = new Mat();
        Cv2.AdaptiveThreshold(gray, binary, 255,
            AdaptiveThresholdTypes.GaussianC,
            ThresholdTypes.BinaryInv, 11, 2);

        // 形态学闭操作填充小孔洞
        var kernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse, new Size(3, 3));
        Cv2.MorphologyEx(binary, binary,
            MorphTypes.Close, kernel);

        return binary;
    }

    private List<RotatedRect> FindPupilCandidates(Mat binaryImage)
    {
        // 查找所有轮廓
        Cv2.FindContours(binaryImage,
            out var contours,
            out _,
            RetrievalModes.List,
            ContourApproximationModes.ApproxSimple);

        var candidates = new List<RotatedRect>();

        foreach (var contour in contours)
        {
            // 跳过太小的区域
            if (contour.Length < 30) continue;

            // 椭圆拟合
            var ellipse = Cv2.FitEllipse(contour);

            // 筛选合理的椭圆参数
            if (IsValidPupilEllipse(ellipse, binaryImage.Size()))
            {
                candidates.Add(ellipse);
            }
        }

        return candidates;
    }

    private bool IsValidPupilEllipse(RotatedRect ellipse, Size imageSize)
    {
        // 椭圆不能太靠近图像边缘
        const int margin = 10;
        if (ellipse.Center.X < margin ||
            ellipse.Center.Y < margin ||
            ellipse.Center.X > imageSize.Width - margin ||
            ellipse.Center.Y > imageSize.Height - margin)
            return false;

        // 长短轴比例应在合理范围内
        float axisRatio = Math.Max(ellipse.Size.Width, ellipse.Size.Height) /
                         Math.Min(ellipse.Size.Width, ellipse.Size.Height);
        if (axisRatio > 2.0f) return false;

        // 瞳孔尺寸应在合理范围内
        float avgRadius = (ellipse.Size.Width + ellipse.Size.Height) / 4;
        float minRadius = imageSize.Height / 20f;
        float maxRadius = imageSize.Height / 5f;

        return avgRadius >= minRadius && avgRadius <= maxRadius;
    }

    private Point SelectBestCandidate(Mat image, List<RotatedRect> candidates)
    {
        if (candidates.Count == 0)
            return new Point(image.Width / 2, image.Height / 2); // 默认中心

        // 选择最圆的候选（长短轴最接近）
        var bestEllipse = candidates
            .OrderBy(e => Math.Abs(e.Size.Width - e.Size.Height))
            .First();

        // 二次精确定位
        return RefinePupilCenter(image, bestEllipse);
    }

    private Point RefinePupilCenter(Mat image, RotatedRect ellipse)
    {
        // 在椭圆ROI内使用灰度加权质心法
        var mask = new Mat(image.Size(), MatType.CV_8UC1, Scalar.All(0));
        Cv2.Ellipse(mask, ellipse, Scalar.All(255), -1);

        // 计算质心
        var moments = Cv2.Moments(image, true);
        if (moments.M00 < float.Epsilon)
            return (Point)ellipse.Center;

        return new Point(
            (int)(moments.M10 / moments.M00),
            (int)(moments.M01 / moments.M00));
    }

    private Point StabilizePosition(Point newPosition)
    {
        _positionBuffer.Enqueue(newPosition);
        if (_positionBuffer.Count > BUFFER_SIZE)
            _positionBuffer.Dequeue();

        // 加权平均，最近帧权重更高
        int sumX = 0, sumY = 0;
        int weight = 1;
        int totalWeight = 0;

        foreach (var p in _positionBuffer)
        {
            sumX += p.X * weight;
            sumY += p.Y * weight;
            totalWeight += weight;
            weight++;
        }

        return new Point(sumX / totalWeight, sumY / totalWeight);
    }
}