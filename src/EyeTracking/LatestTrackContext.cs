using System.Diagnostics.CodeAnalysis;
using OpenCvSharp;
using Cv2 = OpenCvSharp.Cv2;

namespace EyeTracking;

public class LatestTrackContext : EyeTrackContext<EyeDetectResult>
{
    private bool?  isLastLight;

    private EyeDetectResult   Result     { get; set; } = new();

    [field: AllowNull,MaybeNull]
    private CascadeClassifier EyeCascade
    {
        get
        {
            if (field is not null) return field;
            field = new CascadeClassifier();
            if (field.Load(XmlPath)) return field;
            throw new FileLoadException("Error: Unable to load eye cascade classifier!");
        }
    }

    private static readonly string XmlPath =
        Path.Combine(AppContext.BaseDirectory, "Resources/Haarcascade/haarcascade_eye.xml");

    public override void DetectLights(Mat thisMat, out EyeDetectResult? result)
    {
        Debug(DebugHint.Origin, thisMat);
        if (LastMat is not null)
        {
            if (isLastLight is null)
            {
                var s = LastMat.Sum();
                var n = thisMat.Sum();
                isLastLight = s[0] > n[0];
            }

            var light = isLastLight.Value ? LastMat : thisMat;
            var dark  = isLastLight.Value ? thisMat : LastMat;

            DetectPupil(light, dark);
            DetectReflection(light);

            /*using var pupilPosition     = GetPupilPosition(dark, light);
            using var blurredImageLight = ApplyGaussianBlur(light);
            using var binary            = ApplyThreshold(blurredImageLight);
            using var edges             = DetectEdges(binary);

            Cv2.FindContours(edges, out var contours, out _, RetrievalModes.List,
                ContourApproximationModes.ApproxSimple);

            ProcessEllipseAndBlobs(contours, light);
            */

            isLastLight = !isLastLight;
            LastMat.Dispose();
        }

        result = Result;
        // 将成员变量的值赋给 out 参数
        LastMat = thisMat;
        Console.WriteLine("输出: 左眼位置 " + Result.Left);
        Console.WriteLine("输出: 右眼位置 " + Result.Right);
    }

    private bool DetectPupil(Mat lightImage, Mat darkImage)
    {
        var eyes = EyeCascade.DetectMultiScale(lightImage, 1.1, 4, 0, new Size(30, 30));
        if (eyes.Length == 0)
        {
            return false;
        }

        const string filePath = "output.txt";                                       // 输出文件路径
        //using var    outFile  = new StreamWriter(filePath, true);
        foreach (var eye in eyes.OrderBy(x => x.X))
        {
            var eyeLightRegion = lightImage.SubMat(eye);
            var eyeDarkRegion  = darkImage.SubMat(eye);

            var eyePupilPosition = new Mat();
            Cv2.Absdiff(eyeLightRegion, eyeDarkRegion, eyePupilPosition);

            var blurredEye = new Mat();
            Cv2.GaussianBlur(eyePupilPosition, blurredEye, new Size(5, 5), 1.5);

            var edges = new Mat();
            Cv2.Canny(blurredEye, edges, 50, 150);

            Cv2.FindContours(edges, out var contours, out _, RetrievalModes.External,
                ContourApproximationModes.ApproxSimple);

            var leftFin = false;
            foreach (var contour in contours)
            {
                if (contour.Length < 5) continue;
                var ellipse = Cv2.FitEllipse(contour);
                var center  = ellipse.Center;

                center.X += eye.X;
                center.Y += eye.Y;

                if (!leftFin) Result.LeftEyeCenter = new Point(center.X, center.Y);
                else Result.RightEyeCenter         = new Point(center.X, center.Y);
                leftFin = true;
                // 将瞳孔中心坐标写入文件
                //outFile.WriteLine($"Pupil Center: ({center.X:F3}, {center.Y:F3})");
            }
        }

        return true;
    }

    // 检测眼睛
    private Rect[] DetectEyes(Mat image)
    {
        return EyeCascade.DetectMultiScale(image, 1.1, 4, 0, new Size(30, 30));
    }

    // 对眼睛区域进行处理（最大值滤波 + 中值滤波）
    private static Mat ProcessEyeArea(Mat eye, int maxFilterSize, int medianFilterSize)
    {
        // 确保核大小为奇数
        maxFilterSize    = maxFilterSize    % 2 == 0 ? maxFilterSize    + 1 : maxFilterSize;
        medianFilterSize = medianFilterSize % 2 == 0 ? medianFilterSize + 1 : medianFilterSize;

        // 最大值滤波
        Mat maxFiltered = new();
        Cv2.Dilate(eye, maxFiltered,
            Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(maxFilterSize, maxFilterSize)));

        // 中值滤波
        Mat medianFiltered = new();
        Cv2.MedianBlur(eye, medianFiltered, medianFilterSize);

        // 计算最大值滤波结果减去中值滤波结果
        Mat result = new();
        Cv2.Subtract(maxFiltered, medianFiltered, result);

        return result;
    }

    // 提取亮斑中心坐标并绘制
    private bool ExtractBrightSpotCenter(Mat result, Rect eyeRect, Mat original, out Point? output)
    {
        // 阈值分割提取亮斑区域
        Mat binary = new();
        Cv2.Threshold(result, binary, 50, 255, ThresholdTypes.Binary); // 调整阈值以适应亮斑

        // 查找轮廓
        Cv2.FindContours(binary, out var contours, out _, RetrievalModes.External,
            ContourApproximationModes.ApproxSimple);

        if (contours.Length == 0)
        {
            output = null;
            return false;
        }

        // 假设最大的轮廓是亮斑
        var largestContour = contours.OrderByDescending(x => Cv2.ContourArea(x)).First();

        // 计算质心
        var m = Cv2.Moments(largestContour);
        if (m.M00 == 0)
        {
            output = null;
            return false;
        }

        var cx = m.M10 / m.M00;
        var cy = m.M01 / m.M00;

        // 将亮斑中心绘制在原图上
        Cv2.Circle(original, new Point(eyeRect.X + cx, eyeRect.Y + cy), 3, new Scalar(0, 0, 255), -1); // 红色圆点
        //this._leftLightPos = new Point(eyeRect.X + cx, eyeRect.Y + cy);
        //this._rightLightPos = new Point(eyeRect.X + cx, eyeRect.Y + cy);
        output = new Point(eyeRect.X + cx, eyeRect.Y + cy);
        return true;


        // 写入文件
        var filePath = "output.txt";                                 // 输出文件路径
        using (var outFile = new StreamWriter(filePath, true)) // 使用追加模式写入文件
        {
            // 设置小数点后3位
            outFile.WriteLine($"Highlight Center: ({(eyeRect.X + cx):F3}, {(eyeRect.Y + cy):F3})");
        }
    }


    // 反射点检测功能实现
    private void DetectReflection(Mat image)
    {
        // 检测眼睛
        var eyes = DetectEyes(image);
        if (eyes.Length != 2)
        {
            return;
        }

        eyes = eyes.OrderBy(x => x.X).ToArray();

        // 提取左眼和右眼区域
        var leftEye  = image.SubMat(eyes[0]);
        var rightEye = image.SubMat(eyes[1]);

        // 处理眼睛区域
        var resultLeft  = ProcessEyeArea(leftEye, 5, 3);
        var resultRight = ProcessEyeArea(rightEye, 5, 3);

        // 打开输出文件
        // std::ofstream outFile(output_file);
        // if (!outFile.is_open())
        // {
        //     std::cerr << "Error: Unable to open output file!" << std::endl;
        //     return;
        // }

        // 提取左眼和右眼的亮斑中心并绘制
        var leftCenter = Result.LeftEyeCenter;
        if (ExtractBrightSpotCenter(resultLeft, eyes[0], image, out var left)
        && leftCenter != null)
        {
            var cur  = new Point(
                left!.Value.X - leftCenter.Value.X,
                left.Value.Y  - leftCenter.Value.Y);
            if (!(cur.X is 0 && cur.Y is 0))
                Result.Left = cur;
        }

        var rightCenter = Result.RightEyeCenter;
        if (ExtractBrightSpotCenter(resultRight, eyes[1], image, out var right)
        && rightCenter != null)
        {
            var cur = new Point(
                right!.Value.X - rightCenter.Value.X,
                right.Value.Y  - rightCenter.Value.Y);
            if (!(cur.X is 0 && cur.Y is 0))
                Result.Right = cur;
        }

        // 关闭文件

        // 显示结果
        // Cv2.ImShow("Original Image with Bright Spot Centers", image);
        // Cv2.ImShow("Processed Left Eye", resultLeft);
        // Cv2.ImShow("Processed Right Eye", resultRight);

        // 等待按键
        // Cv2.WaitKey(0);
    }


    /*private static Mat GetPupilPosition(Mat darkImage, Mat lightImage)
    {
        var pupilPosition = new Mat();
        Cv2.Absdiff(darkImage, lightImage, pupilPosition);
        Cv2.Normalize(pupilPosition, pupilPosition, 0, 255, NormTypes.MinMax);
        return pupilPosition;
    }

    private static Mat ApplyGaussianBlur(Mat image)
    {
        var blurredImage = new Mat();
        Cv2.GaussianBlur(image, blurredImage, new Size(5, 5), 0);
        return blurredImage;
    }

    private static Mat ApplyThreshold(Mat image)
    {
        var binary = new Mat();
        Cv2.Threshold(image, binary, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
        return binary;
    }

    private static Mat DetectEdges(Mat binaryImage)
    {
        var edges = new Mat();
        Cv2.Canny(binaryImage, edges, 50, 150);
        return edges;
    }

    private List<OpenCvSharp.Point> FindLargestContours(List<List<Point>> contours)
    {
        var largestContours = new List<OpenCvSharp.Point>();
        var maxArea        = 0d;
        foreach (var contour in contours)
        {
            var area = Cv2.ContourArea(contour);
            if (!(area > maxArea)) continue;
            maxArea        = area;
            largestContours = contour;
        }

        return largestContours;
    }

    void ProcessEllipseAndBlobs(Point[][] contours, Mat lightImage)
    {
        var           result        = lightImage.Clone();
        var           detectedBlobs = lightImage.Clone();
        List<Point2f> centers       = [];

        foreach (var contour in contours)
        {
            if(contour.Length < 5) continue;

            var fittedEllipse = Cv2.FitEllipse(contour);
            var area          = Cv2.ContourArea(contour);
            var aspectRatio   = Math.Abs(fittedEllipse.Size.Height / fittedEllipse.Size.Width);
            if (area is > 100 and < 1000 && aspectRatio > 0.5 && aspectRatio < 2.0)
            {
                var center = fittedEllipse.Center;
                var isDuplicate = false;
                foreach (var existingCenter in centers)
                {
                    if (Cv2.Norm(center.DistanceTo(existingCenter)) < 10)
                    {
                        isDuplicate = true;
                        break;
                    }
                }

                if (!isDuplicate)
                {
                    centers.Add(center);
                    Cv2.Ellipse(result, fittedEllipse, Scalar.Green, 2);
                    Cv2.Circle(result, center.ToPoint(), 2, Scalar.Blue, -1);


                    var maskRaw = Mat.Zeros(lightImage.Size(), MatType.CV_8UC1);
                    var mask    = new Mat();
                    Cv2.EqualizeHist(maskRaw, mask);
                    Cv2.Ellipse(mask, fittedEllipse, Scalar.White, -1);
                    var maskedImage = new Mat();
                    lightImage.CopyTo(maskedImage, mask);

                    const int manualThresholdValue = 105;
                    var       binaryMasked         = new Mat();
                    Cv2.Threshold(mask,binaryMasked,manualThresholdValue,255,ThresholdTypes.Binary);

                    Cv2.FindContours(binaryMasked, out var maskedContours, out _,
                        RetrievalModes.List, ContourApproximationModes.ApproxSimple);

                    var blobCount = 0;
                    foreach (var mContour in maskedContours)
                    {
                        var mArea = Cv2.ContourArea(mContour);
                        if (mArea is > 3 and < 10)
                        {
                            Cv2.DrawContours(detectedBlobs, [mContour], -1, Scalar.Red, 2);
                            blobCount++;

                            var moments = Cv2.Moments(mContour);
                            var centroid = new Point2f((float)(moments.M10 / moments.M00), (float)(moments.M01 / moments.M00));

                            Cv2.Circle(detectedBlobs, centroid.ToPoint(), 2, Scalar.Cyan, -1);
                        }
                    }

                    Cv2.MinMaxLoc(maskedImage,
                        out _, out var maxVal,
                        out _, out var brightestPoint, mask);

                }
            }
        }
    }*/
}