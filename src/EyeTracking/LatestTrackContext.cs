using System.Diagnostics.CodeAnalysis;
using OpenCvSharp;
using Cv2 = OpenCvSharp.Cv2;

namespace EyeTracking;

public class DetectionStatistics
{
    public long NoEyesDetectedCount //眼睛识别异常
    {
        get;
        set;
    } = 0;

    public long NoEyesDetectedCount2 //眼睛识别异常
    {
        get;
        set;
    } = 0;

    public long NoCheckLightCount //明瞳监测异常
    {
        get;
        set;
    } = 0;

    public long NoPuilpDetectedCount //瞳孔监测异常
    {
        get;
        set;
    } = 0;

    public long NoReflectionDetectedCount //亮斑识别异常
    {
        get;
        set;
    } = 0;

    public long TotalFrames   { get; set; } = 0; //总帧数
    public long SuccessFrames { get; set; } = 0;

    public double ErrorRate => (NoEyesDetectedCount + NoReflectionDetectedCount + NoPuilpDetectedCount) /
        (double)TotalFrames * 100;
}

public class LatestTrackContext : EyeTrackContext<EyeDetectResult>
{
    private bool? isLastLight;

    private          Rect[] p_eyes           = new Rect[2];
    private readonly Rect[] last_this_center = new Rect[2];

    private static readonly string XmlPath =
        Path.Combine(AppContext.BaseDirectory, "Resources/Haarcascade/haarcascade_eye.xml");

    private static readonly string FrontFaceXmlPath =
        Path.Combine(AppContext.BaseDirectory, "Resources/Haarcascade/haarcascade_frontalface_alt.xml");

    private static readonly string ProfileFaceXmlPath =
        Path.Combine(AppContext.BaseDirectory, "Resources/Haarcascade/haarcascade_profileface.xml");

    [field: AllowNull, MaybeNull]
    private CascadeClassifier FrontFaceCascade
    {
        get
        {
            if (field is not null) return field;
            field = new CascadeClassifier();
            if (field.Load(FrontFaceXmlPath)) return field;
            throw new FileLoadException("Error: Unable to load front face cascade classifier!");
        }
    }

    [field: AllowNull, MaybeNull]
    private CascadeClassifier ProfileFaceCascade
    {
        get
        {
            if (field is not null) return field;
            field = new CascadeClassifier();
            if (field.Load(ProfileFaceXmlPath)) return field;
            throw new FileLoadException("Error: Unable to load profile face cascade classifier!");
        }
    }

    private EyeDetectResult Result { get; set; } = new();

    //做统计
    public DetectionStatistics Stats { get; } = new();

    [field: AllowNull, MaybeNull]
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

    private double CalculateDynamicThreshold(Mat image, Rect rect)
    {
        using var roi    = new Mat(image, rect);
        var       mean   = Cv2.Mean(roi);
        Cv2.MeanStdDev(roi, out _, out var stdDev);

        // 根据均值和标准差计算动态阈值
        return mean[0] * rect.Width * rect.Height * 0.15 + stdDev[0] * rect.Width * rect.Height * 0.1;
    }

    public override void DetectSight(Mat thisMat, out EyeDetectResult? result)
    {
        Debug(DebugHint.Origin, thisMat);
        if (LastMat is not null)
        {
            Stats.TotalFrames++;

            if (DetectEyes(thisMat, last_this_center, 1))
            {
                var s = GetBrightnessOfRectUsingSum(LastMat, last_this_center[0]);
                var n = GetBrightnessOfRectUsingSum(thisMat, last_this_center[1]);
                last_this_center[0] = last_this_center[1];

                var dynamicThreshold = CalculateDynamicThreshold(thisMat, last_this_center[1]);
                var brightnessDiff   = Math.Abs(s[0] - n[0]);

                if (brightnessDiff > dynamicThreshold)
                {
                    isLastLight = s[0] > n[0];
                    var light = isLastLight.Value ? LastMat : thisMat;
                    var dark  = isLastLight.Value ? thisMat : LastMat;
                    if (DetectPupil(light, dark))
                    {
                        if (DetectReflection(light))
                        {
                            Stats.SuccessFrames++;
                            if (true) // 单眼调试模式
                            {
                                using var lightEyeMat      = new Mat(light, p_eyes[0]);
                                using var darkEyeMat       = new Mat(dark, p_eyes[0]);
                                using var lightEyeOriginal = lightEyeMat.Clone();
                                using var darkEyeOriginal  = darkEyeMat.Clone();
                                var       leftCenter       = (Point)Result.LeftEyeCenter;

                                var leftEyeCenterInSub = new Point(
                                    leftCenter.X - p_eyes[0].X,
                                    leftCenter.Y - p_eyes[0].Y
                                );
                                var leftPointInSub = new Point(
                                    Result.Left.X + leftEyeCenterInSub.X,
                                    Result.Left.Y + leftEyeCenterInSub.Y
                                );

                                Cv2.Circle(darkEyeMat, leftEyeCenterInSub, 2, Scalar.Green, -1);
                                Cv2.Circle(darkEyeMat, leftPointInSub, 1, Scalar.White, -1);
                                Cv2.Circle(lightEyeMat, leftEyeCenterInSub, 2, Scalar.Green, -1);
                                Cv2.Circle(lightEyeMat, leftPointInSub, 1, Scalar.White, -1);

                                SaveProcessedEyeImages(lightEyeMat, darkEyeMat, lightEyeOriginal, darkEyeOriginal,
                                    Stats.TotalFrames);

                                Debug(DebugHint.Subtraction, lightEyeMat);
                                Debug(DebugHint.Output, darkEyeMat);
                                Debug(DebugHint.Bin_Subtraction, thisMat);
                            }
                            else // 双眼模式
                            {
                                using (var leftEyeMat = new Mat(light, p_eyes[0]))
                                using (var rightEyeMat = new Mat(light, p_eyes[1]))
                                using (var leftEyeOriginal = leftEyeMat.Clone())
                                using (var rightEyeOriginal = rightEyeMat.Clone())
                                {
                                    var leftCenter  = (Point)Result.LeftEyeCenter;
                                    var rightCenter = (Point)Result.RightEyeCenter;

                                    var leftEyeCenterInSub = new Point(
                                        leftCenter.X - p_eyes[0].X,
                                        leftCenter.Y - p_eyes[0].Y
                                    );
                                    var leftPointInSub = new Point(
                                        Result.Left.X + leftEyeCenterInSub.X,
                                        Result.Left.Y + leftEyeCenterInSub.Y
                                    );
                                    var rightEyeCenterInSub = new Point(
                                        rightCenter.X - p_eyes[1].X,
                                        rightCenter.Y - p_eyes[1].Y
                                    );
                                    var rightPointInSub = new Point(
                                        Result.Right.X + rightEyeCenterInSub.X,
                                        Result.Right.Y + rightEyeCenterInSub.Y
                                    );

                                    Cv2.Circle(leftEyeMat, leftEyeCenterInSub, 2, Scalar.Green, -1);
                                    Cv2.Circle(leftEyeMat, leftPointInSub, 1, Scalar.White, -1);
                                    Cv2.Circle(rightEyeMat, rightEyeCenterInSub, 2, Scalar.Green, -1);
                                    Cv2.Circle(rightEyeMat, rightPointInSub, 1, Scalar.White, -1);

                                    SaveProcessedEyeImages(leftEyeMat, rightEyeMat, leftEyeOriginal, rightEyeOriginal,
                                        Stats.TotalFrames);

                                    Debug(DebugHint.Subtraction, leftEyeMat);
                                    Debug(DebugHint.Output, rightEyeMat);
                                    Debug(DebugHint.Bin_Subtraction, thisMat);
                                }
                            }
                        }
                    }
                }
                else
                {
                    Stats.NoCheckLightCount++;
                }
            }

            Trace(0, "总共处理"                               + Stats.TotalFrames);
            Trace((KeyedEyeDetectTrace.TraceKey)99, "成功"  + Stats.SuccessFrames);
            Trace((KeyedEyeDetectTrace.TraceKey)1, "眼睛异常" + Stats.NoEyesDetectedCount);
            Trace((KeyedEyeDetectTrace.TraceKey)5, "眼睛没有" + Stats.NoEyesDetectedCount2);
            Trace((KeyedEyeDetectTrace.TraceKey)2, "明暗异常" + Stats.NoCheckLightCount);
            Trace((KeyedEyeDetectTrace.TraceKey)3, "瞳孔异常" + Stats.NoPuilpDetectedCount);
            Trace((KeyedEyeDetectTrace.TraceKey)4, "亮斑异常" + Stats.NoReflectionDetectedCount);
        }
        else
        {
            if (DetectEyes(thisMat, last_this_center, 0))
            {
                LastMat = thisMat;
                result  = Result;
                return;
            }
        }

        LastMat?.Dispose();
        LastMat = null;
        result  = Result;
    }

    //
    public class KeyedEyeDetectTrace : EyeDetectTrace
    {
        public enum TraceKey
        {
            All,
            NoEyesDetectedCount,      //眼睛
            NoCheckLightCount,        //明瞳
            NoPuilpDetectedCount,     //瞳孔监测异常
            NoReflectionDetectedCount //亮斑识别异常
        }

        public required TraceKey Key;
    }

    private void Trace(KeyedEyeDetectTrace.TraceKey key, string content)
    {
        var exist = Traces.OfType<KeyedEyeDetectTrace>().FirstOrDefault(x => x.Key == key);
        if (exist == null)
        {
            Traces.Add(new KeyedEyeDetectTrace { Key = key, Content = content });
        }
        else
        {
            exist.Content = content;
        }
    }

    private Scalar GetBrightnessOfRectUsingSum(Mat image, Rect rect)
    {
        // 使用Rect对象从图像中裁剪出区域
        var croppedRegion = new Mat(image, rect);
        // Cv2.ImShow("test", croppedRegion);
        return croppedRegion.Sum();
    }

    private bool DetectEyes(Mat image, Rect[] middleEye, int i)
    {
        // 使用属性获取已加载的分类器
        var facesFront = FrontFaceCascade.DetectMultiScale(
            image, 1.1, 3,
            HaarDetectionTypes.ScaleImage,
            new Size(150, 150)
        );

        if (facesFront.Length > 0)
        {
            // 在人脸区域内检测眼睛
            var faceRect        = facesFront[0];
            var eyeRegionHeight = faceRect.Height / 3;
            var eyeRegionWidth  = faceRect.Width  / 3;
            var eyeRegionTop    = faceRect.Y + faceRect.Height / 4;

            // 左眼区域
            var leftEyeRect = new Rect(
                faceRect.X + faceRect.Width / 6,
                eyeRegionTop,
                eyeRegionWidth,
                eyeRegionHeight
            );

            // 右眼区域
            var rightEyeRect = new Rect(
                faceRect.X + faceRect.Width / 2,
                eyeRegionTop,
                eyeRegionWidth,
                eyeRegionHeight
            );

            // 确保矩形在图像范围内
            leftEyeRect = new Rect(
                Math.Max(0, leftEyeRect.X),
                Math.Max(0, leftEyeRect.Y),
                Math.Min(image.Cols - leftEyeRect.X, leftEyeRect.Width),
                Math.Min(image.Rows - leftEyeRect.Y, leftEyeRect.Height)
            );

            rightEyeRect = new Rect(
                Math.Max(0, rightEyeRect.X),
                Math.Max(0, rightEyeRect.Y),
                Math.Min(image.Cols - rightEyeRect.X, rightEyeRect.Width),
                Math.Min(image.Rows - rightEyeRect.Y, rightEyeRect.Height)
            );

            p_eyes = new[] { leftEyeRect, rightEyeRect };
        }
        else
        {
            // 尝试侧面人脸检测
            var facesProfile = ProfileFaceCascade.DetectMultiScale(
                image, 1.1, 3,
                HaarDetectionTypes.ScaleImage,
                new Size(150, 150)
            );

            if (facesProfile.Length > 0)
            {
                // 使用相同的眼睛区域提取逻辑
                var faceRect        = facesProfile[0];
                var eyeRegionHeight = faceRect.Height / 3;
                var eyeRegionWidth  = faceRect.Width  / 3;
                var eyeRegionTop    = faceRect.Y + faceRect.Height / 4;

                // 根据侧脸调整眼睛位置
                var eyeRect = new Rect(
                    faceRect.X + faceRect.Width / 4,
                    eyeRegionTop,
                    eyeRegionWidth,
                    eyeRegionHeight
                );

                // 确保矩形在图像范围内
                eyeRect = new Rect(
                    Math.Max(0, eyeRect.X),
                    Math.Max(0, eyeRect.Y),
                    Math.Min(image.Cols - eyeRect.X, eyeRect.Width),
                    Math.Min(image.Rows - eyeRect.Y, eyeRect.Height)
                );

                // 在眼睛区域内进行眼睛检测
                var eyes = EyeCascade.DetectMultiScale(
                    new Mat(image, eyeRect),
                    1.05, 6,
                    HaarDetectionTypes.ScaleImage,
                    new Size(40, 40)
                );

                if (eyes.Length >= 1)
                {
                    p_eyes = new[]
                        { new Rect(eyeRect.X + eyes[0].X, eyeRect.Y + eyes[0].Y, eyes[0].Width, eyes[0].Height) };
                }
                else
                {
                    Stats.NoEyesDetectedCount2++;
                    return false;
                }
            }
            else
            {
                // 如果人脸检测失败，尝试直接检测眼睛
                var eyes = EyeCascade.DetectMultiScale(
                    image, 1.05, 6,
                    HaarDetectionTypes.ScaleImage,
                    new Size(40, 40)
                );

                if (eyes.Length < 2)
                {
                    Stats.NoEyesDetectedCount2++;
                    return false;
                }

                // 验证眼睛
                var validatedEyes = new List<Rect>();
                foreach (var eye in eyes.OrderBy(e => e.X))
                {
                    // 验证宽高比
                    var aspectRatio = (float)eye.Width / eye.Height;
                    if (aspectRatio is < 0.4f or > 2.5f) continue;

                    // 验证区域大小相对于图像
                    var relativeSize = (float)(eye.Width * eye.Height) / (image.Rows * image.Cols);
                    if (relativeSize is < 0.01f or > 0.15f) continue;

                    validatedEyes.Add(eye);
                }

                if (validatedEyes.Count < 2)
                {
                    Stats.NoEyesDetectedCount++;
                    return false;
                }

                // 验证两个眼睛的相对位置和大小
                var left  = validatedEyes[0];
                var right = validatedEyes[1];

                // 验证水平距离
                float distance = right.X - (left.X + left.Width);
                if (distance < 0)
                {
                    Stats.NoEyesDetectedCount++;
                    return false;
                }

                // 验证大小相似性
                var sizeRatio = (float)(left.Width * left.Height) / (right.Width * right.Height);
                if (sizeRatio is < 0.5f or > 2.0f)
                {
                    Stats.NoEyesDetectedCount++;
                    return false;
                }

                // 验证垂直位置相似性
                float verticalDiff = Math.Abs(left.Y - right.Y);
                if (verticalDiff > left.Height)
                {
                    Stats.NoEyesDetectedCount++;
                    return false;
                }

                p_eyes = [left, right];
            }
        }

        // 计算中间区域
        if (p_eyes.Length >= 2)
        {
            var center1 = new Point(p_eyes[0].X + p_eyes[0].Width / 2, p_eyes[0].Y + p_eyes[0].Height / 2);
            var center2 = new Point(p_eyes[1].X + p_eyes[1].Width / 2, p_eyes[1].Y + p_eyes[1].Height / 2);
            var center  = new Point((center1.X + center2.X) / 2, (center1.Y + center2.Y) / 2);

            var size    = Math.Max(p_eyes[0].Width, p_eyes[1].Width);
            var newSize = new Size(size, size);
            var topLeft = new Point(center.X - newSize.Width / 2, center.Y - newSize.Height / 2);

            middleEye[i] = new Rect(topLeft, newSize); //眼睛亮度p_eyes[0]
            return true;
        }

        return false;
    }

    //瞳孔检测
    private Mat CreateGrid(int rows, int cols)
    {
        var grid = new Mat(2 * rows - 1, 2 * cols - 1, MatType.CV_32FC2);

        for (var y = 1 - rows; y < rows; y++)
        {
            for (var x = 1 - cols; x < cols; x++)
            {
                var norm          = (float)Math.Sqrt(x * x + y * y);
                if (norm == 0) norm = 1;
                var gridY           = y + rows - 1;
                var gridX           = x + cols - 1;
                grid.Set(gridY, gridX, new Vec2f(y / norm, x / norm));
            }
        }

        return grid;
    }

    private Mat CreateGradient(Mat image)
    {
        var gradX = new Mat();
        var gradY = new Mat();
        Cv2.Sobel(image, gradX, MatType.CV_32F, 1, 0);
        Cv2.Sobel(image, gradY, MatType.CV_32F, 0, 1);

        var gradient = new Mat(image.Size(), MatType.CV_32FC2);
        for (var y = 0; y < image.Rows; y++)
        {
            for (var x = 0; x < image.Cols; x++)
            {
                var gx            = gradX.At<float>(y, x);
                var gy            = gradY.At<float>(y, x);
                var norm          = (float)Math.Sqrt(gx * gx + gy * gy);
                if (norm == 0) norm = 1;
                gradient.Set(y, x, new Vec2f(gy / norm, gx / norm));
            }
        }

        return gradient;
    }

    private Point Locate(Mat image)
    {
        // 转换为浮点图像并归一化
        var floatImage = new Mat();
        image.ConvertTo(floatImage, MatType.CV_32F);
        Cv2.Normalize(floatImage, floatImage, 0, 1, NormTypes.MinMax);

        // 高斯模糊
        var blurred = new Mat();
        Cv2.GaussianBlur(floatImage, blurred, new Size(0, 0), 2); // 修复 Size 参数

        // 边界设置
        const int border = 5;
        var       endY   = image.Rows - border;
        var       endX   = image.Cols - border;

        // 创建网格和梯度
        using var grid     = CreateGrid(image.Rows, image.Cols);
        using var gradient = CreateGradient(floatImage);

        // 计算分数
        // 计算分数
        var scores = new Mat(image.Size(), MatType.CV_32F, new Scalar(0)); // 修复 Scalar.Zero

        for (var cy = border; cy < endY; cy++)
        {
            for (var cx = border; cx < endX; cx++)
            {
                float score   = 0;
                var blurVal = blurred.At<float>(cy, cx);

                var windowSize = 10;
                var startWy    = Math.Max(0, cy          - windowSize);
                var endWy      = Math.Min(image.Rows, cy + windowSize);
                var startWx    = Math.Max(0, cx          - windowSize);
                var endWx      = Math.Min(image.Cols, cx + windowSize);

                for (var y = startWy; y < endWy; y += 2)
                {
                    for (var x = startWx; x < endWx; x += 2)
                    {
                        var   disp = grid.At<Vec2f>(image.Rows - cy - 1 + y, image.Cols - cx - 1 + x);
                        var   grad = gradient.At<Vec2f>(y, x);
                        var dot  = disp.Item0 * grad.Item0 + disp.Item1 * grad.Item1;
                        score += dot * dot;
                    }
                }

                scores.Set(cy, cx, score * (1 - blurVal));
            }
        }

        Cv2.MinMaxLoc(scores, out _, out _, out _, out var maxLoc);
        return maxLoc;
    }

    private bool DetectPupil(Mat lightImage, Mat darkImage)
    {
        // 使用 using 语句确保资源释放
        using var leftEye   = new Mat(darkImage, p_eyes[0]);
        var     leftPupil = Locate(leftEye);

        using var rightEye   = new Mat(darkImage, p_eyes[1]);
        var     rightPupil = Locate(rightEye);

        if (leftPupil == new Point(0, 0) || rightPupil == new Point(0, 0))
        {
            Stats.NoPuilpDetectedCount++;
            return false;
        }

        // 计算绝对坐标
        Result.LeftEyeCenter = new Point(
            p_eyes[0].X + leftPupil.X,
            p_eyes[0].Y + leftPupil.Y);

        Result.RightEyeCenter = new Point(
            p_eyes[1].X + rightPupil.X,
            p_eyes[1].Y + rightPupil.Y);

        // 添加保存图像的代码

        return true;
    }

    private void SaveProcessedEyeImages(Mat leftEye, Mat rightEye, 
                                        Mat leftEyeOriginal, Mat rightEyeOriginal,
                                        long frameNumber)
    {
        if (!Parameters.EnableImageSave) return;

        try
        {
            // 创建保存目录
            var leftEyeDir  = Path.Combine(Parameters.ImageSavePath, "Bright_Pupil");
            var rightEyeDir = Path.Combine(Parameters.ImageSavePath, "Dark_Pupil");

            // 确保目录存在
            Directory.CreateDirectory(leftEyeDir);
            Directory.CreateDirectory(rightEyeDir);
            // 获取坐标信息
            var leftPupilCenter  = Result.LeftEyeCenter  ?? new Point(0, 0);
            var rightPupilCenter = Result.RightEyeCenter ?? new Point(0, 0);
            var leftReflection   = Result.Left;
            var rightReflection  = Result.Right;
            // 生成包含坐标信息的文件名
            var leftCoordInfo =
                $"_P{leftPupilCenter.X:F0}_{leftPupilCenter.Y:F0}_R{leftReflection.X:F0}_{leftReflection.Y:F0}";
            var rightCoordInfo =
                $"_P{rightPupilCenter.X:F0}_{rightPupilCenter.Y:F0}_R{rightReflection.X:F0}_{rightReflection.Y:F0}";

            var timestamp = Parameters.SaveWithTimestamp ? $"_{DateTime.Now:yyyyMMdd_HHmmss}" : "";

            // 修改文件名，加入坐标信息
            var leftOriginalFileName = Path.Combine(leftEyeDir,
                $"left_eye_Bright_Pupil_{frameNumber}{leftCoordInfo}_original{timestamp}.{Parameters.ImageFormat}");
            var leftRenderedFileName = Path.Combine(leftEyeDir,
                $"left_eye_Bright_Pupil_{frameNumber}{leftCoordInfo}_rendered{timestamp}.{Parameters.ImageFormat}");
            var rightOriginalFileName = Path.Combine(rightEyeDir,
                $"left_eye_Dark_Pupil_{frameNumber}{rightCoordInfo}_original{timestamp}.{Parameters.ImageFormat}");
            var rightRenderedFileName = Path.Combine(rightEyeDir,
                $"left_eye_Dark_Pupil_{frameNumber}{rightCoordInfo}_rendered{timestamp}.{Parameters.ImageFormat}");


            // 保存图像
            var imwriteParams = new int[]
            {
                (int)(Parameters.ImageFormat.ToLower() == "png"
                    ? ImwriteFlags.PngCompression
                    : ImwriteFlags.JpegQuality),
                90 // 压缩质量
            };

            // 保存原始图像和渲染后的图像
            leftEyeOriginal.ImWrite(leftOriginalFileName, imwriteParams);
            leftEye.ImWrite(leftRenderedFileName, imwriteParams);
            rightEyeOriginal.ImWrite(rightOriginalFileName, imwriteParams);
            rightEye.ImWrite(rightRenderedFileName, imwriteParams);
        }
        catch (Exception)
        {
            // 暂时忽略保存失败的错误
        }
    }

    // 对眼睛区域进行处理（最大值滤波 + 中值滤波）
    private static Mat ProcessEyeArea(Mat eye, int maxFilterSize, int medianFilterSize)
    {
        maxFilterSize    = maxFilterSize    % 2 == 0 ? maxFilterSize    + 1 : maxFilterSize;
        medianFilterSize = medianFilterSize % 2 == 0 ? medianFilterSize + 1 : medianFilterSize;

        using var kernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new Size(maxFilterSize, maxFilterSize)
        );

        // 使用 using 语句管理资源
        using var maxFiltered    = new Mat();
        using var medianFiltered = new Mat();

        // 直接在目标Mat上操作
        Cv2.Dilate(eye, maxFiltered, kernel);
        Cv2.MedianBlur(eye, medianFiltered, medianFilterSize);

        var result = new Mat();
        Cv2.Subtract(maxFiltered, medianFiltered, result);

        return result;
    }

    // 提取亮斑中心坐标
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
            Stats.NoReflectionDetectedCount++;
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
        //Cv2.Circle(original, new Point(eyeRect.X + cx, eyeRect.Y + cy), 3, new Scalar(0, 0, 255), -1); // 红色圆点
        //this._leftLightPos = new Point(eyeRect.X + cx, eyeRect.Y + cy);
        //this._rightLightPos = new Point(eyeRect.X + cx, eyeRect.Y + cy);
        output = new Point(eyeRect.X + cx, eyeRect.Y + cy);
        return true;


        //// 写入文件
        //var filePath = "output.txt";                                 // 输出文件路径
        //using (var outFile = new StreamWriter(filePath, true)) // 使用追加模式写入文件
        //{
        //    // 设置小数点后3位
        //    outFile.WriteLine($"Highlight Center: ({(eyeRect.X + cx):F3}, {(eyeRect.Y + cy):F3})");
        //}
    }


    // 反射点检测功能实现
    private bool DetectReflection(Mat light_image)
    {
        if (p_eyes.Length != 2)
            return false;
        p_eyes = p_eyes.OrderBy(x => x.X).ToArray();

        // 提取左眼和右眼区域
        var leftEye  = light_image.SubMat(p_eyes[0]);
        var rightEye = light_image.SubMat(p_eyes[1]);

        // 处理眼睛区域
        var resultLeft  = ProcessEyeArea(leftEye, 5, 3);
        var resultRight = ProcessEyeArea(rightEye, 5, 3);

        // 提取左眼和右眼的亮斑中心并绘制
        var leftCenter = Result.LeftEyeCenter;
        if (ExtractBrightSpotCenter(resultLeft, p_eyes[0], light_image, out var left)
            && leftCenter != null)
        {
            var cur = new Point(
                left!.Value.X - leftCenter.Value.X,
                left.Value.Y  - leftCenter.Value.Y);
            if (!(cur.X is 0 && cur.Y is 0))
                Result.Left = cur;
        }
        else
        {
            Stats.NoReflectionDetectedCount++;
            return false;
        }

        var rightCenter = Result.RightEyeCenter;
        if (ExtractBrightSpotCenter(resultRight, p_eyes[1], light_image, out var right)
            && rightCenter != null)
        {
            var cur = new Point(
                right!.Value.X - rightCenter.Value.X,
                right.Value.Y  - rightCenter.Value.Y);
            if (!(cur.X is 0 && cur.Y is 0))
                Result.Right = cur;
        }
        else
        {
            Stats.NoReflectionDetectedCount++;
            return false;
        }

        return true;
    }
}

public class EyeTrack : EyeTrackContext<Point>
{
    public override void DetectSight(Mat thisMat, out Point result)
    {
        throw new NotImplementedException();
    }
}


//public class Demo {


//    public int main() {
//        Funct fun = new Func2();
//        fun.Calculate(1);
//    }


//    public abstract class Funct {
//        public abstract int Calculate(int arg);
//    }


//    public class Fun1 : Funct
//    {
//        public override int Calculate(int arg) => arg * arg + 2 * arg + 1;
//    }

//    public class Func2 : Funct
//    {
//        public override int Calculate(int arg) => 1 / arg;
//     }
//}