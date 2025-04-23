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
    private FastPupilDetector detector = new FastPupilDetector();
    private bool? isLastLight;

    private Rect[] p_eyes = new Rect[2];
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
        using var roi = new Mat(image, rect);
        var mean = Cv2.Mean(roi);
        Cv2.MeanStdDev(roi, out _, out var stdDev);

        // 根据均值和标准差计算动态阈值
        return mean[0] * rect.Width * rect.Height * 0.12 + stdDev[0] * rect.Width * rect.Height * 0.15;
    }

    public override void DetectSight(Mat flipped, out EyeDetectResult? result)
    {
        Mat thisMat = new Mat();
        Cv2.Flip(flipped, thisMat, FlipMode.Y); // 或者 FlipMode.X 表示上下翻转
        //Debug(DebugHint.Origin, thisMat);
        if (LastMat is not null)
        {
            Stats.TotalFrames++;

            if (DetectEyes(thisMat, last_this_center, 1))
            {
                var s = GetBrightnessOfRectUsingSum(LastMat, last_this_center[0]);
                var n = GetBrightnessOfRectUsingSum(thisMat, last_this_center[1]);
                last_this_center[0] = last_this_center[1];

                var dynamicThreshold = CalculateDynamicThreshold(thisMat, last_this_center[1]);
                var brightnessDiff = Math.Abs(s[0] - n[0]);

                if (brightnessDiff > dynamicThreshold)
                {
                    isLastLight = s[0] > n[0];
                    //调试
                    var dark = isLastLight.Value ? LastMat : thisMat;
                    var light = isLastLight.Value ? thisMat : LastMat;
                    if (DetectPupil(light, dark))
                    {
                        if (DetectReflection(light))
                        {
                            Stats.SuccessFrames++;
                            using var lightEyeMat = new Mat(light, p_eyes[0]);
                            using var darkEyeMat = new Mat(dark, p_eyes[0]);
                            using var right_lightEyeMat = new Mat(light, p_eyes[1]);
                            using var right_darkEyeMat = new Mat(dark, p_eyes[1]);

                            using var lightEyeOriginal = lightEyeMat.Clone();
                            using var darkEyeOriginal = darkEyeMat.Clone();
                            using var right_lightEyeOriginal = right_lightEyeMat.Clone();
                            using var right_darkEyeOriginal = right_darkEyeMat.Clone();

                            var leftCenter = (Point)Result.LeftEyeCenter;

                            var leftEyeCenterInSub = new Point(
                                leftCenter.X - p_eyes[0].X,
                                leftCenter.Y - p_eyes[0].Y
                            );
                            var leftPointInSub = new Point(
                                Result.Left.X + leftEyeCenterInSub.X,
                                Result.Left.Y + leftEyeCenterInSub.Y
                            );

                            Cv2.Circle(darkEyeMat, leftPointInSub, 1, Scalar.Green, -1);
                            Cv2.Circle(lightEyeMat, leftPointInSub, 1, Scalar.Green, -1);
                            Cv2.Circle(darkEyeMat, leftEyeCenterInSub, 1, Scalar.White, -1);
                            Cv2.Circle(lightEyeMat, leftEyeCenterInSub, 1, Scalar.White, -1);

                            //SaveProcessedEyeImages(lightEyeMat, darkEyeMat, lightEyeOriginal, darkEyeOriginal,
                            //    Stats.TotalFrames);

                            var right_Center = (Point)Result.RightEyeCenter;

                            var right_EyeCenterInSub = new Point(
                                right_Center.X - p_eyes[1].X,
                                right_Center.Y - p_eyes[1].Y
                            );
                            var right_PointInSub = new Point(
                                Result.Right.X + right_EyeCenterInSub.X,
                                Result.Right.Y + right_EyeCenterInSub.Y
                            );

                            Cv2.Circle(right_darkEyeMat, right_PointInSub, 1, Scalar.Green, -1);
                            Cv2.Circle(right_lightEyeMat, right_PointInSub, 1, Scalar.Green, -1);
                            Cv2.Circle(right_darkEyeMat, right_EyeCenterInSub, 1, Scalar.White, -1);
                            Cv2.Circle(right_lightEyeMat, right_EyeCenterInSub, 1, Scalar.White, -1);

                            //SaveProcessedEyeImages(lightEyeMat, darkEyeMat, lightEyeOriginal, darkEyeOriginal,
                            //    Stats.TotalFrames);

                            Debug(DebugHint.Subtraction, lightEyeMat);
                            Debug(DebugHint.Output, darkEyeMat);
                            Debug(DebugHint.Debug_right_light, right_lightEyeMat);
                            Debug(DebugHint.Debug_right_dark, right_darkEyeMat);

                            Debug(DebugHint.Bin_Subtraction, thisMat);
                        }
                    }
                }
                else
                {
                    Stats.NoCheckLightCount++;
                }
            }

            Trace(0, "总共处理" + Stats.TotalFrames);
            Trace((KeyedEyeDetectTrace.TraceKey)99, "成功" + Stats.SuccessFrames);
            Trace((KeyedEyeDetectTrace.TraceKey)1, "眼睛异常" + Stats.NoEyesDetectedCount);
            Trace((KeyedEyeDetectTrace.TraceKey)5, "无眼睛" + Stats.NoEyesDetectedCount2);
            Trace((KeyedEyeDetectTrace.TraceKey)2, "明暗异常" + Stats.NoCheckLightCount);
            Trace((KeyedEyeDetectTrace.TraceKey)3, "瞳孔异常（总*2）" + Stats.NoPuilpDetectedCount);
            Trace((KeyedEyeDetectTrace.TraceKey)4, "双亮斑异常" + Stats.NoReflectionDetectedCount);
        }
        else
        {
            if (DetectEyes(thisMat, last_this_center, 0))
            {
                LastMat = thisMat;
                result = Result;
                return;
            }
        }

        LastMat?.Dispose();
        LastMat = null;
        result = Result;
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
        //使用属性获取已加载的分类器
        var facesFront = FrontFaceCascade.DetectMultiScale(
            image, 1.1, 3,
            HaarDetectionTypes.ScaleImage,
            new Size(150, 150)
        );

        if (facesFront.Length > 0)
        {
            // 在人脸区域内检测眼睛
            var faceRect = facesFront[0];
            var eyeRegionHeight = (int)(faceRect.Height / 5.7);
            var eyeRegionWidth = (int)(faceRect.Width / 4.3);
            var eyeRegionTop = (int)(faceRect.Y + faceRect.Height / 3.7);

            // 左眼区域
            var leftEyeRect = new Rect(
                (int)(faceRect.X + faceRect.Width / 5.2),
                eyeRegionTop,
                eyeRegionWidth,
                eyeRegionHeight
            );

            // 右眼区域
            var rightEyeRect = new Rect(
                (int)(faceRect.X + faceRect.Width / 1.7),
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
            // 如果人脸检测失败，尝试直接检测眼睛
            var eyes = EyeCascade.DetectMultiScale(
                image, 1.05, 3,
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
            var left = validatedEyes[0];
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

        // 计算中间区域
        if (p_eyes.Length >= 2)
        {
            var center1 = new Point(p_eyes[0].X + p_eyes[0].Width / 2, p_eyes[0].Y + p_eyes[0].Height / 2);
            var center2 = new Point(p_eyes[1].X + p_eyes[1].Width / 2, p_eyes[1].Y + p_eyes[1].Height / 2);
            var center = new Point((center1.X + center2.X) / 2, (center1.Y + center2.Y) / 2);

            var size = Math.Max(p_eyes[0].Width, p_eyes[1].Width);
            var newSize = new Size(size, size);
            var topLeft = new Point(center.X - newSize.Width / 2, center.Y - newSize.Height / 2);

            middleEye[i] = p_eyes[0];//new Rect(topLeft, newSize); //眼睛亮度p_eyes[0]
            return true;
        }

        return false;
    }

    //瞳孔检测
    private Mat CreateGrid(int rows, int cols)
    {
        var grid = new Mat(2 * rows - 1, 2 * cols - 1, MatType.CV_32FC2);

        Parallel.For(0, grid.Rows, y =>
        {
            float actualY = y - rows + 1;
            float ySq = actualY * actualY;

            for (int x = 0; x < grid.Cols; x++)
            {
                float actualX = x - cols + 1;
                float norm = MathF.Sqrt(ySq + actualX * actualX);
                norm = norm == 0 ? 1 : norm;

                grid.Set(y, x, new Vec2f(actualY / norm, actualX / norm));
            }
        });

        return grid;
    }

    private Mat CreateGradient(Mat image)
    {
        using var gradX = new Mat();
        using var gradY = new Mat();
        Cv2.Sobel(image, gradX, MatType.CV_32F, 1, 0, 3, 1, 0, BorderTypes.Default);
        Cv2.Sobel(image, gradY, MatType.CV_32F, 0, 1, 3, 1, 0, BorderTypes.Default);

        var gradient = new Mat(image.Size(), MatType.CV_32FC2);
        var indexer = gradient.GetGenericIndexer<Vec2f>();
        var xIndexer = gradX.GetGenericIndexer<float>();
        var yIndexer = gradY.GetGenericIndexer<float>();

        Parallel.For(0, image.Rows, y =>
        {
            for (int x = 0; x < image.Cols; x++)
            {
                float gx = xIndexer[y, x];
                float gy = yIndexer[y, x];
                float norm = MathF.Sqrt(gx * gx + gy * gy);
                norm = norm == 0 ? 1 : norm;

                indexer[y, x] = new Vec2f(gy / norm, gx / norm);
            }
        });

        return gradient;
    }

    private Point Locate(Mat image, Mat light_image)
    {
        // 预处理保持不变
        using var floatImage = new Mat();
        image.ConvertTo(floatImage, MatType.CV_32F);
        Cv2.Normalize(floatImage, floatImage, 0, 1, NormTypes.MinMax);

        using var blurred = new Mat();
        Cv2.GaussianBlur(floatImage, blurred, new Size(0, 0), 2);

        const int border = 5;
        int rows = image.Rows;
        int cols = image.Cols;
        int endY = rows - border;
        int endX = cols - border;

        // 预计算网格和梯度
        using var grid = CreateGrid(rows, cols);
        using var gradient = CreateGradient(floatImage);

        // 使用索引器替代指针
        var gridIndexer = grid.GetGenericIndexer<Vec2f>();
        var gradIndexer = gradient.GetGenericIndexer<Vec2f>();
        var blurIndexer = blurred.GetGenericIndexer<float>();
        var scores = new Mat(image.Size(), MatType.CV_32F, Scalar.All(0));
        var scoreIndexer = scores.GetGenericIndexer<float>();

        const int windowSize = 10;
        int gridCenterY = rows - 1;
        int gridCenterX = cols - 1;

        Parallel.For(border, endY, cy =>
        {
            for (int cx = border; cx < endX; cx++)
            {
                float score = 0;
                float blurVal = blurIndexer[cy, cx];

                int startWy = Math.Max(0, cy - windowSize);
                int endWy = Math.Min(rows, cy + windowSize);
                int startWx = Math.Max(0, cx - windowSize);
                int endWx = Math.Min(cols, cx + windowSize);

                for (int y = startWy; y < endWy; y += 2)
                {
                    int gridY = gridCenterY - cy + y;
                    for (int x = startWx; x < endWx; x += 2)
                    {
                        int gridX = gridCenterX - cx + x;
                        var disp = gridIndexer[gridY, gridX];
                        var grad = gradIndexer[y, x];

                        float dot = disp.Item0 * grad.Item0 + disp.Item1 * grad.Item1;
                        score += dot * dot;
                    }
                }

                scoreIndexer[cy, cx] = score * (1 - blurVal);
            }
        });

        Cv2.MinMaxLoc(scores, out _, out _, out _, out var maxLoc);
        return maxLoc;
    }

    private bool DetectPupil(Mat lightImage, Mat darkImage)
    {
        // 使用 using 语句确保资源释放
        using var leftEye = new Mat(darkImage, p_eyes[0]);
        using var rightEye = new Mat(darkImage, p_eyes[1]);
        using var leftEye_light = new Mat(lightImage, p_eyes[0]);
        using var rightEye_light = new Mat(lightImage, p_eyes[1]);
#if(false)
        
        var leftPupil = Locate(leftEye, leftEye_light);
        var rightPupil = Locate(rightEye, rightEye_light);
#else
        //用测试类
        var leftPupil = detector.LocateT(leftEye, leftEye_light);
        var rightPupil = detector.LocateT(rightEye, rightEye_light);
#endif

        if (leftPupil == new Point(0, 0) || rightPupil == new Point(0, 0))
        {
            Stats.NoPuilpDetectedCount++;
            return false;
        }


        if (Result.LeftEyeCenter is null || Result.RightEyeCenter is null)
        {
            // 计算绝对坐标
            Result.LeftEyeCenter = new Point(
                p_eyes[0].X + leftPupil.X,
                p_eyes[0].Y + leftPupil.Y);

            Result.RightEyeCenter = new Point(
                p_eyes[1].X + rightPupil.X,
                p_eyes[1].Y + rightPupil.Y);
        }
        else
        {
            double smooth = 0;  // 初始权重

            Point lefteyecenter = new Point(0, 0);
            lefteyecenter = (Point)Result.LeftEyeCenter;
            Point righteyecenter = new Point(0, 0);
            righteyecenter = (Point)Result.RightEyeCenter;
            // 计算绝对坐标
            Result.LeftEyeCenter = new Point(
                (p_eyes[0].X + leftPupil.X) * (1 - smooth) + lefteyecenter.X * smooth,
                (p_eyes[0].Y + leftPupil.Y) * (1 - smooth) + lefteyecenter.Y * smooth);

            Result.RightEyeCenter = new Point(
                (p_eyes[1].X + rightPupil.X) * (1 - smooth) + righteyecenter.X * smooth,
                (p_eyes[1].Y + rightPupil.Y) * (1 - smooth) + righteyecenter.Y * smooth);
        }

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
            var leftEyeDir = Path.Combine(Parameters.ImageSavePath, "Bright_Pupil");
            var rightEyeDir = Path.Combine(Parameters.ImageSavePath, "Dark_Pupil");

            // 确保目录存在
            Directory.CreateDirectory(leftEyeDir);
            Directory.CreateDirectory(rightEyeDir);
            // 获取坐标信息
            var leftPupilCenter = Result.LeftEyeCenter ?? new Point(0, 0);
            var rightPupilCenter = Result.RightEyeCenter ?? new Point(0, 0);
            var leftReflection = Result.Left;
            var rightReflection = Result.Right;
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
        maxFilterSize = maxFilterSize % 2 == 0 ? maxFilterSize + 1 : maxFilterSize;
        medianFilterSize = medianFilterSize % 2 == 0 ? medianFilterSize + 1 : medianFilterSize;

        using var kernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new Size(maxFilterSize, maxFilterSize)
        );

        // 使用 using 语句管理资源
        using var maxFiltered = new Mat();
        using var medianFiltered = new Mat();

        // 直接在目标Mat上操作
        Cv2.Dilate(eye, maxFiltered, kernel);
        Cv2.MedianBlur(eye, medianFiltered, medianFilterSize);

        var result = new Mat();
        Cv2.Subtract(maxFiltered, medianFiltered, result);

        return result;
    }
    // 定义亮斑结构
    public struct BrightSpot
    {
        public Point Center;
        public double Area;
        public double DistanceToPupil;

        public BrightSpot(Point center, double area, double distanceToPupil = 0)
        {
            Center = center;
            Area = area;
            DistanceToPupil = distanceToPupil;
        }
    }

    // 提取亮斑中心坐标
    private bool ExtractBrightSpotCenter(Mat result, Rect eyeRect, Point pupilCenter, out Point? output)
    {
        output = null;
        using var binary = new Mat();
        Cv2.Threshold(result, binary, 50, 255, ThresholdTypes.Binary);

        Cv2.FindContours(binary, out var contours, out _, RetrievalModes.External,
            ContourApproximationModes.ApproxSimple);

        if (contours.Length == 0)
        {
            return false;
        }

        var brightSpots = new List<BrightSpot>();
        foreach (var contour in contours)
        {
            var m = Cv2.Moments(contour);
            if (m.M00 == 0) continue;

            var cx = m.M10 / m.M00;
            var cy = m.M01 / m.M00;
            var absoluteCenter = new Point(eyeRect.X + cx, eyeRect.Y + cy);

            // 计算与瞳孔中心的距离
            var distance = Math.Sqrt(
                Math.Pow(absoluteCenter.X - pupilCenter.X, 2) +
                Math.Pow(absoluteCenter.Y - pupilCenter.Y, 2));

            brightSpots.Add(new BrightSpot(absoluteCenter, Cv2.ContourArea(contour), distance));
        }

        if (brightSpots.Count == 0) return false;

        // 选择距离瞳孔中心最近的亮斑
        var closestSpot = brightSpots.OrderBy(spot => spot.DistanceToPupil).First();

        output = closestSpot.Center;
        return true;
    }


    // 反射点检测功能实现
    private bool DetectReflection(Mat light_image)
    {
        if (p_eyes.Length != 2)
            return false;
        p_eyes = p_eyes.OrderBy(x => x.X).ToArray();

        using var leftEye = light_image.SubMat(p_eyes[0]);
        using var rightEye = light_image.SubMat(p_eyes[1]);
        using var resultLeft = ProcessEyeArea(leftEye, 3, 3);
        using var resultRight = ProcessEyeArea(rightEye, 3, 3);
        bool ret1 = true;
        bool ret2 = true;

        var leftCenter = Result.LeftEyeCenter;
        var rightCenter = Result.RightEyeCenter;

        if (leftCenter == null || rightCenter == null) return false;
        ret1 = ExtractBrightSpotCenter(resultLeft, p_eyes[0], leftCenter.Value, out var left);
        ret2 = ExtractBrightSpotCenter(resultRight, p_eyes[1], rightCenter.Value, out var right);
        if (ret1)
        {
            var cur = new Point(
                left!.Value.X - leftCenter.Value.X,
                left.Value.Y - leftCenter.Value.Y);
            if (!(cur.X is 0 && cur.Y is 0))
                if (cur.Y < 7 && cur.X < 7 && cur.X > -7)
                    Result.Left = cur;
        }

        if (ret2)
        {
            var cur = new Point(
                right!.Value.X - rightCenter.Value.X,
                right.Value.Y - rightCenter.Value.Y);
            if (!(cur.X is 0 && cur.Y is 0))
                if (cur.Y < 7 && cur.X < 7 && cur.X > -7)
                    Result.Right = cur;
        }

        if (!ret1 && !ret2)
        {
            Stats.NoReflectionDetectedCount++;
            return true;
        }

        return true;
    }

    public class EyeTrack : EyeTrackContext<Point>
    {
        public override void DetectSight(Mat thisMat, out Point result)
        {
            throw new NotImplementedException();
        }
    }
}