using System.Diagnostics.CodeAnalysis;
using System.Reflection.Emit;
using EyeTracking.Extensions;
using OpenCvSharp;
using static System.Net.Mime.MediaTypeNames;
using Cv2 = OpenCvSharp.Cv2;

namespace EyeTracking;

public class DetectionStatistics
{
    public long NoEyesDetectedCount//眼睛识别异常
    {
        get;
        set;
    } = 0;
    public long NoEyesDetectedCount2//眼睛识别异常
    {
        get;
        set;
    } = 0;
    public long NoCheckLightCount//明瞳监测异常
    {
        get;
        set;
    } = 0;
    public long NoPuilpDetectedCount//瞳孔监测异常
    {
        get;
        set;
    } = 0;
    public long NoReflectionDetectedCount//亮斑识别异常
    {
        get;
        set;
    } = 0;
    public long TotalFrames
    { 
        get; 
        set; 
    } = 0;//总帧数
    public long SuccessFrames
    {
        get;
        set;
    } = 0;
    public double ErrorRate => (NoEyesDetectedCount + NoReflectionDetectedCount + NoPuilpDetectedCount) / (double)TotalFrames * 100;
}
public class LatestTrackContext : EyeTrackContext<EyeDetectResult>
{
    private bool? isLastLight;

    private Rect[] p_eyes = new Rect[2];
    private Rect[] last_this_center = new Rect[2];

private static readonly string XmlPath =
        Path.Combine(AppContext.BaseDirectory, "Resources/Haarcascade/haarcascade_eye.xml");
    private static readonly string FrontFaceXmlPath = 
        Path.Combine(AppContext.BaseDirectory, "Resources/Haarcascade/haarcascade_frontalface_alt.xml");
    private static readonly string ProfileFaceXmlPath = 
        Path.Combine(AppContext.BaseDirectory, "Resources/Haarcascade/haarcascade_profileface.xml");

    private CascadeClassifier? _frontFaceCascade;
    private CascadeClassifier? _profileFaceCascade;

    [field: AllowNull, MaybeNull]
    private CascadeClassifier FrontFaceCascade
    {
        get
        {
            if (_frontFaceCascade is not null) return _frontFaceCascade;
            _frontFaceCascade = new CascadeClassifier();
            if (_frontFaceCascade.Load(FrontFaceXmlPath)) return _frontFaceCascade;
            throw new FileLoadException("Error: Unable to load front face cascade classifier!");
        }
    }

    [field: AllowNull, MaybeNull]
    private CascadeClassifier ProfileFaceCascade
    {
        get
        {
            if (_profileFaceCascade is not null) return _profileFaceCascade;
            _profileFaceCascade = new CascadeClassifier();
            if (_profileFaceCascade.Load(ProfileFaceXmlPath)) return _profileFaceCascade;
            throw new FileLoadException("Error: Unable to load profile face cascade classifier!");
        }
    }
    private EyeDetectResult Result { get; set; } = new();

    //做统计
    public DetectionStatistics Stats
    {
        get;
        private set;
    } = new DetectionStatistics();

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
    public override void DetectSight(Mat thisMat, out EyeDetectResult? result)
    {
        Debug(DebugHint.Origin, thisMat);          // 原始图像
        if (LastMat is not null)
        {
            Stats.TotalFrames++;//开始总帧计数

            if (DetectEyes(thisMat, last_this_center, 1))
            {
                //识别两个眼睛中间区域亮度，当thismat大于lastmat一定值时，这张为亮，否则小于一定值时，这张为暗，否则缓存这张继续检测
                var s = GetBrightnessOfRectUsingSum(LastMat, last_this_center[0]);
                var n = GetBrightnessOfRectUsingSum(thisMat, last_this_center[1]);
                last_this_center[0] = last_this_center[1];
                double yuzhi = 20000; //XXX:999需要测试 173308
                double debug_num = s[0] - n[0];
                if (s[0] - n[0] > yuzhi || n[0] - s[0] > yuzhi)
                {
                    isLastLight = s[0] > n[0];
                    var light = isLastLight.Value ? LastMat : thisMat;
                    var dark = isLastLight.Value ? thisMat : LastMat;
                    if (DetectPupil(light, dark))
                    {
                        if (DetectReflection(light))
                        {
                            Stats.SuccessFrames++;
                            // 提取左右眼区域
                            Mat leftEyeMat = new Mat(light, p_eyes[0]);
                            Mat rightEyeMat = new Mat(light, p_eyes[1]);
                            Point leftCenter = (Point)Result.LeftEyeCenter;
                            Point rightCenter = (Point)Result.RightEyeCenter;

                            // 将点坐标转换为相对子Mat的坐标
                            Point leftEyeCenterInSub = new Point(
                                leftCenter.X - p_eyes[0].X,
                                leftCenter.Y - p_eyes[0].Y
                            );
                            Point leftPointInSub = new Point(
                                Result.Left.X + leftEyeCenterInSub.X,
                                Result.Left.Y + leftEyeCenterInSub.Y
                            );
                            Point rightEyeCenterInSub = new Point(
                                rightCenter.X - p_eyes[1].X,
                                rightCenter.Y - p_eyes[1].Y
                            );
                            Point rightPointInSub = new Point(
                                Result.Right.X + rightEyeCenterInSub.X,
                                Result.Right.Y + rightEyeCenterInSub.Y
                            );

                            // 在左眼Mat上绘制点
                            Cv2.Circle(leftEyeMat, leftEyeCenterInSub, 2, Scalar.Green, -1);
                            Cv2.Circle(leftEyeMat, leftPointInSub, 1, Scalar.White, -1);

                            // 在右眼Mat上绘制点
                            Cv2.Circle(rightEyeMat, rightEyeCenterInSub, 2, Scalar.Green, -1);
                            Cv2.Circle(rightEyeMat, rightPointInSub, 1, Scalar.White, -1);

                            // 调试显示Candidate
                            Debug(DebugHint.Subtraction, leftEyeMat);  // 显示左眼区域
                            Debug(DebugHint.Output, rightEyeMat);      // 显示右眼区域及绿点
                            Debug(DebugHint.Bin_Subtraction, thisMat);          // 成功图像
                        }
                    }
                }
                else
                {
                    Stats.NoCheckLightCount++;
                }
            }
            
            Trace((KeyedEyeDetectTrace.TraceKey)0, "总共处理" + Stats.TotalFrames.ToString());
            Trace((KeyedEyeDetectTrace.TraceKey)99, "成功" + Stats.SuccessFrames.ToString());
            Trace((KeyedEyeDetectTrace.TraceKey)1, "眼睛异常" + (Stats.NoEyesDetectedCount).ToString());
            Trace((KeyedEyeDetectTrace.TraceKey)5, "眼睛没有" + (Stats.NoEyesDetectedCount2).ToString());
            Trace((KeyedEyeDetectTrace.TraceKey)2, "明暗异常" + (Stats.NoCheckLightCount).ToString());
            Trace((KeyedEyeDetectTrace.TraceKey)3, "瞳孔异常" + (Stats.NoPuilpDetectedCount).ToString());
            Trace((KeyedEyeDetectTrace.TraceKey)4, "亮斑异常" + (Stats.NoReflectionDetectedCount).ToString());
            LastMat.Dispose();
        }
        else
        {
            if(!DetectEyes(thisMat, last_this_center, 0))
            {
                result = Result;
                return;
            }
        }

        result = Result;
        LastMat = thisMat;
    }

    //
    public class KeyedEyeDetectTrace : EyeDetectTrace {
        public enum TraceKey {
            All,
            NoEyesDetectedCount,//眼睛
            NoCheckLightCount,//明瞳
            NoPuilpDetectedCount,//瞳孔监测异常
            NoReflectionDetectedCount//亮斑识别异常
        }

        public required TraceKey Key;
    }

    private void Trace(KeyedEyeDetectTrace.TraceKey key, string content)
    {
        var exist = Traces.OfType<KeyedEyeDetectTrace>().FirstOrDefault(x => x.Key == key);
        if(exist == null)
        {
            Traces.Add(new KeyedEyeDetectTrace() { Key = key, Content = content });
        }
        else
        {
            exist.Content = content;
        }
    }
    private Scalar GetBrightnessOfRectUsingSum(Mat image, Rect rect)
    {

        // 使用Rect对象从图像中裁剪出区域
        Mat croppedRegion = new Mat(image, rect);              
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
            var faceRect = facesFront[0];
            var eyeRegionHeight = faceRect.Height / 3;
            var eyeRegionWidth = faceRect.Width / 3;
            var eyeRegionTop = faceRect.Y + faceRect.Height / 4;

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
                var faceRect = facesProfile[0];
                var eyeRegionHeight = faceRect.Height / 3;
                var eyeRegionWidth = faceRect.Width / 3;
                var eyeRegionTop = faceRect.Y + faceRect.Height / 4;

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
                    p_eyes = new[] { new Rect(eyeRect.X + eyes[0].X, eyeRect.Y + eyes[0].Y, eyes[0].Width, eyes[0].Height) };
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
                    float aspectRatio = (float)eye.Width / eye.Height;
                    if (aspectRatio < 0.4f || aspectRatio > 2.5f) continue;

                    // 验证区域大小相对于图像
                    float relativeSize = (float)(eye.Width * eye.Height) / (image.Rows * image.Cols);
                    if (relativeSize < 0.01f || relativeSize > 0.15f) continue;

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
                float sizeRatio = (float)(left.Width * left.Height) / (right.Width * right.Height);
                if (sizeRatio < 0.5f || sizeRatio > 2.0f)
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

                p_eyes = new[] { left, right };
            }
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

            middleEye[i] = p_eyes[0];//new Rect(topLeft, newSize);
            return true;
        }

        return false;
    }

    //瞳孔检测
    private Mat CreateGrid(int rows, int cols)
    {
        var grid = new Mat(2 * rows - 1, 2 * cols - 1, MatType.CV_32FC2);

        for (int y = 1 - rows; y < rows; y++)
        {
            for (int x = 1 - cols; x < cols; x++)
            {
                float norm = (float)Math.Sqrt(x * x + y * y);
                if (norm == 0) norm = 1;
                int gridY = y + rows - 1;
                int gridX = x + cols - 1;
                grid.Set(gridY, gridX, new Vec2f(y / norm, x / norm));
            }
        }
        return grid;
    }

    private Mat CreateGradient(Mat image)
    {
        Mat gradX = new Mat();
        Mat gradY = new Mat();
        
        Cv2.Sobel(image, gradX, MatType.CV_32F, 1, 0);
        Cv2.Sobel(image, gradY, MatType.CV_32F, 0, 1);

        Mat gradient = new Mat(image.Size(), MatType.CV_32FC2);
        for (int y = 0; y < image.Rows; y++)
        {
            for (int x = 0; x < image.Cols; x++)
            {
                float gx = gradX.At<float>(y, x);
                float gy = gradY.At<float>(y, x);
                float norm = (float)Math.Sqrt(gx * gx + gy * gy);
                if (norm == 0) norm = 1;
                gradient.Set(y, x, new Vec2f(gy / norm, gx / norm));
            }
        }
        return gradient;
    }

    public Point Locate(Mat image, double sigma = 2, int accuracy = 1)
    {
        using var floatImage = new Mat();
        image.ConvertTo(floatImage, MatType.CV_32F);
        Cv2.Normalize(floatImage, floatImage, 0, 1, NormTypes.MinMax);

        using var blurred = new Mat();
        Cv2.GaussianBlur(floatImage, blurred, new Size(0, 0), sigma, sigma);

        int border = 5;
        int startY = border;
        int endY = image.Rows - border;
        int startX = border;
        int endX = image.Cols - border;

        // 预计算网格和梯度，避免重复计算
        using var grid = CreateGrid(image.Rows, image.Cols);
        using var gradient = CreateGradient(blurred);
        
        // 使用单个数组存储分数，避免Mat操作的开销
        float[,] scores = new float[image.Rows, image.Cols];
        object lockObj = new object();
        Point maxLoc = new Point(0, 0);
        float maxScore = float.MinValue;

        // 优化并行计算
        int threadCount = Environment.ProcessorCount;
        int rowsPerThread = (endY - startY) / threadCount;
        
        Parallel.For(0, threadCount, threadIndex =>
        {
            int localStartY = startY + threadIndex * rowsPerThread;
            int localEndY = threadIndex == threadCount - 1 ? endY : localStartY + rowsPerThread;
            
            float localMaxScore = float.MinValue;
            Point localMaxLoc = new Point(0, 0);

            for (int cy = localStartY; cy < localEndY; cy += accuracy)
            {
                for (int cx = startX; cx < endX; cx += accuracy)
                {
                    float score = 0;
                    float blurVal = blurred.At<float>(cy, cx);

                    // 优化窗口大小计算
                    int windowSize = 10; // 减小窗口大小以提高性能
                    int startWy = Math.Max(0, cy - windowSize);
                    int endWy = Math.Min(image.Rows, cy + windowSize);
                    int startWx = Math.Max(0, cx - windowSize);
                    int endWx = Math.Min(image.Cols, cx + windowSize);

                    // 使用向量化计算
                    for (int y = startWy; y < endWy; y++)
                    {
                        for (int x = startWx; x < endWx; x++)
                        {
                            var disp = grid.At<Vec2f>(image.Rows - cy - 1 + y, image.Cols - cx - 1 + x);
                            var grad = gradient.At<Vec2f>(y, x);
                            float dot = disp.Item0 * grad.Item0 + disp.Item1 * grad.Item1;
                            score += dot * dot;  
                        }
                    }

                    score *= (1 - blurVal);
                    scores[cy, cx] = score;

                    if (score > localMaxScore)
                    {
                        localMaxScore = score;
                        localMaxLoc = new Point(cx, cy);
                    }
                }
            }

            // 合并局部最大值
            lock (lockObj)
            {
                if (localMaxScore > maxScore)
                {
                    maxScore = localMaxScore;
                    maxLoc = localMaxLoc;
                }
            }
        });

        return maxLoc;
    }

    private bool DetectPupil(Mat lightImage, Mat darkImage)
    {
        // 使用 using 语句确保资源释放
        using var leftEye = new Mat(darkImage, p_eyes[0]);
        Point leftPupil = Locate(leftEye);
        
        using var rightEye = new Mat(darkImage, p_eyes[1]);
        Point rightPupil = Locate(rightEye);

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

        return true;
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
        var leftEye = light_image.SubMat(p_eyes[0]);
        var rightEye = light_image.SubMat(p_eyes[1]);

        // 处理眼睛区域
        var resultLeft = ProcessEyeArea(leftEye, 5, 3);
        var resultRight = ProcessEyeArea(rightEye, 5, 3);

        // 提取左眼和右眼的亮斑中心并绘制
        var leftCenter = Result.LeftEyeCenter;
        if (ExtractBrightSpotCenter(resultLeft, p_eyes[0], light_image, out var left)
        && leftCenter != null)
        {
            var cur = new Point(
                left!.Value.X - leftCenter.Value.X,
                left.Value.Y - leftCenter.Value.Y);
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
                right.Value.Y - rightCenter.Value.Y);
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