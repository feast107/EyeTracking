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
        Debug(DebugHint.Origin, thisMat);
        if (LastMat is not null)
        {
            Stats.TotalFrames++;//开始总帧计数

            if (DetectEyes(thisMat, last_this_center, 1))
            {
                //识别两个眼睛中间区域亮度，当thismat大于lastmat一定值时，这张为亮，否则小于一定值时，这张为暗，否则缓存这张继续检测
                var s = GetBrightnessOfRectUsingSum(LastMat, last_this_center[0]);
                var n = GetBrightnessOfRectUsingSum(thisMat, last_this_center[1]);
                last_this_center[0] = last_this_center[1];
                double yuzhi = 10000; //XXX:999需要测试 173308
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

        p_eyes = EyeCascade.DetectMultiScale(image, 1.1, 4, (HaarDetectionTypes)8, new Size(50, 50));
        if (p_eyes.Length < 2)
        {
            Stats.NoEyesDetectedCount2++;
            return false;
        }
        if (p_eyes.Length > 2)
        {
            Stats.NoEyesDetectedCount++;
            //return false;
        }
        p_eyes = p_eyes.ToArray().OrderBy(e => e.X).ToArray();
        // 计算两个眼睛区域的中心点
        Point center1 = new Point(p_eyes[0].X + p_eyes[0].Width / 2, p_eyes[0].Y + p_eyes[0].Height / 2);
        Point center2 = new Point(p_eyes[1].X + p_eyes[1].Width / 2, p_eyes[1].Y + p_eyes[1].Height / 2);
        Point center = new Point((center1.X + center2.X) / 2, (center1.Y + center2.Y) / 2);

        // 计算新区域的宽度和高度，使其与其中一个眼睛区域一样大
        int size = Math.Max(p_eyes[0].Width, p_eyes[1].Width);
        Size newSize = new Size(size, size);

        Point topLeft = new Point(center.X - newSize.Width / 2, center.Y - newSize.Height / 2);

        // 创建一个新的Rect对象，表示中间区域
        middleEye[i] = new Rect(topLeft, newSize);
        return true;

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
                float gx = gradY.At<float>(y, x);
                float gy = gradX.At<float>(y, x);
                float norm = (float)Math.Sqrt(gx * gx + gy * gy);
                if (norm == 0) norm = 1;
                gradient.Set(y, x, new Vec2f(gy / norm, gx / norm));
            }
        }
        return gradient;
    }

    public Point Locate(Mat image, double sigma = 2, int accuracy = 1)
    {
        using (Mat floatImage = new Mat())
        {
            image.ConvertTo(floatImage, MatType.CV_32F);
            Cv2.Normalize(floatImage, floatImage, 0, 1, NormTypes.MinMax);

            using (Mat blurred = new Mat())
            {
                Cv2.GaussianBlur(floatImage, blurred, new Size(0, 0), sigma);

                int border = 5;
                int startY = border;
                int endY = image.Rows - border;
                int startX = border;
                int endX = image.Cols - border;

                using (Mat grid = CreateGrid(image.Rows, image.Cols))
                using (Mat gradient = CreateGradient(floatImage))
                {
                    Mat scores = Mat.Zeros(image.Size(), MatType.CV_32F);

                    Parallel.For(startY, endY, cy =>
                    {
                        if (cy % accuracy != 0) return ;

                        for (int cx = startX; cx < endX; cx += accuracy)
                        {
                            float score = 0;
                            float blurVal = blurred.At<float>(cy, cx);

                            int windowSize = 15;
                            int startWy = Math.Max(0, cy - windowSize);
                            int endWy = Math.Min(image.Rows, cy + windowSize);
                            int startWx = Math.Max(0, cx - windowSize);
                            int endWx = Math.Min(image.Cols, cx + windowSize);

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
                            scores.Set(cy, cx, score * (1 - blurVal));
                        }
                    });

                    Cv2.MinMaxLoc(scores, out _, out double maxVal, out _, out Point maxLoc);
                    return maxLoc;
                }
            }
        }
    }

    private bool DetectPupil(Mat lightImage, Mat darkImage)
    {
        // 加载眼睛检测器
        //using (var eyeCascade = new CascadeClassifier("haarcascades/haarcascade_eye.xml"))
        //{
        //    if (eyeCascade.Empty())
        //    {
        //        Console.WriteLine("Error: Unable to load eye cascade classifier!");
        //        return false;
        //    }

        //    // 眼睛检测
        //    var eyes = eyeCascade.DetectMultiScale(image, 1.1, 4, HaarDetectionTypes.ScaleImage, new Size(30, 30));
        //    if (eyes.Length != 2)
        //    {
        //        Console.WriteLine("Error: Exactly two eyes are required!");
        //        return false;
        //    }
        // 处理左眼
        using (Mat leftEye = new Mat(darkImage, p_eyes[0]))
        {
            Point leftPupil = Locate(leftEye);
            // 处理右眼
            using (Mat rightEye = new Mat(darkImage, p_eyes[1]))
            {
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
            }
        }
        return true;
        //var leftFin = false;
        //foreach (var eye in p_eyes.OrderBy(x => x.X))
        //{
        //    bool check_circle = false;
        //    var eyeLightRegion = lightImage.SubMat(eye);
        //    var eyeDarkRegion = darkImage.SubMat(eye);

        //    var eyePupilPosition = new Mat();
        //    Cv2.Absdiff(eyeLightRegion, eyeDarkRegion, eyePupilPosition);

        //    var blurredEye = new Mat();
        //    Cv2.GaussianBlur(eyePupilPosition, blurredEye, new Size(5, 5), 1.5);

        //    var edges = new Mat();
        //    Cv2.Canny(blurredEye, edges, 50, 150);

        //    Cv2.FindContours(edges, out var contours, out _, RetrievalModes.External,
        //        ContourApproximationModes.ApproxSimple);
        //    foreach (var contour in contours)
        //    {
        //        if (contour.Length < 5) continue;
        //        var ellipse = Cv2.FitEllipse(contour);
        //        var center = ellipse.Center;
        //        check_circle = true;

        //        center.X += eye.X;
        //        center.Y += eye.Y;

        //        if (!leftFin) Result.LeftEyeCenter = new Point(center.X, center.Y);
        //        else Result.RightEyeCenter = new Point(center.X, center.Y);

        //        leftFin = true;
        //    }
        //    if (!check_circle)
        //    {
        //        Stats.NoPuilpDetectedCount++;
        //        return false;
        //    }
        //}

        //return true;
    }

    // 对眼睛区域进行处理（最大值滤波 + 中值滤波）
    private static Mat ProcessEyeArea(Mat eye, int maxFilterSize, int medianFilterSize)
    {
        // 确保核大小为奇数
        maxFilterSize = maxFilterSize % 2 == 0 ? maxFilterSize + 1 : maxFilterSize;
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

// 显示结果
// Cv2.ImShow("Original Image with Bright Spot Centers", image);
// Cv2.ImShow("Processed Left Eye", resultLeft);
// Cv2.ImShow("Processed Right Eye", resultRight);

// 等待按键
// Cv2.WaitKey(0);



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