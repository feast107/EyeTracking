using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

public class FastPupilDetector
{
    private Queue<Point> _positionBuffer = new Queue<Point>();
    private const int BUFFER_SIZE = 5;
    private Mat _visualization;

    public Point LocateT3(Mat darkPupilImage, Mat lightPupilImage)
    {// 1. 转为灰度图
        Mat gray = new Mat();
        if (darkPupilImage.Channels() > 1)
            Cv2.CvtColor(darkPupilImage, gray, ColorConversionCodes.BGR2GRAY);
        else
            darkPupilImage.CopyTo(gray);

        // 2. 高斯模糊降噪
       // Cv2.GaussianBlur(gray, gray, new Size(5, 5), 1);

        // 3. 霍夫圆检测
        var circles = Cv2.HoughCircles(
            gray,
            HoughModes.Gradient,
            dp: 2,           // 分辨率比例
            minDist: 20,     // 圆之间的最小距离
            param1: 100,     // Canny边缘检测阈值
            param2: 30,      // 圆心累加阈值（越小检测越多）
            minRadius: 2,   // 最小半径
            maxRadius: 10    // 最大半径
        );

        // 4. 绘制检测到的所有圆
        if (circles != null && circles.Length > 0)
        {
            // 创建一个掩膜
            var mask = new Mat();
            mask = darkPupilImage.Clone();

            // 遍历所有检测到的圆
            for (int i = 0; i < circles.Length; i++)
            {
                var circle = circles[i];
                Point center = new Point((int)circle.Center.X, (int)circle.Center.Y);
                int radius = (int)circle.Radius;

                // 在原始图像上绘制圆（红色边框）
                Cv2.Circle(mask, center, radius, Scalar.Red, 2);
                Debug.WriteLine($"圆 {i + 1}: 中心={center}, 半径={radius}, 圆形度={radius:F2}");
            }

            // 显示掩膜（可选）
            DebugMat(mask);

            // 返回第一个圆的中心（或根据需求调整）
            Point firstCenter = new Point((int)circles[0].Center.X, (int)circles[0].Center.Y);
            return firstCenter;
        }
        return new Point(0, 0); // 未检测到圆时返回默认值
    }

    public Point LocateT2(Mat darkPupilImage, Mat lightPupilImage)
    {
        const int minPupilDiameter = 70;
        const int maxPupilDiameter = 100;
        const int morphKernelSize = 3;

        try
        {
            // === 1. 专用瞳孔增强 ===
            Debug.WriteLine($"图像类型: {darkPupilImage.Type()}"); // 应输出CV_8UC1或CV_16UC1
            if (darkPupilImage.Empty())
                throw new ArgumentException("输入图像为空");
            Mat enhanced = new Mat();
            if (darkPupilImage.Channels() > 1)
                Cv2.CvtColor(darkPupilImage, enhanced, ColorConversionCodes.BGR2GRAY);
            else
                darkPupilImage.ConvertTo(enhanced, MatType.CV_8UC1);

            // 1.1 高斯拉普拉斯增强边缘 (LoG)
            // 保留浮点结果
            Mat laplacian = new Mat();
            Cv2.Laplacian(enhanced, laplacian, MatType.CV_32F);

            // 归一化到可视范围
            Cv2.Normalize(laplacian, enhanced, 0, 255, NormTypes.MinMax);
            enhanced.ConvertTo(enhanced, MatType.CV_8UC1);
            Cv2.GaussianBlur(enhanced, enhanced, new Size(5, 5), 0);

            // 1.2 降低 CLAHE 的 clipLimit，避免过度增强
            using (var clahe = Cv2.CreateCLAHE(clipLimit: 8.0, tileGridSize: new Size(8, 8))) // 降低 clipLimit
                clahe.Apply(enhanced, enhanced);

            // === 2. 圆形检测优化 ===
            Mat binary = new Mat();
            //Cv2.Threshold(enhanced, binary, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
            binary = enhanced.Clone();

            // 2.1 形态学操作强化圆形
            var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(morphKernelSize, morphKernelSize));
            Cv2.MorphologyEx(binary, binary, MorphTypes.Close, kernel);

            DebugMat(binary);

            // === 3. 霍夫圆检测 (专为70-100像素优化) ===
            var circles = Cv2.HoughCircles(
                binary,
                HoughModes.Gradient,
                dp: 1.2,  // 提高检测精度
                minDist: maxPupilDiameter * 1.5, // 避免重叠检测
                param1: 100, // Canny边缘阈值
                param2: 30,  // 圆心累加阈值
                minRadius: minPupilDiameter / 2,
                maxRadius: maxPupilDiameter / 2
            );

            // 3.1 选择最佳候选圆
            if (circles != null && circles.Length > 0)
            {
                // 优先选择最接近图像中心的圆
                var centerPoint = new Point2f(enhanced.Width / 2f, enhanced.Height / 2f);
                var bestCircle = circles.OrderBy(c => Distance(c.Center, centerPoint)).First();

                // 二次验证：检查圆形度
                var mask = new Mat(binary.Size(), MatType.CV_8UC1, Scalar.Black);
                Cv2.Circle(mask, (Point)bestCircle.Center, (int)bestCircle.Radius, Scalar.White, -1);

                double circleArea = Math.PI * bestCircle.Radius * bestCircle.Radius;
                double actualArea = Cv2.CountNonZero(mask);
                double circularity = actualArea / circleArea;

                if (circularity > 0.7) // 圆形度阈值
                    return new Point((int)bestCircle.Center.X, (int)bestCircle.Center.Y);
            }

            // === 4. 备用方案：轮廓检测 ===
            var contours = Cv2.FindContoursAsArray(binary, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
            var validContours = contours.Where(c =>
            {
                double area = Cv2.ContourArea(c);
                double equivalentRadius = Math.Sqrt(area / Math.PI);
                return equivalentRadius >= minPupilDiameter / 2 &&
                       equivalentRadius <= maxPupilDiameter / 2;
            }).ToList();

            if (validContours.Any())
            {
                var largestContour = validContours.OrderByDescending(c => Cv2.ContourArea(c)).First();
                var moments = Cv2.Moments(largestContour);
                return new Point((int)(moments.M10 / moments.M00), (int)(moments.M01 / moments.M00));
            }

            return new Point(0, 0);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"瞳孔定位失败: {ex.Message}");
            return new Point(0, 0);
        }
    }

    // 计算两点间距离
    private static float Distance(Point2f p1, Point2f p2)
    {
        return (float)Math.Sqrt(Math.Pow(p1.X - p2.X, 2) + Math.Pow(p1.Y - p2.Y, 2));
    }

    public Point LocateT(Mat darkPupilImage, Mat lightPupilImage)
    {
        //Cv2.EqualizeHist(darkPupilImage, darkPupilImage);

        // 3. 计算差异图（亮瞳-暗瞳）
        // 使用加权减法而非简单差值（暗瞳环境光补偿）
        //Mat binaryDarkPupil = new Mat();
        //Cv2.AdaptiveThreshold(darkPupilImage, binaryDarkPupil, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.BinaryInv, 11, 2);
        //using var diff_pre = new Mat();
        //Cv2.AddWeighted(lightPupilImage, 1.0, darkPupilImage, -0.8, 0, diff_pre);

        //Mat diff = new Mat();
        //Cv2.BitwiseAnd(diff_pre, binaryDarkPupil, diff);

        //Cv2.GaussianBlur(diff, diff, new Size(3, 3), 0);
        //Cv2.Threshold(diff, diff, 32, 255, ThresholdTypes.Binary);

        // 初始化参数
        Mat binaryDarkPupil = new Mat();
        Mat gaussianBlurDark = new Mat();
        double darkWeight = -0.8;  // 初始权重
        int maxAttempts = 10;       // 最大尝试次数
        int minWhiteArea = 20;    // 您需要设定的最小白色区域面积阈值
        int maxWhiteArea = 77;    // 您需要设定的最大白色区域面积阈值
        int currentAttempt = 0;
        int currentBlackHandle = 0;
        int threshold_v = 22;

        int newBlockSize = 27;   // 减小窗口大小
        double newC = 52;           // 减小常数C
        Mat diff = new Mat();
        using var diff_pre = new Mat();
        using var diff_sec = new Mat();
        Cv2.GaussianBlur(darkPupilImage, gaussianBlurDark, new Size(7, 7), 1);

        while (currentAttempt < maxAttempts)
        {
            currentAttempt++;
            // 第一步：创建暗瞳孔的二进制掩码
            binaryDarkPupil = new Mat();
            Cv2.AdaptiveThreshold(gaussianBlurDark, binaryDarkPupil, 255,
                             AdaptiveThresholdTypes.MeanC,
                             ThresholdTypes.BinaryInv, newBlockSize, newC);
            double blackArea = Cv2.CountNonZero(binaryDarkPupil);
            //string windowName = "Pupil Detection (Press ESC to close)";
            //Cv2.NamedWindow(windowName, WindowFlags.Normal);
            //Cv2.ResizeWindow(windowName, 1200, 800);
            //using var resizedImage = new Mat();
            //Cv2.Resize(binaryDarkPupil, resizedImage, new Size(0, 0), fx: 2.0, fy: 2.0, InterpolationFlags.Linear);
            //Cv2.ImShow(windowName, resizedImage);
            //Cv2.WaitKey(0);
            if (blackArea >= 20)
            {
                if (blackArea <= 87)
                {
                    break;
                }
                else
                {
                    newC += 3;
                }

            }
            else
            {
                newC -= 2;
            }
        }

        while (currentAttempt < maxAttempts)
        {
            currentAttempt++;

            //// 1. 检测并标记反光点（高亮区域）
            //Mat mask = new Mat();
            //Cv2.Threshold(lightPupilImage, mask, 250, 255, ThresholdTypes.Binary); // 250为亮度阈值，可根据实际情况调整

            //// 2. 对暗瞳孔图像也进行同样处理（确保一致性）
            //Mat maskDark = new Mat();
            //Cv2.Threshold(darkPupilImage, maskDark, 250, 255, ThresholdTypes.Binary);

            //// 3. 合并两个掩模
            //Cv2.BitwiseOr(mask, maskDark, mask);

            //// 4. 对掩模进行膨胀操作，确保覆盖整个反光区域
            //Cv2.Dilate(mask, mask, Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(5, 5)));

            //// 5. 使用邻域均值填充反光区域
            //Mat inpaintedLight = new Mat();
            //Cv2.Inpaint(lightPupilImage, mask, inpaintedLight, 3, InpaintMethod.NS);

            //Mat inpaintedDark = new Mat();
            //Cv2.Inpaint(darkPupilImage, mask, inpaintedDark, 3, InpaintMethod.NS);

            //// 6. 现在可以安全地进行加权相加
            //Cv2.AddWeighted(inpaintedLight, 1.0, inpaintedDark, darkWeight, 0, diff_pre);

            // 第二步：加权混合图像
            Cv2.AddWeighted(lightPupilImage, 1.0, darkPupilImage, darkWeight, 0, diff_pre);

            // 第三步：与暗瞳孔掩码进行与运算
            Cv2.BitwiseAnd(diff_pre, binaryDarkPupil, diff_sec);

            // 第四步：高斯模糊和二值化
            Cv2.GaussianBlur(diff_pre, diff_sec, new Size(7, 7), 1);
            Cv2.Threshold(diff_sec, diff_sec, threshold_v, 255, ThresholdTypes.Binary);

            // 计算白色区域面积
            double whiteArea = Cv2.CountNonZero(diff_sec);

            // 保存当前结果
            diff_sec.CopyTo(diff);

            // 检查是否满足条件
            if (whiteArea >= minWhiteArea)
            {
                if (whiteArea <= maxWhiteArea)
                {
                    break;  // 满足条件，退出循环
                }
                else
                {
                    threshold_v += Math.Min(3, (int)((whiteArea / maxWhiteArea) * 3));
                    //darkWeight -= 0.02;
                }
            }
            else
            {
                // 调整权重（增加绝对值，使暗区域影响更大）
                //darkWeight += 0.1;  // 每次调整0.1的步长
                threshold_v -= Math.Min(4, (int)(minWhiteArea / whiteArea * 3));
                if (threshold_v <= 0) threshold_v = 1;
            }
        }


        var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
        // 使用临时 Mat 作为输出（避免可能的冲突）
        //Cv2.ImShow("windowName", diff);
        //Cv2.WaitKey(0);
        using var temp = new Mat();
        Cv2.MorphologyEx(diff, temp, MorphTypes.Close, kernel);

        // 将结果复制回原图像
        temp.CopyTo(diff);

        // 1. 查找轮廓（只检测外轮廓）
        var contours = Cv2.FindContoursAsArray(
            diff,
            RetrievalModes.External,
            ContourApproximationModes.ApproxSimple
        );

        // 5.创建可视化结果图像
        using var result = darkPupilImage.Clone();
        Cv2.CvtColor(result, result, ColorConversionCodes.GRAY2BGR); // 转为彩色以便画彩色图形

        Point pupilCenter = new Point(0, 0); // 默认返回 (0,0)
        /*if (contours.Length > 0)
        {
            // 取面积最大的轮廓（假设是瞳孔）
            var maxContour = contours.OrderByDescending(c => Cv2.ContourArea(c)).FirstOrDefault();

            // 确保找到了轮廓
            if (maxContour != null && maxContour.Length > 0)
            {
                // 创建一个 Mat 并填充 maxContour
                Mat maxContourMat = new Mat(maxContour.Length, 1, MatType.CV_32SC2);

                for (int i = 0; i < maxContour.Length; i++)
                {
                    maxContourMat.Set(i, 0, maxContour[i]);
                }

                // 使用 approxPolyDP 清理轮廓
                var epsilon = Cv2.ArcLength(maxContourMat, true) * 0.02;
                Mat approxContour = new Mat();
                Cv2.ApproxPolyDP(maxContourMat, approxContour, epsilon, true);

                // 检查轮廓点数量
                if (approxContour.Rows < 6)
                {
                    // 轮廓点太少，可能需要重新检测或使用其他方法
                    throw new InvalidOperationException("轮廓点数量太少，无法进行进一步处理。");
                }

                // 使用 FitEllipse 计算最小外接椭圆
                RotatedRect ellipse = Cv2.FitEllipse(approxContour);

                // 绘制最小外接椭圆
                Cv2.Ellipse(result, ellipse, Scalar.Red, 2);

                // 记录椭圆中心
                pupilCenter = new Point((int)ellipse.Center.X, (int)ellipse.Center.Y);
            }
            else
            {

            }
            // 取面积最大的轮廓（假设是瞳孔）
            //var maxContour = contours.OrderByDescending(c => Cv2.ContourArea(c)).First();

            //// 拟合最小外接圆
            //Cv2.MinEnclosingCircle(maxContour, out var center, out var radius);
            //pupilCenter = new Point((int)center.X, (int)center.Y);

            //// 可视化：绘制轮廓（蓝色）
            //Cv2.DrawContours(result, new[] { maxContour }, -1, Scalar.Blue, 1);

            //// 可视化：绘制最小外接圆（红色）
            //Cv2.Circle(result, pupilCenter, (int)radius, Scalar.Red, 1);

            //// 可视化：绘制圆心（绿色实心点）
            //Cv2.Circle(result, pupilCenter, 1, Scalar.Green, -1);
        }
        else
        {

        }*/
        if (contours != null && contours.Length > 0)
        {
            try
            {
                // 1. 取面积最大的有效轮廓（过滤掉过小的轮廓）
                const double minContourArea = 10.0; // 根据实际场景调整最小面积阈值
                var validContours = contours.Where(c => Cv2.ContourArea(c) > minContourArea);

                if (validContours.Any())
                {
                    var maxContour = validContours.OrderByDescending(c => Cv2.ContourArea(c)).First();

                    // 2. 椭圆拟合（增加拟合验证）
                    if (maxContour.Length >= 5) // FitEllipse要求至少5个点
                    {
                        var fittedEllipse = Cv2.FitEllipse(maxContour);

                        // 验证椭圆参数合理性
                        if (fittedEllipse.Size.Width > 0 && fittedEllipse.Size.Height > 0)
                        {
                            pupilCenter = (Point)fittedEllipse.Center;

                            // 3. 计算椭圆的几何特性（增加边界检查）
                            double majorAxis = Math.Max(fittedEllipse.Size.Width, fittedEllipse.Size.Height);
                            double minorAxis = Math.Min(fittedEllipse.Size.Width, fittedEllipse.Size.Height);
                            double axisRatio = minorAxis / majorAxis;
                            double angle = fittedEllipse.Angle % 180; // 规范化角度到0-180度

                            // 4. 动态修正圆心（增加修正限制）
                            const double minAxisRatioForCorrection = 0.8;
                            const double maxCorrectionOffset = 1.0; // 最大修正偏移量  //10

                            if (axisRatio < minAxisRatioForCorrection)
                            {
                                // 计算修正偏移量（带限幅）
                                double offsetScale = Math.Min((1 - axisRatio) * 5, maxCorrectionOffset);

                                // 转换为弧度并计算偏移
                                double angleRad = angle * Math.PI / 180;
                                int offsetX = (int)(offsetScale * Math.Cos(angleRad));
                                int offsetY = (int)(offsetScale * Math.Sin(angleRad));

                                // 应用修正（确保不越界）
                                pupilCenter.X = Math.Max(0, Math.Min(darkPupilImage.Width - 1, pupilCenter.X + offsetX));
                                pupilCenter.Y = Math.Max(0, Math.Min(darkPupilImage.Height - 1, pupilCenter.Y + offsetY));
                            }

                            // 5. 可视化（增加注释和样式区分）
                            // 绘制轮廓（蓝色，1px宽）
                            Cv2.DrawContours(result, new[] { maxContour }, -1, new Scalar(255, 0, 0), 1);

                            // 绘制拟合椭圆（黄色虚线）
                            Cv2.Ellipse(result, fittedEllipse, new Scalar(0, 255, 255), 1, LineTypes.Link4);

                            // 绘制修正后的圆心（绿色实心点，1px半径）
                            Cv2.Circle(result, pupilCenter, 1, new Scalar(0, 255, 0), -1);

                            // 可选：绘制原始最小外接圆（红色虚线，1px宽）
                            Cv2.MinEnclosingCircle(maxContour, out var circleCenter, out var circleRadius);
                            Cv2.Circle(result, (Point)circleCenter, (int)circleRadius,
                                      new Scalar(0, 0, 255), 1, LineTypes.Link4);
                        }
                        else
                        {
                            Debug.WriteLine("警告：拟合椭圆参数无效");
                        }
                    }
                    else
                    {
                        Debug.WriteLine("警告：轮廓点数不足，无法拟合椭圆");
                    }
                }
                else
                {
                    Debug.WriteLine("警告：未找到有效轮廓（面积过小）");
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"瞳孔检测异常：{ex.Message}");
                // 可考虑回退到其他检测方法或使用默认值
            }
        }
        else
        {
            Debug.WriteLine("警告：未检测到任何轮廓");
        }

        // 6.显示结果（放大2倍）
        //string windowName = "Pupil Detection (Press ESC to close)";
        //Cv2.NamedWindow(windowName, WindowFlags.Normal);
        //Cv2.ResizeWindow(windowName, 1200, 800);

        //using var resizedImage = new Mat();
        //Cv2.Resize(result, resizedImage, new Size(0, 0), fx: 2.0, fy: 2.0, InterpolationFlags.Linear);
        //Cv2.ImShow(windowName, resizedImage);

        //// 按ESC退出
        //while (Cv2.WaitKey(10) != 27) { }
        //Cv2.DestroyWindow(windowName);

        return pupilCenter;
    }

    public (Point, Mat) DetectWithVisualization(Mat eyeImage, Mat eyeImage_light)
    {
        if (eyeImage.Empty())
            throw new ArgumentException("输入图像为空"); 

        // 创建可视化画布
        _visualization = new Mat();
        Cv2.CvtColor(eyeImage, _visualization, ColorConversionCodes.BGR2RGB);

        try
        {
            // 1. 改进的预处理
            using var processed = EnhancedPupilPreprocess(eyeImage, eyeImage_light);

            // 显示预处理结果
            AddToVisualizationSafe("1. processed", processed);

            // 2. 候选区域检测
            var candidates = FindPupilCandidates(processed);

            // 显示候选区域
            //var candidatesVis = DrawCandidates(processed, candidates);
            //AddToVisualizationSafe("2. candidatesVis", candidatesVis);

            // 3. 选择最佳候选
            var bestCandidate = SelectBestCandidate(processed, candidates);

            // 显示最佳候选
            var bestVis = DrawBestCandidate(processed, bestCandidate);
            AddToVisualizationSafe("3. bestVis", bestVis);

            // 4. 时间域滤波稳定结果
            var finalPosition = StabilizePosition(bestCandidate);

            // 在原始图像上绘制最终结果
            //if (IsPointInImage(finalPosition, _visualization))
            //{
            //    Cv2.Circle(_visualization, finalPosition, 5, new Scalar(0, 255, 0), -1);
            //    Cv2.PutText(_visualization, "finalPosition", finalPosition + new Point(10, -10),
            //               HersheyFonts.HersheySimplex, 0.5, new Scalar(0, 255, 0), 1);
            //}

            return (finalPosition, _visualization);
        }
        catch (Exception ex)
        {
            Cv2.PutText(_visualization, $"错误: {ex.Message}", new Point(10, 30),
                       HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 0, 255), 2);
            return (new Point(eyeImage.Width / 2, eyeImage.Height / 2), _visualization);
        }
    }

    private Mat EnhancedPupilPreprocess(Mat input, Mat light)
    {
        // 转为灰度图
        var gray = input.Channels() == 3 ?
            input.CvtColor(ColorConversionCodes.BGR2GRAY) :
            input.Clone();

        // 适度高斯模糊降噪（保留边缘）
        Cv2.GaussianBlur(gray, gray, new Size(3, 3), 0.5);

        // 极端对比度增强（三阶段增强）
        // 阶段1：全局非线性增强（伽马校正）
        gray.ConvertTo(gray, -1, 1.5, -30); // 提高对比度并降低亮度

        // 阶段2：局部对比度极限增强
        var clahe = CLAHE.Create();
        clahe.ClipLimit = 4;  // 极高的对比度限制
        clahe.TilesGridSize = new Size(2, 2);  // 极小的网格尺寸
        clahe.Apply(gray, gray);

        // 阶段3：直方图拉伸
        Cv2.Normalize(gray, gray, 0, 255, NormTypes.MinMax);

        // 动态阈值处理（聚焦最黑区域）
        var binary = new Mat();
        double minVal, maxVal;
        Cv2.MinMaxLoc(gray, out minVal, out maxVal);

        // 基于最黑像素值的自适应阈值
        double thresholdValue = minVal + 0.2 * (maxVal - minVal);
        Cv2.Threshold(gray, binary, thresholdValue, 255, ThresholdTypes.BinaryInv);

        // 形态学后处理
        var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(2, 2));
        Cv2.MorphologyEx(binary, binary, MorphTypes.Close, kernel); // 填充小孔
        Cv2.MorphologyEx(binary, binary, MorphTypes.Open, kernel);  // 去除孤立噪点

        return binary;
    }

    private Mat ImprovedPreprocess(Mat input)//Y:\O_Space_Work\Eye\src\EyeTracking.Desktop\bin\x64\Debug\net8.0-windows\保存
    {
        // 转为灰度图
        var gray = input.Channels() == 3 ?
            input.CvtColor(ColorConversionCodes.BGR2GRAY) :
            input.Clone();

        // 高斯模糊降噪
        Cv2.GaussianBlur(gray, gray, new Size(3, 3), 0.8);

        // 自适应直方图均衡化
        var clahe = CLAHE.Create();
        clahe.ClipLimit = 1.8;
        clahe.TilesGridSize = new Size(6, 6);
        clahe.Apply(gray, gray);

        return gray;

        // 改进的阈值处理 - 结合自适应阈值
        var binary = new Mat();
        Cv2.AdaptiveThreshold(gray, binary, 255,
            AdaptiveThresholdTypes.MeanC,  // 尝试MeanC可能比GaussianC更稳定
            ThresholdTypes.BinaryInv,
            15,  // 增大块大小使阈值计算更稳定
            3);   // 提高常数项使阈值更高

        // 可选: 形态学操作去除小噪点
        var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
        Cv2.MorphologyEx(binary, binary, MorphTypes.Open, kernel);

        return binary;

        // 自适应阈值
        //var binary = new Mat();
        //Cv2.Threshold(gray, binary, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);

        //// 形态学操作
        //var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(5, 5));
        //Cv2.MorphologyEx(binary, binary, MorphTypes.Close, kernel, iterations: 2);

        //return binary;
    }

    private List<RotatedRect> FindPupilCandidates(Mat binaryImage)
    {
        var candidates = new List<RotatedRect>();

        // 查找所有轮廓
        Cv2.FindContours(binaryImage, out var contours, out _,
                       RetrievalModes.List, ContourApproximationModes.ApproxSimple);

        foreach (var contour in contours)
        {
            // 面积筛选
            var area = Cv2.ContourArea(contour);
            if (contour.Length < 30 || area < 50 || area > binaryImage.Width * binaryImage.Height / 2)
                continue;

            // 椭圆拟合
            try
            {
                //原始方案
                //var ellipse = Cv2.FitEllipse(contour);
                //if (IsValidPupilEllipse(ellipse, binaryImage.Size()))
                //    candidates.Add(ellipse);
                // 方案1：最小外接圆
                //Cv2.MinEnclosingCircle(contour, out var center, out var radius);
                //var circleRect = new RotatedRect(
                //    center: center,
                //    size: new Size2f((float)radius * 2, (float)radius * 2),
                //    angle: 0);

                // 方案2：基于矩的圆拟合（更精确但计算量稍大）
                //var moments = Cv2.Moments(contour);
                //var center = new Point2f(
                //    (float)(moments.M10 / moments.M00),
                //    (float)(moments.M01 / moments.M00));
                //var radius = (float)Math.Sqrt(area / Math.PI);
                //var circleRect = new RotatedRect(
                //    center: center,
                //    size: new Size2f(radius * 2, radius * 2),
                //    angle: 0);

                //if (IsValidPupilCircle(circleRect, binaryImage.Size()))
                //    candidates.Add(circleRect);
                // 实现最大内切圆查找
                var (center, radius) = FindMaxInscribedCircle(contour, binaryImage.Size());

                var circleRect = new RotatedRect(
                    center: center,
                    size: new Size2f(radius * 2, radius * 2),
                    angle: 0);

                if (IsValidPupilCircle_nei(circleRect, binaryImage.Size()))
                    candidates.Add(circleRect);
            }
            catch
            {
                continue;
            }
        }

        return candidates;
    }

    // 实现最大内切圆查找
    private (Point2f Center, float Radius) FindMaxInscribedCircle(IEnumerable<Point> contour, Size imageSize)
    {
        // 创建一个空白图像用于绘制轮廓
        using var contourImage = new Mat(imageSize, MatType.CV_8UC1, Scalar.Black);
        Cv2.DrawContours(contourImage, new[] { contour }, -1, Scalar.White, -1);

        // 计算距离变换
        using var distTransform = new Mat();
        Cv2.DistanceTransform(contourImage, distTransform, DistanceTypes.L2, (DistanceTransformMasks)5);

        // 找到最大距离值及其位置
        Cv2.MinMaxLoc(distTransform, out _, out double maxVal, out _, out Point maxLoc);

        return (new Point2f(maxLoc.X, maxLoc.Y), (float)maxVal);
    }

    private bool IsValidPupilCircle_nei(RotatedRect circle, Size imageSize)
    {
        // 基础位置检查
        if (circle.Center.X < 0 || circle.Center.Y < 0 ||
            circle.Center.X >= imageSize.Width || circle.Center.Y >= imageSize.Height)
            return false;

        // 大小检查
        //float diameter = circle.Size.Width;
        //if (diameter < 10 || diameter > 100)
        //    return false;

        return true;
    }
    private bool IsValidPupilCircle(RotatedRect circle, Size imageSize)
    {
        // 基础位置检查
        if (circle.Center.X < 0 || circle.Center.Y < 0 ||
            circle.Center.X >= imageSize.Width || circle.Center.Y >= imageSize.Height)
            return false;

        // 大小检查（假设瞳孔直径在10-100像素之间）
        float diameter = circle.Size.Width;
        if (diameter < 10 || diameter > 100)
            return false;

        // 圆形度检查（面积/外接矩形面积比）
        float circleArea = (float)(Math.PI * Math.Pow(diameter / 2, 2));
        float rectArea = circle.Size.Width * circle.Size.Height;
        float circularity = circleArea / rectArea;
        if (circularity < 0.7) // 越接近1越圆
            return false;

        return true;
    }

    private bool IsValidPupilEllipse(RotatedRect ellipse, Size imageSize)
    {
        // 边界检查
        const int margin = 10;
        if (ellipse.Center.X < margin || ellipse.Center.Y < margin ||
            ellipse.Center.X > imageSize.Width - margin ||
            ellipse.Center.Y > imageSize.Height - margin)
            return false;

        // 尺寸检查
        float avgRadius = (ellipse.Size.Width + ellipse.Size.Height) / 4;
        float minRadius = imageSize.Height / 30f;
        float maxRadius = imageSize.Height / 4f;

        return avgRadius >= minRadius && avgRadius <= maxRadius;
    }

    private Point SelectBestCandidate(Mat image, List<RotatedRect> candidates)
    {
        if (candidates.Count == 0)
            return new Point(image.Width / 2, image.Height / 2);

        // 计算每个候选的综合评分（圆形度+大小）
        var scoredCandidates = candidates.Select(c => new
        {
            Candidate = c,
            // 圆形度评分（1.0为完美圆）
            CircularityScore = 1 - Math.Abs(c.Size.Width - c.Size.Height) / Math.Max(c.Size.Width, c.Size.Height),
            // 大小评分（标准化到0-1范围，越大分越高）
            SizeScore = (c.Size.Width + c.Size.Height) / (2 * 100f) // 假设最大预期直径为100
        })
        .Select(x => new
        {
            x.Candidate,
            // 综合评分（圆形度权重60%，大小权重40%）
            CombinedScore = 0.6f * x.CircularityScore + 0.4f * x.SizeScore
        })
        .OrderByDescending(x => x.CombinedScore)
        .ToList();

        // 返回综合评分最高的候选
        return (Point)scoredCandidates.First().Candidate.Center;
    }

    private Point StabilizePosition(Point newPosition)
    {
        _positionBuffer.Enqueue(newPosition);
        if (_positionBuffer.Count > BUFFER_SIZE)
            _positionBuffer.Dequeue();

        // 简单平均
        return new Point(
            (int)_positionBuffer.Average(p => p.X),
            (int)_positionBuffer.Average(p => p.Y));
    }

    private Mat DrawCandidates(Mat baseImage, List<RotatedRect> candidates)
    {
        var vis = baseImage.Channels() == 1 ?
            baseImage.CvtColor(ColorConversionCodes.GRAY2BGR) :
            baseImage.Clone();

        foreach (var candidate in candidates)
        {
            Cv2.Ellipse(vis, candidate, new Scalar(0, 0, 255), 1);
        }

        return vis;
    }

    private Mat DrawBestCandidate(Mat baseImage, Point bestCandidate)
    {
        var vis = baseImage.Channels() == 1 ?
            baseImage.CvtColor(ColorConversionCodes.GRAY2BGR) :
            baseImage.Clone();

        if (IsPointInImage(bestCandidate, vis))
        {
            Cv2.Circle(vis, bestCandidate, 5, new Scalar(0, 255, 0), 2);
            Cv2.PutText(vis, "最佳", bestCandidate + new Point(10, -10),
                       HersheyFonts.HersheySimplex, 0.5, new Scalar(0, 255, 0), 1);
        }

        return vis;
    }

    private void AddToVisualizationSafe(string label, Mat stepImage)
    {
        if (stepImage.Empty() || _visualization.Empty())
            return;

        try
        {
            // 确保可视化区域足够大
            int visWidth = Math.Max(_visualization.Width, 400);
            int visHeight = Math.Max(_visualization.Height, 300);
            if (_visualization.Width < visWidth || _visualization.Height < visHeight)
            {
                Cv2.Resize(_visualization, _visualization, new Size(visWidth, visHeight));
            }

            // 调整步骤图像大小
            var resizedStep = new Mat();
            Cv2.Resize(stepImage, resizedStep, new Size(200, 200));

            // 转换为彩色显示
            if (resizedStep.Channels() == 1)
            {
                Cv2.CvtColor(resizedStep, resizedStep, ColorConversionCodes.GRAY2BGR);
            }

            // 添加标签
            Cv2.PutText(resizedStep, label, new Point(10, 20),
                       HersheyFonts.HersheySimplex, 0.5, new Scalar(255, 255, 255), 1);

            // 安全复制到可视化区域
            int startX = _visualization.Width - 210;
            int startY = 10;

            if (startX >= 0 && startY >= 0 &&
                startX + resizedStep.Width <= _visualization.Width &&
                startY + resizedStep.Height <= _visualization.Height)
            {
                var roi = new Rect(startX, startY, resizedStep.Width, resizedStep.Height);
                resizedStep.CopyTo(new Mat(_visualization, roi));
            }
        }
        catch
        {
            // 忽略可视化错误，不影响主流程
        }
    }

    private bool IsPointInImage(Point p, Mat img)
    {
        return p.X >= 0 && p.Y >= 0 && p.X < img.Width && p.Y < img.Height;
    }

    private void DebugMat (Mat binary)
    {
        string windowName = "Pupil Detection (Press ESC to close)";
        Cv2.NamedWindow(windowName, WindowFlags.Normal);
        Cv2.ResizeWindow(windowName, 1200, 800);
        using var resizedImage = new Mat();
        Cv2.Resize(binary, resizedImage, new Size(0, 0), fx: 2.0, fy: 2.0, InterpolationFlags.Linear);
        Cv2.ImShow(windowName, resizedImage);
        // 按ESC退出
        while (Cv2.WaitKey(10) != 27) { }
        Cv2.DestroyWindow(windowName);
    }
}