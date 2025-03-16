using System.Diagnostics.CodeAnalysis;
using OpenCvSharp;

namespace EyeTracking;

public class GradientEyeTrackContext : EyeTrackContext<EyeDetectResult>
{
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
    private static readonly string XmlPath =
        Path.Combine(AppContext.BaseDirectory, "Resources/Haarcascade/haarcascade_eye.xml");


    public override void DetectSight(Mat thisMat, out EyeDetectResult? result)
    {
        // 眼睛检测
        var eyes = EyeCascade.DetectMultiScale(thisMat, 1.1, 4, 0, new Size(30, 30));
        if (eyes.Length != 2)
        {
            throw new Exception("Error: Exactly two eyes are required for this operation!");
        }

        // 按x坐标排序，确保左眼在前，右眼在后
        eyes = eyes.OrderBy(x => x.X).ToArray();

        // 处理左眼
        var left_eye      = thisMat.SubMat(eyes[0]);
        var left_pupil    = Locate(left_eye);
        var left_gradient = GetGradientImage(left_eye);

        // 处理右眼
        var right_eye      = thisMat.SubMat(eyes[1]);
        var right_pupil    = Locate(right_eye);
        var right_gradient = GetGradientImage(right_eye);

        // 计算原图中的瞳孔绝对坐标
        var left_pupil_abs  = new Point(eyes[0].X + left_pupil.X, eyes[0].Y  + left_pupil.Y);
        var right_pupil_abs = new Point(eyes[1].X + right_pupil.X, eyes[1].Y + right_pupil.Y);

        // 在原图上标记结果
        var clone = thisMat.Clone();

        // 标记左眼
        Cv2.Circle(clone, left_pupil_abs, 3, new Scalar(255), -1);
        Cv2.Circle(clone, left_pupil_abs, 10, new Scalar(255), 1);

        // 标记右眼
        Cv2.Circle(clone, right_pupil_abs, 3, new Scalar(255), -1);
        Cv2.Circle(clone, right_pupil_abs, 10, new Scalar(255), 1);

        // 创建并显示窗口
        Cv2.NamedWindow("Original");
        Cv2.NamedWindow("Result" );
        Cv2.NamedWindow("Left Eye");
        Cv2.NamedWindow("Right Eye");
        Cv2.NamedWindow("Left Gradient");
        Cv2.NamedWindow("Right Gradient");

        // 设置窗口大小
        Cv2.ResizeWindow("Original", 800, 600);
        Cv2.ResizeWindow("Result", 800, 600);
        Cv2.ResizeWindow("Left Eye", 400, 300);
        Cv2.ResizeWindow("Right Eye", 400, 300);
        Cv2.ResizeWindow("Left Gradient", 400, 300);
        Cv2.ResizeWindow("Right Gradient", 400, 300);

        // 显示结果
        Cv2.ImShow("Original", thisMat);
        Cv2.ImShow("Result", clone);
        Cv2.ImShow("Left Eye", left_eye);
        Cv2.ImShow("Right Eye", right_eye);
        Cv2.ImShow("Left Gradient", left_gradient);
        Cv2.ImShow("Right Gradient", right_gradient);

        // // 保存结果
        // cv::imwrite("result.jpg", result);
        // cv::imwrite("left_gradient.jpg", left_gradient);
        // cv::imwrite("right_gradient.jpg", right_gradient);
        //
        // std::cout << "Left pupil absolute position: (" << left_pupil_abs.x << ", " << left_pupil_abs.y << ")" <<
        //     std::endl;
        // std::cout << "Right pupil absolute position: (" << right_pupil_abs.x << ", " << right_pupil_abs.y << ")" <<
        //     std::endl;
        // std::cout << "Press any key to exit..." << std::endl;

        Cv2.WaitKey(0);
        Cv2.DestroyAllWindows();
        
        // TODO:返回结果
        result = null;
    }

    private Mat CreateGrid(int rows, int cols)
    {
        var grid = new Mat(2 * rows - 1, 2 * cols - 1, MatType.CV_32SC2);
        for (var y = 1 - rows; y < rows; y++)
        {
            for (var x = 1 - cols; x < cols; x++)
            {
                var norm            = Math.Sqrt(x * x + y * y);
                if (norm == 0) norm = 1;
                var gridY           = y + rows - 1;
                var gridX           = x + cols - 1;
                grid.At<Vec2f>(gridY, gridX) = new Vec2f((float)(y / norm), (float)(x / norm));
            }
        }

        return grid;
    }
    
    private Mat CreateGradient(Mat image)
    {
        using Mat gradX = new(), gradY = new();
        Cv2.Sobel(image, gradX, MatType.CV_32F, 1, 0);
        Cv2.Sobel(image, gradY, MatType.CV_32F, 0, 1);

        Mat gradient = new(image.Size(), MatType.CV_32FC2);
        for (var y = 0; y < image.Rows; y++) {
            for (var x = 0; x < image.Cols; x++) {
                var gx            = gradX.At<float>(y, x);
                var gy            = gradY.At<float>(y, x);
                var norm          = Math.Sqrt(gx * gx + gy * gy);
                if (norm == 0) norm = 1;
                gradient.At<Vec2f>(y, x) = new Vec2f((float)(gy / norm), (float)(gx / norm));
            }
        }
        return gradient;
    }
    
    private Point Locate(Mat image, double sigma = 2, int accuracy = 1) {  // 增加accuracy默认值
        Mat floatImage = new();
        image.ConvertTo(floatImage, MatType.CV_32F);
        Cv2.Normalize(floatImage, floatImage, 0, 1, NormTypes.MinMax);

        Mat blurred = new();
        Cv2.GaussianBlur(floatImage, blurred, new Size(0, 0), sigma);

        // 限制搜索范围到中心区域
        const int border = 5;  // 边界像素
        var       endY   = image.Rows - border;
        var       endX   = image.Cols - border;

        var grid = CreateGrid(image.Rows, image.Cols);
        var gradient = CreateGradient(floatImage);

        var scores = Mat.Zeros(image.Size(), MatType.CV_32F).ToMat();

        // 使用OpenMP并行计算
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (var cy = border; cy < endY; cy += accuracy) {
            for (var cx = border; cx < endX; cx += accuracy) {
                float score = 0;
                // 预先获取需要的数据以优化内存访问
                var blurVal = blurred.At<float>(cy, cx);
                
                // 减少计算范围，只考虑周围区域
                const int windowSize = 15;  // 可调整的窗口大小
                var       startWy    = Math.Max(0, cy          - windowSize);
                var       endWy      = Math.Min(image.Rows, cy + windowSize);
                var       startWx    = Math.Max(0, cx          - windowSize);
                var       endWx      = Math.Min(image.Cols, cx + windowSize);

                for (var y = startWy; y < endWy; y++) {
                    for (var x = startWx; x < endWx; x++)
                    {
                        var disP = grid.At<Vec2f>(
                            image.Rows - cy - 1 + y,
                            image.Cols - cx - 1 + x
                        );
                        var grad = gradient.At<Vec2f>(y, x);
                        var dot = disP[0] * grad[0] + disP[1] * grad[1];
                        score += dot * dot;
                    }
                }
                scores.At<float>(cy, cx) = score * (1 - blurVal);
            }
        }

        Cv2.MinMaxLoc(scores, out _, out _, out _, out var maxLoc);
        return maxLoc;
    }

    // 添加获取梯度图的方法
    Mat GetGradientImage(Mat image)
    {
        Mat floatImage = new();
        image.ConvertTo(floatImage, MatType.CV_32F);
        Cv2.Normalize(floatImage, floatImage, 0, 1, NormTypes.MinMax);

        var gradient = CreateGradient(floatImage);
        
        // 转换梯度为可视化图像
        Mat gradientVis = new(image.Size(), MatType.CV_8UC1);
        for (var y = 0; y < image.Rows; y++) {
            for (var x = 0; x < image.Cols; x++) {
                var grad = gradient.At<Vec2f>(y, x);
                var magnitude = Math.Sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
                gradientVis.At<byte>(y, x) = (byte)Math.Max(magnitude * 255, 255);
            }
        }
        return gradientVis;
    }
}