using MathNet.Numerics.LinearAlgebra;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks.Dataflow;

namespace EyeTracking
{
    //[AutoObservable]
    public sealed class GazeCalibration
    {
        private double _lastX = 960; // 屏幕中心默认值
        private double _lastY = 540;
        private readonly Vector<double> _coefficientsX;
        private readonly Vector<double> _coefficientsY;
        private const double RegularizationLambda = 1e-6;

        private EyeStabilizer stabilizer = new EyeStabilizer(
            medianWindowSize: 3,   // 中值窗口大小
            processNoise: 0.1f,    // 过程噪声（Q）
            measurementNoise: 1f   // 测量噪声（R）
        );

        public GazeCalibration(
            IEnumerable<(double deltaX, double deltaY, double screenX, double screenY)> calibrationData)
        {
            var dataPoints = calibrationData?.ToList() ?? throw new ArgumentNullException();

            if (dataPoints.Count < 3)
                throw new ArgumentException("At least 3 calibration points required");

            var (deltaX, deltaY, screenX, screenY) = ExtractFeatures(dataPoints);

            _coefficientsX = FitWithRegularization(deltaX, deltaY, screenX);
            _coefficientsY = FitWithRegularization(deltaX, deltaY, screenY);
        }

        private static Vector<double> FitWithRegularization(
            IList<double> deltaX,
            IList<double> deltaY,
            IList<double> screenCoords)
        {
            var designMatrix = BuildDesignMatrix(deltaX, deltaY);
            var regularization = Matrix<double>.Build.DenseDiagonal(designMatrix.ColumnCount, designMatrix.ColumnCount, RegularizationLambda);

            return (designMatrix.Transpose() * designMatrix + regularization)
                .Inverse()
                * designMatrix.Transpose()
                * Vector<double>.Build.Dense(screenCoords.ToArray());
        }

        private static Matrix<double> BuildDesignMatrix(IList<double> deltaX, IList<double> deltaY)
        {
            return Matrix<double>.Build.DenseOfRowArrays(
                deltaX.Select((x, i) => new[]
                {
                    x,                   // Δx
                    deltaY[i],           // Δy
                    x * deltaY[i],       // Interaction term
                    x * x,               // Δx²
                    deltaY[i] * deltaY[i], // Δy²
                    1                    // Intercept
                }).ToArray());
        }
        private static Matrix<double> BuildEnhancedDesignMatrix(IList<double> deltaX, IList<double> deltaY) //添加三次项拟合，增加头部姿态评估
        {
            return Matrix<double>.Build.DenseOfRowArrays(
                deltaX.Select((x, i) => new[]
                {
            x,                   // Δx
            deltaY[i],           // Δy
            x * deltaY[i],       // ΔxΔy
            x * x,               // Δx²
            deltaY[i] * deltaY[i], // Δy²
            x * x * deltaY[i],   // Δx²Δy
            deltaX[i] * deltaY[i] * deltaY[i], // ΔxΔy²
            x * x * x,           // Δx³
            deltaY[i] * deltaY[i] * deltaY[i], // Δy³
            1                    // Intercept
                }).ToArray());
        }

        // 类成员变量新增
        private readonly Queue<PointF> _positionBuffer = new Queue<PointF>(5); // 历史位置缓存
        private double _velocityEMA = 0;
        private const double JitterThreshold = 50.0; // 像素/帧（200像素抖动对应值）
        private const double DeadZoneRadius = 15.0; // 死区半径
        private readonly object _bufferLock = new object();

        public (double screenX, double screenY) CalculateGazePoint(double deltaX, double deltaY)
        {
            // 1. 原始坐标计算（保持原有逻辑）
            var features = new[] { deltaX, deltaY, deltaX * deltaY, deltaX * deltaX, deltaY * deltaY, 1 };
            double rawX = Math.Clamp(_coefficientsX.Zip(features, (c, f) => c * f).Sum(), 70, 1850);
            double rawY = Math.Clamp(_coefficientsY.Zip(features, (c, f) => c * f).Sum(), 70, 1000);

            // 2. 动态抖动检测
            double instantVelocity = (_lastX == 0 && _lastY == 0) ? 0 :
                Math.Sqrt(Math.Pow(rawX - _lastX, 2) + Math.Pow(rawY - _lastY, 2));
            _velocityEMA = double.IsNaN(instantVelocity) ? 0 :
                0.8 * instantVelocity + 0.2 * _velocityEMA;

            // 3. 多级处理
            PointF result;
            if (_positionBuffer.Count == 0)
            {
                result = new PointF((float)rawX, (float)rawY);
            }
            else if (_velocityEMA > JitterThreshold)
            {
                result = ApplyKalmanFilter(rawX, rawY);
            }
            else if (_velocityEMA > 5.0)
            {
                result = new PointF(
                    (float)(_lastX + (rawX - _lastX) / 1.2),
                    (float)(_lastY + (rawY - _lastY) / 1.2));
            }
            else
            {
                result = ApplyPrecisionStabilization(rawX, rawY);
            }

            // 4. 死区控制（消除剩余微抖）
            if (Math.Abs(result.X - _lastX) < DeadZoneRadius &&
                Math.Abs(result.Y - _lastY) < DeadZoneRadius)
            {
                result = new PointF((float)_lastX, (float)_lastY);
            }

            // 5. 更新状态
            lock (_bufferLock)
            {
                _positionBuffer.Enqueue(result);
                if (_positionBuffer.Count > 5) _positionBuffer.Dequeue();
                _lastX = result.X;
                _lastY = result.Y;
            }
            return (result.X, result.Y);
        }

        private PointF ApplyKalmanFilter(double x, double y)
        {
            try
            {
                // 1. 安全检查缓冲区
                if (_positionBuffer == null || _positionBuffer.Count < 2)
                    return new PointF((float)x, (float)y);

                // 2. 安全访问缓冲区元素
                PointF prev1, prev2;
                lock (_bufferLock) // 确保线程安全
                {
                    prev1 = _positionBuffer.LastOrDefault();
                    prev2 = _positionBuffer.Count >= 2 ?
                           _positionBuffer.ElementAt(_positionBuffer.Count - 2) : prev1;
                }

                // 3. 验证数据有效性
                if (float.IsNaN(prev1.X)) prev1 = new PointF((float)x, (float)y);
                if (float.IsNaN(prev2.X)) prev2 = prev1;

                // 4. 动态计算卡尔曼增益
                double baseGain = 0.2;
                double velocity = Math.Sqrt(Math.Pow(prev1.X - prev2.X, 2) +
                                 Math.Pow(prev1.Y - prev2.Y, 2));
                double dynamicGain = Math.Clamp(baseGain * (1 + velocity / 100.0), 0.1, 0.5);

                // 5. 带保护的预测计算
                double predictedX, predictedY;
                try
                {
                    predictedX = prev1.X + (prev1.X - prev2.X);
                    predictedY = prev1.Y + (prev1.Y - prev2.Y);

                    // 防止预测值溢出
                    predictedX = Math.Clamp(predictedX, 70, 1850);
                    predictedY = Math.Clamp(predictedY, 70, 1000);
                }
                catch
                {
                    predictedX = prev1.X;
                    predictedY = prev1.Y;
                }

                // 6. 最终结果处理
                float resultX = (float)(predictedX + dynamicGain * (x - predictedX));
                float resultY = (float)(predictedY + dynamicGain * (y - predictedY));

                // 二次验证
                if (float.IsNaN(resultX)) resultX = (float)x;
                if (float.IsNaN(resultY)) resultY = (float)y;

                return new PointF(resultX, resultY);
            }
            catch (Exception ex)
            {
                //Debug.WriteLine($"Kalman Filter Error: {ex.Message}");
                return new PointF((float)x, (float)y); // 故障安全返回
            }
        }

        private PointF ApplyPrecisionStabilization(double x, double y)
        {
            // 高精度稳定算法
            double avgX = _positionBuffer.Average(p => p.X);
            double avgY = _positionBuffer.Average(p => p.Y);
            double weight = 0.7; // 历史权重

            return new PointF(
                (float)(weight * avgX + (1 - weight) * x),
                (float)(weight * avgY + (1 - weight) * y));
        }

        private static (double[] dx, double[] dy, double[] sx, double[] sy) ExtractFeatures(
            IList<(double dx, double dy, double sx, double sy)> data)
        {
            return (
                data.Select(p => p.dx).ToArray(),
                data.Select(p => p.dy).ToArray(),
                data.Select(p => p.sx).ToArray(),
                data.Select(p => p.sy).ToArray()
            );
        }
    }
}

//using Antelcat.AutoGen.ComponentModel;
//using MathNet.Numerics.LinearAlgebra;

//namespace EyeTracking;

//public class GazeCalibration
//{
//    // 映射函数的参数
//    private readonly IList<double> paramsX; // a_x, b_x, c_x
//    private readonly IList<double> paramsY; // a_y, b_y, c_y

//    // 构造函数：传入校准数据并拟合映射函数
//    public GazeCalibration(IEnumerable<(double deltaX, double deltaY, double screenX, double screenY)> calibrationData)
//    {
//        // 提取校准数据
//        //var deltaX  = new List<double>();
//        //var deltaY  = new List<double>();
//        //var screenX = new List<double>();
//        //var screenY = new List<double>();

//        //var count = 0;
//        //foreach (var (dX, dY, sX, sY) in calibrationData)
//        //{
//        //    count++;
//        //    deltaX.Add(dX);
//        //    deltaY.Add(dY);
//        //    screenX.Add(sX);
//        //    screenY.Add(sY);
//        //}

//        //if (count < 3)
//        //{
//        //    throw new ArgumentException("参数必须大于3");
//        //}
//        var dataPoints = calibrationData.ToList();
//        if (dataPoints.Count < 3) throw  new ArgumentException("参数必须大于3");

//        var deltaX = dataPoints.Select(p => p.deltaX).ToList();
//        var deltaY = dataPoints.Select(p => p.deltaY).ToList();
//        var screenX = dataPoints.Select(p => p.screenX).ToList();
//        var screenY = dataPoints.Select(p => p.screenY).ToList();

//        // 拟合 X 坐标的映射函数
//        paramsX = FitMappingFunction(deltaX, deltaY, screenX);

//        // 拟合 Y 坐标的映射函数
//        paramsY = FitMappingFunction(deltaX, deltaY, screenY);
//    }

//    // 使用最小二乘法拟合映射函数
//    private static IList<double> FitMappingFunction(IList<double> deltaX, IList<double> deltaY, IList<double> screenCoord)
//    {
//        // 构建行数据集合
//        var rows = new double[deltaX.Count][];
//        for (var i = 0; i < deltaX.Count; i++)
//        {
//            rows[i] = [deltaX[i], deltaY[i], 1];
//        }

//        // 通过行集合构建矩阵
//        var a = Matrix<double>.Build.DenseOfRowArrays(rows);

//        // 构建向量b
//        var b = Vector<double>.Build.Dense(screenCoord.ToArray());

//        // 使用最小二乘法求解 (A^T * A)^-1 * A^T * b
//        var pseudoInverse = (a.Transpose() * a).Inverse() * a.Transpose();
//        var paramsVec = pseudoInverse * b;
//        // 改用SVD分解（更稳定）
//        //var svd = (a.Transpose() * a).Svd();
//        //svd.Solve(b); // 自动处理秩亏情况
//        // 防止过拟合
//        //var lambda = 1e-6; // 正则化系数
//        //var regularization = Matrix<double>.Build.DenseDiagonal(3, 3, lambda);
//        //var pseudoInverse = (a.Transpose() * a + regularization).Inverse() * a.Transpose();

//        return paramsVec.ToArray();
//    }

//    // 根据瞳孔-亮斑向量计算视点坐标
//    public (double screenX, double screenY) CalculateGazePoint(double deltaX, double deltaY)
//    {
//        var screenX = paramsX[0] * deltaX + paramsX[1] * deltaY + paramsX[2];
//        var screenY = paramsY[0] * deltaX + paramsY[1] * deltaY + paramsY[2];

//        return (screenX, screenY);
//    }
//}


// using System;
// using System.Collections.Generic;
//
// class Program
// {
//     static void Main()
//     {
//         // 定义校准数据（9 组瞳孔-亮斑向量和对应的屏幕坐标）
//         var calibrationData = new List<(double deltaX, double deltaY, double screenX, double screenY)>
//         {
//             (10, 5, 100, 200),
//             (20, 10, 300, 400),
//             (30, 15, 500, 600),
//             (40, 20, 700, 800),
//             (50, 25, 900, 1000),
//             (60, 30, 1100, 1200),
//             (70, 35, 1300, 1400),
//             (80, 40, 1500, 1600),
//             (90, 45, 1700, 1800)
//         };
//
//         // 创建 GazeCalibration 对象并拟合映射函数
//         var gazeCalibration = new GazeCalibration(calibrationData);
//
//         // 测试：传入瞳孔-亮斑向量，计算视点坐标
//         double deltaX = 25;
//         double deltaY = 12;
//         var (screenX, screenY) = gazeCalibration.CalculateGazePoint(deltaX, deltaY);
//
//         Console.WriteLine($"Calculated Screen Coordinates: ({screenX}, {screenY})");
//     }
// }