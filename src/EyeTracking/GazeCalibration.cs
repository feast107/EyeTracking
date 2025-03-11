using MathNet.Numerics.LinearAlgebra;
using System.Linq;

namespace EyeTracking
{
    //[AutoObservable]
    public sealed class GazeCalibration
    {
        private readonly Vector<double> _coefficientsX;
        private readonly Vector<double> _coefficientsY;
        private double _lastX = 0;
        private double _lastY = 0;
        private const double RegularizationLambda = 1e-6;

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

        public (double screenX, double screenY) CalculateGazePoint(double deltaX, double deltaY)
        {
            var features = new[] { deltaX, deltaY, deltaX * deltaY, deltaX * deltaX, deltaY * deltaY, 1 };
            if (_lastX == 0 && _lastY == 0)
            {
                _lastX = _coefficientsX.ToArray().Zip(features, (c, f) => c * f).Sum() > 1900 ? 1900 : _coefficientsX.ToArray().Zip(features, (c, f) => c * f).Sum();
                _lastY = _coefficientsY.ToArray().Zip(features, (c, f) => c * f).Sum() > 1000 ? 1000 : _coefficientsY.ToArray().Zip(features, (c, f) => c * f).Sum();
                return (_lastX, _lastY);
            }
            else
            {
                double _thisX = _coefficientsX.ToArray().Zip(features, (c, f) => c * f).Sum() > 1900 ? 1900 : _coefficientsX.ToArray().Zip(features, (c, f) => c * f).Sum();
                double _thisY = _coefficientsY.ToArray().Zip(features, (c, f) => c * f).Sum() > 1000 ? 1000 : _coefficientsY.ToArray().Zip(features, (c, f) => c * f).Sum();
                double _stepX = _lastX + (_thisX - _lastX) / 10;
                double _stepY = _lastY + (_thisY - _lastY) / 10;
                _lastX = _stepX;
                _lastY = _stepY;
                return (_stepX, _stepY);
            }
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