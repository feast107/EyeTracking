using Antelcat.AutoGen.ComponentModel;
using MathNet.Numerics.LinearAlgebra;

namespace EyeTracking;

public class GazeCalibration
{
    // 映射函数的参数
    private readonly IList<double> paramsX; // a_x, b_x, c_x
    private readonly IList<double> paramsY; // a_y, b_y, c_y

    // 构造函数：传入校准数据并拟合映射函数
    public GazeCalibration(IEnumerable<(double deltaX, double deltaY, double screenX, double screenY)> calibrationData)
    {
        // 提取校准数据
        var deltaX  = new List<double>();
        var deltaY  = new List<double>();
        var screenX = new List<double>();
        var screenY = new List<double>();

        var count = 0;
        foreach (var (dX, dY, sX, sY) in calibrationData)
        {
            count++;
            deltaX.Add(dX);
            deltaY.Add(dY);
            screenX.Add(sX);
            screenY.Add(sY);
        }

        if (count < 3)
        {
            throw new ArgumentException("参数必须大于3");
        }
        
        // 拟合 X 坐标的映射函数
        paramsX = FitMappingFunction(deltaX, deltaY, screenX);

        // 拟合 Y 坐标的映射函数
        paramsY = FitMappingFunction(deltaX, deltaY, screenY);
    }

    // 使用最小二乘法拟合映射函数
    private static IList<double> FitMappingFunction(IList<double> deltaX, IList<double> deltaY, IList<double> screenCoord)
    {
        // 构建行数据集合
        var rows = new double[deltaX.Count][];
        for (var i = 0; i < deltaX.Count; i++)
        {
            rows[i] = [deltaX[i], deltaY[i], 1];
        }

        // 通过行集合构建矩阵
        var a = Matrix<double>.Build.DenseOfRowArrays(rows);

        // 构建向量b
        var b = Vector<double>.Build.Dense(screenCoord.ToArray());

        // 使用最小二乘法求解 (A^T * A)^-1 * A^T * b
        var pseudoInverse = (a.Transpose() * a).Inverse() * a.Transpose();
        var paramsVec     = pseudoInverse                 * b;

        return paramsVec.ToArray();
    }

    // 根据瞳孔-亮斑向量计算视点坐标
    public (double screenX, double screenY) CalculateGazePoint(double deltaX, double deltaY)
    {
        var screenX = paramsX[0] * deltaX + paramsX[1] * deltaY + paramsX[2];
        var screenY = paramsY[0] * deltaX + paramsY[1] * deltaY + paramsY[2];

        return (screenX, screenY);
    }
}


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