using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

public class GazeCalibration
{
    // 映射函数的参数
    private double[] paramsX; // a_x, b_x, c_x
    private double[] paramsY; // a_y, b_y, c_y

    // 构造函数：传入校准数据并拟合映射函数
    public GazeCalibration(List<(double deltaX, double deltaY, double screenX, double screenY)> calibrationData)
    {
        if (calibrationData.Count < 3)
        {
            throw new ArgumentException("至少需要 3 组校准数据");
        }

        // 提取校准数据
        var deltaX = new double[calibrationData.Count];
        var deltaY = new double[calibrationData.Count];
        var screenX = new double[calibrationData.Count];
        var screenY = new double[calibrationData.Count];

        for (int i = 0; i < calibrationData.Count; i++)
        {
            deltaX[i] = calibrationData[i].deltaX;
            deltaY[i] = calibrationData[i].deltaY;
            screenX[i] = calibrationData[i].screenX;
            screenY[i] = calibrationData[i].screenY;
        }

        // 拟合 X 坐标的映射函数
        paramsX = FitMappingFunction(deltaX, deltaY, screenX);

        // 拟合 Y 坐标的映射函数
        paramsY = FitMappingFunction(deltaX, deltaY, screenY);
    }

    // 使用最小二乘法拟合映射函数
    private double[] FitMappingFunction(double[] deltaX, double[] deltaY, double[] screenCoord)
    {
        // 构建矩阵 A 和向量 b
        var A = Matrix<double>.Build.DenseOfRowArrays(
            deltaX.Length,
            3,
            (i) => new double[] { deltaX[i], deltaY[i], 1 }
        );

        var b = Vector<double>.Build.Dense(screenCoord);

        // 使用最小二乘法求解参数
        var paramsVec = A.TransposeThisAndMultiply(A).Inverse() * A.TransposeThisAndMultiply(b);

        return paramsVec.ToArray();
    }

    // 根据瞳孔-亮斑向量计算视点坐标
    public (double screenX, double screenY) CalculateGazePoint(double deltaX, double deltaY)
    {
        double screenX = paramsX[0] * deltaX + paramsX[1] * deltaY + paramsX[2];
        double screenY = paramsY[0] * deltaX + paramsY[1] * deltaY + paramsY[2];

        return (screenX, screenY);
    }
}



示例
using System;
using System.Collections.Generic;

class Program
{
    static void Main()
    {
        // 定义校准数据（9 组瞳孔-亮斑向量和对应的屏幕坐标）
        var calibrationData = new List<(double deltaX, double deltaY, double screenX, double screenY)>
        {
            (10, 5, 100, 200),
            (20, 10, 300, 400),
            (30, 15, 500, 600),
            (40, 20, 700, 800),
            (50, 25, 900, 1000),
            (60, 30, 1100, 1200),
            (70, 35, 1300, 1400),
            (80, 40, 1500, 1600),
            (90, 45, 1700, 1800)
        };

        // 创建 GazeCalibration 对象并拟合映射函数
        var gazeCalibration = new GazeCalibration(calibrationData);

        // 测试：传入瞳孔-亮斑向量，计算视点坐标
        double deltaX = 25;
        double deltaY = 12;
        var (screenX, screenY) = gazeCalibration.CalculateGazePoint(deltaX, deltaY);

        Console.WriteLine($"Calculated Screen Coordinates: ({screenX}, {screenY})");
    }
}