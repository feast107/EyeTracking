using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

public class EyeStabilizer
{
    private readonly MedianFilter medianX;
    private readonly MedianFilter medianY;
    private KalmanFilter1D kalmanX;
    private KalmanFilter1D kalmanY;
    private readonly float q;
    private readonly float r;

    public EyeStabilizer(int medianWindowSize = 3, float processNoise = 0.1f, float measurementNoise = 1f)
    {
        medianX = new MedianFilter(medianWindowSize);
        medianY = new MedianFilter(medianWindowSize);
        q = processNoise;
        r = measurementNoise;
    }

    public PointF Update(PointF newPoint)
    {
        // 中值滤波去除离群点
        var medianPoint = new PointF(
            medianX.Update(newPoint.X),
            medianY.Update(newPoint.Y)
        );

        // 初始化卡尔曼滤波器
        if (kalmanX == null)
        {
            kalmanX = new KalmanFilter1D(q, r, medianPoint.X);
            kalmanY = new KalmanFilter1D(q, r, medianPoint.Y);
        }

        // 卡尔曼滤波平滑
        return new PointF(
            kalmanX.Update(medianPoint.X),
            kalmanY.Update(medianPoint.Y)
        );
    }
}

/// <summary>
/// 中值滤波器（滑动窗口）
/// </summary>
public class MedianFilter
{
    private readonly Queue<float> window = new Queue<float>();
    private readonly int windowSize;

    public MedianFilter(int windowSize)
    {
        this.windowSize = windowSize;
    }

    public float Update(float value)
    {
        window.Enqueue(value);
        while (window.Count > windowSize)
            window.Dequeue();

        var ordered = window.OrderBy(v => v).ToArray();
        return ordered[ordered.Length / 2];
    }
}

/// <summary>
/// 简化版一维卡尔曼滤波器
/// </summary>
public class KalmanFilter1D
{
    private float q;       // 过程噪声
    private float r;       // 测量噪声
    private float p = 1;  // 估计误差协方差
    private float x;       // 状态值
    private float k;       // 卡尔曼增益

    public KalmanFilter1D(float q, float r, float initialValue)
    {
        this.q = q;
        this.r = r;
        x = initialValue;
    }

    public float Update(float measurement)
    {
        // 预测阶段
        p += q;

        // 更新阶段
        k = p / (p + r);
        x += k * (measurement - x);
        p *= (1 - k);

        return x;
    }
}

//使用
//var stabilizer = new EyeStabilizer(
//    medianWindowSize: 3,   // 中值窗口大小
//    processNoise: 0.1f,    // 过程噪声（Q）
//    measurementNoise: 1f   // 测量噪声（R）
//);

//// 模拟持续获取眼动数据
//foreach (var rawPoint in GetEyeTrackingData())
//{
//    var stabilizedPoint = stabilizer.Update(rawPoint);
//    // 使用平滑后的坐标...
//}