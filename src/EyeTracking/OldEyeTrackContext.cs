using EyeTracking.Extensions;
using OpenCvSharp;

namespace EyeTracking;

public class OldEyeTrackContext : EyeTrackContext<EyeDetectResult>
{

    public override void DetectSight(Mat thisMat, out EyeDetectResult? result)
    {
        /*leftEyeVector  = null;
        rightEyeVector = null;*/

        //-----------------------------------------------------------------------------------
        //选择前一帧作为背景（读入第一帧时，第一帧作为背景）
        if (LastMat == null)
        {
            LastMat = thisMat;
            result  = null;
            return;
        }

        using var foreground = FrameSubtraction(thisMat, LastMat);
        Debug(DebugHint.Bin_Subtraction, foreground);

        //imshow("foreground_BW", foreground_BW);
        //threshold(foreground, foreground_BW, 30, 255, 0);//二值化通常设置为50  255
        ////threshold(foreground, foreground_BW, 0, 255 ,CV_THRESH_BINARY | CV_THRESH_OTSU) ;  //此处使用大津法  自适应取阈值
        //imshow("foreground_BW", foreground_BW);
        //medianBlur(foreground_BW, mid_filer, 3);     //中值滤波法
        //imshow("mid_filer", mid_filer);
        //dilate(mid_filer, gray_dilate1, element);
        //imshow("gray_dilate1", gray_dilate1);
        //dilate(gray_dilate1, gray_dilate2, element);
        //imshow("gray_dilate2", gray_dilate2);
        //dilate(gray_dilate2, gray_dilate3, element);
        //imshow("gray_dilate3", gray_dilate3);

        //imshow("background", background);
        //imshow("gray", gray);
        var dstImage = Connect(foreground);
        //imshow("dstImage", dstImage);

        /*Open(foreground_BW, dstImage);
        imshow("dstImage", dstImage);*/
        dstImage.Show();
        var area = GetArea(dstImage);

        //找到瞳孔的最小外切圆
        GetMyMinEnclosingCircle(area);
        //imshow("frame_0", frame_0);
        LastMat = thisMat;
        result  = null;
    }

    private Mat FrameSubtraction(Mat gray, Mat background)
    {
        using var tmp = new Mat();
        Cv2.Absdiff(gray, background, tmp);
        Debug(DebugHint.Subtraction, tmp);
        //var element = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
        //使用局部自适应阈值分割
        //  *参数说明
        //	参数1：InputArray类型的src，输入图像，填单通道，单8位浮点类型Mat即可。
        //	参数2：函数运算后的结果存放在这。即为输出图像（与输入图像同样的尺寸和类型）。
        //	参数3：预设满足条件的最大值。
        //	参数4：指定自适应阈值算法。可选择ADAPTIVE_THRESH_MEAN_C 或 ADAPTIVE_THRESH_GAUSSIAN_C两种。（具体见下面的解释）。
        //	参数5：指定阈值类型。可选择THRESH_BINARY或者THRESH_BINARY_INV两种。（即二进制阈值或反二进制阈值）。
        //	参数6：表示邻域块大小，用来计算区域阈值，一般选择为3、5、7......等。
        //	参数7：参数C表示与算法有关的参数，它是一个从均值或加权均值提取的常数，可以是负数。（具体见下面的解释）。*/
        return tmp.AdaptiveThreshold(byte.MaxValue, AdaptiveThresholdTypes.MeanC,
            ThresholdTypes.BinaryInv, 5, 10);
        //大津方法
        //threshold(foreground, foreground_BW, 0, 255, CV_THRESH_OTSU);
    }

    private static Mat Connect(Mat background)
    {
        var element  = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
        var dstImage = background.MorphologyEx(MorphTypes.Close, element);
        Mat inPaint = Mat.Ones(dstImage.Size(), MatType.CV_8U);
        Cv2.Inpaint(dstImage, inPaint, dstImage, 3, InpaintMethod.Telea);
        return dstImage;
    }

    private static Mat GetArea(Mat image)
    {
        const int size = 5; //面积因子
        //srcImage = imread("Road2.png");
        /*imshow("原图", srcImage);
        cvtColor(srcImage, grayImage, COLOR_RGB2GRAY);
        threshold(grayImage, dstImage, 100, 255, THRESH_BINARY);
        imshow("二值图", dstImage);*/

        //var element = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));   //针对高亮部分
        //erode(imgHSVMask, imgHSVMask, element);
        //imshow("腐蚀", imgHSVMask);

        // 提取连通区域，并剔除小面积联通区域
        image.FindContours(out var contours, 
            out var hierarchy, 
            RetrievalModes.List,
            ContourApproximationModes.ApproxNone); //检测所有轮廓
        //contours.erase(remove_if(contours.begin(), contours.end(),[](const vector<Point>& c) {return contourArea(c) < 800; }), contours.end());  //vector.erase 删除元素
        // 显示图像并保存
        /*imgHSVMask.setTo(0);
        drawContours(imgHSVMask, contours, -1, Scalar(255), FILLED);
        imshow("处理图", imgHSVMask); */

        Mat imageContours = Mat.Zeros(image.Size(), MatType.CV_8UC1); //绘制
        using Mat imgContours   = Mat.Zeros(image.Size(), MatType.CV_8UC1);

        List<Point[]> qualified = [];
        foreach (var points in contours)
        {
            if (Cv2.ContourArea(points) >= size) qualified.Add(points);
        }

        foreach (var (points, index) in qualified.Select((x, i) => (x, i)))
        {
            foreach (var point in points)
            {
                imgContours.At<byte>(point.X, point.Y) = 255;
            }

            imageContours.DrawContours(contours, index, Scalar.White, -1);
        }
        

        /*imshow("轮廓", ImageContours);
        imshow("轮廓点集合", ImgContours);*/
        //waitKey(0);
        return imageContours;
    }


    //找到外切圆的同时将圆心坐标输出到文件上
    private static void GetMyMinEnclosingCircle(Mat srcGray)
    {
        const int thresh = 100;

        // 使用Threshold检测边缘
        var threshold_output = srcGray.Threshold(thresh, 255, ThresholdTypes.Binary);
        // 找到轮廓
        Cv2.FindContours(threshold_output, out var contours, out var hierarchy,
            RetrievalModes.Tree,
            ContourApproximationModes.ApproxSimple,
            new Point(0, 0));

        /// 多边形逼近轮廓 + 获取矩形和圆形边界框
        List<Point[]> contoursPoly = [];
        List<Rect>    boundRect    = [];
        List<Point2f> center       = [];
        List<float>   radius       = [];

        //for (int i = 0; i < contours.size(); i++)
        //{
        // approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        // boundRect[i] = boundingRect(Mat(contours_poly[i]));
        // minEnclosingCircle(contours_poly[i], center[i], radius[i]);
        //}
        foreach (var (points, index) in contours.Select((x, i) => (x, i)))
        {
            contoursPoly.Add(Cv2.ApproxPolyDP(points, 3, true));
            boundRect.Add(Cv2.BoundingRect(points));
            Cv2.MinEnclosingCircle(points, out var c, out var r);
            center.Add(c);
            radius.Add(r);
        }


        // 画多边形轮廓 + 包围的矩形框 + 圆形框
        Mat drawing = Mat.Zeros(threshold_output.Size(), MatType.CV_8UC3);
        foreach (var (points, index) in contours.Select((x, i) => (x, i)))
        {
            var color = new Scalar(0, 255, 0);
            Cv2.Circle(drawing, (Point)center[index], (int)radius[index], color);
            //            ofs << center[i]<<",";
        }

        drawing.Show("Contours");
    }

}