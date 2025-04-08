using System.Collections.Specialized;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Threading;
using EyeTracking.Desktop.ViewModels;
using Point = System.Drawing.Point;
using Avalonia.Animation;
using Avalonia.Media;
using Avalonia.Styling;
using System.Diagnostics;

namespace EyeTracking.Desktop.Views.Windows;

public partial class EyeTrackWindow : Window
{
    public EyeTrackWindow(EyeTrackViewModel vm)
    {
        InitializeComponent();
        DataContext = vm;
        Canvas.SetLeft(LeftPosition, 960);
        Canvas.SetTop(LeftPosition, 540);
        bool play = false;
        Loaded += (_, _) =>
        {
            vm.ClickCircles.CollectionChanged += (o,e) =>
            {
                switch (e.Action)
                {
                    case NotifyCollectionChangedAction.Reset:
                        Canvas.Children.RemoveAll(Canvas.Children.Except([MousePosition, LeftPosition, RightPosition]));
                        break;
                    case NotifyCollectionChangedAction.Add:
                        if (e.NewItems is null) return;
                        foreach (var ccv in e.NewItems.OfType<ClickCircleViewModel>())
                        {
                            var ellipse = NewEllipse();
                            Canvas.Children.Add(ellipse);
                            var p = Canvas.PointToClient(new PixelPoint(
                                (int)ccv.ScreenPoint.X,
                                (int)ccv.ScreenPoint.Y));
                            Canvas.SetLeft(ellipse, p.X - ellipse.Width  / 2);
                            Canvas.SetTop(ellipse, p.Y  - ellipse.Height / 2);
                            var text = new TextBlock
                            {
                                Text = $"屏幕坐标:{ccv.ScreenPoint.X},{ccv.ScreenPoint.Y}\n" +
                                       $"左:{ccv.LeftEyeVector.X},{ccv.LeftEyeVector.Y} 右:{ccv.RightEyeVector.X},{ccv.RightEyeVector.Y}"
                            };
                            Canvas.Children.Add(text);
                            Canvas.SetLeft(text, p.X + ellipse.Width  / 2);
                            Canvas.SetTop(text, p.Y  - ellipse.Height / 2);
                        }

                        break;

                }
            };
            offset = Canvas.PointToScreen(new Avalonia.Point());
            var mouse = MousePosition;
            vm.PropertyChanged += (o, e) =>
            {
                switch (e.PropertyName)
                {
                    case nameof(EyeTrackViewModel.MousePos):
                        var point = vm.MousePos;
                        Dispatcher.UIThread.Invoke(() =>
                        {
                            var p = Canvas.PointToClient(new PixelPoint(point.X, point.Y));
                            Canvas.SetLeft(mouse, p.X - mouse.Width  / 2);
                            Canvas.SetTop(mouse, p.Y  - mouse.Height / 2);
                            //mouse.Fill();
                            vm.CanvasPos = p;
                        });
                        break;
                    case nameof(EyeTrackViewModel.RightEyeVector):
                        if (vm.LeftGazeCalibration is null || vm.RightGazeCalibration is null || play)
                        {
                            return;
                        }
                        play = true;
                        var left = vm.LeftGazeCalibration.CalculateGazePoint(
                            vm.LeftEyeVector.X, vm.LeftEyeVector.Y);
                        var right = vm.RightGazeCalibration.CalculateGazePoint(
                            vm.RightEyeVector.X, vm.RightEyeVector.Y);

                        Dispatcher.UIThread.InvokeAsync(async () =>
                        {
                            // 计算目标位置
                            var targetX = (left.screenX + right.screenX) / 2 - LeftPosition.Width / 2;
                            var targetY = (left.screenY + right.screenY) / 2 - LeftPosition.Height / 2;

                            // 获取当前位置
                            var currentX = Canvas.GetLeft(LeftPosition);
                            var currentY = Canvas.GetTop(LeftPosition);

                            // 创建动画
                            var animation = new Animation
                            {
                                Duration = TimeSpan.FromMilliseconds(100),
                                FillMode = FillMode.Forward, // 保持动画结束状态
                                Children =
                                {
                                    new KeyFrame
                                    {
                                        Cue = new Cue(0),
                                        Setters =
                                        {
                                            new Setter(Canvas.LeftProperty, currentX),
                                            new Setter(Canvas.TopProperty, currentY)
                                        }
                                    },
                                    new KeyFrame
                                    {
                                        Cue = new Cue(1),
                                        Setters =
                                        {
                                            new Setter(Canvas.LeftProperty, targetX),
                                            new Setter(Canvas.TopProperty, targetY)
                                        },
                                        KeySpline = new KeySpline(0.33, 0, 0.66, 1) // 缓动曲线
                                    }
                                }
                            };

                            // 运行动画
                            await animation.RunAsync(LeftPosition);

                            // 确保最终位置准确（可选）
                            Canvas.SetLeft(LeftPosition, targetX);
                            Canvas.SetTop(LeftPosition, targetY);
                            play = false;
                        });
                        //Dispatcher.UIThread.InvokeAsync(() =>
                        //{
                        //    Canvas.SetLeft(LeftPosition, (left.screenX + right.screenX) / 2 - LeftPosition.Width / 2);
                        //    Canvas.SetTop(LeftPosition, (left.screenY + right.screenY) / 2 - LeftPosition.Height / 2);
                        //    //Canvas.SetLeft(RightPosition, right.screenX - RightPosition.Width  / 2);
                        //    //Canvas.SetTop(RightPosition, right.screenY - RightPosition.Height / 2);
                        //});
                        break;
                }
            };
            Canvas.PointerPressed += async (o, e) =>
            {
                try
                {
                    await vm.RecordTrackAsync(); // 异步调用，不阻塞 UI
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"录制失败: {ex.Message}");
                }
            };
        };
    }

    private Avalonia.Point Map(Point point)
    {
        if (ratio is not null)
        {
            var r = ratio.Value;
            return new Avalonia.Point(point.X * r.XR, point.Y * r.YR);
        }

        var p      = Canvas.PointToClient(new PixelPoint(point.X, point.Y));
        ratio = ((p.X - offset.X) / point.X, (p.Y - offset.Y) / point.Y);
        return p;
    }

    private (double XR, double YR)? ratio;
    private PixelPoint              offset;

    private static Ellipse NewEllipse() => new()
    {
        Height = 30,
        Width  = 30,
        Fill   = Brushes.Cyan
    };
    
}