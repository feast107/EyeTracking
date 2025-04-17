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
    private CancellationTokenSource _animationCts;
    private bool _isAnimating;
    private double left_x = 0;
    private double left_y = 0;
    private double right_x = 0;
    private double right_y = 0;
    public EyeTrackWindow(EyeTrackViewModel vm)
    {
        InitializeComponent();
        DataContext = vm;
        Canvas.SetLeft(LeftPosition, 960);
        Canvas.SetTop(LeftPosition, 540);
        bool play = false;
        vm.Calibration_default();
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
                        if (vm.LeftGazeCalibration is null || vm.RightGazeCalibration is null)
                        {
                            return;
                        }
                        OnEyeDataChanged(vm);

                        /*if (vm.LeftGazeCalibration is null || vm.RightGazeCalibration is null || play)
                        {
                            return;
                        }
                        play = true;

                        // 计算目标位置
                        var left = vm.LeftGazeCalibration.CalculateGazePoint(vm.LeftEyeVector.X, vm.LeftEyeVector.Y);
                        var right = vm.RightGazeCalibration.CalculateGazePoint(vm.RightEyeVector.X, vm.RightEyeVector.Y);

                        Dispatcher.UIThread.InvokeAsync(async () =>
                        {
                            // 基础目标位置（中心点）
                            var baseTargetX = (left.screenX + right.screenX) / 2 - LeftPosition.Width / 2;
                            var baseTargetY = (left.screenY + right.screenY) / 2 - LeftPosition.Height / 2;

                            // 添加随机"气泡"偏移量（模拟蠕动效果）
                            var random = new Random();
                            var bubbleOffsetX = (random.NextDouble() - 0.5) * 50; // ±25像素随机偏移
                            var bubbleOffsetY = (random.NextDouble() - 0.5) * 50;

                            // 最终目标位置（基础位置+气泡偏移）
                            var targetX = baseTargetX + bubbleOffsetX;
                            var targetY = baseTargetY + bubbleOffsetY;

                            // 获取当前位置
                            var currentX = Canvas.GetLeft(LeftPosition);
                            var currentY = Canvas.GetTop(LeftPosition);

                            // 创建主移动动画（带弹性效果）
                            var moveAnimation = new Animation
                            {
                                Duration = TimeSpan.FromMilliseconds(150), // 延长动画时间
                                FillMode = FillMode.Forward,
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
                                        Cue = new Cue(0.7), // 70%时间到达目标位置
                                        Setters =
                                        {
                                            new Setter(Canvas.LeftProperty, targetX),
                                            new Setter(Canvas.TopProperty, targetY)
                                        },
                                        KeySpline = new KeySpline(0.25, 0.1, 0.25, 1) // 缓入缓出
                                    },
                                    new KeyFrame
                                    {
                                        Cue = new Cue(1),
                                        Setters =
                                        {
                                            new Setter(Canvas.LeftProperty, targetX),
                                            new Setter(Canvas.TopProperty, targetY)
                                        },
                                        KeySpline = new KeySpline(0.2, 0.8, 0.2, 1) // 弹性效果
                                    }
                                }
                             };

                            // 创建尺寸变化动画（模拟气泡效果）
                            var scaleAnimation = new Animation
                            {
                                Duration = TimeSpan.FromMilliseconds(800),
                                Children =
                                {
                                    new KeyFrame
                                    {
                                        Cue = new Cue(0),
                                        Setters =
                                        {
                                            new Setter(ScaleTransform.ScaleXProperty, 1.0),
                                            new Setter(ScaleTransform.ScaleYProperty, 1.0)
                                        }
                                    },
                                    new KeyFrame
                                    {
                                        Cue = new Cue(0.3),
                                        Setters =
                                        {
                                            new Setter(ScaleTransform.ScaleXProperty, 1.05),
                                            new Setter(ScaleTransform.ScaleYProperty, 0.95)
                                        }
                                    },
                                    new KeyFrame
                                    {
                                        Cue = new Cue(0.6),
                                        Setters =
                                        {
                                            new Setter(ScaleTransform.ScaleXProperty, 0.98),
                                            new Setter(ScaleTransform.ScaleYProperty, 1.02)
                                        }
                                    },
                                    new KeyFrame
                                    {
                                        Cue = new Cue(1),
                                        Setters =
                                        {
                                            new Setter(ScaleTransform.ScaleXProperty, 1.0),
                                            new Setter(ScaleTransform.ScaleYProperty, 1.0)
                                        }
                                    }
                                }
                            };

                            // 确保有变换组
                            if (LeftPosition.RenderTransform is not TransformGroup)
                            {
                                LeftPosition.RenderTransform = new TransformGroup
                                {
                                    Children = { new ScaleTransform(), new TranslateTransform() }
                                };
                            }

                            // 并行运行动画
                            await Task.WhenAll(
                                moveAnimation.RunAsync(LeftPosition),
                                scaleAnimation.RunAsync(LeftPosition)
                            );

                            // 确保最终位置准确
                            Canvas.SetLeft(LeftPosition, targetX);
                            Canvas.SetTop(LeftPosition, targetY);

                            // 重置变换（可选）
                            if (LeftPosition.RenderTransform is TransformGroup group)
                            {
                                group.Children[0] = new ScaleTransform(1, 1);
                            }

                            play = false;
                        });*/


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

    // 修改后的方法
    private async Task UpdateEyePositionAsync(EyeTrackViewModel vm)
    {
        if (vm.LeftGazeCalibration is null || vm.RightGazeCalibration is null)
            return;

        // 取消之前的动画（如果正在运行）
        _animationCts?.Cancel();
        _animationCts = new CancellationTokenSource();
        var cancellationToken = _animationCts.Token;

        try
        {
            _isAnimating = true;

            // 初始化位置
            var currentX = Canvas.GetLeft(LeftPosition);
            var currentY = Canvas.GetTop(LeftPosition);

            // 确保有变换组
            if (LeftPosition.RenderTransform is not TransformGroup)
            {
                LeftPosition.RenderTransform = new TransformGroup
                {
                    Children = { new ScaleTransform(), new TranslateTransform() }
                };
            }

            // 动画循环
            while (!cancellationToken.IsCancellationRequested)
            {
                // 获取最新数据
                var left = vm.LeftGazeCalibration.CalculateGazePoint(
                    vm.LeftEyeVector.X, vm.LeftEyeVector.Y);
                var right = vm.RightGazeCalibration.CalculateGazePoint(
                    vm.RightEyeVector.X, vm.RightEyeVector.Y);

                // 计算目标位置（带随机气泡偏移）
                var random = new Random();
                var targetX = (left.screenX + right.screenX) / 2 - LeftPosition.Width / 2
                             + (random.NextDouble() - 0.5) * 6;
                var targetY = (left.screenY + right.screenY) / 2 - LeftPosition.Height / 2
                             + (random.NextDouble() - 0.5) * 6;

                // 计算移动距离决定动画时长（动态速度）
                var distance = Math.Sqrt(
                    Math.Pow(targetX - currentX, 2) +
                    Math.Pow(targetY - currentY, 2));
                var duration = Math.Min(250, Math.Max(150, distance * 2)); // 150-250ms之间

                // 创建动画
                var animation = new Animation
                {
                    Duration = TimeSpan.FromMilliseconds(duration),
                    FillMode = FillMode.Forward,
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
                        KeySpline = new KeySpline(0.2, 0.8, 0.2, 1) // 弹性效果
                    }
                }
                };

                // 运行动画并等待
                await animation.RunAsync(LeftPosition, cancellationToken);

                // 更新当前位置
                currentX = targetX;
                currentY = targetY;

                // 短暂延迟（避免CPU占用过高）
                // await Task.Delay(16, cancellationToken); // ~60fps
            }
        }
        catch (OperationCanceledException)
        {
            // 动画被取消是正常情况
        }
        finally
        {
            _isAnimating = false;
        }
    }

    // 调用方式（在数据变化时调用）
    private void OnEyeDataChanged(EyeTrackViewModel vm)
    {
        Dispatcher.UIThread.Post(async () =>
        {
            if (!_isAnimating)
            {
                await UpdateEyePositionAsync(vm);
            }
            else
            {
                await UpdateEyePositionAsync(vm);
            }
            // 否则动画已经在运行，会自动处理新位置
        });
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