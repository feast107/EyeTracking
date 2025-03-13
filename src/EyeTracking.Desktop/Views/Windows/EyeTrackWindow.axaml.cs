using System.Collections.Specialized;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Threading;
using EyeTracking.Desktop.ViewModels;
using Point = System.Drawing.Point;

namespace EyeTracking.Desktop.Views.Windows;

public partial class EyeTrackWindow : Window
{
    public EyeTrackWindow(EyeTrackViewModel vm)
    {
        InitializeComponent();
        DataContext = vm;
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
                        if (vm.LeftGazeCalibration is null || vm.RightGazeCalibration is null) return;
                        var left = vm.LeftGazeCalibration.CalculateGazePoint(
                            vm.LeftEyeVector.X, vm.LeftEyeVector.Y);
                        var right = vm.RightGazeCalibration.CalculateGazePoint(
                            vm.RightEyeVector.X, vm.RightEyeVector.Y);
                        Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            Canvas.SetLeft(LeftPosition, (left.screenX + right.screenX) / 2 - LeftPosition.Width   / 2);
                            Canvas.SetTop(LeftPosition, (left.screenY + right.screenY) / 2 - LeftPosition.Height  / 2);
                            //Canvas.SetLeft(RightPosition, right.screenX - RightPosition.Width  / 2);
                            //Canvas.SetTop(RightPosition, right.screenY - RightPosition.Height / 2);
                        });
                        break;
                }
            };
            Canvas.PointerPressed += (o, e) =>
            {
                vm.RecordTrack();
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