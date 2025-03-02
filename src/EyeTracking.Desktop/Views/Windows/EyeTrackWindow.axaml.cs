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
                        Canvas.Children.RemoveAll(Canvas.Children.Except([EyeSight]));
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
                            var text = new TextBlock()
                            {
                                Text = $"屏幕坐标:{ccv.ScreenPoint.X},{ccv.ScreenPoint.Y}\n" +
                                       $"左:{ccv.LeftEyePoint.X},{ccv.LeftEyePoint.Y} 右:{ccv.RightEyePoint.X},{ccv.RightEyePoint.Y}"
                            };
                            Canvas.Children.Add(text);
                            Canvas.SetLeft(text, p.X + ellipse.Width  / 2);
                            Canvas.SetTop(text, p.Y  - ellipse.Height / 2);
                        }

                        break;

                }
            };
            offset = Canvas.PointToScreen(new Avalonia.Point());
            var ellipse = EyeSight;
            vm.PropertyChanged += (o, e) =>
            {
                if (e.PropertyName is not nameof(EyeTrackViewModel.MousePos)) return;
                var point = vm.MousePos;
                Dispatcher.UIThread.Invoke(() =>
                {
                    var p = Canvas.PointToClient(new PixelPoint(point.X, point.Y));
                    Canvas.SetLeft(ellipse, p.X - ellipse.Width  / 2);
                    Canvas.SetTop(ellipse, p.Y  - ellipse.Height / 2);
                    vm.CanvasPos = p;
                });
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