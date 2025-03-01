using System.Reflection;
using Antelcat.AutoGen.ComponentModel.Diagnostic;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EyeTracking.Desktop.Extensions;
using EyeTracking.Desktop.Views.Windows;
using OpenCvSharp;

namespace EyeTracking.Desktop.ViewModels;

[AutoMetadataFrom(typeof(EyeDetectParameters), MemberTypes.Property,
    Template =
        """
        #if {GetMethod.IsPublic}
        public {PropertyType} {Name} { get => parameters.{Name};
        #if {CanWrite}
            set {
                parameters.{Name} = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(Output));
                OnPropertyChanged(nameof(Sum));
            }
        #endif
        }
        #endif

        """)]
public partial class TrackDebugViewModel(
    Mat mat,
    string point,
    IImmutableSolidColorBrush brush,
    EyeDetectParameters parameters) : ObservableObject, IDisposable
{
    public IImmutableSolidColorBrush Brush  { get; } = brush;
    public string                    Point  { get; } = point;
    public WriteableBitmap           Origin { get; } = mat.ToWriteableBitmap();

    public double Sum
    {
        get
        {
            using var tmp = parameters.Threshold(mat);
            return parameters.Weighted(tmp);
        }
    }

    public WriteableBitmap? Output
    {
        get
        {
            field?.Dispose();
            using var tmp = parameters.Threshold(mat);
            field = tmp.ToWriteableBitmap();
            return field;
        }
    }

    [RelayCommand]
    private void Detail() => new DebugWindow(this).Show();

    public void Dispose()
    {
        Origin.Dispose();
        mat.Dispose();
    }
}