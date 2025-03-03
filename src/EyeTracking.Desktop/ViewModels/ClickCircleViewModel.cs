using Avalonia;
using CommunityToolkit.Mvvm.ComponentModel;

namespace EyeTracking.Desktop.ViewModels;

public partial class ClickCircleViewModel : ObservableObject
{
    [ObservableProperty] public partial Point ScreenPoint { get; set; }

    [ObservableProperty] public partial Point LeftEyeVector  { get; set; }
    [ObservableProperty] public partial Point RightEyeVector { get; set; }
}