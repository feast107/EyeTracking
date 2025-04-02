using EyeTracking.Desktop.ViewModels;
using Window = Avalonia.Controls.Window;

namespace EyeTracking.Desktop.Views.Windows;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        var vm = new EyeTrackViewModel(this);
        DataContext = vm;
        this.Closed += MainWindow_Closed;
        new EyeTrackWindow(vm).Show();
    }
    private async void MainWindow_Closed(object sender, EventArgs e)
    {
        Environment.Exit(0);
    }
}