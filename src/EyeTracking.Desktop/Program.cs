using System.CodeDom.Compiler;
using Avalonia;
using Avalonia.Dialogs;
using EyeTracking.Desktop.Views.Windows;
using ShowMeTheXaml;

namespace EyeTracking.Desktop;

internal static class Program
{
    // Initialization code. Don't use any Avalonia, third-party APIs or any
    // SynchronizationContext-reliant code before AppMain is called: things aren't initialized
    // yet and stuff might break.
    [STAThread]
    public static void Main(string[] args) => BuildAvaloniaApp()
        .StartWithClassicDesktopLifetime(args);

    // Avalonia configuration, don't remove; also used by visual designer.
    public static AppBuilder BuildAvaloniaApp()
    {
        AppDomain.CurrentDomain.UnhandledException += (o, e) =>
        {
            MessageBox.Show(e.ExceptionObject.ToString() ?? string.Empty);
        };
        var app = AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace()
            .UseXamlDisplay();

        if (OperatingSystem.IsWindows() || OperatingSystem.IsMacOS() || OperatingSystem.IsLinux())
            app.UseManagedSystemDialogs();
        return app;
    }
}