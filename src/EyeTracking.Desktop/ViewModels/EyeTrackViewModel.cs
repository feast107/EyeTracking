using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using System.Web;
using Windows.Win32;
using Antelcat.AutoGen.ComponentModel;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EyeTracking.Desktop.Extensions;
using EyeTracking.Desktop.Views.Windows;
using EyeTracking.Extensions;
using EyeTracking.Utils;
using EyeTracking.Windows.Capture;
using MathNet.Numerics.Distributions;
using Microsoft.Extensions.DependencyInjection;
using OpenCvSharp;
using Window = Avalonia.Controls.Window;

namespace EyeTracking.Desktop.ViewModels;

[AutoKeyAccessor]
public partial class EyeTrackViewModel : ObservableObject, IDisposable
{
    public EyeTrackViewModel(Window window)
    {
        this.window = window;
        var interval = 1000 ;
        Task.Run(() =>
        {
            while (true)
            {
                PInvoke.GetCursorPos(out var point);
                MousePos = point;
                if (interval-- > 0) continue;
                interval = 1000;
                OnPropertyChanged(nameof(Fps));
                OnPropertyChanged(nameof(AlgFps));
            }
        });
    }

    private readonly Window window;

    private string? VideoPath
    {
        get;
        set
        {
            SetProperty(ref field, value);
            Dispose();
            if (value == null) return;
            Reset();
            Decoder    = new VideoDecoder(value);
            Enumerator = Decoder.Decode().GetEnumerator();
        }
    }

    private string? PicturesPath
    {
        get;
        set
        {
            SetProperty(ref field, value);
            Dispose();
            if (value == null) return;
            Reset();
            Enumerator = Directory
                .GetFiles(value)
                .Select(static x => Mat.FromStream(File.OpenRead(x), ImreadModes.Grayscale))
                .GetEnumerator();
        }
    }

    private readonly DoubleBuffer doubleBuffer = new();
    
    private readonly UsbKCapture capture = new();


    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanNext))]
    [NotifyPropertyChangedFor(nameof(CanPlay))]
    [NotifyPropertyChangedFor(nameof(CanPlay))]
    [NotifyPropertyChangedFor(nameof(PlayVisible))]
    [NotifyPropertyChangedFor(nameof(StopVisible))]
    public partial IEnumerator<Mat>? Enumerator { get; set; }

    [ObservableProperty] public partial EyeTrackContext<EyeDetectResult>?    Tracker        { get; set; }
    [ObservableProperty] public partial EyeDetectParameters Parameters     { get; set; } = new();
    [ObservableProperty] public partial VideoDecoder?       Decoder        { get; set; }
    [ObservableProperty] public partial WriteableBitmap?    Bitmap         { get; set; }
    [ObservableProperty] public partial WriteableBitmap?    Output         { get; set; }
    [ObservableProperty] public partial WriteableBitmap?    Subtraction    { get; set; }
    [ObservableProperty] public partial WriteableBitmap?    BinSubtraction { get; set; }
    [ObservableProperty] public partial WriteableBitmap?    Origin         { get; set; }

    [NotifyPropertyChangedFor(nameof(CanNext))]
    [NotifyPropertyChangedFor(nameof(PlayVisible))]
    [NotifyPropertyChangedFor(nameof(StopVisible))]
    [ObservableProperty]
    public partial bool AutoPlay { get; set; }

    [ObservableProperty] public partial bool                 Capturing            { get; set; }
    [ObservableProperty] public partial bool                 Saving               { get; set; }
    public                              int                  Fps                  => doubleBuffer.InputFps;
    public                              int                  AlgFps               => doubleBuffer.OutputFps;
    [ObservableProperty] public partial System.Drawing.Point MousePos             { get; set; }
    [ObservableProperty] public partial Point                LeftEyeVector        { get; set; }
    [ObservableProperty] public partial Point                RightEyeVector       { get; set; }
    [ObservableProperty] public partial Avalonia.Point       CanvasPos            { get; set; }
    [ObservableProperty] public partial bool                 EnableDetect         { get; set; }
    [ObservableProperty] public partial bool                 EnableSave           { get; set; }
    [ObservableProperty] public partial GazeCalibration?     LeftGazeCalibration  { get; set; }
    [ObservableProperty] public partial GazeCalibration?     RightGazeCalibration { get; set; }

    [field: AllowNull, MaybeNull] public ObservableCollection<ClickCircleViewModel> ClickCircles => field ??= [];

    public bool CanNext => CanPlay && !AutoPlay;
    public bool CanPlay => Enumerator is not null;

    public bool PlayVisible => CanPlay && !AutoPlay;
    public bool StopVisible => CanPlay && AutoPlay;

    public ObservableCollection<TrackDebugViewModel> Debugs { get; } = [];

    private string SavePath
    {
        get
        {
            if (created || Directory.Exists(field)) return field;
            Directory.CreateDirectory(field);
            created = true;
            return field;
        }
    } = (FilePath)AppContext.BaseDirectory / "保存";

    private bool created;

    [RelayCommand]
    public void CleanRecord()
    {
        ClickCircles.Clear();
    }
    
    public void RecordTrack()
    {
        ClickCircles.Add(new()
        {
            LeftEyeVector  = new(LeftEyeVector.X, LeftEyeVector.Y),
            RightEyeVector = new(RightEyeVector.X, RightEyeVector.Y),
            ScreenPoint    = new(MousePos.X, MousePos.Y),
        });
    }

    [RelayCommand]
    public void Calibration()
    {
        LeftGazeCalibration = new GazeCalibration(ClickCircles.Select(x =>
            (x.LeftEyeVector.X, x.LeftEyeVector.Y, x.ScreenPoint.X, x.ScreenPoint.Y)));
        RightGazeCalibration = new GazeCalibration(ClickCircles.Select(x =>
            (x.RightEyeVector.X, x.RightEyeVector.Y, x.ScreenPoint.X, x.ScreenPoint.Y)));
    }

    private void SetMat(ref WriteableBitmap? field, string propName, Mat mat)
    {
        if (field != null)
        {
            var tmp = field;
            field = null;
            OnPropertyChanged(propName);
            tmp.Dispose();
        }

        field = mat.ToWriteableBitmap();
        OnPropertyChanged(propName);
    }

    private void OnDebug(EyeTrackContext<EyeDetectResult>.DebugHint hint, params object[] args)
    {
        var mat = args[0].AsNotNull<Mat>();
        switch (hint)
        {
            case EyeTrackContext<EyeDetectResult>.DebugHint.Origin:
                Origin = mat.ToWriteableBitmap();
                return;
            case EyeTrackContext<EyeDetectResult>.DebugHint.Bin_Subtraction:
                BinSubtraction = mat.ToWriteableBitmap();
                return;
            case EyeTrackContext<EyeDetectResult>.DebugHint.Subtraction:
                Subtraction = mat.ToWriteableBitmap();
                return;
            case EyeTrackContext<EyeDetectResult>.DebugHint.Output:
                Output = mat.ToWriteableBitmap();
                return;
            case EyeTrackContext<EyeDetectResult>.DebugHint.Candidate:
                return;
                var clone = mat.Clone();
                Dispatcher.UIThread.Invoke(() =>
                {
                    Debugs.Add(new(
                        clone,
                        $"X:{args[2].AsNotNull<Point>().X}, Y:{args[2].AsNotNull<Point>().Y}",
                        args[1] is true ? Brushes.CornflowerBlue : Brushes.Red,
                        Tracker.Parameters with { }));
                });
                return;
        }
    }

    private void Reset()
    {
        if (Tracker != null) return;
        
        // 配置图像保存参数
        Parameters.EnableImageSave = true;
        Parameters.ImageSavePath = @"F:\EyeTrackingData";
        Parameters.SaveWithTimestamp = true;
        Parameters.ImageFormat = "png";
        
        // 创建跟踪器实例
        Tracker = this.ServiceProvider().GetRequiredService<EyeTrackContext<EyeDetectResult>>();
        Tracker.Parameters = Parameters;
        Tracker.OnDebug += OnDebug;
    }


    [RelayCommand]
    private async Task SelectVideo()
    {
        var result = await window.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            SuggestedStartLocation = await window.StorageProvider.TryGetFolderFromPathAsync(
                VideoPath is null
                    ? @"F:\Shared\Files\视线追踪项目材料\明暗瞳视线追踪技术材料\开发板资料\视频1"
                    : ((FilePath)VideoPath).DirectoryName!),
            AllowMultiple = false
        });
        foreach (var file in result)
        {
            VideoPath = HttpUtility.UrlDecode(file.Path.AbsolutePath);
            break;
        }
    }

    [RelayCommand]
    private async Task SelectPictures()
    {
        var result = await window.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions()
        {
            SuggestedStartLocation = await window.StorageProvider.TryGetFolderFromPathAsync(
                VideoPath is null
                    ? @"E:\WeChat\WeChat Files\wxid_8218gwuowoqm22\FileStorage\File\2025-02\Eye_Release_V1.0_2025_2_5\NEU_Release_Flip_20230805\中"
                    : ((FilePath)VideoPath).DirectoryName!),
            AllowMultiple = false
        });
        foreach (var folder in result)
        {
            PicturesPath = HttpUtility.UrlDecode(folder.Path.AbsolutePath);
            break;
        }
    }

    [RelayCommand]
    private async Task Start()
    {
        if (AutoPlay) return;
        AutoPlay = true;
        while (Enumerator is not null && AutoPlay)
        {
            Next();
            await Task.Delay(20);
        }
    }

    [RelayCommand]
    private void Stop() => AutoPlay = false;

    [RelayCommand]
    private void Next()
    {
        if (Enumerator is null) return;
        if (!Enumerator.MoveNext())
        {
            Enumerator.Dispose();
            Enumerator = null;
            Decoder?.Dispose();
            Decoder  = null;
            AutoPlay = false;
            return;
        }

        Detect(Enumerator.Current);
    }

    private void Detect(Mat mat)
    {
        var items = Debugs.ToArray();
        Debugs.Clear();
        foreach (var debug in items) debug.Dispose();
        if (Tracker is null) return;
        Tracker.DetectSight(mat, out var result);
        if (result != null)
        {
            LeftEyeVector  = result.Left;
            RightEyeVector = result.Right;
        }
    }

    [RelayCommand]
    private void StartCapture()
    {
        if (UsbKCapture.EnumUsbDevices(out var names) <= 0)
        {
            MessageBox.Show("未检测到驱动", window);
            return;
        }

        if (!capture.OpenDevice(0))
        {
            MessageBox.Show("设备 0 启动失败", window);
            return;
        }

        Reset();
        unsafe
        {
            Capturing = true;
            doubleBuffer.Output(arr =>
            {
                var mat = Mat.FromPixelData(capture.Height, capture.Width, MatType.CV_8UC1, arr);
                //CopyCost = cost.ElapsedMilliseconds;
                if (EnableSave) mat.SaveImage((FilePath)SavePath / DateTimeOffset.Now.Ticks.ToString() + ".png");
                if (EnableDetect) Detect(mat);
                //Origin = mat.ToWriteableBitmap();
            });
            var input = doubleBuffer.CreateInput();
            capture.Start((buffer, length) => input(buffer, length));
        }
    }

    [RelayCommand]
    private void StopCapture()
    {
        capture.Stop();
        Capturing = false;
        Tracker   = null;
    }

    public void Dispose()
    {
        Tracker?.Dispose();
        Tracker = null;
        Decoder?.Dispose();
        Decoder = null;
        Enumerator?.Dispose();
        Enumerator = null;
        doubleBuffer.Dispose();
    }
}