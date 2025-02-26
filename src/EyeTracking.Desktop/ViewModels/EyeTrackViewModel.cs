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
using EyeTracking.Windows.Capture;
using Microsoft.Extensions.DependencyInjection;
using OpenCvSharp;
using Window = Avalonia.Controls.Window;

namespace EyeTracking.Desktop.ViewModels;

[AutoKeyAccessor]
public partial class EyeTrackViewModel : ObservableObject , IDisposable
{
   
    public EyeTrackViewModel(Window window)
    {
        this.window = window;
        Task.Run(() =>
        {
            while (true)
            {
                PInvoke.GetCursorPos(out var point);
                MousePos = point;
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
            if (value == null)return;
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


    private readonly UsbKCapture capture = new();


    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanNext))]
    [NotifyPropertyChangedFor(nameof(CanPlay))]
    [NotifyPropertyChangedFor(nameof(CanPlay))]
    [NotifyPropertyChangedFor(nameof(PlayVisible))]
    [NotifyPropertyChangedFor(nameof(StopVisible))]
    private IEnumerator<Mat>? enumerator;

    [ObservableProperty] private EyeTrackContext?    tracker;
    [ObservableProperty] private EyeDetectParameters parameters = new();
    [ObservableProperty] private VideoDecoder?       decoder;
    [ObservableProperty] private WriteableBitmap?    bitmap;
    [ObservableProperty] private WriteableBitmap?    output;
    [ObservableProperty] private WriteableBitmap?    subtraction;
    [ObservableProperty] private WriteableBitmap?    binSubtraction;
    [ObservableProperty] private WriteableBitmap?    origin;

    [NotifyPropertyChangedFor(nameof(CanNext))]
    [NotifyPropertyChangedFor(nameof(PlayVisible))]
    [NotifyPropertyChangedFor(nameof(StopVisible))]
    [ObservableProperty] private bool autoPlay;
    [ObservableProperty] private bool                 capturing;
    [ObservableProperty] private bool                 saving;
    [ObservableProperty] private int                  fps;
    [ObservableProperty] private long                 copyCost;
    [ObservableProperty] private System.Drawing.Point mousePos;
    [ObservableProperty] private Point                leftEyePos;
    [ObservableProperty] private Point                rightEyePos;
    [ObservableProperty] private Avalonia.Point       canvasPos;
    [ObservableProperty] private bool                 enableDetect;
    [ObservableProperty] private bool                 enableSave;
    
    public bool CanNext => CanPlay && !AutoPlay;
    public bool CanPlay => Enumerator is not null;

    public bool PlayVisible => CanPlay && !AutoPlay;
    public bool StopVisible => CanPlay && AutoPlay;
    
    public ObservableCollection<TrackDebugViewModel> Debugs  { get; }= [];

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

    private void OnDebug(EyeTrackContext.DebugHint hint, params object[] args)
    {
        var mat = args[0].AsNotNull<Mat>();
        switch (hint)
        {
            case EyeTrackContext.DebugHint.Origin:
                Origin = mat.ToWriteableBitmap();
                return;
            case EyeTrackContext.DebugHint.Bin_Subtraction:
                BinSubtraction = mat.ToWriteableBitmap();
                return;
            case EyeTrackContext.DebugHint.Subtraction:
                Subtraction = mat.ToWriteableBitmap();
                return;
            case EyeTrackContext.DebugHint.Output:
                Output = mat.ToWriteableBitmap();
                return;
            case EyeTrackContext.DebugHint.Candidate:
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
        Tracker            =  this.ServiceProvider().GetRequiredService<EyeTrackContext>();
        Tracker.Parameters =  Parameters;
        Tracker.OnDebug    += OnDebug;
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
                    ? @"C:\Users\13532\Documents\WeChat Files\wxid_9wusmarmogu622\FileStorage\File\2024-09\保存\保存"
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
        Tracker.DetectLights(mat, out var left, out var right);
        if (left.HasValue) LeftEyePos   = left.Value;
        if (right.HasValue) RightEyePos = right.Value;
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
            var watch    = Stopwatch.StartNew();
            var lastTick = watch.ElapsedMilliseconds;
            capture.Start((buffer, length) =>
            {
                var cur = watch.ElapsedMilliseconds;
                Fps       = (int)(1000 / (cur - lastTick));
                lastTick  = cur;
                var arr  = new byte[length];
                var ptr  = new IntPtr(buffer);
                var cost = Stopwatch.StartNew();
                Marshal.Copy(ptr, arr, 0, (int)length);
                var mat = Mat.FromPixelData(capture.Height, capture.Width, MatType.CV_8UC1, arr);
                CopyCost = cost.ElapsedMilliseconds;
                if (EnableSave) mat.SaveImage((FilePath)SavePath / DateTimeOffset.Now.Ticks.ToString() + ".png");
                if (EnableDetect) Detect(mat);
                else
                {
                    Origin = mat.ToWriteableBitmap();
                }
            });
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
    }
}