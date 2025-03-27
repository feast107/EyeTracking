using System.Diagnostics;
using System.Runtime.InteropServices;

namespace EyeTracking.Utils;

public class DoubleBuffer
{
    public unsafe delegate void DataHandler(byte* buffer, long length);

    public int InputFps { get; private set; }

    public int OutputFps { get; private set; }

    private          byte[]?                  pending;
    private          (byte[], byte[])?        current;
    private          CancellationTokenSource? cancel;
    
    public unsafe DataHandler CreateInput()
    {
        var watch = Stopwatch.StartNew();
        var last  = watch.ElapsedMilliseconds;
        return (buffer, length) =>
        {
            var now = watch.ElapsedMilliseconds;
            InputFps = (int)(1000 / (now - last));
            last = now;
            Console.WriteLine(InputFps);
            var arr = new byte[length];
            var ptr = new IntPtr(buffer);
            Marshal.Copy(ptr, arr, 0, (int)length);
            if (pending is null) pending = arr;
            else
            {
                lock (this)
                {
                    current = (pending, arr);
                    pending = null;
                }
            }
        };
    }
    
    public void Output(Action<byte[]> action)
    {
        cancel?.Cancel();
        var tmp = cancel = new();
        var watch = Stopwatch.StartNew();
        var last  = watch.ElapsedMilliseconds;
        new Thread(() =>
        {
            while (!tmp.IsCancellationRequested)
            {
                if (current is null) continue;
                (byte[], byte[])? cur;
                lock (this)
                {
                    cur = current;
                }
                current = null;
                action(cur.Value.Item1);
                CalculateFps();
                action(cur.Value.Item2);
                CalculateFps();
            }
        }).Start();
        return;

        void CalculateFps()
        {
            var now = watch.ElapsedMilliseconds;
            if (now <= last) return;
            OutputFps = (int)(1000 / (now - last));
            last      = now;
        }
    }
}