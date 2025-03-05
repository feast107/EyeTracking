using OpenCvSharp;
using Sdcb.FFmpeg.Codecs;
using Sdcb.FFmpeg.Formats;
using Sdcb.FFmpeg.Raw;
using Sdcb.FFmpeg.Utils;

namespace EyeTracking.Tests;

public class Tests
{
    private FilePath      file => @"F:\Shared\Files\视线追踪项目材料\明暗瞳视线追踪技术材料\开发板资料\视频1\视频1.mp4";
    private FormatContext format;
    private MediaStream   stream;
    private Codec         codec;
    private CodecContext  context;
    [SetUp]
    public void Setup()
    {
        format  = FormatContext.OpenInputUrl(file);
        stream = format.GetVideoStream();
        var param = stream.Codecpar!;
        codec   = Codec.FindDecoderById(param.CodecId);
        context = new CodecContext(codec);
        context.FillParameters(param);
        context.Open(codec);
    }

    

    [Test]
    public void Test1()
    {
        Mat?   last     = null;
        Point? leftEye  = null;
        Point? rightEye = null;
        var    context  = new OldEyeTrackContext();
        Loop(mat =>
        {
            context.DetectSight(mat, out _);
        }, false);
        last?.Dispose();
    }

    public void Loop(Action<Mat> action, bool dispose = true)
    {
        using var packet = new Packet();
        using var frame  = new Frame();
        while (true)
        {
            switch (format.ReadFrame(packet))
            {
                case CodecResult.EOF:
                default:
                    goto final;
                case CodecResult.Again:
                    packet.Unref();
                    continue;
                case CodecResult.Success:
                    break;
            }
            
            unsafe
            {
                ffmpeg.avcodec_send_packet(context, packet);
                while (true)
                {
                    switch (context.ReceiveFrame(frame))
                    {
                        case CodecResult.Again:
                        default:
                            goto outer;
                        case CodecResult.EOF:
                            goto final;
                        case CodecResult.Success:
                        {
                            var mat    = Mat.FromPixelData(context.Height, context.Width, MatType.CV_8UC1, frame.ToImageBuffer());
                            action(mat);
                            if (dispose) mat.Dispose();
                            break;
                        }
                    }
                }
                outer:
                continue;
            }
        }
        final:
        return;
    }

    [TearDown]
    public void Dispose()
    {
        format.Dispose();
        context.Dispose();
    }
}

