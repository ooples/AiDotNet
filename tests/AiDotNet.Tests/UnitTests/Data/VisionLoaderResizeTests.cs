using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Data;

/// <summary>
/// Reference values for the vision loaders' resize: bilinear with align_corners = true
/// (torch.nn.functional.interpolate), HWC output, and BT.601 luma (PIL convert("L")) for a single target channel.
/// </summary>
public class VisionLoaderResizeTests
{
    private const double Tolerance = 1e-9;

    // 2x2 RGB image, pixel (y, x) -> (R, G, B).
    private static readonly byte[,,] Pixels =
    {
        { { 0, 10, 20 }, { 100, 110, 120 } },
        { { 200, 210, 220 }, { 50, 60, 70 } },
    };

    private static string WritePpm()
    {
        string path = Path.Combine(Path.GetTempPath(), "ew57b-" + Guid.NewGuid().ToString("N") + ".ppm");
        using (var stream = File.Create(path))
        {
            byte[] header = Encoding.ASCII.GetBytes("P6\n2 2\n255\n");
            stream.Write(header, 0, header.Length);
            for (int y = 0; y < 2; y++)
            {
                for (int x = 0; x < 2; x++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        stream.WriteByte(Pixels[y, x, c]);
                    }
                }
            }
        }

        return path;
    }

    private static double Bilinear(int channel, double sy, double sx)
    {
        double v00 = Pixels[0, 0, channel], v01 = Pixels[0, 1, channel];
        double v10 = Pixels[1, 0, channel], v11 = Pixels[1, 1, channel];
        return (v00 * (1 - sx) * (1 - sy) + v01 * sx * (1 - sy) + v10 * (1 - sx) * sy + v11 * sx * sy) / 255.0;
    }

    [Fact(Timeout = 60000)]
    public async Task LoadAndResizeImage_UpsamplesWithAlignedCornersInHwcOrder()
    {
        await Task.Yield();
        string path = WritePpm();
        try
        {
            var pixels = VisionLoaderHelper.LoadAndResizeImage<double>(path, 3, 3, 3);

            Assert.Equal(27, pixels.Length);
            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        // align_corners: source coordinate = target * (2 - 1) / (3 - 1).
                        Assert.Equal(Bilinear(c, y * 0.5, x * 0.5), pixels[(y * 3 + x) * 3 + c], Tolerance);
                    }
                }
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task LoadAndResizeImage_SingleChannelFromColour_IsBt601Luma()
    {
        await Task.Yield();
        string path = WritePpm();
        try
        {
            var pixels = VisionLoaderHelper.LoadAndResizeImage<double>(path, 2, 2, 1);

            for (int y = 0; y < 2; y++)
            {
                for (int x = 0; x < 2; x++)
                {
                    double luma = (0.299 * Pixels[y, x, 0] + 0.587 * Pixels[y, x, 1] + 0.114 * Pixels[y, x, 2]) / 255.0;
                    Assert.Equal(luma, pixels[y * 2 + x], Tolerance);
                }
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task ResizeChwToHwc_ZeroFill_LeavesChannelsBeyondTheSourceAtZero()
    {
        await Task.Yield();
        string path = WritePpm();
        try
        {
            var image = AiDotNet.Helpers.ImageHelper<double>.LoadImage(path, true);

            var resized = VisionLoaderHelper.ResizeChwToHwc(image, 2, 2, 4, VisionLoaderHelper.ExtraChannelFill.Zero);
            var repeated = VisionLoaderHelper.ResizeChwToHwc(image, 2, 2, 4, VisionLoaderHelper.ExtraChannelFill.RepeatSource);

            Assert.Equal(new[] { 1, 2, 2, 4 }, resized.Shape.ToArray());
            var zeroData = resized.ToArray();
            var repeatData = repeated.ToArray();
            for (int p = 0; p < 4; p++)
            {
                int y = p / 2, x = p % 2;
                for (int c = 0; c < 3; c++)
                {
                    Assert.Equal(Pixels[y, x, c] / 255.0, zeroData[p * 4 + c], Tolerance);
                }

                Assert.Equal(0.0, zeroData[p * 4 + 3], Tolerance);
                Assert.Equal(Pixels[y, x, 0] / 255.0, repeatData[p * 4 + 3], Tolerance);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }
}
