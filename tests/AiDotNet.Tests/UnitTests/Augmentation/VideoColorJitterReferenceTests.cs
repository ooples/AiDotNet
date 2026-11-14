using AiDotNet.Augmentation;
using AiDotNet.Augmentation.Image;
using AiDotNet.Augmentation.Video;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Augmentation;

/// <summary>
/// Reference tests for VideoColorJitter against torchvision's ColorJitter (adjust_brightness,
/// adjust_contrast, adjust_saturation).
/// </summary>
public class VideoColorJitterReferenceTests
{
    private static ImageTensor<double> Frame(double r, double g, double b)
    {
        var frame = new ImageTensor<double>(2, 2, 3, ChannelOrder.CHW) { IsNormalized = true };
        for (int y = 0; y < 2; y++)
        {
            for (int x = 0; x < 2; x++)
            {
                frame.SetPixel(y, x, 0, r);
                frame.SetPixel(y, x, 1, g);
                frame.SetPixel(y, x, 2, b);
            }
        }

        return frame;
    }

    [Fact(Timeout = 60000)]
    public async Task Brightness_IsMultiplicative()
    {
        await Task.Yield();

        // torchvision adjust_brightness: clip(factor * img). Every channel scales by the same factor.
        var jitter = new VideoColorJitter<double>(brightnessRange: 0.5, contrastRange: 0.0, saturationRange: 0.0, probability: 1.0);

        var result = jitter.Apply(new[] { Frame(0.2, 0.4, 0.6) }, new AugmentationContext<double>(seed: 11));

        double factor = result[0].GetPixel(0, 0, 0) / 0.2;
        Assert.InRange(factor, 0.5, 1.5);

        // The seeded factor is a real draw from [0.5, 1.5]: an identity transform (factor 1) cannot pass.
        Assert.True(Math.Abs(factor - 1.0) > 1e-6, $"brightness factor {factor} is the identity");
        Assert.Equal(0.4 * factor, result[0].GetPixel(0, 0, 1), 9);
        Assert.Equal(0.6 * factor, result[0].GetPixel(0, 0, 2), 9);
        Assert.True(result[0].IsNormalized);
    }

    [Fact(Timeout = 60000)]
    public async Task Contrast_BlendsWithMeanGrayAndLeavesAUniformGrayImageUnchanged()
    {
        await Task.Yield();

        // torchvision adjust_contrast: blend(img, mean(grayscale(img)), factor). For a uniform gray image
        // the mean equals every pixel up to torchvision's luma weights summing to 0.9999, so any factor
        // leaves it unchanged to 1e-5 (the old midpoint contrast moved it by up to 0.15).
        var jitter = new VideoColorJitter<double>(brightnessRange: 0.0, contrastRange: 0.5, saturationRange: 0.0, probability: 1.0);

        var result = jitter.Apply(new[] { Frame(0.2, 0.2, 0.2) }, new AugmentationContext<double>(seed: 3));

        for (int c = 0; c < 3; c++)
        {
            Assert.Equal(0.2, result[0].GetPixel(1, 1, c), 4);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Contrast_MatchesTorchvisionOnANonUniformFrame()
    {
        await Task.Yield();

        // torchvision adjust_contrast: out = clip(f * img + (1 - f) * m), m = mean(0.2989 R + 0.587 G + 0.114 B).
        // The pixels differ, so returning the input (f = 1) or blending towards another centre fails.
        double[][] pixels =
        {
            new[] { 0.2, 0.4, 0.6 }, new[] { 0.6, 0.4, 0.2 },
            new[] { 0.1, 0.9, 0.3 }, new[] { 0.8, 0.2, 0.5 }
        };
        var frame = new ImageTensor<double>(2, 2, 3, ChannelOrder.CHW) { IsNormalized = true };
        for (int p = 0; p < 4; p++)
        {
            for (int c = 0; c < 3; c++)
            {
                frame.SetPixel(p / 2, p % 2, c, pixels[p][c]);
            }
        }

        double mean = pixels.Average(px => 0.2989 * px[0] + 0.587 * px[1] + 0.114 * px[2]);
        var jitter = new VideoColorJitter<double>(brightnessRange: 0.0, contrastRange: 0.5, saturationRange: 0.0, probability: 1.0);

        var result = jitter.Apply(new[] { frame }, new AugmentationContext<double>(seed: 5));

        double factor = (result[0].GetPixel(0, 0, 0) - mean) / (pixels[0][0] - mean);
        Assert.InRange(factor, 0.5, 1.5);
        Assert.True(Math.Abs(factor - 1.0) > 1e-6, $"contrast factor {factor} is the identity");
        for (int p = 0; p < 4; p++)
        {
            for (int c = 0; c < 3; c++)
            {
                double expected = Math.Min(Math.Max(factor * pixels[p][c] + (1 - factor) * mean, 0.0), 1.0);
                Assert.Equal(expected, result[0].GetPixel(p / 2, p % 2, c), 9);
            }
        }
    }

    [Fact(Timeout = 60000)]
    public async Task HalfContrast_MeanStaysFiniteForA255RangeFrame()
    {
        await Task.Yield();

        // A uniform 32x32 frame of 200 in the 0-255 range: contrast leaves it unchanged for any factor. In Half the
        // plain sum of its grayscale (204780) passes 65504, so the mean became infinite and the blend clipped the
        // whole frame to 0 or 255.
        var frame = new ImageTensor<Half>(32, 32, 3, ChannelOrder.CHW) { IsNormalized = false };
        for (int y = 0; y < 32; y++)
        {
            for (int x = 0; x < 32; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    frame.SetPixel(y, x, c, (Half)200.0);
                }
            }
        }

        var jitter = new VideoColorJitter<Half>(brightnessRange: 0.0, contrastRange: 0.5, saturationRange: 0.0, probability: 1.0);

        var result = jitter.Apply(new[] { frame }, new AugmentationContext<Half>(seed: 3));

        for (int c = 0; c < 3; c++)
        {
            Assert.InRange((double)result[0].GetPixel(31, 31, c), 199.5, 200.5);
        }
    }
}
