using System;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for keyed image watermarkers, judged on their embed/detect round trip (#2139).
/// </summary>
/// <remarks>
/// <para>
/// A watermark is only worth anything if four things hold: a marked image is detected, an unmarked one is not, a
/// watermarker with another key does not detect the mark, and the mark survives mild noise and 8-bit quantization.
/// Each is checked on fixed, seeded images: a smooth natural-like image (gradients and low-frequency texture with
/// sensor noise), uniform noise, and a flat image with a gradient, all in [0, 1].
/// </para>
/// <para>
/// The mark must also be invisible, stay in the image's value range, leave the input untouched and be
/// deterministic. These watermarkers were previously detect-only heuristics with no way to embed a mark, and their
/// tests only checked that a score lay in [0, 1].
/// </para>
/// </remarks>
public abstract class ImageWatermarkerTestBase
{
    /// <summary>Creates the watermarker under test with its default key.</summary>
    protected abstract IImageWatermarker<double> CreateModule();

    /// <summary>
    /// Creates the watermarker with another key. The default finds the module's constructor parameter named
    /// <c>key</c> and passes every other parameter its default.
    /// </summary>
    protected virtual IImageWatermarker<double> CreateModuleWithKey(int key)
    {
        var type = CreateModule().GetType();
        foreach (var ctor in type.GetConstructors())
        {
            var parameters = ctor.GetParameters();
            if (!parameters.Any(p => p.Name == "key" && p.ParameterType == typeof(int))) continue;
            if (!parameters.All(p => p.Name == "key" || p.HasDefaultValue)) continue;
            var args = parameters.Select(p => p.Name == "key" ? key : p.DefaultValue).ToArray();
            return (IImageWatermarker<double>)ctor.Invoke(args);
        }

        throw new InvalidOperationException(
            $"{type.Name} has no constructor taking a 'key' with defaults for the rest; override CreateModuleWithKey.");
    }

    /// <summary>The number of channels of the test images.</summary>
    protected virtual int ImageChannels => 3;

    /// <summary>The height and width of the test images.</summary>
    protected virtual int ImageSize => 64;

    private const int OtherKey = 0x1234567;

    [Fact(Timeout = 60000)]
    public async Task Embed_ThenDetect_IsConfident()
    {
        await Task.Yield();
        var module = CreateModule();
        foreach (var (name, image) in Images())
        {
            double score = module.DetectWatermark(module.EmbedWatermark(image));
            Assert.True(score >= 0.9, $"{module.ModuleName}: confidence {score:F3} on its own mark in the {name} image; expected >= 0.9.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task UnmarkedImages_AreNotDetected()
    {
        await Task.Yield();
        var module = CreateModule();
        foreach (var (name, image) in Images())
        {
            double score = module.DetectWatermark(image);
            Assert.True(score == 0.0, $"{module.ModuleName}: confidence {score:F3} on the unmarked {name} image; expected 0.");
            Assert.Empty(module.EvaluateImage(image));
        }
    }

    [Fact(Timeout = 60000)]
    public async Task OtherKey_DoesNotDetectTheMark()
    {
        await Task.Yield();
        var module = CreateModule();
        var other = CreateModuleWithKey(OtherKey);
        foreach (var (name, image) in Images())
        {
            var marked = module.EmbedWatermark(image);
            double score = other.DetectWatermark(marked);
            Assert.True(score == 0.0, $"{module.ModuleName}: another key scored {score:F3} on the {name} image's mark; expected 0.");
            Assert.True(other.DetectWatermark(other.EmbedWatermark(image)) >= 0.9,
                $"{module.ModuleName}: the other key does not detect its own mark in the {name} image.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Mark_SurvivesNoiseAndQuantization()
    {
        await Task.Yield();
        var module = CreateModule();
        var random = RandomHelper.CreateSeededRandom(7);
        foreach (var (name, image) in Images())
        {
            var marked = module.EmbedWatermark(image).ToArray();
            for (int i = 0; i < marked.Length; i++)
            {
                double noisy = marked[i] + 0.01 * Gaussian(random);
                marked[i] = Math.Round(Math.Max(0, Math.Min(1, noisy)) * 255) / 255;
            }

            var degraded = new Tensor<double>(marked, image.Shape.ToArray());
            double score = module.DetectWatermark(degraded);
            Assert.True(score >= 0.5, $"{module.ModuleName}: confidence {score:F3} after noise and 8-bit quantization of the {name} image; expected >= 0.5.");
            var findings = module.EvaluateImage(degraded);
            Assert.Contains(findings, f => f.Category == SafetyCategory.Watermarked && f.Confidence == score);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Mark_IsInvisible_InRange_AndLeavesTheInputUntouched()
    {
        await Task.Yield();
        var module = CreateModule();
        var image = NaturalImage();
        var before = image.ToArray();
        var marked = module.EmbedWatermark(image);

        Assert.Equal(before, image.ToArray());
        Assert.Equal(image.Shape.ToArray(), marked.Shape.ToArray());
        var after = marked.ToArray();
        double squared = 0;
        for (int i = 0; i < after.Length; i++)
        {
            Assert.True(after[i] >= 0 && after[i] <= 1, $"{module.ModuleName}: value {after[i]} at {i} left [0, 1].");
            squared += (after[i] - before[i]) * (after[i] - before[i]);
        }

        double psnr = 10 * Math.Log10(1.0 / (squared / after.Length));
        Assert.True(psnr >= 38, $"{module.ModuleName}: PSNR {psnr:F1} dB on the natural image; a visible mark (expected >= 38 dB).");
        Assert.Equal(after, module.EmbedWatermark(image).ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task TooSmallImage_CannotBeMarked_AndScoresZero()
    {
        await Task.Yield();
        var module = CreateModule();
        var tiny = new Tensor<double>(new double[ImageChannels * 4 * 4], new[] { ImageChannels, 4, 4 });
        Assert.Throws<ArgumentException>(() => module.EmbedWatermark(tiny));
        Assert.Equal(0.0, module.DetectWatermark(tiny));
    }

    private (string Name, Tensor<double> Image)[] Images() => new[]
    {
        ("natural", NaturalImage()),
        ("uniform-noise", NoiseImage()),
        ("flat-gradient", FlatGradientImage()),
    };

    private Tensor<double> NaturalImage()
    {
        int c = ImageChannels, n = ImageSize;
        var random = RandomHelper.CreateSeededRandom(42);
        var data = new double[c * n * n];
        for (int ch = 0; ch < c; ch++)
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    double v = 0.45 + 0.05 * ch + 0.2 * (double)x / n
                        + 0.15 * Math.Sin(2 * Math.PI * 1.5 * x / n) * Math.Cos(2 * Math.PI * y / n)
                        + 0.02 * Gaussian(random);
                    data[(ch * n + y) * n + x] = Math.Max(0, Math.Min(1, v));
                }

        return new Tensor<double>(data, new[] { c, n, n });
    }

    private Tensor<double> NoiseImage()
    {
        var random = RandomHelper.CreateSeededRandom(43);
        var data = new double[ImageChannels * ImageSize * ImageSize];
        for (int i = 0; i < data.Length; i++) data[i] = random.NextDouble();
        return new Tensor<double>(data, new[] { ImageChannels, ImageSize, ImageSize });
    }

    private Tensor<double> FlatGradientImage()
    {
        int c = ImageChannels, n = ImageSize;
        var data = new double[c * n * n];
        for (int ch = 0; ch < c; ch++)
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++) data[(ch * n + y) * n + x] = 0.3 + 0.4 * (double)y / n;
        return new Tensor<double>(data, new[] { c, n, n });
    }

    private static double Gaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}