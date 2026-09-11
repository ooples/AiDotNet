using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Safety.Image;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>The artifact an image deepfake detector is built to find.</summary>
public enum DeepfakeArtifact
{
    /// <summary>Not declared for this detector.</summary>
    Undeclared = 0,

    /// <summary>
    /// Pasted regions whose noise level and colour statistics differ from the rest of the image: the blending
    /// boundary that natural-consistency detectors look for (NACO, Nie et al. 2024).
    /// </summary>
    SplicedRegion,

    /// <summary>
    /// A periodic grid in the pixels, which makes every row's Fourier magnitudes periodic and flat: the
    /// upsampling fingerprint that spectral detectors look for (Durall et al. 2019).
    /// </summary>
    PeriodicGrid,

    /// <summary>
    /// Flat 8x8 blocks with hard edges and no sensor noise: heavy block compression, a provenance trace no
    /// camera leaves.
    /// </summary>
    BlockCompression,
}

/// <summary>
/// Family invariants for image deepfake detectors, judged on images through <c>GetDeepfakeScore</c> (#2139).
/// </summary>
/// <remarks>
/// <para>
/// The generic safety fixture evaluated a content vector of random values and required a finding. Random values
/// are not a deepfake, and a detector reports a finding only above its threshold, so the fixture could only watch
/// these detectors stay quiet and then fail them for it.
/// </para>
/// <para>
/// This base gives each detector an image carrying the artifact it is built to find, which it must flag, and
/// natural images - smooth, correlated colour with a little sensor noise - which it must pass. Both are
/// [3, 64, 64] in [0, 1], the layout the detectors read.
/// </para>
/// </remarks>
public abstract class DeepfakeDetectorTestBase
{
    private const int Channels = 3;
    private const int Size = 64;

    /// <summary>The threshold every detector's default constructor uses.</summary>
    private const double DefaultThreshold = 0.5;

    /// <summary>Subclasses return their concrete detector, built with its default threshold.</summary>
    protected abstract IDeepfakeDetector<double> CreateDetector();

    /// <summary>The artifact this detector targets.</summary>
    protected virtual DeepfakeArtifact TargetArtifact => CreateDetector() switch
    {
        ConsistencyDeepfakeDetector<double> => DeepfakeArtifact.SplicedRegion,
        FrequencyDeepfakeDetector<double> => DeepfakeArtifact.PeriodicGrid,
        ProvenanceDeepfakeDetector<double> => DeepfakeArtifact.BlockCompression,
        _ => DeepfakeArtifact.Undeclared,
    };

    private static double Gaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static double Clamp(double value) => Math.Max(0.001, Math.Min(0.999, value));

    private static Tensor<double> ToTensor(double[,,] pixels)
    {
        var tensor = new Tensor<double>(new[] { Channels, Size, Size });
        int i = 0;
        for (int c = 0; c < Channels; c++)
            for (int y = 0; y < Size; y++)
                for (int x = 0; x < Size; x++)
                    tensor[i++] = pixels[c, y, x];
        return tensor;
    }

    /// <summary>Smooth low-frequency luminance, a warm tint, a camera-like gamma and sensor noise.</summary>
    private static double[,,] NaturalPixels(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var pixels = new double[Channels, Size, Size];
        var waves = Enumerable.Range(0, 6).Select(_ => (fx: rng.NextDouble() * 3, fy: rng.NextDouble() * 3,
            phase: rng.NextDouble() * 2 * Math.PI, amplitude: 0.5 + rng.NextDouble())).ToArray();
        double[] tint = { 1.0, 0.85, 0.7 };
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                double luma = 0;
                foreach (var w in waves)
                    luma += w.amplitude * Math.Sin(2 * Math.PI * (w.fx * x + w.fy * y) / Size + w.phase);
                luma = 0.5 + 0.08 * luma;
                for (int c = 0; c < Channels; c++)
                    pixels[c, y, x] = Clamp(Math.Pow(Clamp(luma * tint[c]), 1.8) + 0.01 * Gaussian(rng));
            }
        }

        return pixels;
    }

    private static Tensor<double> ArtifactImage(DeepfakeArtifact artifact)
    {
        switch (artifact)
        {
            case DeepfakeArtifact.SplicedRegion:
            {
                // Two diagonal quadrants pasted in from another source: heavier noise, red channel inverted.
                var pixels = NaturalPixels(seed: 1);
                var rng = RandomHelper.CreateSeededRandom(3);
                for (int y = 0; y < Size; y++)
                {
                    for (int x = 0; x < Size; x++)
                    {
                        if ((y < Size / 2) != (x < Size / 2)) continue;
                        pixels[0, y, x] = 1.0 - pixels[0, y, x];
                        for (int c = 0; c < Channels; c++) pixels[c, y, x] = Clamp(pixels[c, y, x] + 0.30 * Gaussian(rng));
                    }
                }

                return ToTensor(pixels);
            }

            case DeepfakeArtifact.PeriodicGrid:
            {
                // Columns every 32 pixels, alternating strong and weak, on a flat ground: each row's spectrum
                // alternates between two levels, periodic in frequency and flat overall.
                var pixels = new double[Channels, Size, Size];
                for (int c = 0; c < Channels; c++)
                {
                    for (int y = 0; y < Size; y++)
                    {
                        for (int x = 0; x < Size; x++)
                        {
                            double value = 0.4;
                            if (x % 64 == 0) value += 0.4;
                            else if (x % 64 == 32) value += 0.2;
                            pixels[c, y, x] = Clamp(value);
                        }
                    }
                }

                return ToTensor(pixels);
            }

            case DeepfakeArtifact.BlockCompression:
            {
                var rng = RandomHelper.CreateSeededRandom(4);
                var pixels = new double[Channels, Size, Size];
                for (int by = 0; by < Size; by += 8)
                {
                    for (int bx = 0; bx < Size; bx += 8)
                    {
                        var colour = new[] { rng.NextDouble(), rng.NextDouble(), rng.NextDouble() };
                        for (int y = by; y < by + 8; y++)
                            for (int x = bx; x < bx + 8; x++)
                                for (int c = 0; c < Channels; c++)
                                    pixels[c, y, x] = Clamp(colour[c]);
                    }
                }

                return ToTensor(pixels);
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(artifact), artifact, "No image for this artifact.");
        }
    }

    private IDeepfakeDetector<double> CreateReadyDetector()
    {
        var detector = CreateDetector();
        Assert.True(detector.IsReady, $"Detector '{detector.ModuleName}' reports IsReady=false.");
        return detector;
    }

    private DeepfakeArtifact DeclaredArtifact(IDeepfakeDetector<double> detector)
    {
        var artifact = TargetArtifact;
        Assert.True(artifact != DeepfakeArtifact.Undeclared,
            $"{detector.ModuleName} declares no target artifact. Map it in DeepfakeDetectorTestBase.TargetArtifact "
            + "to the artifact its paper detects, so the fixture can show it flags that artifact.");
        return artifact;
    }

    [Fact(Timeout = 60000)]
    public async Task GetDeepfakeScore_IsAProbability()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var detector = CreateReadyDetector();

        foreach (var image in new[] { ToTensor(NaturalPixels(1)), ArtifactImage(DeclaredArtifact(detector)) })
        {
            double score = detector.GetDeepfakeScore(image);
            Assert.True(score >= 0.0 && score <= 1.0, $"Score {score} is outside [0, 1]; it is a probability.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task FlagsTheArtifactItTargets()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var detector = CreateReadyDetector();
        var artifact = DeclaredArtifact(detector);
        var image = ArtifactImage(artifact);

        double score = detector.GetDeepfakeScore(image);
        var findings = detector.EvaluateImage(image);

        Assert.True(score >= DefaultThreshold, $"{detector.ModuleName} scored {score:F3} on {artifact}, below its threshold.");
        Assert.NotEmpty(findings);
        foreach (var finding in findings)
        {
            Assert.True(finding.Category == SafetyCategory.Deepfake || finding.Category == SafetyCategory.AIGenerated,
                $"Finding category {finding.Category} is not a deepfake category.");
            Assert.InRange(finding.Confidence, 0.0, 1.0);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task PassesNaturalImages()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var detector = CreateReadyDetector();

        foreach (int seed in new[] { 1, 7, 9 })
        {
            var image = ToTensor(NaturalPixels(seed));
            double score = detector.GetDeepfakeScore(image);
            Assert.True(score < DefaultThreshold, $"{detector.ModuleName} scored natural image {seed} at {score:F3}.");
            Assert.Empty(detector.EvaluateImage(image));
        }
    }

    [Fact(Timeout = 60000)]
    public async Task GetDeepfakeScore_IsDeterministic()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var detector = CreateReadyDetector();
        var image = ArtifactImage(DeclaredArtifact(detector));

        Assert.Equal(detector.GetDeepfakeScore(image), detector.GetDeepfakeScore(image));
    }

    [Fact(Timeout = 60000)]
    public async Task RefusesAnImageTooSmallToAnalyse()
    {
        // A score for an image the detector could not analyse would read as "authentic".
        await Task.Yield();
        using var arena = TensorArena.Create();
        var detector = CreateReadyDetector();
        var tiny = new Tensor<double>(new[] { Channels, 4, 4 });

        Assert.Throws<ArgumentException>(() => detector.GetDeepfakeScore(tiny));
    }
}
