using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Diffusion.Guidance;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Guidance;

/// <summary>
/// Adaptive projected guidance and rescaled CFG reproduce diffusers 0.35
/// (<c>tools/reference-data/diffusers_diffusion_reference.py</c>) on a batch of two samples.
/// </summary>
/// <remarks>
/// Before this, APG scaled the orthogonal component and kept the parallel one (the paper keeps the orthogonal
/// component and down-weights the oversaturating parallel one), and both methods pooled their projection and
/// standard deviations over the whole batch instead of per sample.
/// </remarks>
public class DiffusersGuidanceReferenceTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(2048)]
    [InlineData(70000)]
    public void HanningWindow_HalfAndWideTypesMatchTheSymmetricReference(int size)
    {
        var half = new AiDotNet.WindowFunctions.HanningWindow<Half>().Create(size);
        var single = new AiDotNet.WindowFunctions.HanningWindow<float>().Create(size);
        var precise = new AiDotNet.WindowFunctions.HanningWindow<double>().Create(size);
        Assert.All(half.ToArray(), value => Assert.True(Half.IsFinite(value)));
        foreach (int index in new[] { 0, size / 3, size / 2, size - 1 })
        {
            double expected = size == 1 ? 1 : 0.5 * (1 - Math.Cos(2 * Math.PI * index / (size - 1)));
            Assert.InRange(Math.Abs(precise[index] - expected), 0, 1e-12);
            Assert.InRange(Math.Abs(single[index] - expected), 0, 1e-6);
            Assert.InRange(Math.Abs((double)half[index] - expected), 0, 0.0005);
            Assert.Equal(half[index], half[size - index - 1]);
        }
    }

    [Fact]
    public void HalfConceptErasure_ZeroDirectionLeavesTheEmbeddingFiniteAndUnchanged()
    {
        var eraser = new AiDotNet.Diffusion.Safety.ConceptEraser<Half>();
        eraser.AddConceptDirection(new Vector<Half>(2));
        var embedding = new Vector<Half>(new Half[] { (Half)1, (Half)2 });
        Assert.Equal(embedding.ToArray(), eraser.EraseFromEmbedding(embedding).ToArray());
    }

    [Fact]
    public void SparseImageStencil_DoesNotReadZeroCoefficientPixels()
    {
        var pixels = new Tensor<double>(new double[] { 0, 0, 0, 0, double.NaN, 0, 0, 0, 0 }, new[] { 1, 1, 3, 3 });
        var actual = AiDotNet.Diffusion.Preprocessing.PreprocessorFilters.Interior3x3(pixels,
            AiDotNet.Diffusion.Preprocessing.PreprocessorFilters.CentralDifferences);
        Assert.Equal(new[] { 0.0, 0.0 }, actual.ToArray());
    }

    [Fact]
    public void HalfLabelColors_MatchWideIntegerHashes()
    {
        var labels = new Tensor<Half>(new Half[] { (Half)111 }, new[] { 1, 1, 1, 1 });
        var expected = AiDotNet.Diffusion.Preprocessing.PreprocessorFilters.LabelColors(labels.Cast<double>(), 37, 67, 97).Cast<Half>();
        Assert.Equal(expected.ToArray(), AiDotNet.Diffusion.Preprocessing.PreprocessorFilters.LabelColors(labels, 37, 67, 97).ToArray());
    }

    [Fact]
    public void HalfThreshold_DoesNotRoundTheComparisonBoundary()
    {
        var value = new Tensor<Half>(new Half[] { (Half)0.5 }, new[] { 1 });
        Assert.Equal((Half)1, AiDotNet.Diffusion.Preprocessing.PreprocessorFilters.Above(value, 0.49999).GetFlat(0));
    }

    [Fact]
    public void HalfRescaledCfg_ConstantSamplesStayFinite()
    {
        var unconditional = new Tensor<Half>(new Half[] { (Half)1, (Half)1 }, new[] { 1, 2 });
        var conditional = new Tensor<Half>(new Half[] { (Half)2, (Half)2 }, new[] { 1, 2 });
        Assert.Equal(new Half[] { (Half)3, (Half)3 },
            new RescaledCFG<Half>().Apply(unconditional, conditional, 2, 0).ToArray());
    }

    [Fact]
    public void HalfProjectedGuidance_LargeFiniteValuesMatchWideReference()
    {
        var unconditional = new Tensor<Half>(new[] { 1, 2 });
        var conditional = new Tensor<Half>(new Half[] { (Half)1000, (Half)2000 }, new[] { 1, 2 });
        Assert.Equal(new Half[] { (Half)500, (Half)1000 },
            new AdaptiveProjectedGuidance<Half>().Apply(unconditional, conditional, 1, 0).ToArray());
    }

    [Fact]
    public void HalfAudioNormalization_PreservesSilenceAndNormalizesTinyPeaks()
    {
        var processor = new AiDotNet.Diffusion.Audio.AudioProcessor<Half>();
        var silence = new Tensor<Half>(new[] { 2 });
        Assert.Equal(silence.ToArray(), processor.NormalizeAudio(silence).ToArray());
        var tiny = new Tensor<Half>(new[] { (Half)0, Half.Epsilon }, new[] { 2 });
        var normalized = processor.NormalizeAudio(tiny).ToArray();
        Assert.Equal((Half)0, normalized[0]);
        Assert.Equal((Half)0.95, normalized[1]);
    }

    [Fact]
    public void HalfAudioPowerToDb_SilenceHasAFiniteFloor()
    {
        var processor = new AiDotNet.Diffusion.Audio.AudioProcessor<Half>();
        var method = processor.GetType().GetMethod("PowerToDb",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        var result = (Tensor<Half>)method.Invoke(processor, new object[] { new Tensor<Half>(new[] { 1 }) })!;
        Assert.Equal((Half)(-100), result.GetFlat(0));
    }

    [Fact]
    public void ContentSort_UsesStableTiesAndNativeNaNOrdering()
    {
        var values = new Tensor<double>(new[] { double.NaN, 2.0, 1.0, double.NaN, 1.0, 2.0 }, new[] { 6 });
        Assert.Equal(new[] { 0, 3, 2, 4, 1, 5 },
            AiDotNet.Diffusion.Preprocessing.PreprocessorFilters.StableArgsort(values));
    }

    [Fact]
    public void ContentSort_LargePatchGridRemainsBoundedAndStable()
    {
        const int count = 16384;
        var values = new Tensor<double>(System.Linq.Enumerable.Range(0, count)
            .Select(index => (double)(index % 7)).ToArray(), new[] { count });
        var actual = AiDotNet.Diffusion.Preprocessing.PreprocessorFilters.StableArgsort(values);
        var expected = System.Linq.Enumerable.Range(0, count).OrderBy(index => index % 7).ToArray();
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void HalfPointMask_LargeRadiusPreservesTheCircularBoundary()
    {
        var generator = new AiDotNet.Diffusion.MaskUtilities.MaskFromPoints<Half>(Array.Empty<(int y, int x)>(), pointRadius: 300);
        var mask = generator.Generate(641, 641, new[] { (320, 320) });
        Assert.Equal((Half)1, mask[320, 320]);
        Assert.Equal((Half)1, mask[320, 620]);
        Assert.Equal((Half)0, mask[320, 621]);
        Assert.Equal((Half)0, mask[0, 0]);
    }

    private const double Tolerance = 1e-12;

    private static Tensor<double> Batch(double[] values) => new(new[] { 2, 1, 2, 3 }, new Vector<double>(values));

    private static readonly double[] Unconditional = { 0.1, -0.4, 0.7, 0.2, -0.3, 0.5, 1.2, -0.8, 0.3, 0.0, 0.6, -1.1 };
    private static readonly double[] Conditional = { 0.3, -0.1, 0.9, -0.2, 0.1, 0.4, 0.7, -0.2, 0.8, 0.5, 0.1, -0.9 };

    private static void AssertClose(double[] expected, Tensor<double> actual)
    {
        var values = actual.ToArray();
        Assert.Equal(expected.Length, values.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - values[i]) <= Tolerance, $"[{i}] expected {expected[i]:R}, got {values[i]:R}");
    }

    [Fact(Timeout = 120000)]
    public async Task AdaptiveProjectedGuidance_MatchesDiffusersNormalizedGuidance()
    {
        await Task.Yield();
        var guidance = new AdaptiveProjectedGuidance<double>(projectionStrength: 0.5);
        var result = guidance.Apply(Batch(Unconditional), Batch(Conditional), 7.5, 0.5);
        AssertClose(new[]
        {
            1.30870535714286, 1.94709821428571, 1.32611607142857, -2.60580357142857, 2.60290178571429, -0.638392857142857, -2.49140625, 3.68325892857143, 4.11696428571429, 3.79185267857143, -3.14162946428571, 0.324665178571429,
        }, result);
    }

    [Fact(Timeout = 120000)]
    public async Task RescaledCFG_MatchesDiffusersRescaleNoiseCfg()
    {
        await Task.Yield();
        var guidance = new RescaledCFG<double>(rescaleWeight: 0.7);
        var result = guidance.Apply(Batch(Unconditional), Batch(Conditional), 7.5, 0.5);
        AssertClose(new[]
        {
            0.695981480687581, 0.804728587045016, 0.956974535945424, -1.21796759120327, 1.17446874866029, -0.108747106357434, -1.11369009479563, 1.61594249048778, 1.76880191526366, 1.63777955117005, -1.37573482298284, 0.174696485458139,
        }, result);
    }
}
