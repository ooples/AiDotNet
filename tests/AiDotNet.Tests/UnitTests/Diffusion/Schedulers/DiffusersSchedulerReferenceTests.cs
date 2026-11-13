using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.Guidance;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Schedulers;

/// <summary>
/// The Euler, Euler ancestral and DPM-Solver++ multistep schedulers reproduce diffusers 0.35
/// (<c>tools/reference-data/diffusers_diffusion_reference.py</c>): ten trailing-spaced steps over Stable Diffusion's
/// scaled-linear betas with the toy model output m(x, t) = 0.5 x + c (1 + t / 1000) end at the reference samples.
/// </summary>
/// <remarks>
/// Before this, DPM-Solver++ built its log-SNR from the Karras sigma instead of the VP sigma and added
/// alpha_t (exp(-h) - 1) D0 where the solver subtracts it, and the Euler schedulers converted a v-prediction with
/// the VP coefficients although their sample lives in sigma space. The consistency model re-noised a multistep
/// sample with sigma_next instead of Algorithm 1's sqrt(sigma_next^2 - sigma_min^2).
/// </remarks>
public class DiffusersSchedulerReferenceTests
{
    private const double Tolerance = 1e-10;
    private static readonly double[] InitialSample = { 1.0, -0.5, 0.25, 2.0 };
    private static readonly double[] Offset = { 0.1, -0.2, 0.3, -0.4 };

    private static SchedulerConfig<double> StableDiffusionConfig(DiffusionPredictionType predictionType) =>
        new(1000, 0.00085, 0.012, BetaSchedule.ScaledLinear, clipSample: false, predictionType: predictionType);

    private static double[] Run(INoiseScheduler<double> scheduler, bool ancestralNoise)
    {
        scheduler.SetTimesteps(10);
        var sample = new Vector<double>(InitialSample);
        int step = 0;
        foreach (int t in scheduler.Timesteps)
        {
            var output = new Vector<double>(sample.Length);
            var noise = new Vector<double>(sample.Length);
            for (int i = 0; i < sample.Length; i++)
            {
                output[i] = 0.5 * sample[i] + Offset[i] * (1.0 + t / 1000.0);
                noise[i] = Math.Sin(1.3 * (step + 1) * (i + 1));
            }

            sample = scheduler.Step(output, t, sample, 1.0, ancestralNoise ? noise : null);
            step++;
        }

        return sample.ToArray();
    }

    private static void AssertClose(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= Tolerance, $"[{i}] expected {expected[i]:R}, got {actual[i]:R}");
    }

    [Fact(Timeout = 120000)]
    public async Task Timesteps_AreDiffusersTrailingSpacing()
    {
        await Task.Yield();
        var scheduler = new EulerDiscreteScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.Epsilon));
        scheduler.SetTimesteps(10);
        Assert.Equal(new[] { 999, 899, 799, 699, 599, 499, 399, 299, 199, 99 }, scheduler.Timesteps);
    }

    [Fact(Timeout = 120000)]
    public async Task Euler_Epsilon_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new EulerDiscreteScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.Epsilon));
        AssertClose(new[] { -0.262875569327016, 0.555948232611201, -0.843988046902525, 1.17229065313674 },
            Run(scheduler, ancestralNoise: false));
    }

    [Fact(Timeout = 120000)]
    public async Task Euler_VPrediction_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new EulerDiscreteScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.VPrediction));
        AssertClose(new[] { -0.113811287473735, 0.236431801580701, -0.357584111248795, 0.490482056427865 },
            Run(scheduler, ancestralNoise: false));
    }

    [Fact(Timeout = 120000)]
    public async Task EulerAncestral_Epsilon_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new EulerAncestralDiscreteScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.Epsilon));
        AssertClose(new[] { -0.390360182153464, 0.533637051239677, -1.00917052576202, 0.927967158970461 },
            Run(scheduler, ancestralNoise: true));
    }

    [Fact(Timeout = 120000)]
    public async Task EulerAncestral_VPrediction_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new EulerAncestralDiscreteScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.VPrediction));
        AssertClose(new[] { -0.284217462288841, 0.154337565838926, -0.444837736354689, 0.301483692977084 },
            Run(scheduler, ancestralNoise: true));
    }

    [Fact(Timeout = 120000)]
    public async Task DPMSolverMultistep_SecondOrder_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new DPMSolverMultistepScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.Epsilon), solverOrder: 2);
        AssertClose(new[] { 1.91301665868667, 0.264969760654408, -1.76112233365753, 8.71194567736432 },
            Run(scheduler, ancestralNoise: false));
    }

    [Fact(Timeout = 120000)]
    public async Task DPMSolverMultistep_ThirdOrder_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new DPMSolverMultistepScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.Epsilon), solverOrder: 3);
        AssertClose(new[] { 1.93788526846478, 0.280834257993723, -1.80678631863168, 8.87487810583401 },
            Run(scheduler, ancestralNoise: false));
    }

    [Fact(Timeout = 120000)]
    public async Task ConsistencyRenoise_MatchesDiffusers()
    {
        await Task.Yield();
        // diffusers CMStochasticIterativeScheduler.step from the first to the second of two Karras sigmas
        const double sigmaNext = 2.2404396533966064;
        var denoised = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 20.0522322038777, -10.1011127621746, 5.15055192806836, 39.8044777668122 }));
        var noise = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { Math.Sin(1.3), Math.Sin(2.6), Math.Sin(3.9), Math.Sin(5.2) }));

        var renoised = ConsistencyModel<double>.RenoiseToSigma(denoised, noise, sigmaNext, 0.002);

        AssertClose(new[] { 22.2110253106901, -8.94616350754432, 3.60965396672485, 37.8251517128031 }, renoised.ToArray());
    }
}

/// <summary>
/// Adaptive projected guidance and rescaled CFG reproduce diffusers 0.35
/// (<c>tools/reference-data/diffusers_diffusion_reference.py</c>) on a batch of two samples.
/// </summary>
/// <remarks>
/// Before this, APG scaled the orthogonal component and kept the parallel one (the paper keeps the orthogonal
/// component and down-weights the oversaturating parallel one), and both methods pooled their projection and
/// standard deviations over the whole batch instead of per sample.
/// <para>
/// The class also holds the low-precision (Half) regressions of the diffusion sweep. It lives beside the scheduler
/// references so the "Unit - 03a" shard (UnitTests.Diffusion.Schedulers) runs it; in its own Guidance namespace no
/// shard selected it.
/// </para>
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
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
            ?? throw new MissingMethodException(nameof(AiDotNet.Diffusion.Audio.AudioProcessor<Half>), "PowerToDb");
        var result = method.Invoke(processor, new object[] { new Tensor<Half>(new[] { 1 }) }) as Tensor<Half>
            ?? throw new InvalidOperationException("PowerToDb did not return a Half tensor.");
        Assert.Equal((Half)(-100), result.GetFlat(0));
    }

    [Fact]
    public void HalfRewardGuidance_ClipsByTheTrueGradientNorm()
    {
        // 64 entries of 64: |g| = 512, but the squares (262144 in total) overflow Half. The old double norm clips the
        // gradient to g / 512 = 0.125; an overflowed norm clips it to zero and drops the reward guidance.
        var guidance = new AiDotNet.Diffusion.Alignment.RewardGuidance<Half>(
            guidanceScale: 1.0, truncationTimestep: 1.0, gradientClipNorm: 1.0);
        var gradient = new Vector<Half>(Enumerable.Repeat((Half)64, 64).ToArray());
        var guided = guidance.ApplyGuidance(new Vector<Half>(64), gradient, 0.0);
        Assert.All(guided.ToArray(), value => Assert.Equal((Half)(-0.125), value));
    }

    [Fact]
    public void HalfRelativeSquaredError_LargeMagnitudesStayFinite()
    {
        // The Griffin-Lim convergence and spectral convergence ratio: STFT magnitudes of 300 square past Half's range.
        var target = new Tensor<Half>(Enumerable.Repeat((Half)300, 81).ToArray(), new[] { 81 });
        var silence = new Tensor<Half>(new[] { 81 });
        Assert.Equal((Half)1, AiDotNet.Diffusion.DiffusionVectorOps.RelativeSquaredError(target, silence));
    }

    [Fact]
    public void HalfMelFilterbank_LargeFftKeepsTheDoublePrecisionSupport()
    {
        // nFft = 8192 has 4097 frequency bins; Half cannot represent the odd bin indices above 2048, so building the
        // triangles in Half moves filter edges. The support must match the double filterbank exactly.
        var half = MelFilterbank<Half>(64, 8192, 22050).ToArray();
        var wide = MelFilterbank<double>(64, 8192, 22050).ToArray();
        Assert.Equal(wide.Length, half.Length);
        for (int i = 0; i < wide.Length; i++)
            Assert.True((wide[i] != 0) == ((double)half[i] != 0), $"[{i}] double {wide[i]:R}, Half {(double)half[i]:R}");
    }

    private static Tensor<TNum> MelFilterbank<TNum>(int nMels, int nFft, int sampleRate)
    {
        var method = typeof(AiDotNet.Diffusion.Audio.MelSpectrogram<TNum>).GetMethod("CreateMelFilterbank",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)
            ?? throw new MissingMethodException("MelSpectrogram", "CreateMelFilterbank");
        return method.Invoke(null, new object[] { nMels, nFft, sampleRate, 0.0, sampleRate / 2.0 }) as Tensor<TNum>
            ?? throw new InvalidOperationException("CreateMelFilterbank did not return a tensor.");
    }

    [Fact]
    public void HalfTileBlur_LargeRadiusAveragesTheWholeWindow()
    {
        // A 47 x 47 window holds 2209 pixels; Half convolution stops counting at 2048, which turned the mean of a
        // constant 0.75 image into about 0.995.
        var image = new Tensor<Half>(Enumerable.Repeat((Half)0.75, 47 * 47).ToArray(), new[] { 1, 1, 47, 47 });
        var blurred = new AiDotNet.Diffusion.Preprocessing.TilePreprocessor<Half>(blurRadius: 23).Transform(image);
        for (int c = 0; c < 3; c++)
            Assert.Equal((Half)0.75, blurred[0, c, 23, 23]);
    }

    [Fact]
    public void HalfConsistencySigmaToTimestep_CountsMoreThan2048Levels()
    {
        // 4097 exactly representable ascending Half levels; the level at index 3000 maps to timestep 3000. A 0/1 sum
        // in Half rounds the 3001 levels at or below it to 3000 and lands one bracket low.
        var sigmas = Enumerable.Range(0, 4097)
            .Select(k => (Half)((1.0 + (k % 1024) / 1024.0) * Math.Pow(2, k / 1024)))
            .ToArray();
        Assert.Equal(3000,
            AiDotNet.Diffusion.FastGeneration.ConsistencyModel<Half>.SigmaToTimestep(sigmas, (double)sigmas[3000]));
    }

    [Fact]
    public void HalfTimestepEmbedding_MatchesTheDoubleAngles()
    {
        // FlagDiT's embedding at t = 999 with hidden size 32: sin/cos(t f_j), f_j = exp(-ln(10000) j / 15).
        var actual = AiDotNet.Diffusion.DiffusionVectorOps.SinusoidalTimestepEmbedding<Half>(999, 32).ToArray();
        for (int j = 0; j < 16; j++)
        {
            double angle = 999 * Math.Exp(-Math.Log(10000.0) * j / 15);
            Assert.InRange(Math.Abs((double)actual[j] - Math.Sin(angle)), 0, 1e-3);
            Assert.InRange(Math.Abs((double)actual[16 + j] - Math.Cos(angle)), 0, 1e-3);
        }
    }

    [Fact]
    public void HalfPositionEmbedding_MatchesTheDoubleAnglesPast2048Positions()
    {
        // MMDiT's interleaved table for 4096 patches and hidden size 8: Half rounds positions above 2048.
        var actual = AiDotNet.Diffusion.DiffusionVectorOps.InterleavedPositionEmbedding<Half>(4096, 8);
        foreach (int position in new[] { 0, 1, 2049, 3001, 4095 })
        {
            for (int k = 0; k < 4; k++)
            {
                double angle = position * Math.Pow(10000.0, -2.0 * k / 8);
                Assert.InRange(Math.Abs((double)actual[position, 2 * k] - Math.Sin(angle)), 0, 1e-3);
                Assert.InRange(Math.Abs((double)actual[position, 2 * k + 1] - Math.Cos(angle)), 0, 1e-3);
            }
        }
    }

    [Fact]
    public void HalfSpeakerEmbedding_LongClipStatisticsMatchTheWideReference()
    {
        // 1000 frames x 128 mel channels around log(0.01): per-channel sums pass 2048, where Half drops whole units.
        const int frames = 1000, channels = 128;
        var values = new Half[frames * channels];
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < channels; c++)
                values[f * channels + c] = (Half)(-4.6 + 0.25 * ((f * 7 + c * 3) % 5));
        var mel = new Tensor<Half>(values, new[] { frames, channels });

        // 128 channels leave room for two statistics each: mean and sqrt(population variance + 1e-6), L2-normalized.
        var expected = new double[2 * channels];
        for (int c = 0; c < channels; c++)
        {
            double mean = 0, variance = 0;
            for (int f = 0; f < frames; f++) mean += (double)values[f * channels + c];
            mean /= frames;
            for (int f = 0; f < frames; f++) variance += Math.Pow((double)values[f * channels + c] - mean, 2);
            expected[2 * c] = mean;
            expected[2 * c + 1] = Math.Sqrt(variance / frames + 1e-6);
        }

        double norm = Math.Sqrt(expected.Sum(v => v * v) + 1e-8);
        var actual = AiDotNet.Diffusion.AudioDiffusionModelBase<Half>.SpeakerStatisticsEmbedding(mel, 256).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(Math.Abs((double)actual[i] - expected[i] / norm), 0, 1e-3);
    }

    [Fact]
    public void HalfStylePresence_LargeParameterVectorsStayFinite()
    {
        // delta = style = 4 over 4096 parameters: the dot products (65536) overflow Half; the cosine is exactly 1.
        var eraser = new AiDotNet.Diffusion.Safety.SGRACEEraser<Half>();
        var style = new Vector<Half>(Enumerable.Repeat((Half)4, 4096).ToArray());
        eraser.AddStyleVector(style);
        Assert.Equal((Half)1, eraser.MeasureStylePresence(style, new Vector<Half>(4096), 0));
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
