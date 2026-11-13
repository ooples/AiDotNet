using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.StyleEmotion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// References for the TTS engine-op helpers in a narrow T (Half, float): every case here overflowed, lost its index, or
/// moved a constant table by more than T's own rounding when the values were formed in T.
/// </summary>
public sealed class TtsNarrowPrecisionReferenceTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static T[] RoundedOnce<T>(double[] values) => values.Select(v => MathHelper.GetNumericOperations<T>().FromDouble(v)).ToArray();

    // Glow-TTS maximum_path_each, in double.
    private static int[] ReferenceDurations(double[,] scores)
    {
        const double MaxNegative = -1e9;
        int tx = scores.GetLength(0), ty = scores.GetLength(1);
        var value = (double[,])scores.Clone();
        for (int y = 0; y < ty; y++)
            for (int x = Math.Max(0, tx + y - ty); x < Math.Min(tx, y + 1); x++)
            {
                double vCur = x == y ? MaxNegative : value[x, y - 1];
                double vPrev = x == 0 ? (y == 0 ? 0.0 : MaxNegative) : value[x - 1, y - 1];
                value[x, y] = Math.Max(vCur, vPrev) + value[x, y];
            }
        var durations = new int[tx];
        int index = tx - 1;
        for (int y = ty - 1; y >= 0; y--)
        {
            durations[index]++;
            if (index != 0 && (index == y || value[index, y - 1] < value[index - 1, y - 1])) index--;
        }
        return durations;
    }

    [Fact]
    public void MonotonicDurations_InHalf_DoNotOverflowTheAccumulatedPathScores()
    {
        // Accumulated along 40 frames the path scores reach -80,000, past Half's -65,504: without renormalization
        // both tokens' values become -inf, the trace-back compares -inf < -inf and the path is lost.
        const int frames = 40;
        var scores = new double[2, frames];
        for (int y = 0; y < frames; y++)
        {
            scores[0, y] = y < 38 ? -2000 : -4000;
            scores[1, y] = y < 38 ? -4000 : -2000;
        }
        var expected = ReferenceDurations(scores);
        Assert.Equal(new[] { 38, 2 }, expected);

        var flat = new Half[2 * frames];
        for (int i = 0; i < 2; i++)
            for (int y = 0; y < frames; y++) flat[i * frames + y] = (Half)scores[i, y];
        Assert.Equal(expected, TtsModelBase<Half>.MonotonicDurations(Engine, new Tensor<Half>(flat, new[] { 2, frames })));
    }

    [Fact]
    public void FirstArgMax_InHalf_ReturnsIndicesPastHalfRange()
    {
        // An index carried in Half is inexact past 2,048 and infinite past 65,504.
        var values = new Half[70000];
        values[60000] = (Half)1.0;
        values[65000] = (Half)1.0;
        Assert.Equal(60000, TtsModelBase<Half>.FirstArgMax(Engine, new Tensor<Half>(values, new[] { values.Length })));

        var small = new Half[] { (Half)0.5, (Half)2.0, (Half)2.0, (Half)1.0 };
        Assert.Equal(1, TtsModelBase<Half>.FirstArgMax(Engine, new Tensor<Half>(small, new[] { 4 })));
    }

    [Fact]
    public void ScaledCharacterCodes_InHalf_AreFiniteAboveHalfRange()
    {
        // U+FFFD (65,533) cast to Half is +inf before the 1/128 scaling.
        var codes = TtsModelBase<Half>.ScaledCharacterCodes(Engine, "A\uFFFD", 2).ToArray();
        Assert.Equal((Half)(65.0 / 128.0), codes[0]);
        Assert.Equal((Half)(65533.0 / 128.0), codes[1]);
        Assert.False(double.IsInfinity((double)codes[1]));

        var exact = TtsModelBase<double>.ScaledCharacterCodes(Engine, "z\uFFFD", 2).ToArray();
        Assert.Equal(new[] { 'z' / 128.0, 65533.0 / 128.0 }, exact);
    }

    [Theory]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NaN)]
    [InlineData(-65504.0)]
    public void CheckedInt32_InHalf_FollowsCheckedConversion(double value)
    {
        // -2^31 rounds to -inf in Half, so a bound compared in Half lets -inf through; NaN must throw as well.
        var tensor = new Tensor<Half>(new[] { (Half)3.0, (Half)value }, new[] { 2 });
        if (double.IsNaN(value) || double.IsInfinity(value))
        {
            var error = Assert.Throws<OverflowException>(() => TtsModelBase<Half>.CheckedInt32(Engine, tensor, "overflow"));
            Assert.Equal("overflow", error.Message);
        }
        else
        {
            Assert.Equal(new[] { 3, (int)value }, TtsModelBase<Half>.CheckedInt32(Engine, tensor, "overflow"));
        }
    }

    [Fact]
    public void CheckedInt32_InDouble_RejectsValuesOutsideInt32()
    {
        Assert.Throws<OverflowException>(() =>
            TtsModelBase<double>.CheckedInt32(Engine, new Tensor<double>(new[] { 2147483648.0 }, new[] { 1 }), "m"));
        Assert.Throws<OverflowException>(() =>
            TtsModelBase<double>.CheckedInt32(Engine, new Tensor<double>(new[] { -2147483649.0 }, new[] { 1 }), "m"));
        Assert.Equal(new[] { int.MinValue, int.MaxValue },
            TtsModelBase<double>.CheckedInt32(Engine, new Tensor<double>(new[] { -2147483648.0, 2147483647.0 }, new[] { 2 }), "m"));
    }

    [Fact]
    public void PhonemeMeanEnergy_InHalf_DoesNotOverflowTheSquares()
    {
        // exp(5)^2 = 22,026 per channel; 80 channels sum to 1.76e6, past Half's range.
        var mel = new Half[4 * 80];
        for (int i = 0; i < mel.Length; i++) mel[i] = (Half)5.0;
        var energy = ForwardTacotron<Half>.PhonemeMeanEnergy(Engine, new Tensor<Half>(mel, new[] { 4, 80 }), new[] { 2, 0, 2 }).ToArray();
        double expected = Math.Sqrt(80) * Math.Exp(5);
        Assert.Equal(3, energy.Length);
        Assert.Equal(expected, (double)energy[0], expected * 4e-3);
        Assert.Equal(0.0, (double)energy[1]);
        Assert.Equal(expected, (double)energy[2], expected * 4e-3);
    }

    [Fact]
    public void DeepVoice3PositionalEncoding_InFloat_IsTheDoubleTableRoundedOnce()
    {
        const int length = 600, dimension = 64;
        const double rate = 1.3, weight = 1.0;
        var table = new double[length * dimension];
        for (int i = 0; i < length; i++)
            for (int k = 0; k < dimension; k++)
            {
                double angle = rate * (i + 1) / Math.Pow(10000.0, (k - k % 2) / (double)dimension);
                table[i * dimension + k] = weight * (k % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
            }
        Assert.Equal(RoundedOnce<float>(table), DeepVoice3<float>.PositionalEncoding(length, dimension, rate, weight).ToArray());
    }

    [Fact]
    public void SpeedySpeechPhonemePositions_InFloat_AreTheDoubleTableRoundedOnce()
    {
        const int frames = 300, channels = 64;
        var table = new double[frames * channels];
        for (int p = 0; p < frames; p++)
            for (int i = 0; i < channels / 2; i++)
            {
                double angle = p / Math.Pow(10000.0, 2.0 * i / channels);
                table[p * channels + 2 * i] = Math.Sin(angle);
                table[p * channels + 2 * i + 1] = Math.Cos(angle);
            }
        Assert.Equal(RoundedOnce<float>(table), SpeedySpeech<float>.PhonemePositions(new[] { frames }, channels).ToArray());
    }

    [Fact]
    public void TacotronHannWindow_IsTheDoubleWindowRoundedOnce()
    {
        const int size = 1024;
        var window = new double[size];
        for (int i = 0; i < size; i++) window[i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / size);
        Assert.Equal(RoundedOnce<float>(window), new Tacotron<float>.CenteredHannWindow(size).Create(size).ToArray());
        Assert.Equal(RoundedOnce<Half>(window), new Tacotron<Half>.CenteredHannWindow(size).Create(size).ToArray());
    }

    [Fact]
    public void GradTtsStepScales_InHalf_AreRoundedOnce()
    {
        const int steps = 50;
        const double betaMin = 0.05, betaMax = 20.0, h = 1.0 / steps;
        var scales = new double[steps];
        for (int i = 0; i < steps; i++) scales[i] = 0.5 * (betaMin + (betaMax - betaMin) * (1.0 - (i + 0.5) * h)) * h;
        Assert.Equal(RoundedOnce<Half>(scales), GradTTS<Half>.StepScales(steps, betaMin, betaMax));
        Assert.Equal(RoundedOnce<float>(scales), GradTTS<float>.StepScales(steps, betaMin, betaMax));
    }

    [Fact]
    public void CoMoSpeechCoefficients_InHalf_AreRoundedOnce()
    {
        const int n = 50;
        const double sigmaMin = 0.002, sigmaMax = 80.0, rho = 7.0;
        double lo = Math.Pow(sigmaMin, 1 / rho), hi = Math.Pow(sigmaMax, 1 / rho);
        var levels = new double[n];
        for (int i = 0; i < n; i++) levels[i] = Math.Pow(lo + (double)i / (n - 1) * (hi - lo), rho);

        var inverse = levels.Select(t => 1.0 / t).ToArray();
        var step = levels.Select((t, i) => (i > 0 ? levels[i - 1] : 0.0) - t).ToArray();
        var renoise = levels.Select(t => Math.Sqrt(t * t - sigmaMin * sigmaMin)).ToArray();

        var (inverseLevel, stepSize) = CoMoSpeech<Half>.EulerCoefficients(levels);
        Assert.Equal(RoundedOnce<Half>(inverse), inverseLevel);
        Assert.Equal(RoundedOnce<Half>(step), stepSize);
        // At t = sigma_min the scale is exactly 0; formed in T the difference of squares can fall below zero (NaN).
        var scales = CoMoSpeech<Half>.RenoiseScales(levels, sigmaMin);
        Assert.Equal(RoundedOnce<Half>(renoise), scales);
        Assert.Equal((Half)0.0, scales[0]);
        Assert.Equal(RoundedOnce<float>(renoise), CoMoSpeech<float>.RenoiseScales(levels, sigmaMin));
    }

    [Fact]
    public void DiTToTtsNoiseLevels_InHalf_AreRoundedOnce()
    {
        const int steps = 50;
        const double scale = 0.5;
        var signal = new double[steps + 1];
        var noise = new double[steps + 1];
        for (int k = 0; k <= steps; k++)
        {
            double cosinePower = Math.Pow(Math.Cos(k / (double)steps * Math.PI / 2), 2);
            cosinePower = Math.Min(Math.Max(cosinePower, 1e-9), 1 - 1e-9);
            double logSnr = Math.Log(cosinePower) - Math.Log(1 - cosinePower) + 2 * Math.Log(scale);
            double signalPower = 1 / (1 + Math.Exp(-logSnr));
            signal[k] = Math.Sqrt(signalPower);
            noise[k] = Math.Sqrt(1 - signalPower);
        }
        var levels = DiTToTTS<Half>.NoiseLevels(steps, scale);
        Assert.Equal(RoundedOnce<Half>(signal), levels.Signal);
        Assert.Equal(RoundedOnce<Half>(noise), levels.Noise);
    }

    [Fact]
    public void E3TtsReverseCoefficients_InHalf_AreRoundedOnce()
    {
        const int steps = 50;
        const double minimumAlphaBar = 1e-4;
        double AlphaBar(int s) => Math.Exp(Math.Log(minimumAlphaBar) * Math.Pow(1 - Math.Cos(s / (double)steps * Math.PI / 2), 1.5));
        var expected = new double[5][];
        for (int c = 0; c < 5; c++) expected[c] = new double[steps];
        for (int s = 0; s < steps; s++)
        {
            double alphaBar = AlphaBar(s + 1), previous = AlphaBar(s), alpha = alphaBar / previous, beta = 1 - alpha;
            double noiseStd = Math.Sqrt(Math.Max(1e-12, 1 - alphaBar));
            expected[0][s] = Math.Sqrt(alphaBar);
            expected[1][s] = 1.0 / noiseStd;
            expected[2][s] = beta / noiseStd;
            expected[3][s] = 1.0 / Math.Sqrt(alpha);
            expected[4][s] = Math.Sqrt(beta * (1 - previous) / Math.Max(1e-12, 1 - alphaBar));
        }
        var actual = E3TTS<Half>.ReverseCoefficients(steps, minimumAlphaBar);
        Assert.Equal(RoundedOnce<Half>(expected[0]), actual.SqrtAlphaBar);
        Assert.Equal(RoundedOnce<Half>(expected[1]), actual.InverseNoiseStd);
        Assert.Equal(RoundedOnce<Half>(expected[2]), actual.NoiseScale);
        Assert.Equal(RoundedOnce<Half>(expected[3]), actual.InverseSqrtAlpha);
        Assert.Equal(RoundedOnce<Half>(expected[4]), actual.PosteriorStd);
    }

    [Fact]
    public void StyleTtsZsAngularSchedule_IsRoundedOnce()
    {
        const int steps = 50;
        var alpha = new double[steps + 1];
        var sigma = new double[steps + 1];
        for (int n = 0; n <= steps; n++)
        {
            double tau = (double)n / steps;
            alpha[n] = Math.Cos(Math.PI * tau / 2);
            sigma[n] = Math.Sin(Math.PI * tau / 2);
        }
        var schedule = StyleTTSZS<float>.AngularSchedule(steps);
        Assert.Equal(RoundedOnce<float>(alpha), schedule.Alpha);
        Assert.Equal(RoundedOnce<float>(sigma), schedule.Sigma);
        var half = StyleTTSZS<Half>.AngularSchedule(steps);
        Assert.Equal(RoundedOnce<Half>(alpha), half.Alpha);
        Assert.Equal(RoundedOnce<Half>(sigma), half.Sigma);
    }

    [Fact]
    public void MeloTtsSymbolIds_SaturateLikeTheIntConversionAndClamp()
    {
        var input = new Tensor<double>(new[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN, 3.5, 2.5, 12.0, -3.0 }, new[] { 7 });
        Assert.Equal(new[] { 9, 0, 0, 4, 2, 9, 0 }, MeloTTS<double>.SymbolIds(Engine, input, 10));
    }
}
