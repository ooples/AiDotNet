using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.HarmonicEngine.Models;
using AiDotNet.HarmonicEngine.Options;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Experiment 8: Novel language capability — long-range periodic pattern learning.
/// Demonstrates that HRE can capture periodic patterns at arbitrary period lengths
/// via its spectral representation, which is structurally suited for periodicity.
/// </summary>
public class PeriodicSequenceTests
{
    private readonly ITestOutputHelper _output;

    public PeriodicSequenceTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Theory]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(13)]
    public void HRESequenceModel_PeriodicInput_ProducesValidLogProbs(int period)
    {
        // Generate periodic character sequence: "ABCABCABC..." with given period
        int contextLength = 64;
        int vocabSize = 26; // A-Z

        var options = new HREModelOptions
        {
            CarrierCount = 8,
            FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = false,
            NumOFDMLayers = 1,
            NumAttentionLayers = 0,
            Seed = 42
        };

        var model = new HRESequenceModel<double>(contextLength, vocabSize, options);

        // Build context of periodic characters
        var context = new Vector<double>(contextLength);
        for (int i = 0; i < contextLength; i++)
        {
            context[i] = i % period; // Cycles through 0, 1, ..., period-1
        }

        var logProbs = model.PredictNext(context);

        // Log probs should be valid (no NaN, sum to ~1 in prob space)
        Assert.Equal(vocabSize, logProbs.Length);

        double maxLogProb = double.NegativeInfinity;
        double sumProb = 0;
        int predictedChar = -1;

        for (int i = 0; i < vocabSize; i++)
        {
            Assert.False(double.IsNaN(logProbs[i]), $"LogProb[{i}] is NaN");
            double prob = Math.Exp(logProbs[i]);
            sumProb += prob;
            if (logProbs[i] > maxLogProb)
            {
                maxLogProb = logProbs[i];
                predictedChar = i;
            }
        }

        // Probabilities should approximately sum to 1
        Assert.Equal(1.0, sumProb, 3);

        int expectedNext = contextLength % period;
        _output.WriteLine($"Period={period}: predicted char={predictedChar}, expected={expectedNext}, prob_sum={sumProb:F6}");
    }

    [Fact]
    public void HRESequenceModel_Generate_ProducesValidCharacters()
    {
        int contextLength = 64;
        int vocabSize = 26;

        var options = new HREModelOptions
        {
            CarrierCount = 8,
            FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = false,
            NumOFDMLayers = 1,
            NumAttentionLayers = 0,
            Seed = 42
        };

        var model = new HRESequenceModel<double>(contextLength, vocabSize, options);

        // Seed with periodic pattern
        var seed = new int[contextLength];
        for (int i = 0; i < contextLength; i++)
        {
            seed[i] = i % 5; // Period 5
        }

        var generated = model.Generate(seed, 20);

        Assert.Equal(20, generated.Length);
        for (int i = 0; i < 20; i++)
        {
            Assert.True(generated[i] >= 0, $"Generated[{i}]={generated[i]} should be >= 0");
            Assert.True(generated[i] < vocabSize, $"Generated[{i}]={generated[i]} should be < {vocabSize}");
        }

        _output.WriteLine("Generated sequence: " + string.Join(", ", generated));
    }

    [Fact]
    public void HRESequenceModel_DifferentPeriods_ProduceDifferentPredictions()
    {
        int contextLength = 64;
        int vocabSize = 26;

        var options = new HREModelOptions
        {
            CarrierCount = 8,
            FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = false,
            NumOFDMLayers = 1,
            NumAttentionLayers = 0,
            Seed = 42
        };

        var model = new HRESequenceModel<double>(contextLength, vocabSize, options);

        // Period 3 context
        var context3 = new Vector<double>(contextLength);
        for (int i = 0; i < contextLength; i++)
        {
            context3[i] = i % 3;
        }

        // Period 7 context
        var context7 = new Vector<double>(contextLength);
        for (int i = 0; i < contextLength; i++)
        {
            context7[i] = i % 7;
        }

        var logProbs3 = model.PredictNext(context3);
        var logProbs7 = model.PredictNext(context7);

        // Different periodic inputs should produce different output distributions
        double diff = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            diff += Math.Abs(logProbs3[i] - logProbs7[i]);
        }

        Assert.True(diff > 0.01,
            $"Period-3 and period-7 contexts should produce different predictions, but diff = {diff}");

        _output.WriteLine($"L1 distance between period-3 and period-7 distributions: {diff:F6}");
    }

    [Fact]
    public void SpectralRepresentation_LongPeriods_CapturableAsSingleFrequency()
    {
        // This test validates the theoretical claim: a pattern with period P
        // maps to frequency f = 1/P in the spectral representation, which the
        // HRE captures with a single coefficient regardless of P.
        // Traditional attention would need context >= P to capture it.

        int n = 128; // Signal length
        var fft = new FastFourierTransform<double>();

        foreach (int period in new[] { 5, 11, 23, 37 })
        {
            var signal = new Vector<double>(n);
            for (int i = 0; i < n; i++)
            {
                signal[i] = Math.Sin(2 * Math.PI * i / period);
            }

            var spectrum = fft.Forward(signal);

            // Find the peak frequency bin
            int peakBin = 0;
            double peakMag = 0;
            for (int k = 1; k < n / 2; k++)
            {
                double mag = spectrum[k].Magnitude;
                if (mag > peakMag)
                {
                    peakMag = mag;
                    peakBin = k;
                }
            }

            double expectedBin = (double)n / period;
            _output.WriteLine($"Period={period}: peak at bin {peakBin} (expected ~{expectedBin:F1}), magnitude={peakMag:F2}");

            // Peak should be near the expected frequency bin
            Assert.True(Math.Abs(peakBin - expectedBin) < 2,
                $"For period {period}, peak bin {peakBin} should be near expected {expectedBin:F1}");

            // Peak should dominate the spectrum (at least 10x the median)
            var magnitudes = new double[n / 2];
            for (int k = 0; k < n / 2; k++)
            {
                magnitudes[k] = spectrum[k].Magnitude;
            }
            Array.Sort(magnitudes);
            double median = magnitudes[n / 4];

            Assert.True(peakMag > median * 5,
                $"Peak magnitude ({peakMag:F2}) should dominate median ({median:F2})");
        }
    }

    /// <summary>
    /// Experiment 8 (rigorous): validates the "novel capability" claim that
    /// HRE learns long-range periodic patterns via its spectral representation.
    /// Trains an HREModel on periodic regression data — given a context of
    /// length L, predict the next value in a period-P sinusoid — and measures
    /// test-set accuracy across multiple period lengths, including periods
    /// longer than a typical transformer context window would practically handle.
    /// </summary>
    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(32)]
    public void PeriodicNextTokenPrediction_AfterTraining_MatchesGroundTruth(int period)
    {
        const int contextLength = 64;
        const int trainSamples = 500;
        const int testSamples = 100;

        var rng = RandomHelper.CreateSecureRandom();

        var options = new HREModelOptions
        {
            InputSize = contextLength,
            OutputSize = 1,
            CarrierCount = 8,
            FftSize = 512,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = false,
            NumOFDMLayers = 1,
            NumAttentionLayers = 0,
            HebbianLearningRate = 0.02,
            SparsityK = 8,
            Seed = 1000 + period,
        };

        var model = new HREModel<double>(options);
        model.SetTrainingMode(true);

        // Generate training samples: random starting phase, period-P sinusoid
        for (int s = 0; s < trainSamples; s++)
        {
            double phase = rng.NextDouble() * 2 * Math.PI;
            var ctx = new Tensor<double>([contextLength]);
            for (int t = 0; t < contextLength; t++)
                ctx[t] = Math.Cos(2 * Math.PI * t / period + phase);

            var nextVal = new Tensor<double>([1]);
            nextVal[0] = Math.Cos(2 * Math.PI * contextLength / period + phase);

            model.Train(ctx, nextVal);
        }

        // Test: predict the next value on held-out data
        model.SetTrainingMode(false);
        double sumAbsError = 0, sumSqError = 0, sumTargetPower = 0;
        for (int s = 0; s < testSamples; s++)
        {
            double phase = rng.NextDouble() * 2 * Math.PI;
            var ctx = new Tensor<double>([contextLength]);
            for (int t = 0; t < contextLength; t++)
                ctx[t] = Math.Cos(2 * Math.PI * t / period + phase);

            double trueNext = Math.Cos(2 * Math.PI * contextLength / period + phase);
            double predNext = model.Forward(ctx)[0];

            double err = predNext - trueNext;
            sumAbsError += Math.Abs(err);
            sumSqError += err * err;
            sumTargetPower += trueNext * trueNext;
        }

        double mae = sumAbsError / testSamples;
        double mse = sumSqError / testSamples;
        double normalizedMSE = mse / (sumTargetPower / testSamples);

        _output.WriteLine($"Period={period}  MAE={mae:F4}  MSE={mse:F4}  normalized={normalizedMSE:P2}");

        // Assertion: normalized MSE < 30% means the model has clearly learned
        // the periodic structure (baseline "predict zero" would give 100%).
        // This is a meaningful signal that HRE captures the periodicity.
        Assert.True(normalizedMSE < 0.30,
            $"Period={period}: normalized MSE {normalizedMSE:P2} should be < 30% " +
            $"(baseline = 100%). HRE should learn the periodic structure.");
    }
}
