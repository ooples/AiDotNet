using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LossFunctions;

public class APNet2GeneratorLossGradientTests
{
    [Theory(Timeout = 30000)]
    [InlineData(1.0, 0.0, 0.0, 0.0, "amplitude")]
    [InlineData(0.0, 1.0, 0.0, 0.0, "phase")]
    [InlineData(0.0, 0.0, 1.0, 0.0, "STFT")]
    [InlineData(0.0, 0.0, 0.0, 1.0, "mel")]
    [InlineData(45.0, 100.0, 20.0, 45.0, "combined")]
    public async Task ComputeTapeLoss_EachObjectiveTermMatchesFiniteDifference(
        double amplitudeWeight,
        double phaseWeight,
        double stftWeight,
        double melWeight,
        string term)
    {
        await Task.Yield();

        const int fftSize = 8;
        const int bins = (fftSize / 2) + 1;
        var predicted = CreateSpectrum(frames: 2, bins, offset: 0.03);
        var target = CreateSpectrum(frames: 2, bins, offset: -0.08);
        var loss = new APNet2GeneratorLoss<double>(
            amplitudeWeight,
            phaseWeight,
            stftWeight,
            melWeight,
            fftSize,
            hopSize: 4,
            sampleRate: 16000,
            melChannels: 3);

        Tensor<double> analytical;
        using (var tape = new GradientTape<double>())
        {
            var objective = loss.ComputeTapeLoss(predicted, target);
            var gradients = tape.ComputeGradients(objective, new[] { predicted });
            Assert.True(gradients.TryGetValue(predicted, out analytical));
        }

        const double epsilon = 1e-6;
        for (int i = 0; i < predicted.Length; i++)
        {
            double original = predicted[i];
            predicted[i] = original + epsilon;
            double plus = loss.ComputeTapeLoss(predicted, target)[0];
            predicted[i] = original - epsilon;
            double minus = loss.ComputeTapeLoss(predicted, target)[0];
            predicted[i] = original;

            double numerical = (plus - minus) / (2.0 * epsilon);
            double scale = Math.Max(1e-8, Math.Abs(analytical[i]) + Math.Abs(numerical));
            double relativeError = Math.Abs(analytical[i] - numerical) / scale;
            Assert.True(
                relativeError < 1e-5,
                $"APNet2 {term} gradient[{i}] differs: analytical={analytical[i]:G17}, " +
                $"numerical={numerical:G17}, relativeError={relativeError:G6}.");
        }
    }

    private static Tensor<double> CreateSpectrum(int frames, int bins, double offset)
    {
        var values = new double[frames * bins * 3];
        for (int frame = 0; frame < frames; frame++)
        {
            int row = frame * bins * 3;
            for (int bin = 0; bin < bins; bin++)
            {
                values[row + bin] = -0.25 + offset + (0.04 * frame) + (0.03 * bin);
                values[row + bins + bin] = 0.55 + offset + (0.02 * frame) + (0.025 * bin);
                values[row + (2 * bins) + bin] = -0.35 + offset - (0.015 * frame) + (0.02 * bin);
            }
        }

        return new Tensor<double>(new[] { frames, bins * 3 }, new Vector<double>(values));
    }
}
