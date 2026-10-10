using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// The APNet2 generator's gradient of its frame-level and mel objective is the tape's full gradient restricted to any
/// parameter subset, and agrees with central finite differences — through the ConvNeXt v2 predictors with GRN, the
/// two-argument arctangent, the inverse STFT and the STFT consistency loss.
/// </summary>
public class APNet2NetworkGradientTests
{
    private sealed class Probe : APNet2<double>
    {
        public Probe()
            : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                    inputSize: 8, outputSize: 16) { RandomSeed = 3 },
                new APNet2Options
                {
                    MelChannels = 8, FftSize = 32, HopSize = 8, WindowSize = 32, SampleRate = 4000, MelMaxFrequency = 2000,
                    ConvNeXtChannels = 8, ConvNeXtIntermediateChannels = 16, NumConvNeXtBlocks = 2,
                    DiscriminatorPeriods = [2], DiscriminatorWidthDivisor = 32,
                    ResolutionFftSizes = [32], ResolutionHopSizes = [8], ResolutionWindowSizes = [32], ResolutionDiscriminatorChannels = 64,
                })
        {
        }

        public Tensor<double> Objective(Tensor<double> mel, Tensor<double> audio) => ReconstructionObjective(mel, audio);
    }

    [Fact(Timeout = 60000)]
    public async Task ComputeGradients_MatchesFullTapeAndFiniteDifference()
    {
        await Task.Yield();
        using var model = new Probe();
        var audio = new Tensor<double>(new[] { 128 });
        for (int i = 0; i < 128; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 375 * i / 4000.0) + 0.05 * Math.Cos(i * 0.9);
        var mel = model.ComputeMel(audio);

        var parameters = TapeTrainingStep<double>.CollectParameters(model.Layers, structureVersion: -1);
        IReadOnlyDictionary<Tensor<double>, Tensor<double>> full;
        using (var tape = new GradientTape<double>())
            full = tape.ComputeGradients(model.Objective(mel, audio), sources: null);
        // The objective reaches the generator, not the discriminators.
        var generator = parameters.Where(full.ContainsKey).ToList();
        Assert.NotEmpty(generator);

        IReadOnlyDictionary<Tensor<double>, Tensor<double>> selective;
        using (var tape = new GradientTape<double>())
            selective = tape.ComputeGradients(model.Objective(mel, audio), generator);
        foreach (var parameter in generator)
        {
            Assert.True(selective.TryGetValue(parameter, out var s));
            var f = full[parameter];
            for (int i = 0; i < f.Length; i++)
                Assert.True(Math.Abs(f[i] - s[i]) < 1e-12, $"Selective gradient differs at index {i}: {s[i]:G17} vs {f[i]:G17}.");
        }

        double Loss()
        {
            using var _ = new NoGradScope<double>();
            return model.Objective(mel, audio)[0];
        }
        foreach (var parameter in new[] { generator[0], generator[generator.Count / 2], generator[^1] })
        {
            int coordinate = parameter.Length / 3;
            double original = parameter[coordinate];
            const double epsilon = 1e-6;
            parameter[coordinate] = original + epsilon;
            double plus = Loss();
            parameter[coordinate] = original - epsilon;
            double minus = Loss();
            parameter[coordinate] = original;
            double numerical = (plus - minus) / (2 * epsilon);
            double analytical = full[parameter][coordinate];
            double scale = Math.Max(Math.Abs(analytical), Math.Abs(numerical));
            Assert.True(Math.Abs(analytical - numerical) <= Math.Max(1e-7, scale * 1e-4),
                $"Parameter of length {parameter.Length}, coordinate {coordinate}: analytical={analytical:G17}, numerical={numerical:G17}.");
        }
    }
}
