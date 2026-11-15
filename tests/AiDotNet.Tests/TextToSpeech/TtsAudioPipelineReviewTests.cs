using System;
using System.Linq;
using AiDotNet.Audio.Effects;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Latest;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class TtsAudioPipelineReviewTests
{
    [Theory]
    [InlineData(11025, 0.37, 0.73)]
    [InlineData(16000, 0.4, 0.75)]
    public void Loudness_BatchedBlocksMatchScalarBoundsAndGates(int rate, double blockSize, double overlap)
    {
        // More than one gather batch, with a partial final batch and non-integral block bounds in the first case.
        var audio = new double[(int)(8.2 * rate)];
        for (int i = 0; i < audio.Length; i++)
        {
            double seconds = (double)i / rate;
            double amplitude = seconds < 2 ? 0.00001 : seconds < 5 ? 0.5 : 0.03;
            audio[i] = amplitude * (Math.Sin(2 * Math.PI * 997 * seconds) + 0.2 * Math.Cos(2 * Math.PI * 173 * seconds));
        }
        var waveform = new Tensor<double>(audio, new[] { audio.Length });
        var filtered = LoudnessMeter.KWeighted(AiDotNetEngine.Current, waveform, rate, clamp: false).ToArray();
        double step = 1 - overlap;
        int blocks = (int)(Math.Round((audio.Length / (double)rate - blockSize) / (blockSize * step), MidpointRounding.ToEven) + 1);
        var energies = new double[blocks];
        for (int j = 0; j < blocks; j++)
        {
            int lower = (int)(blockSize * (j * step) * rate);
            int upper = Math.Min((int)(blockSize * (j * step + 1) * rate), filtered.Length);
            double sum = 0;
            for (int i = lower; i < upper; i++) sum += filtered[i] * filtered[i];
            energies[j] = sum / (blockSize * rate);
        }
        double Level(double energy) => -0.691 + 10 * Math.Log10(energy);
        var absolute = energies.Where(e => Level(e) >= -70).ToArray();
        double relative = Level(absolute.Average()) - 10;
        var gated = energies.Where(e => Level(e) > relative && Level(e) > -70).ToArray();
        Assert.NotEmpty(gated);
        Assert.True(gated.Length < blocks, "The recording must exercise the loudness gates.");
        double expected = Level(gated.Average());

        double actual = LoudnessMeter.IntegratedLoudness(waveform, rate, blockSize, overlap);

        Assert.Equal(expected, actual, precision: 9);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MegaTts2_HalfProsodyKeepsNonStartProbabilitiesUntilTheDraw(bool withStyle)
    {
        Assert.Equal(new[] { 0, 0, 0 }, ProsodyCodes<Half>(withStyle));
        Assert.Equal(new[] { 0, 0, 0 }, ProsodyCodes<double>(withStyle));
    }

    private static int[] ProsodyCodes<T>(bool withStyle)
    {
        var options = MegaTts2ModelReferenceTests.TinyOptions();
        options.Codes = 2;
        options.TopK = 1;
        using var model = new MegaTTS2<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        model.SetTrainingMode(false);
        using var noGrad = new NoGradScope<T>();
        var decoder = model.Network.ProsodyModel;
        var pooled = new Tensor<T>(new[] { 3, options.HiddenDim });
        var row = new Tensor<T>(new[] { 1, options.HiddenDim });
        // Materialize the final readout, then make every step's logits exactly [20, 0, 0].
        decoder.Extend(new[] { model.Network.StartToken }, row, new MegaTts3DurationCache<T>());
        var parameters = new Vector<T>(decoder.Output.GetParameters().Length);
        var ops = MathHelper.GetNumericOperations<T>();
        parameters[parameters.Length - decoder.Outputs] = ops.FromDouble(20);
        decoder.Output.SetParameters(parameters);
        var logits = decoder.Extend(new[] { model.Network.StartToken }, row, new MegaTts3DurationCache<T>()).ToArray();
        Assert.Equal(new[] { 20.0, 0.0, 0.0 }, logits.Select(ops.ToDouble).ToArray());

        var prompts = Array.Empty<MegaTTS2<T>.Prompt>();
        return model.SampleProsody(prompts, withStyle ? prompts : null, pooled, new Random(5));
    }
}
