using System;
using System.Threading.Tasks;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MARS5 end to end at test scale: deep and shallow cloning, and one training step of each network (the generated suite
/// trains only the default AR stage).
/// </summary>
public class Mars5ModelTests
{
    private static MARS5TTSOptions Tiny(Mars5Stage stage) => new()
    {
        CurrentStage = stage, HiddenDim = 16, NumHeads = 2, ArLayers = 1, ArFeedForward = 32, ArSpeakerLayers = 1, SlidingWindow = 512,
        NarDim = 16, NarHeads = 2, NarEncoderLayers = 1, NarDecoderLayers = 1, NarSpeakerLayers = 1, TimeDim = 8, DiffusionSteps = 4,
        Q0OverrideSteps = 1, MaxGenerationLength = 64, NumCodebooks = 4, CodebookSize = 64, HopSize = 2560, DecoderDim = 16,
        DecoderIntermediateDim = 32, DecoderLayers = 1, WarmupSteps = 0,
        Codec = new EnCodecOptions
        {
            SampleRate = 24000, NumQuantizers = 4, CodebookSize = 64, Filters = 4, Ratios = new[] { 8, 5, 4, 4, 4 }, Dimension = 8,
            ResidualKernelSizes = new[] { 3, 1 }, TargetBandwidthKbps = 0.225,
        },
    };

    private static Tensor<double> Tone(int samples, double frequency)
    {
        var x = new double[samples];
        for (int i = 0; i < samples; i++) x[i] = 0.3 * Math.Sin(2 * Math.PI * frequency * i / 24000.0);
        return new Tensor<double>(new[] { samples }, new Vector<double>(x));
    }

    private static MARS5TTS<double> Model(Mars5Stage stage) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), Tiny(stage));

    [Fact(Timeout = 900000)]
    public async Task DeepAndShallowClones_AndBothStagesTrain()
    {
        await Task.Yield();
        var model = Model(Mars5Stage.Autoregressive);
        model.Voice = model.CreateVoice(Tone(51200, 180), 24000, "hello there");
        var deep = model.Synthesize("good morning");
        model.Voice = model.CreateVoice(Tone(51200, 180), 24000);
        var shallow = model.Synthesize("good morning");
        foreach (var audio in new[] { deep, shallow })
            for (int i = 0; i < audio.Length; i++) Assert.False(double.IsNaN(audio.GetFlat(i)) || double.IsInfinity(audio.GetFlat(i)));

        var sample = new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Text = "a short test", Audio = Tone(25600, 220) };
        Assert.False(double.IsNaN(model.Train(sample)));
        var nar = Model(Mars5Stage.NonAutoregressive);
        double before = nar.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 3; i++) Assert.False(double.IsNaN(nar.Train(sample)));
        Assert.False(double.IsNaN(before));
    }
}
