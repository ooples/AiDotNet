using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// NaturalSpeech through every stage at test scale (phoneme ids in, so no eSpeak NG): mixed-phoneme pre-training with
/// learned sup-phonemes, warm-up, the memory bank's K-means initialization, joint training with soft-DTW, tuning, and
/// synthesis; the paper's and heatz123's variants both.
/// </summary>
public class NaturalSpeechModelTests
{
    private static NaturalSpeechOptions Tiny(NaturalSpeechOptions o)
    {
        o.HiddenDim = 16; o.NumHeads = 2; o.NumEncoderLayers = 1; o.FilterChannels = 32; o.DurationFilter = 8; o.UpsamplingFilter = 4;
        o.Couplings = 2; o.CouplingLayers = 1; o.UpsampleRates = new[] { 4, 4 }; o.UpsampleKernels = new[] { 8, 8 }; o.UpsampleInitialChannels = 16;
        o.ResblockKernels = new[] { 3 }; o.ResblockDilations = new[] { new[] { 1, 3 } }; o.DiscriminatorPeriods = new[] { 2, 3 }; o.DiscriminatorWidthDivisor = 16; o.MemorySize = 8;
        o.FftSize = 64; o.HopSize = 16; o.SampleRate = 4000; o.MelChannels = 8; o.SegmentFrames = 8;
        return o;
    }

    private static Tensor<double> Tone(int samples, double frequency) =>
        new(new[] { samples }, new Vector<double>(Enumerable.Range(0, samples).Select(i => 0.3 * Math.Sin(2 * Math.PI * frequency * i / 4000.0)).ToArray()));

    private static Tensor<double> Ids(params int[] ids) => new(new[] { ids.Length }, new Vector<double>(ids.Select(i => (double)i).ToArray()));

    [Theory(Timeout = 900000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task EveryStage_TrainsAndSynthesizes(bool heatz123)
    {
        await Task.Yield();
        int space = NaturalSpeech<double>.SpaceId;
        var corpus = new[] { new[] { 60, 70, 80, space, 60, 70, 90 }, new[] { 61, 70, 80, space, 62, 70, 80 } };
        var options = Tiny(heatz123 ? NaturalSpeechOptions.Heatz123() : new NaturalSpeechOptions());
        options.SupPhonemeMerges = NaturalSpeech<double>.LearnSupPhonemes(corpus, options.VocabSize, options.VocabSize + 4);
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new NaturalSpeech<double>(architecture, options);
        var sample = new TtsTrainingSample<double> { Tokens = Ids(60, 70, 80, space, 60, 70, 90), Audio = Tone(512, 300) };

        options.CurrentStage = NaturalSpeechStage.PhonemePretraining;
        Assert.False(double.IsNaN(model.Train(sample)));
        options.CurrentStage = NaturalSpeechStage.Warmup;
        Assert.False(double.IsNaN(model.Train(sample)));
        Assert.Throws<InvalidOperationException>(() =>
        {
            options.CurrentStage = NaturalSpeechStage.Joint;
            model.Train(sample);
        });
        model.InitializeMemoryBank(new[] { Tone(512, 300), Tone(512, 500) });
        Assert.False(double.IsNaN(model.Train(sample)));
        options.CurrentStage = NaturalSpeechStage.Tuning;
        Assert.False(double.IsNaN(model.Train(sample)));
        Assert.False(double.IsNaN(model.EvaluateTrainingObjective(sample)));

        var audio = model.Predict(Ids(60, 70, 80));
        Assert.True(audio.Length > 0);
        for (int i = 0; i < audio.Length; i++) Assert.False(double.IsNaN(audio.GetFlat(i)));
    }
}
