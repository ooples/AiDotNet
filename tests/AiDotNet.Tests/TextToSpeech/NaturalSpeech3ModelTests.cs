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
/// NaturalSpeech 3 at test scale: FACodec's training (reconstruction, adversarial, supervision and gradient-reversal
/// losses), the redecoder, the factorized diffusion (with and without the phoneme-level prosody stage), synthesis after a
/// prompt, and voice conversion.
/// </summary>
public class NaturalSpeech3ModelTests
{
    [Theory]
    [InlineData("CodecBeta1", -0.1)]
    [InlineData("CodecBeta1", 1)]
    [InlineData("CodecBeta1", double.NaN)]
    [InlineData("CodecBeta1", double.PositiveInfinity)]
    [InlineData("CodecBeta2", -0.1)]
    [InlineData("CodecBeta2", 1)]
    [InlineData("CodecBeta2", double.NaN)]
    [InlineData("CodecBeta2", double.PositiveInfinity)]
    public void InvalidCodecAdamBetasAreRejectedBeforeTraining(string option, double value)
    {
        var options = Tiny();
        typeof(NaturalSpeech3Options).GetProperty(option)!.SetValue(options, value);
        Assert.Throws<ArgumentOutOfRangeException>(() => Model(options));
    }

    [Theory]
    [InlineData("HopSize")]
    [InlineData("CodecEncoderRates")]
    [InlineData("CodecDecoderRates")]
    public void UnpairedCodecGeometryIsRejected(string option)
    {
        var options = Tiny();
        if (option == "HopSize") options.HopSize = 6;
        else typeof(NaturalSpeech3Options).GetProperty(option)!.SetValue(options, new[] { 2, 3 });
        Assert.Throws<ArgumentException>(() => Model(options));
    }

    [Theory]
    [InlineData(2, 4)]
    [InlineData(3, 40)]
    [InlineData(4, 128)]
    public void PairedStrideProductsRoundTripWaveforms(int secondStride, int inheritedMelChannels)
    {
        var options = Tiny();
        options.CodecEncoderRates = new[] { 2, secondStride };
        options.CodecDecoderRates = new[] { secondStride, 2 };
        options.HopSize = 2 * secondStride;
        options.MelChannels = inheritedMelChannels;
        using var model = Model(options);
        var (codes, timbre) = model.EncodeFactors(Tone(12 * options.HopSize, .05));
        Assert.Equal(12, codes.GetLength(1));
        var audio = model.DecodeFactors(codes, timbre);
        Assert.Equal(12 * options.HopSize, audio.Length);
        Assert.All(audio.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    private static NaturalSpeech3Options Tiny(bool phonemeProsody = false) => new()
    {
        SampleRate = 1000, HopSize = 4, CodecEncoderDim = 4, CodecEncoderRates = new[] { 2, 2 }, CodecDim = 16, CodecDecoderChannels = 16,
        CodecDecoderRates = new[] { 2, 2 }, CodebookDim = 4, CodebookSize = 16, CodecPhones = 100, CodecSpeakers = 4, IncludeRedecoder = true,
        MelWindows = new[] { 16, 32 }, MelBins = new[] { 4, 8 }, DiscriminatorPeriods = new[] { 2, 3 }, DiscriminatorWindows = new[] { 64, 32 },
        DiscriminatorWidthDivisor = 16, HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, FilterChannels = 32, EncoderKernel = 3, DiffusionDim = 16,
        DiffusionHeads = 2, DiffusionFilter = 32, FrameDiffusionLayers = 1, PhonemeDiffusionLayers = 1, MaxDuration = 15, SamplingSteps = 2, TopK = 4,
        WarmupSteps = 1, PhonemeProsody = phonemeProsody,
    };

    private static Tensor<double> Tone(int samples, double cycles) =>
        new(new[] { samples }, new Vector<double>(Enumerable.Range(0, samples).Select(i => 0.3 * Math.Sin(2 * Math.PI * cycles * i) + 0.1 * Math.Sin(2 * Math.PI * 0.21 * i)).ToArray()));

    private static Tensor<double> Ids(params int[] ids) => new(new[] { ids.Length }, new Vector<double>(ids.Select(i => (double)i).ToArray()));

    private static NaturalSpeech3<double> Model(NaturalSpeech3Options options) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);

    private static TtsTrainingSample<double> Sample()
    {
        var durations = new[] { 3, 2, 4, 1, 3, 2, 3, 2 };
        int frames = durations.Sum();
        return new TtsTrainingSample<double>
        {
            Tokens = Ids(3, 17, 40, 86, 52, 9, 66, 30),
            Durations = durations,
            Audio = Tone(frames * 4, 0.05),
            Pitch = Enumerable.Range(0, frames).Select(f => f % 5 == 4 ? 0.0 : 120.0 + 10 * Math.Sin(f)).ToArray(),
            SpeakerId = 1,
        };
    }

    [Theory(Timeout = 900000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task EveryStage_Trains_ThenSynthesizesAfterAPrompt(bool phonemeProsody)
    {
        await Task.Yield();
        var options = Tiny(phonemeProsody);
        var model = Model(options);
        var sample = Sample();

        foreach (var stage in new[] { NaturalSpeech3Stage.Codec, NaturalSpeech3Stage.Redecoder, NaturalSpeech3Stage.Diffusion })
        {
            options.CurrentStage = stage;
            double before = model.EvaluateTrainingObjective(sample);
            Assert.False(double.IsNaN(before) || double.IsInfinity(before), $"{stage} objective");
            for (int i = 0; i < 2; i++) Assert.False(double.IsNaN(model.Train(sample)), $"{stage} step");
            Assert.NotEqual(before, model.EvaluateTrainingObjective(sample));
        }

        model.Voice = model.CreateVoice(Tone(40, 0.08), new[] { 12, 33, 50 }, new[] { 3, 4, 3 });
        var audio = model.Predict(Ids(3, 17, 40, 86));
        Assert.True(audio.Length > 0 && audio.Length % 4 == 0);
        for (int i = 0; i < audio.Length; i++) Assert.False(double.IsNaN(audio.GetFlat(i)));

        var converted = model.ConvertVoice(Tone(48, 0.05), Tone(40, 0.11));
        Assert.Equal(48, converted.Length);
        var (codes, timbre) = model.EncodeFactors(Tone(48, 0.05));
        Assert.Equal(6, codes.GetLength(0));
        Assert.Equal(48, model.DecodeFactors(codes, timbre).Length);
    }
}
