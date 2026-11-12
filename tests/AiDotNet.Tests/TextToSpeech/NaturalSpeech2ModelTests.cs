using System;
using System.Linq;
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
/// NaturalSpeech 2 at test scale: training on durations, a recording (encoded by the codec, its pitch extracted by WORLD)
/// with a prompt cut from it, then synthesis in a reference's voice; the paper's configuration, Amphion's departures,
/// and the flow-matching variant.
/// </summary>
public class NaturalSpeech2ModelTests
{
    private static NaturalSpeech2Options Tiny(NaturalSpeech2Options o)
    {
        o.SampleRate = 4000; o.HopSize = 16; o.HiddenDim = 16; o.NumHeads = 2; o.NumEncoderLayers = 1; o.FilterChannels = 32; o.EncoderKernel = 3;
        o.NumPromptLayers = 1; o.DurationLayers = 2; o.PitchLayers = 2; o.PitchBins = 16; o.QueryTokens = 4; o.DenoiserLayers = 3; o.SamplingSteps = 2;
        o.WarmupSteps = 1;
        o.Codec = new SoundStreamOptions
        {
            SampleRate = 4000, SegmentSize = 4000, Filters = 2, Ratios = [4, 4], Dimension = 8, NumQuantizers = 2, CodebookSize = 16,
            TargetBandwidthKbps = 2.0, FilmPosition = SoundStreamFilmPosition.None, IncludeDiscriminators = false,
        };
        return o;
    }

    private static Tensor<double> Tone(int samples, double frequency) =>
        new(new[] { samples }, new Vector<double>(Enumerable.Range(0, samples).Select(i => 0.3 * Math.Sin(2 * Math.PI * frequency * i / 4000.0)).ToArray()));

    private static Tensor<double> Ids(params int[] ids) => new(new[] { ids.Length }, new Vector<double>(ids.Select(i => (double)i).ToArray()));

    [Theory]
    [InlineData(1)]
    [InlineData(-1)]
    public void NativeCodecRejectsUnsupportedMelWidths(int width)
    {
        var options = Tiny(new NaturalSpeech2Options { MelChannels = width });
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1);
        Assert.Throws<ArgumentException>(() => new NaturalSpeech2<double>(architecture, options));
    }

    [Fact]
    public void LegacyMelDefaultStillLoads()
    {
        var options = Tiny(Newtonsoft.Json.JsonConvert.DeserializeObject<NaturalSpeech2Options>("{\"MelChannels\":80}")!);
        using var model = new NaturalSpeech2<double>(
            new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: 1, outputSize: 1), options);
        Assert.True(model.GetParameters().Length > 0);
    }

    [Theory(Timeout = 900000)]
    [InlineData("paper")]
    [InlineData("amphion")]
    [InlineData("flow")]
    public async Task Trains_AndSynthesizesInTheReferenceVoice(string variant)
    {
        await Task.Yield();
        var options = Tiny(variant == "amphion" ? NaturalSpeech2Options.Amphion() : new NaturalSpeech2Options());
        if (variant == "flow") options.Diffusion = NaturalSpeech2Diffusion.FlowMatching;
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new NaturalSpeech2<double>(architecture, options);
        var durations = new[] { 12, 8, 16, 4, 12, 8, 12, 8 };
        var sample = new TtsTrainingSample<double>
        {
            Tokens = Ids(3, 17, 40, 86, 52, 9, 66, 30),
            Durations = durations,
            Audio = Tone(durations.Sum() * 16, 220),
        };

        double before = model.EvaluateTrainingObjective(sample);
        Assert.False(double.IsNaN(before) || double.IsInfinity(before));
        for (int i = 0; i < 3; i++) Assert.False(double.IsNaN(model.Train(sample)));
        double after = model.EvaluateTrainingObjective(sample);
        Assert.False(double.IsNaN(after) || double.IsInfinity(after));
        Assert.NotEqual(before, after);

        model.Voice = model.CreateVoice(Tone(800, 180));
        var audio = model.Predict(Ids(3, 17, 40, 86));
        Assert.True(audio.Length > 0 && audio.Length % 16 == 0);
        for (int i = 0; i < audio.Length; i++) Assert.False(double.IsNaN(audio.GetFlat(i)));
    }

    [Fact(Timeout = 120000)]
    public async Task TextToPhonemes_FollowsAmphionsFrontEnd()
    {
        await Task.Yield();
        var model = new NaturalSpeech2<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), Tiny(new NaturalSpeech2Options()));
        // g2p_en: "hello, world." → HH AH0 L OW1 {sp} W ER1 L D (the trailing period stripped, the comma a short pause).
        var phones = model.TextToPhonemes("hello, world.").Select(i => NaturalSpeech2<double>.Symbols[i]).ToArray();
        Assert.Equal(new[] { "HH", "AH0", "L", "OW1", "sp", "W", "ER1", "L", "D" }, phones);
    }
}
