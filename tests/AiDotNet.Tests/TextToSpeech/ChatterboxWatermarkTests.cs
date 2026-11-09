using System;
using System.Linq;
using AiDotNet.Audio.Watermarking;
using AiDotNet.ModelLoading;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.VoiceCloning;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Chatterbox puts Resemble's Perth watermark on its output as the release's <c>generate()</c> does (the watermark
/// itself is checked against resemble-perth in <c>PerthWatermarkerReferenceTests</c>).</summary>
public class ChatterboxWatermarkTests
{
    [Fact]
    public void Watermark_IsOnByDefault_AndRefusesWithoutTheModel()
    {
        Assert.True(new ChatterboxOptions().Watermark);
        var options = ChatterboxFixture.Options();
        options.Watermark = true;
        var model = new Chatterbox<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(AiDotNet.Enums.InputType.OneDimensional,
            AiDotNet.Enums.NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);
        var error = Assert.Throws<InvalidOperationException>(() => model.Watermarked(ChatterboxFixture.Recording<double>()));
        Assert.Contains("WatermarkModelPath", error.Message);
    }

    [Fact]
    public void Watermarked_IsPerthsApplyAtTheModelsRate()
    {
        var f = WhisperSpeechT2SReferenceTests.Load("perth_reference.json");
        var state = (JObject)f["tiny_state"]!;
        var perth = new PerthWatermarker<double>(AiDotNetEngine.Current, hidden: 8);
        perth.LoadTorchWeights(TorchTensorSource.From(n => state[n] is { } v ? v.Select(x => (double)x).ToArray() : null));
        var recording = ChatterboxFixture.Recording<double>();

        var options = ChatterboxFixture.Options();
        options.Watermark = true;
        var model = new Chatterbox<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(AiDotNet.Enums.InputType.OneDimensional,
            AiDotNet.Enums.NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options) { Watermarker = perth };
        var expected = perth.Apply(recording.ToArray(), 24000);
        var actual = model.Watermarked(recording).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i]);
        Assert.NotEqual(recording.ToArray().Take(expected.Length), actual);

        var unmarked = ChatterboxFixture.Create<double>();
        Assert.Equal(recording.ToArray(), unmarked.Watermarked(recording).ToArray());
    }
}
