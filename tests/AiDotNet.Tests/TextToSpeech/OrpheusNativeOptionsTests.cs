using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class OrpheusNativeOptionsTests
{
    private static OrpheusTTSOptions Tiny() => new()
    {
        HiddenDim = 16, Layers = 1, Heads = 2, KeyValueHeads = 1, HeadDim = 8, Intermediate = 32,
        MaxNewTokens = 7, ConstrainToAudioTokens = true,
        CodecEncoderDim = 2, CodecDecoderDim = 32, CodecCodebookDim = 2, LearningRate = 1e-3,
    };

    private static OrpheusTTS<double> Create(OrpheusTTSOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 3 }, options);

    [Fact]
    public void DefaultsDescribeTheReleasedDecoderOnlyLlama()
    {
        var options = new OrpheusTTSOptions();
        Assert.Equal((0, 28, 24, 0.0),
            (options.NumEncoderLayers, options.NumDecoderLayers, options.NumHeads, options.DropoutRate));
    }

    [Theory]
    [InlineData("NumHeads", 4)]
    [InlineData("NumDecoderLayers", 2)]
    public void BaseTypedSettingsReachNativeAttentionAndDecoderDepth(string property, int value)
    {
        var options = Tiny();
        using var control = Create(new OrpheusTTSOptions(options));
        typeof(TtsModelOptions).GetProperty(property)!.SetValue(options, value);
        var copied = (OrpheusTTSOptions)CloneEngine.CopyConfiguration(options);
        using var changed = Create(copied);
        Assert.NotEqual(control.GetParameters().Length, changed.GetParameters().Length);
        Assert.Equal(value, property == "NumHeads" ? changed.LanguageModel.Configuration.Heads : changed.LanguageModel.Configuration.Layers);
        Assert.Equal(value, typeof(TtsModelOptions).GetProperty(property)!.GetValue(copied));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LegacySharedMetadataPreservesNativeGeometryAndDecoderOnlyTopology(bool reverse)
    {
        var json = JObject.FromObject(Tiny());
        json["NumHeads"] = 8;
        json["NumDecoderLayers"] = 6;
        json["NumEncoderLayers"] = 6;
        if (reverse) json = new JObject(json.Properties().Reverse().Select(p => new JProperty(p.Name, p.Value.DeepClone())));
        var restored = JsonConvert.DeserializeObject<OrpheusTTSOptions>(json.ToString())!;
        Assert.Equal((2, 1, 0), (restored.NumHeads, restored.NumDecoderLayers, restored.NumEncoderLayers));
        using var model = Create(restored);
        Assert.Equal((2, 1), (model.LanguageModel.Configuration.Heads, model.LanguageModel.Configuration.Layers));
    }

    [Fact]
    public void PartialSharedAliasesConfigureNativeDimensions()
    {
        var options = JsonConvert.DeserializeObject<OrpheusTTSOptions>("{\"NumHeads\":4,\"NumDecoderLayers\":2}")!;
        Assert.Equal((4, 2, 0), (options.Heads, options.Layers, options.NumEncoderLayers));
    }

    [Fact]
    public void RequestingAnEncoderIsRejectedByTheActualConstructor()
    {
        var options = Tiny();
        options.NumEncoderLayers = 1;
        Assert.Contains("NumEncoderLayers must be zero", Assert.Throws<ArgumentException>(() => Create(options)).Message);
    }

    [Fact]
    public void AttentionDropoutTrainsTheNativeLmAndEvaluationRemainsDeterministic()
    {
        var options = Tiny();
        options.DropoutRate = 0.5;
        using var model = Create(options);
        using var zero = Create(new OrpheusTTSOptions(options) { DropoutRate = 0 });
        Assert.Equal(model.LanguageModelLayers.SelectMany(layer => layer.GetParameters()).ToArray(),
            zero.LanguageModelLayers.SelectMany(layer => layer.GetParameters()).ToArray());
        var codes = new Tensor<double>(new[] { 2, 7 });
        for (int frame = 0; frame < 2; frame++)
            for (int slot = 0; slot < 7; slot++) codes[frame, slot] = frame + slot + 1;
        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 })),
            Text = "Hello.", CodecTokens = codes,
        };
        Assert.Equal(zero.EvaluateTrainingObjective(sample), model.EvaluateTrainingObjective(sample));
        var ids = model.TrainingIds(sample);
        model.SetTrainingMode(true);
        double stochastic = model.Loss(ids)[0];
        Assert.True(TtsPaperNumerics.IsFinite(stochastic));
        Assert.NotEqual(model.EvaluateTrainingObjective(sample), stochastic);
        var lmBefore = model.LanguageModelLayers.SelectMany(layer => layer.GetParameters()).ToArray();
        var codecBefore = model.CodecLayers.SelectMany(layer => layer.GetParameters()).ToArray();
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(sample)));
        Assert.False(lmBefore.SequenceEqual(model.LanguageModelLayers.SelectMany(layer => layer.GetParameters())));
        Assert.Equal(codecBefore, model.CodecLayers.SelectMany(layer => layer.GetParameters()).ToArray());
        Assert.Equal(model.EvaluateTrainingObjective(sample), model.EvaluateTrainingObjective(sample));
    }
}
