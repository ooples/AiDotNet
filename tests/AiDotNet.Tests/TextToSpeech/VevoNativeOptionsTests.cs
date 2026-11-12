using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class VevoNativeOptionsTests
{
    private static Vevo<double> Model(VevoOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 4, outputSize: 1) { RandomSeed = 11 }, options);

    private static Tensor<double> Recording() => new(new[] { 1024 }, new Vector<double>(Enumerable.Range(0, 1024)
        .Select(i => .4 * Math.Sin(2 * Math.PI * (120 + i * .1) * i / 24000)).ToArray()));

    [Theory]
    [InlineData("VocabSize", 24)]
    [InlineData("NumHeads", 4)]
    [InlineData("NumEncoderLayers", 3)]
    [InlineData("NumDecoderLayers", 3)]
    public void SharedFields_ConfigureTheActualNativeComponents(string field, int value)
    {
        var options = VevoPaperTests.Options();
        typeof(VevoOptions).GetProperty(field)!.SetValue(options, value);
        using var model = Model(options);
        Assert.Equal(options.VocabSize, model.TextToContentStyle.InputVocabulary);
        Assert.Equal(options.NumHeads, model.TextToContentStyle.Network.Configuration.Heads);
        Assert.Equal(options.NumDecoderLayers, model.TextToContentStyle.Network.Configuration.Layers);
        Assert.Equal(options.NumEncoderLayers, Vevo<double>.HubertConfiguration(options).Layers);
        var features = model.HubertFeatures(model.AtContentRate(Recording()));
        Assert.All(features.ToArray(), sample => Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(sample)));
    }

    [Theory]
    [InlineData("PhoneVocabulary", "VocabSize", 20)]
    [InlineData("TextArHeads", "NumHeads", 2)]
    [InlineData("TextArLayers", "NumDecoderLayers", 2)]
    [InlineData("HubertLayer", "NumEncoderLayers", 2)]
    public void SavedNativeFields_WinConflictingAliasesInEitherOrder(string native, string alias, int value)
    {
        foreach (string json in new[] { $"{{\"{native}\":{value},\"{alias}\":99}}", $"{{\"{alias}\":99,\"{native}\":{value}}}" })
        {
            var options = Newtonsoft.Json.JsonConvert.DeserializeObject<VevoOptions>(json)!;
            Assert.Equal(value, typeof(VevoOptions).GetProperty(alias)!.GetValue(options));
        }
        var legacy = Newtonsoft.Json.JsonConvert.DeserializeObject<VevoOptions>($"{{\"{alias}\":{value}}}")!;
        Assert.Equal(value, typeof(VevoOptions).GetProperty(native)!.GetValue(legacy));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AttentionHeadCounts_ChangeActualOwningNetworkOutputs(bool voice)
    {
        var controlOptions = VevoPaperTests.Options();
        var changedOptions = VevoPaperTests.Options();
        if (voice) changedOptions.VoiceArHeads = 1;
        else changedOptions.HubertHeads = 1;
        using var control = Model(controlOptions);
        using var changed = Model(changedOptions);
        Assert.Equal(control.GetParameters().ToArray(), changed.GetParameters().ToArray());
        if (voice)
        {
            var tokens = new[] { 20, 21, 22, 23 };
            var codes = new[] { 1, 2, 3, 4 };
            Assert.False(control.ContentToContentStyle.Logits(tokens, codes).ToArray()
                .SequenceEqual(changed.ContentToContentStyle.Logits(tokens, codes).ToArray()));
        }
        else
            Assert.False(control.HubertFeatures(control.AtContentRate(Recording())).ToArray()
                .SequenceEqual(changed.HubertFeatures(changed.AtContentRate(Recording())).ToArray()));
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(-.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ReleasedTransformers_RejectGenericDropout(double rate)
    {
        var options = VevoPaperTests.Options(); options.DropoutRate = rate;
        Assert.Throws<ArgumentException>(() => Model(options));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(1)]
    [InlineData(2048)]
    public void NativeTokenBudget_RejectsAnUnrelatedMelFrameLimit(int frames)
    {
        var options = VevoPaperTests.Options(); options.MaxMelLength = frames;
        Assert.Throws<ArgumentException>(() => Model(options));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void PhoneLimit_MustBePositive(int maximum)
    {
        var options = VevoPaperTests.Options(); options.MaxTextLength = maximum;
        Assert.Throws<ArgumentException>(() => Model(options));
    }

    [Fact]
    public void PhoneLimit_RejectsOversizedInputBeforeVoiceOrGeneration()
    {
        var options = VevoPaperTests.Options(); options.MaxTextLength = 1;
        using var model = Model(options);
        var phones = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 3.0, 7.0 }));
        Assert.Throws<ArgumentException>(() => model.Predict(phones));
        Assert.Throws<ArgumentException>(() => model.Train(new TtsTrainingSample<double> { Tokens = phones, Audio = Recording() }));
    }

    [Theory]
    [InlineData("FlowLearningRate", .00015)]
    [InlineData("FlowSigma", .25)]
    [InlineData("FlowPromptDropProbability", 1.0)]
    [InlineData("TrainingSeed", 119.0)]
    public void FlowControls_ChangeActualAcousticTrainingUpdates(string field, double candidate)
    {
        var controlOptions = VevoPaperTests.Options();
        controlOptions.CurrentStage = VevoStage.Acoustic; controlOptions.WarmupSteps = 1;
        if (field == "FlowPromptDropProbability") controlOptions.FlowPromptDropProbability = 0;
        var changedOptions = Newtonsoft.Json.JsonConvert.DeserializeObject<VevoOptions>(Newtonsoft.Json.JsonConvert.SerializeObject(controlOptions))!;
        typeof(VevoOptions).GetProperty(field)!.SetValue(changedOptions,
            field == "TrainingSeed" ? (object)(int)candidate : candidate);
        using var control = Model(controlOptions);
        using var changed = Model(changedOptions);
        Assert.Equal(control.GetParameters().ToArray(), changed.GetParameters().ToArray());
        var phones = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 3.0, 7.0, 11.0, 4.0 }));
        for (int step = 0; step < 2; step++)
        {
            Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(control.Train(new TtsTrainingSample<double> { Tokens = phones, Audio = Recording() })));
            Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(changed.Train(new TtsTrainingSample<double> { Tokens = phones, Audio = Recording() })));
        }
        Assert.False(control.GetParameters().ToArray().SequenceEqual(changed.GetParameters().ToArray()));
    }

    [Fact]
    public void RepetitionPenalty_ChangesActualPromptedDecodingWithoutTraining()
    {
        using var model = Model(VevoPaperTests.Options());
        var prompt = model.ContentStyleCodes(model.AtContentRate(Recording())).Take(8).ToArray();
        int[] Decode(double penalty) => model.TextToContentStyle.Generate(new[] { 3, 7, 11, 4 }, prompt,
            new VevoDecoding(true, .8, 50, .9, penalty, 24, 39, true), new Random(1));
        Assert.False(Decode(1).SequenceEqual(Decode(.5)));
    }
}
