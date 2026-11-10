using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class InfillingNativeOptionsTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 4, outputSize: 8) { RandomSeed = 11 };

    private static F5TTSOptions F5() => new()
    {
        HiddenDim = 16, NumHeads = 2, HeadDim = 8, NumLayers = 2, TextDim = 8, TextConvLayers = 1,
        VocabSize = 32, MelChannels = 8, DropoutRate = 0, WarmupSteps = 0, LearningRate = 1e-3, WeightDecay = 0,
        MaskFractionMin = .5, MaskFractionMax = .5, ConditionDropProbability = 0, AudioDropProbability = 0,
    };

    private static E2TTSOptions E2() => new()
    {
        HiddenDim = 16, NumHeads = 2, HeadDim = 8, NumLayers = 2, VocabSize = 32, MelChannels = 8,
        DropoutRate = 0, WarmupSteps = 0, LearningRate = 1e-3, WeightDecay = 0,
        MaskFractionMin = .5, MaskFractionMax = .5, ConditionDropProbability = 0, AudioDropProbability = 0,
    };

    private static (Tensor<double> Tokens, Tensor<double> Mel) Example()
    {
        var tokens = new Tensor<double>(new[] { 4 });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = i + 1;
        var mel = new Tensor<double>(new[] { 16, 8 });
        for (int i = 0; i < mel.Length; i++) mel[i] = .4 * Math.Sin(.17 * i) + .2 * Math.Cos(.31 * i);
        return (tokens, mel);
    }

    private static object Field(object instance, string name) => instance.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(instance)!;

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 3)]
    [InlineData(3, 4)]
    public void F5_LayerAliasesBuildActualTextAndFlowBlocks(int textDepth, int flowDepth)
    {
        var options = F5();
        options.NumEncoderLayers = textDepth;
        options.NumDecoderLayers = flowDepth;
        var model = new F5TTS<double>(Architecture(), options);
        Assert.Equal(textDepth, ((IList)Field(Field(model, "_text"), "_blocks")).Count);
        Assert.Equal(flowDepth, ((IList)Field(model, "_blocks")).Count);
        var (tokens, mel) = Example();
        var before = model.GetParameters().ToArray();
        model.Train(tokens, mel);
        Assert.True(double.IsFinite(((ITrainingObjectiveProvider<double>)model).EvaluateTrainingObjective(tokens, mel)));
        Assert.False(before.SequenceEqual(model.GetParameters().ToArray()));
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(2, false)]
    [InlineData(2, true)]
    public void E2_LayerAliasesBuildBothMirroredHalves(int depth, bool decoderAlias)
    {
        var options = E2();
        if (decoderAlias) options.NumDecoderLayers = depth;
        else options.NumEncoderLayers = depth;
        var model = new E2TTS<double>(Architecture(), options);
        Assert.Equal(depth, options.NumEncoderLayers);
        Assert.Equal(depth, options.NumDecoderLayers);
        Assert.Equal(2 * depth, ((IList)Field(model, "_layers")).Count);
        var (tokens, mel) = Example();
        var before = model.GetParameters().ToArray();
        model.Train(tokens, mel);
        Assert.True(double.IsFinite(((ITrainingObjectiveProvider<double>)model).EvaluateTrainingObjective(tokens, mel)));
        Assert.False(before.SequenceEqual(model.GetParameters().ToArray()));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeAdamWUsesAuthorScheduleAndOnlyConfiguredModelClip(bool f5)
    {
        var a = F5(); a.WarmupSteps = 2; a.TotalSteps = 6; a.MaxGradientNorm = 0;
        var b = E2(); b.WarmupSteps = 2; b.TotalSteps = 6; b.MaxGradientNorm = 0;
        TtsModelBase<double> model = f5 ? new F5TTS<double>(Architecture(), a) : new E2TTS<double>(Architecture(), b);
        var optimizer = Assert.IsType<AdamWOptimizer<double, Tensor<double>, Tensor<double>>>(Field(model, "_optimizer"));
        var optimizerOptions = Assert.IsType<AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        Assert.Equal(0, model.MaxGradNormValue);
        Assert.False(optimizerOptions.EnableGradientClipping);
        Assert.Equal(.9, optimizerOptions.Beta1);
        Assert.Equal(.999, optimizerOptions.Beta2);
        Assert.Equal(1e-8, optimizerOptions.Epsilon);
        // Independently emitted by PyTorch 2.6 LinearLR(start=1e-8, end=1, total=2), followed by
        // LinearLR(start=1, end=1e-8, total=4), SequentialLR milestone=2; first value precedes update 1.
        double[] factors = { 1e-8, .500000005, 1, .7500000025, .500000005, .2500000075, 1.000000007269183e-8 };
        var (tokens, mel) = Example();
        for (int i = 0; i < factors.Length; i++)
        {
            Assert.InRange(Math.Abs(optimizer.LearningRateScheduler!.CurrentLearningRate - 1e-3 * factors[i]), 0, 1e-17);
            Assert.InRange(Math.Abs(optimizer.GetCurrentLearningRate() - 1e-3 * factors[i]), 0, 1e-17);
            if (i < factors.Length - 1) model.Train(tokens, mel);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AudioDropChangesRealInfillingObjectiveAfterZeroProjectionOpens(bool f5)
    {
        var a = F5(); var b = E2();
        TtsModelBase<double> model = f5 ? new F5TTS<double>(Architecture(), a) : new E2TTS<double>(Architecture(), b);
        var (tokens, mel) = Example();
        for (int i = 0; i < 3; i++) model.Train(tokens, mel);
        double conditioned = ((ITrainingObjectiveProvider<double>)model).EvaluateTrainingObjective(tokens, mel);
        if (f5) a.AudioDropProbability = 1; else b.AudioDropProbability = 1;
        double dropped = ((ITrainingObjectiveProvider<double>)model).EvaluateTrainingObjective(tokens, mel);
        Assert.True(double.IsFinite(conditioned) && double.IsFinite(dropped));
        Assert.True(Math.Abs(conditioned - dropped) > 1e-12, $"Audio condition had no effect: {conditioned} vs {dropped}.");
    }

    public sealed class DerivedF5 : F5TTSOptions { public string Label { get; set; } = "derived"; }
    public sealed class DerivedE2 : E2TTSOptions { public string Label { get; set; } = "derived"; }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeJsonDepthWinsStaleSharedCountsInEitherOrder(bool reverse)
    {
        string aliases = "\"NumEncoderLayers\":6,\"NumDecoderLayers\":6";
        string f5Native = "\"NumLayers\":3,\"TextConvLayers\":2";
        string e2Native = "\"NumLayers\":4";
        var f5 = JsonConvert.DeserializeObject<DerivedF5>("{" + (reverse ? f5Native + "," + aliases : aliases + "," + f5Native) + "}")!;
        var e2 = JsonConvert.DeserializeObject<DerivedE2>("{" + (reverse ? e2Native + "," + aliases : aliases + "," + e2Native) + "}")!;
        Assert.Equal(2, f5.NumEncoderLayers); Assert.Equal(3, f5.NumDecoderLayers);
        Assert.Equal(2, e2.NumEncoderLayers); Assert.Equal(2, e2.NumDecoderLayers);
        Assert.Equal("derived", f5.Label); Assert.Equal("derived", e2.Label);
        Assert.Equal(4, new F5TTSOptions().NumEncoderLayers); Assert.Equal(22, new F5TTSOptions().NumDecoderLayers);
        Assert.Equal(12, new E2TTSOptions().NumEncoderLayers); Assert.Equal(12, new E2TTSOptions().NumDecoderLayers);
    }
}
