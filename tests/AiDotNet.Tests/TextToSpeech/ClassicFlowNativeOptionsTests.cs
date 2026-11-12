using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class ClassicFlowNativeOptionsTests
{
    private static AcousticModelOptions Options(bool glow) => glow
        ? new GlowTTSOptions
        {
            HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, FilterChannels = 32,
            DropoutRate = 0, PrenetDropout = 0, DurationPredictorFilterChannels = 16,
            MelChannels = 8, DecoderHiddenChannels = 16, NumFlowBlocks = 2, CouplingLayers = 2,
            DecoderDropout = 0,
        }
        : new GradTTSOptions
        {
            HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, FilterChannels = 32,
            DropoutRate = 0, PrenetDropout = 0, DurationPredictorFilterChannels = 16,
            MelChannels = 8, DecoderDim = 8, SegmentFrames = 12, NumDiffusionSteps = 2,
        };

    private static TtsModelBase<double> Create(bool glow, AcousticModelOptions options)
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 8) { RandomSeed = 11 };
        return glow ? new GlowTTS<double>(architecture, (GlowTTSOptions)options)
            : new GradTTS<double>(architecture, (GradTTSOptions)options);
    }

    private static T Field<T>(object model, string name) => (T)model.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    private static GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>> Optimizer(TtsModelBase<double> model)
        => Field<GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>>>(model, "_optimizer");

    private static void Train(TtsModelBase<double> model)
    {
        var tokens = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++) tokens[i] = 10 + 7 * i;
        var mel = new Tensor<double>(new[] { 14, 8 });
        for (int f = 0; f < 14; f++)
            for (int c = 0; c < 8; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
        model.Train(tokens, mel);
        Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(
            ((ITrainingObjectiveProvider<double>)model).EvaluateTrainingObjective(tokens, mel)));
    }

    [Fact]
    public void Defaults_PreserveReleasedNativeGeometryAndOptimizerScales()
    {
        var grad = new GradTTSOptions();
        Assert.Equal(192, grad.EncoderDim); Assert.Equal(64, grad.DecoderDim);
        Assert.Equal(3, grad.NumDecoderLayers); Assert.Equal(new[] { 1, 2, 4 }, grad.DecoderDimMultipliers);
        Assert.Equal(1e-4, grad.LearningRate); Assert.Equal(0, grad.WeightDecay);
        var glow = new GlowTTSOptions();
        Assert.Equal(192, glow.EncoderDim); Assert.Equal(192, glow.DecoderDim);
        Assert.Equal(12, glow.NumDecoderLayers); Assert.Equal(12, glow.NumFlowBlocks);
        Assert.Equal(1.0, glow.LearningRate); Assert.Equal(0, glow.WeightDecay);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EncoderAlias_ChangesTheActualNativeEmbedding(bool glow)
    {
        var options = Options(glow); options.EncoderDim = 32;
        using var model = Create(glow, options);
        Assert.Equal(32, options.HiddenDim);
        var embedding = Field<AiDotNet.NeuralNetworks.Layers.EmbeddingLayer<double>>(model, "_embedding");
        Assert.Equal(options.VocabSize * 32, embedding.GetParameters().Length);
        Train(model);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void GradDecoderDepth_BuildsAndTrainsTheRequestedNativeUNet(int depth)
    {
        var options = (GradTTSOptions)Options(false); options.NumDecoderLayers = depth;
        using var model = Create(false, options);
        var score = Field<GradTtsScoreEstimator<double>>(model, "_scoreNetwork");
        Assert.Equal(depth, Field<ICollection>(score, "_downs").Count);
        Assert.Equal(depth - 1, Field<ICollection>(score, "_ups").Count);
        Train(model);
        var before = score.Layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray(); Train(model);
        Assert.False(before.SequenceEqual(score.Layers.SelectMany(layer => layer.GetParameters().ToArray())));
        var mel = model.Synthesize("hello");
        Assert.Equal(8, mel.Shape[1]);
        Assert.All(mel.ToVector().ToArray(), x => Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(x)));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void GlowDecoderAliases_ChangeActualCouplingsAndFlowDepth(int depth)
    {
        var options = (GlowTTSOptions)Options(true);
        options.NumDecoderLayers = depth; options.DecoderDim = 24;
        using var model = Create(true, options);
        Assert.Equal(24, options.DecoderHiddenChannels);
        Assert.Equal(depth * 3, Field<ICollection>(model, "_flows").Count);
        Train(model);
        Assert.Equal(8, model.Synthesize("hello").Shape[1]);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void LegacyJson_NativeFieldsWinStaleAliasesInEitherOrder(bool glow, bool reverse)
    {
        var data = glow
            ? JObject.Parse("{ 'HiddenDim': 16, 'EncoderDim': 192, 'DecoderHiddenChannels': 24, 'DecoderDim': 80, 'NumFlowBlocks': 2, 'NumDecoderLayers': 6 }")
            : JObject.Parse("{ 'HiddenDim': 16, 'EncoderDim': 192, 'DecoderDimMultipliers': [1,3,5], 'NumDecoderLayers': 6 }");
        if (reverse) data = new JObject(data.Properties().Reverse().Select(p => new JProperty(p.Name, p.Value.DeepClone())));
        var options = (AcousticModelOptions)JsonConvert.DeserializeObject(data.ToString(), glow ? typeof(GlowTTSOptions) : typeof(GradTTSOptions))!;
        Assert.Equal(16, options.EncoderDim);
        if (options is GlowTTSOptions g)
        {
            Assert.Equal(24, g.DecoderDim); Assert.Equal(2, g.NumDecoderLayers);
            var copy = new GlowTTSOptions(g);
            Assert.Equal(24, copy.DecoderDim); Assert.Equal(2, copy.NumFlowBlocks);
        }
        else
        {
            var gradOptions = Assert.IsType<GradTTSOptions>(options);
            Assert.Equal(new[] { 1, 3, 5 }, gradOptions.DecoderDimMultipliers); Assert.Equal(3, gradOptions.NumDecoderLayers);
            var copy = new GradTTSOptions(gradOptions);
            Assert.Equal(gradOptions.DecoderDimMultipliers, copy.DecoderDimMultipliers);
            Assert.NotSame(gradOptions.DecoderDimMultipliers, copy.DecoderDimMultipliers);
        }
    }

    public class DerivedGradOptions : GradTTSOptions { public string Marker { get; set; } = "grad"; }
    public class DerivedGlowOptions : GlowTTSOptions { public string Marker { get; set; } = "glow"; }

    [Fact]
    public void AliasOnlyJsonAndDerivedOptions_KeepTheNativeTypeAndGeometry()
    {
        const string json = "{ 'EncoderDim':24, 'DecoderDim':16, 'NumDecoderLayers':2, 'Marker':'restored' }";
        var grad = JsonConvert.DeserializeObject<DerivedGradOptions>(json)!;
        Assert.Equal("restored", grad.Marker); Assert.Equal(24, grad.HiddenDim);
        Assert.Equal(16, grad.DecoderDim); Assert.Equal(new[] { 1, 2 }, grad.DecoderDimMultipliers);
        var glow = JsonConvert.DeserializeObject<DerivedGlowOptions>(json)!;
        Assert.Equal("restored", glow.Marker); Assert.Equal(24, glow.HiddenDim);
        Assert.Equal(16, glow.DecoderHiddenChannels); Assert.Equal(2, glow.NumFlowBlocks);
    }

    [Theory]
    [InlineData(false, 1e-4, 0.0)]
    [InlineData(false, 0.002, 0.2)]
    [InlineData(true, 1.0, 0.0)]
    [InlineData(true, 0.002, 0.2)]
    public void NativeAdam_UsesConfiguredRateDecayAndReleasedBetas(bool glow, double rate, double decay)
    {
        var options = Options(glow); options.LearningRate = rate; options.WeightDecay = decay;
        using var model = Create(glow, options);
        var settings = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(Optimizer(model).GetOptions());
        Assert.Equal(rate, settings.InitialLearningRate);
        Assert.Equal(0.9, settings.Beta1); Assert.Equal(glow ? 0.98 : 0.999, settings.Beta2);
        Assert.Equal(glow ? 1e-9 : 1e-8, settings.Epsilon);
        Assert.False(settings.UseAdaptiveBetas); Assert.False(settings.EnableGradientClipping);
        Assert.Equal(decay > 0, settings.RegularizationExplicitlySet);
        if (decay > 0)
            Assert.Equal(decay, Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(settings.Regularization).GetOptions().Strength);
        Train(model);
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(0.002)]
    public void GlowNoam_MatchesAuthorCommonsAdamAtAndBeyondWarmup(double factor)
    {
        var options = Options(true); options.LearningRate = factor;
        using var model = Create(true, options);
        var optimizer = Optimizer(model);
        double Expected(int completed)
        {
            double t = completed + 1.0;
            return factor * Math.Pow(options.HiddenDim, -0.5) * Math.Min(Math.Pow(t, -0.5), t * Math.Pow(4000, -1.5));
        }
        foreach (int completed in new[] { 0, 1, 99, 3999, 4000, 9999, 240000 })
            Assert.Equal(Expected(completed), optimizer.LearningRateScheduler!.GetLearningRateAtStep(completed), 14);
        Assert.Equal(Expected(0), optimizer.GetCurrentLearningRate(), 14);
        Train(model);
        Assert.Equal(Expected(1), optimizer.GetCurrentLearningRate(), 14);
    }

    [Fact]
    public void GlowClip_ClampsValuesToFiveWithoutApplyingANormClip()
    {
        using var model = Create(true, Options(true)); Train(model);
        var embedding = Field<AiDotNet.NeuralNetworks.Layers.EmbeddingLayer<double>>(model, "_embedding");
        var parameter = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { embedding }, -1)[0];
        var gradient = new Tensor<double>(parameter.Shape.ToArray());
        for (int i = 0; i < gradient.Length; i++) gradient[i] = (i % 4) switch { 0 => 6, 1 => -6, 2 => 4, _ => -4 };
        var gradients = new Dictionary<Tensor<double>, Tensor<double>> { [parameter] = gradient };
        typeof(NeuralNetworkBase<double>).GetMethod("ClipCustomObjectiveGradients", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(model, new object[] { gradients });
        for (int i = 0; i < gradient.Length; i++)
            Assert.Equal((i % 4) switch { 0 => 5.0, 1 => -5.0, 2 => 4.0, _ => -4.0 }, gradients[parameter][i]);
    }
}
