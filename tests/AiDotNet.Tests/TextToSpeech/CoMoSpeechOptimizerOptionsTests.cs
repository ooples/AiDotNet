using System;
using System.Linq;
using System.Reflection;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Classic;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class CoMoSpeechOptimizerOptionsTests
{
    private static CoMoSpeechOptions Options() => new()
    {
        HiddenDim = 16, EncoderDim = 16, NumHeads = 2, NumEncoderLayers = 1,
        FilterChannels = 32, PrenetDropout = 0, DropoutRate = 0,
        DurationPredictorFilterChannels = 16, FlowDim = 8, MelChannels = 8,
        SegmentFrames = 12, TeacherSamplingSteps = 4, DistillationSteps = 6,
    };

    private static CoMoSpeech<double> Create(CoMoSpeechOptions options,
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? optimizer = null)
        => new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 8) { RandomSeed = 11 }, options, optimizer);

    private static T Field<T>(object model, string name) => (T)(model.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!);

    private static AdamOptimizer<double, Tensor<double>, Tensor<double>> Optimizer(CoMoSpeech<double> model, string field)
        => Field<AdamOptimizer<double, Tensor<double>, Tensor<double>>>(model, field);

    private static void Train(CoMoSpeech<double> model)
    {
        var tokens = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = 10 + 7 * i;
        var mel = new Tensor<double>(new[] { 14, 8 });
        for (int f = 0; f < 14; f++)
            for (int c = 0; c < 8; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
        model.Train(tokens, mel);
    }

    [Theory]
    [InlineData(1e-4, 0.0)]
    [InlineData(0.002, 0.2)]
    public void NativeTeacherAndStudent_UseConfiguredAdamWithoutBatchScaling(double rate, double decay)
    {
        // Author train.py creates ordinary torch.optim.Adam for both stages, with params.py's rate.
        var options = Options(); options.LearningRate = rate; options.WeightDecay = decay;
        using var model = Create(options);
        var teacher = Optimizer(model, "_optimizer");
        model.BeginDistillation();
        var student = Optimizer(model, "_distillationOptimizer");
        Assert.NotSame(teacher, student);
        foreach (var optimizer in new[] { teacher, student })
        {
            var settings = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
            Assert.Equal(rate, settings.InitialLearningRate);
            Assert.Equal(0.9, settings.Beta1); Assert.Equal(0.999, settings.Beta2); Assert.Equal(1e-8, settings.Epsilon);
            Assert.False(settings.UseAdaptiveBetas);
            Assert.False(settings.EnableGradientClipping);
            Assert.Equal(decay > 0, settings.RegularizationExplicitlySet);
            if (decay > 0)
                Assert.Equal(decay, Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(settings.Regularization).GetOptions().Strength);
        }
    }

    [Fact]
    public void Distillation_StartsFreshAdam_AndFreezesEncoderTeacherAndDurationPredictor()
    {
        var options = Options(); options.WeightDecay = 0.2;
        using var model = Create(options);
        Train(model);
        var teacherOptimizer = Optimizer(model, "_optimizer");
        Assert.Equal(1, Field<int>(teacherOptimizer, "_tapeStep"));
        model.BeginDistillation();
        var studentOptimizer = Optimizer(model, "_distillationOptimizer");
        Assert.Equal(0, Field<int>(studentOptimizer, "_tapeStep"));
        var encoder = model.Layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray();
        var duration = Field<AiDotNet.NeuralNetworks.Layers.VariancePredictorLayer<double>>(model, "_durationPredictor").GetParameters().ToArray();
        var frozenTeacher = Snapshot(model, "_teacher");
        var denoiser = Snapshot(model, "_denoiser");
        Train(model);
        Assert.Equal(1, Field<int>(teacherOptimizer, "_tapeStep"));
        Assert.Equal(1, Field<int>(studentOptimizer, "_tapeStep"));
        Assert.Equal(encoder, model.Layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray());
        Assert.Equal(duration, Field<AiDotNet.NeuralNetworks.Layers.VariancePredictorLayer<double>>(model, "_durationPredictor").GetParameters().ToArray());
        Assert.Equal(frozenTeacher, Snapshot(model, "_teacher"));
        Assert.False(denoiser.SequenceEqual(Snapshot(model, "_denoiser")));
    }

    private static double[] Snapshot(CoMoSpeech<double> model, string field)
        => Field<GradTtsScoreEstimator<double>>(model, field).Layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray();

    [Fact]
    public void NativeClip_NormalizesEncoderAndDecoderIndependently()
    {
        using var model = Create(Options());
        Train(model);
        var groups = (IReadOnlyList<IReadOnlyList<Tensor<double>>>)typeof(CoMoSpeech<double>)
            .GetMethod("GradientClippingGroups", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(model, new object[] { Array.Empty<Tensor<double>>() })!;
        Assert.Equal(2, groups.Count);
        var gradients = new Dictionary<Tensor<double>, Tensor<double>>();
        foreach (var (group, norm) in new[] { (groups[0], 3.0), (groups[1], 4.0) })
        {
            var parameter = group.First(p => p.Length > 0);
            var gradient = new Tensor<double>(parameter.Shape.ToArray());
            for (int i = 0; i < gradient.Length; i++) gradient[i] = norm / Math.Sqrt(gradient.Length);
            gradients.Add(parameter, gradient);
        }
        typeof(NeuralNetworkBase<double>).GetMethod("ClipCustomObjectiveGradients", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(model, new object[] { gradients });
        foreach (var group in groups)
        {
            double squaredNorm = group.Where(gradients.ContainsKey)
                .Sum(p => gradients[p].ToVector().ToArray().Sum(g => g * g));
            Assert.InRange(Math.Sqrt(squaredNorm), 0.999999, 1.000001);
        }
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(0.95)]
    [InlineData(1.0)]
    public void NativeTarget_FollowsExactPreUpdateEma_WhileTeacherStaysFrozen(double mu)
    {
        var options = Options(); options.EmaDecay = mu;
        using var model = Create(options);
        Train(model); model.BeginDistillation(); Train(model);
        var target = Snapshot(model, "_target");
        var online = Snapshot(model, "_denoiser");
        var teacher = Snapshot(model, "_teacher");
        Assert.False(target.SequenceEqual(online));
        Train(model);
        var actual = Snapshot(model, "_target");
        Assert.Equal(target.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(mu * target[i] + (1 - mu) * online[i], actual[i], 12);
        Assert.Equal(teacher, Snapshot(model, "_teacher"));
    }

    [Fact]
    public void ExplicitOptimizer_RemainsCallerControlledAcrossDistillation()
    {
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null);
        using var model = Create(Options(), optimizer);
        Train(model); model.BeginDistillation();
        Assert.Same(optimizer, Optimizer(model, "_optimizer"));
        Assert.Same(optimizer, Optimizer(model, "_distillationOptimizer"));
        Train(model);
        Assert.Equal(2, Field<int>(optimizer, "_tapeStep"));
    }
}
