using System;
using System.Linq;
using AiDotNet.SelfSupervisedLearning;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.SelfSupervisedLearning;

/// <summary>
/// What a self-supervised training step promises beyond moving the weights: it runs at the configured schedule's rate,
/// applies the configured weight decay for every optimizer, rejects configurations it cannot honour, and its heads are
/// written in place and keep running statistics for evaluation.
/// </summary>
public class SelfSupervisedTrainingContractTests
{
    private static Tensor<double> Batch(int seed, params int[] shape)
    {
        var t = new Tensor<double>(shape.Length == 0 ? new[] { 8, 5 } : shape);
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static SimCLR<double> SimClr(SelfSupervisedLearningConfig<double> config) =>
        new(SslFixtureParts.Encoder(), SslFixtureParts.Mlp(1), config);

    [Fact]
    public void A_step_runs_at_the_rate_it_reports()
    {
        var config = new SelfSupervisedLearningConfig<double>
        {
            Seed = 1, LearningRate = 0.4, BatchSize = 64, WarmupEpochs = 4, UseCosineDecay = true, PretrainingEpochs = 20,
        };
        var method = SimClr(config);

        // Epoch 0 of a 4-epoch warmup: a quarter of 0.4 x 64 / 256.
        var first = method.TrainStep(Batch(1));
        Assert.Equal(0.4 * 64 / 256 / 4, first.CurrentLearningRate ?? double.NaN, 12);
        Assert.Equal(first.CurrentLearningRate ?? double.NaN, method.TrainingOptimizerLearningRate, 12);

        // Halfway through the cosine decay after warmup: half the rate.
        method.OnEpochStart(12);
        var later = method.TrainStep(Batch(2));
        Assert.Equal(0.4 * 64 / 256 * 0.5, later.CurrentLearningRate ?? double.NaN, 12);
        Assert.Equal(later.CurrentLearningRate ?? double.NaN, method.TrainingOptimizerLearningRate, 12);
    }

    [Fact]
    public void SGD_applies_the_configured_weight_decay()
    {
        SelfSupervisedLearningConfig<double> Config(double wd) => new()
        {
            Seed = 1, LearningRate = 0.5, WarmupEpochs = 0, UseCosineDecay = false,
            OptimizerType = SelfSupervisedLearningOptimizerType.SGD, WeightDecay = wd,
        };
        var plain = SimClr(Config(0.0));
        var decayed = SimClr(Config(0.1));
        var batch = Batch(3);
        var context = new SelfSupervisedLearningAugmentationContext<double> { PrecomputedView = Batch(4) };

        var plainStep = plain.TrainStep(batch, context);
        var decayedStep = decayed.TrainStep(batch, context);

        // The same objective from the same weights; the decay changes only the update.
        Assert.Equal(plainStep.Loss, decayedStep.Loss, 10);
        Assert.NotEqual(plain.GetParameters().ToArray(), decayed.GetParameters().ToArray());
    }

    [Fact]
    public void MAE_refuses_an_image_of_one_patch()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new MAE<double>(SslFixtureParts.Encoder(4, 4), null, patchSize: 4, imageSize: 4, maskRatio: 0.75,
                config: SslFixtureParts.Config()));
        Assert.Contains("at least 2", ex.Message);
    }

    [Fact]
    public void MoCo_refuses_a_momentum_projector_of_another_size()
    {
        var ex = Assert.Throws<ArgumentException>(() => new MoCoV3<double>(SslFixtureParts.Encoder(),
            SslFixtureParts.Momentum(), SslFixtureParts.Mlp(1), new MLPProjector<double>(8, 32, 8, seed: 2),
            config: SslFixtureParts.Config()));
        Assert.Contains("momentum projector", ex.Message);
        Assert.Throws<ArgumentException>(() => new MoCo<double>(SslFixtureParts.Encoder(), SslFixtureParts.Momentum(),
            SslFixtureParts.Mlp(1), new MLPProjector<double>(8, 32, 8, seed: 2), embeddingDim: 8,
            config: SslFixtureParts.Config()));
    }

    [Fact]
    public void DINO_trains_its_student_on_the_configured_local_crops()
    {
        DINO<double> Build(int localCrops) => new(SslFixtureParts.Encoder(), SslFixtureParts.Momentum(),
            SslFixtureParts.Mlp(1), SslFixtureParts.Mlp(2), outputDim: 8,
            config: new SelfSupervisedLearningConfig<double> { Seed = 1, DINO = new DINOConfig { NumLocalCrops = localCrops } });
        var images = Batch(5, 8, 1, 4, 4);
        var globalOnly = Build(0);
        var multiCrop = Build(2);

        var a = globalOnly.TrainStep(images);
        var b = multiCrop.TrainStep(images);

        Assert.False(double.IsNaN(a.Loss) || double.IsInfinity(a.Loss) || double.IsNaN(b.Loss) || double.IsInfinity(b.Loss));
        // The local crops add student terms to the objective, so the same batch gives another loss.
        Assert.NotEqual(a.Loss, b.Loss);
        Assert.Throws<ArgumentException>(() => new DINO<double>(SslFixtureParts.Encoder(), SslFixtureParts.Momentum(),
            SslFixtureParts.Mlp(1), SslFixtureParts.Mlp(2), outputDim: 8,
            config: new SelfSupervisedLearningConfig<double> { Seed = 1, DINO = new DINOConfig { NumGlobalCrops = 3 } }));
    }

    [Fact]
    public void Projector_parameters_are_written_into_the_live_tensors()
    {
        IProjectorHead<double>[] heads =
        {
            new MLPProjector<double>(8, 16, 8, useBatchNormOnOutput: true, seed: 1),
            new LinearProjector<double>(8, 4, seed: 1),
            new SymmetricProjector<double>(8, 16, 8, 16, seed: 1),
        };
        foreach (var head in heads)
        {
            var live = head.GetTrainableTensors().ToArray();
            var values = new Vector<double>((int)head.ParameterCount);
            for (int i = 0; i < values.Length; i++) values[i] = i * 1e-3;

            head.SetParameters(values);

            // The same tensor objects, now holding the new values: an optimizer keyed by reference keeps its state.
            Assert.Equal(live, head.GetTrainableTensors().ToArray());
            Assert.Equal(values.ToArray(), head.GetParameters().ToArray());
            Assert.Throws<ArgumentException>(() => head.SetParameters(new Vector<double>(values.Length - 1)));
            Assert.Throws<NotSupportedException>(() => head.GetParameterGradients());
        }
    }

    [Fact]
    public void The_symmetric_projector_evaluates_with_its_running_statistics()
    {
        var head = new SymmetricProjector<double>(8, 16, 8, 16, seed: 1);
        var batch = Batch(6, 16, 8);
        for (int i = 0; i < 20; i++) head.Project(batch);
        head.SetTrainingMode(false);

        // With running statistics, a sample's projection no longer depends on the rest of its batch.
        var whole = head.Project(batch);
        var first = head.Project(FirstRow(batch));
        for (int j = 0; j < 8; j++) Assert.Equal(whole[0, j], first[0, j], 10);

        static Tensor<double> FirstRow(Tensor<double> b)
        {
            var row = new Tensor<double>(new[] { 1, 8 });
            for (int j = 0; j < 8; j++) row[0, j] = b[0, j];
            return row;
        }
    }
}
