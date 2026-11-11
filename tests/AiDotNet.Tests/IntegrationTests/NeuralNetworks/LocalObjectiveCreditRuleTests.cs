using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.CreditAssignment;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Local-objective credit rules (<see cref="TiedReadoutLocalErrorCreditRule{T}"/>,
/// <see cref="LayerwiseDistillationCreditRule{T}"/>): exact gradient isolation against hand-built tapes, the rule is
/// honoured by <c>Train()</c> (which previously ran back-propagation regardless of the configured rule), and both
/// rules produce real held-out learning.
/// </summary>
public partial class LocalObjectiveCreditRuleTests
{
    private const int Dim = 8, Classes = 3, Width = 16;

    private static (Tensor<double> x, Tensor<double> y, int[] labels) MakeBlobs(int samples, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var centreRng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(20261105);
        var centres = new double[Classes][];
        for (int c = 0; c < Classes; c++)
        {
            centres[c] = new double[Dim];
            for (int d = 0; d < Dim; d++) centres[c][d] = (centreRng.NextDouble() * 2.0 - 1.0) * 3.0;
        }
        var x = new Tensor<double>(new[] { samples, Dim });
        var y = new Tensor<double>(new[] { samples, Classes });
        var labels = new int[samples];
        for (int n = 0; n < samples; n++)
        {
            int c = n % Classes; labels[n] = c; y[n, c] = 1.0;
            for (int d = 0; d < Dim; d++) x[n, d] = centres[c][d] + (rng.NextDouble() * 2.0 - 1.0) * 1.5;
        }
        return (x, y, labels);
    }

    /// <summary>Hidden FC layers of equal width (so the shared readout applies to each) + a softmax readout layer.</summary>
    private static NeuralNetwork<double> BuildNet(int hidden, int seed)
    {
        var layers = new List<ILayer<double>> { new FullyConnectedLayer<double>(Dim, Width, new TanhActivation<double>()) };
        for (int i = 1; i < hidden; i++) layers.Add(new FullyConnectedLayer<double>(Width, Width, new TanhActivation<double>()));
        layers.Add(new FullyConnectedLayer<double>(Width, Classes, new SoftmaxActivation<double>()));
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple, inputSize: Dim, outputSize: Classes, layers: layers) { RandomSeed = seed };
        return new NeuralNetwork<double>(arch);
    }

    private static double Accuracy(Func<Tensor<double>, Tensor<double>> predict, Tensor<double> x, int[] labels)
    {
        var p = predict(x); int ok = 0;
        for (int n = 0; n < labels.Length; n++)
        {
            int best = 0;
            for (int c = 1; c < Classes; c++) if (p[n, c] > p[n, best]) best = c;
            if (best == labels[n]) ok++;
        }
        return (double)ok / labels.Length;
    }

    private static Tensor<double> Detach(Tensor<double> t) => new Tensor<double>(t.Shape.ToArray(), t.ToVector());

    private static Dictionary<Tensor<double>, Tensor<double>> ManualGrads(Func<Tensor<double>> objective, IEnumerable<Tensor<double>> sources)
    {
        using var tape = new GradientTape<double>();
        var loss = objective();
        return tape.ComputeGradients(loss, sources: sources.ToList());
    }

    private static List<Tensor<double>> Params(ILayer<double> l) => ((ITrainableLayer<double>)l).GetTrainableParameters().ToList();

    [Fact]
    public void TiedReadout_EachLayerGradient_IsExactlyItsOwnLocalObjective()
    {
        var net = BuildNet(hidden: 3, seed: 7);
        var (x, y, _) = MakeBlobs(24, seed: 3);
        var loss = new CategoricalCrossEntropyLoss<double>();
        net.SetCreditRule(new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1));
        var flat = net.ComputeGradients(x, y, loss);

        // Slice the flat gradient back into chunks.
        var got = new Dictionary<Tensor<double>, double[]>(TensorReferenceComparer<Tensor<double>>.Instance);
        int off = 0;
        foreach (var chunk in net.GetParameterChunks())
        {
            if (chunk is null || chunk.Length == 0) continue;
            got[chunk] = Enumerable.Range(off, chunk.Length).Select(i => flat[i]).ToArray();
            off += chunk.Length;
        }

        net.SetTrainingMode(true);
        var L = net.Layers;
        Tensor<double> Scalar(Tensor<double> t) => AiDotNetEngine.Current.ReduceSum(t, Enumerable.Range(0, t.Shape.Length).ToArray(), keepDims: false);
        Tensor<double> LossOf(Tensor<double> pred) => Scalar(loss.ComputeTapeLoss(pred, y));

        var h0 = Detach(L[0].Forward(x));
        var h1 = Detach(L[1].Forward(h0));
        // Layer 0: loss(readout(L0(x))). Layer 1: loss(readout(L1(h0))). Layer 2 (last hidden): loss(readout(L2(h1))).
        var expect0 = ManualGrads(() => LossOf(L[3].Forward(L[0].Forward(x))), Params(L[0]));
        var expect1 = ManualGrads(() => LossOf(L[3].Forward(L[1].Forward(h0))), Params(L[1]));
        var expect2 = ManualGrads(() => LossOf(L[3].Forward(L[2].Forward(h1))), Params(L[2]));

        int checkedValues = 0;
        foreach (var (layer, expect) in new[] { (L[0], expect0), (L[1], expect1), (L[2], expect2) })
        {
            foreach (var p in Params(layer))
            {
                Assert.True(got.ContainsKey(p), "parameter chunk missing from the flat gradient");
                Assert.True(expect.TryGetValue(p, out var e), "manual tape produced no gradient");
                var g = got[p];
                double maxAbs = 0;
                for (int i = 0; i < g.Length; i++)
                {
                    Assert.True(Math.Abs(g[i] - e[i]) <= 1e-10 * Math.Max(1.0, Math.Abs(e[i])),
                        $"layer {L.IndexOf(layer)} gradient [{i}] = {g[i]:R}, its own local objective gives {e[i]:R}");
                    maxAbs = Math.Max(maxAbs, Math.Abs(g[i]));
                    checkedValues++;
                }
                Assert.True(maxAbs > 0, $"layer {L.IndexOf(layer)} received an all-zero gradient");
            }
        }
        Assert.True(checkedValues > 0);
    }

    [Fact]
    public void LayerwiseDistillation_LocalTaskWeightOnly_EqualsTiedReadout()
    {
        // localTaskWeight gives each detached hidden layer its own task loss through the shared readout. With the
        // teacher terms switched off (hiddenWeight 0, kdAlpha 0) the rule must be exactly TiedReadoutLocalError.
        var (x, y, _) = MakeBlobs(24, seed: 3);
        var loss = new CategoricalCrossEntropyLoss<double>();
        var init = BuildNet(hidden: 4, seed: 9).GetParameters();
        Vector<double> Grad(ICreditRule<double> rule)
        {
            var net = BuildNet(hidden: 4, seed: 9);
            net.SetParameters(init);
            net.SetCreditRule(rule);
            return net.ComputeGradients(x, y, loss);
        }
        var tied = Grad(new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1));
        var kdTask = Grad(new LayerwiseDistillationCreditRule<double>(
            _ => throw new InvalidOperationException("teacher must not run when hiddenWeight is 0"), teacherLogits: null,
            readoutLayerCount: 1, hiddenWeight: 0.0, kdAlpha: 0.0, seed: 3, localTaskWeight: 1.0));
        var kdOnly = Grad(new LayerwiseDistillationCreditRule<double>(
            input => { var h = input; var states = new List<Tensor<double>>(); var t = BuildNet(hidden: 2, seed: 5);
                       for (int i = 0; i < t.Layers.Count - 1; i++) { h = t.Layers[i].Forward(h); states.Add(Detach(h)); } return states; },
            teacherLogits: null, readoutLayerCount: 1, hiddenWeight: 1.0, kdAlpha: 0.0, seed: 3));
        Assert.Equal(tied.Length, kdTask.Length);
        double gap = 0, kdGap = 0;
        for (int i = 0; i < tied.Length; i++) { gap = Math.Max(gap, Math.Abs(tied[i] - kdTask[i])); kdGap = Math.Max(kdGap, Math.Abs(tied[i] - kdOnly[i])); }
        Assert.Equal(0.0, gap);
        Assert.True(kdGap > 1e-9, "teacher-regression-only gradients matched the tied rule; the comparison cannot distinguish the terms");
    }

    [Fact]
    public void LayerwiseDistillation_RunsTheFrozenTeacherOffTheTape()
    {
        // The teacher is frozen. Run on the student's tape, its forward kept every teacher activation alive until the
        // student's backward: a SmolLM2-360M teacher at batch 16x256 ran a 12 GB GPU out of memory.
        var (x, y, _) = MakeBlobs(24, seed: 3);
        var loss = new CategoricalCrossEntropyLoss<double>();
        var teacher = BuildNet(hidden: 2, seed: 5);
        bool hiddenRan = false, hiddenOnTape = false, logitsRan = false, logitsOnTape = false;
        var net = BuildNet(hidden: 4, seed: 9);
        net.SetCreditRule(new LayerwiseDistillationCreditRule<double>(
            input =>
            {
                hiddenRan = true;
                hiddenOnTape |= !NoGradScope<double>.IsSuppressed;
                var h = input; var states = new List<Tensor<double>>();
                for (int i = 0; i < teacher.Layers.Count - 1; i++) { h = teacher.Layers[i].Forward(h); states.Add(Detach(h)); }
                return states;
            },
            input =>
            {
                logitsRan = true;
                logitsOnTape |= !NoGradScope<double>.IsSuppressed;
                var h = input;
                foreach (var layer in teacher.Layers) h = layer.Forward(h);
                return Detach(h);
            },
            readoutLayerCount: 1, hiddenWeight: 1.0, kdAlpha: 0.5, seed: 3));
        net.ComputeGradients(x, y, loss);
        Assert.True(hiddenRan && logitsRan, "the teacher did not run; the test cannot observe where it ran");
        Assert.False(hiddenOnTape, "the teacher's hidden-state forward ran while the student's tape was recording");
        Assert.False(logitsOnTape, "the teacher's logit forward ran while the student's tape was recording");
    }

    [Fact]
    public void SetCreditRule_IsHonouredByTrain()
    {
        var (x, y, _) = MakeBlobs(30, seed: 4);
        // Explicitly supplied layers draw their init from the shared RNG, not architecture.RandomSeed, so every net
        // starts from one fixed parameter vector.
        var init = BuildNet(hidden: 3, seed: 11).GetParameters();
        Vector<double> AfterOneStep(ICreditRule<double>? rule)
        {
            var net = BuildNet(hidden: 3, seed: 11);
            net.SetParameters(init);
            net.SetCreditRule(rule);
            net.Train(x, y);
            return net.GetParameters();
        }
        var bp1 = AfterOneStep(null);
        var bp2 = AfterOneStep(null);
        var local = AfterOneStep(new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1));

        double bpDrift = 0, ruleGap = 0;
        for (int i = 0; i < bp1.Length; i++) { bpDrift = Math.Max(bpDrift, Math.Abs(bp1[i] - bp2[i])); ruleGap = Math.Max(ruleGap, Math.Abs(bp1[i] - local[i])); }
        Assert.Equal(0.0, bpDrift); // same init, same step: deterministic
        Assert.True(ruleGap > 1e-9, $"Train() with a local-objective rule produced the back-propagation update (max diff {ruleGap:E2}): the rule was ignored.");
    }

    [Fact]
    public void DeepCopy_CarriesTheCreditRule()
    {
        // The optimizer trains RequireModel().DeepCopy(); a copy without the rule trains by back-propagation.
        var net = BuildNet(hidden: 3, seed: 11);
        var rule = new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1);
        net.SetCreditRule(rule, seed: 5);
        var copy = (NeuralNetworkBase<double>)net.DeepCopy();
        Assert.Same(rule, copy.ActiveCreditRule);
        Assert.Same(rule, ((NeuralNetworkBase<double>)net.Clone()).ActiveCreditRule);
    }

    [Fact]
    public async Task ConfigureCreditRule_ChangesTheFacadeTrainedModel()
    {
        // Same init, same data, same optimizer: the facade-trained parameters must depend on the configured rule.
        var init = BuildNet(hidden: 3, seed: 11).GetParameters();
        async Task<Vector<double>> Facade(ICreditRule<double>? rule)
        {
            var net = BuildNet(hidden: 3, seed: 11);
            net.SetParameters(init);
            var (trainX, trainY, _) = MakeBlobs(64, seed: 1);
            var adam = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.02, MaxIterations = 2, BatchSize = 32 });
            var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>().ConfigureModel(net).ConfigureOptimizer(adam);
            if (rule is not null) builder = builder.ConfigureCreditRule(rule, 5);
            var result = await builder
                .ConfigureLossFunction(new CategoricalCrossEntropyLoss<double>())
                .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(trainX, trainY))
                .BuildAsync();
            return ((NeuralNetworkBase<double>)result.Model!).GetParameters();
        }
        var bp = await Facade(null);
        var local = await Facade(new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1));
        double gap = 0;
        for (int i = 0; i < bp.Length; i++) gap = Math.Max(gap, Math.Abs(bp[i] - local[i]));
        Assert.True(gap > 1e-9, $"facade training with a local-objective rule matched back-propagation (max diff {gap:E2}): the rule never reached the trained model.");
    }

    [Fact]
    public async Task ConfigureCreditRule_ChangesTheStreamingFacadeTrainedModel()
    {
        // The streaming build path (a StreamingDataLoaderBase loader) never wired the configured rule, so every
        // streamed run trained by back-propagation whatever ConfigureCreditRule said.
        var init = BuildNet(hidden: 3, seed: 11).GetParameters();
        var (trainX, trainY, _) = MakeBlobs(64, seed: 1);
        async Task<Vector<double>> Streamed(ICreditRule<double>? rule)
        {
            var net = BuildNet(hidden: 3, seed: 11);
            net.SetParameters(init);
            var loader = new StreamingDataLoader<double, Tensor<double>, Tensor<double>>(64, (i, _) =>
            {
                var xi = new Tensor<double>(new[] { Dim });
                var yi = new Tensor<double>(new[] { Classes });
                for (int d = 0; d < Dim; d++) xi[d] = trainX[i, d];
                for (int c = 0; c < Classes; c++) yi[c] = trainY[i, c];
                return Task.FromResult((xi, yi));
            }, batchSize: 32, prefetchCount: 1, numWorkers: 1);
            var adam = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.02, MaxIterations = 2, BatchSize = 32 });
            var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>().ConfigureModel(net).ConfigureOptimizer(adam);
            if (rule is not null) builder = builder.ConfigureCreditRule(rule, 5);
            var result = await builder
                .ConfigureLossFunction(new CategoricalCrossEntropyLoss<double>())
                .ConfigureDataLoader(loader)
                // A seeded shuffle: unseeded, the streaming loader draws its order from a secure RNG and no two
                // runs see the same batches.
                .ConfigureStreamingTraining(new StreamingTrainingOptions<double, Tensor<double>, Tensor<double>> { Seed = 7 })
                .BuildAsync();
            return ((NeuralNetworkBase<double>)result.Model!).GetParameters();
        }
        var bp1 = await Streamed(null);
        var bp2 = await Streamed(null);
        var local = await Streamed(new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1));
        double drift = 0, gap = 0;
        for (int i = 0; i < bp1.Length; i++) { drift = Math.Max(drift, Math.Abs(bp1[i] - bp2[i])); gap = Math.Max(gap, Math.Abs(bp1[i] - local[i])); }
        Assert.Equal(0.0, drift); // a streamed back-propagation run is reproducible, so the gap below is the rule's
        Assert.True(gap > 1e-9, $"streamed training with a local-objective rule matched back-propagation (max diff {gap:E2}): the rule was never wired.");
    }

    private static async Task<double> TrainAndScore(NeuralNetwork<double> net, ICreditRule<double>? rule, int iterations)
    {
        var (trainX, trainY, _) = MakeBlobs(300, seed: 1);
        var (testX, _, testLabels) = MakeBlobs(150, seed: 999);
        var adam = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.02, MaxIterations = iterations, BatchSize = 32 });
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(net)
            .ConfigureOptimizer(adam);
        if (rule is not null) builder = builder.ConfigureCreditRule(rule);
        var result = await builder
            .ConfigureLossFunction(new CategoricalCrossEntropyLoss<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(trainX, trainY))
            .BuildAsync();
        return Accuracy(result.Predict, testX, testLabels);
    }

    [Fact]
    public async Task TiedReadout_DeepStack_LearnsHeldOut()
    {
        var net = BuildNet(hidden: 6, seed: 21);
        var (testX, _, testLabels) = MakeBlobs(150, seed: 999);
        double before = Accuracy(net.Predict, testX, testLabels);
        double after = await TrainAndScore(net, new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1), 80);
        Assert.True(after >= 0.90, $"held-out accuracy {after:F3} (before {before:F3}, chance {1.0 / Classes:F3})");
    }

    [Fact]
    public async Task LayerwiseDistillation_FromTrainedTeacher_LearnsHeldOut()
    {
        // Hidden-state distillation only (kdAlpha = 0): this net's readout emits softmax probabilities, while the
        // soft-logit term expects raw logits (as a language model's head produces).
        // Teacher: a net trained by ordinary back-propagation; its hidden states are the targets.
        var teacher = BuildNet(hidden: 2, seed: 5);
        double teacherAcc = await TrainAndScore(teacher, rule: null, 80);
        Assert.True(teacherAcc >= 0.90, $"teacher only reached {teacherAcc:F3}");
        teacher.SetTrainingMode(false);
        IReadOnlyList<Tensor<double>> TeacherHidden(Tensor<double> input)
        {
            var states = new List<Tensor<double>>();
            var h = input;
            for (int i = 0; i < teacher.Layers.Count - 1; i++) { h = teacher.Layers[i].Forward(h); states.Add(Detach(h)); }
            return states;
        }
        var student = BuildNet(hidden: 4, seed: 33);
        var rule = new LayerwiseDistillationCreditRule<double>(TeacherHidden, teacherLogits: null, readoutLayerCount: 1,
            hiddenWeight: 1.0, kdAlpha: 0.0, projectionLearningRate: 0.05, seed: 3);
        double after = await TrainAndScore(student, rule, 80);
        Assert.True(after >= 0.90, $"distilled student held-out accuracy {after:F3} (teacher {teacherAcc:F3})");
        Assert.NotEmpty(rule.AuxiliaryParameters);
    }
}
