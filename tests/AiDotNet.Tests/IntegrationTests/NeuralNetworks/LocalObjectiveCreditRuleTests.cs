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

    private static (NeuralNetwork<float> net, Tensor<float> tokens, Tensor<float> targets) BuildSequenceLm(
        int blocks, int width, int vocab, bool tiedHead, int seed)
    {
        var emb = new EmbeddingLayer<float>(vocab, width);
        var layers = new List<ILayer<float>> { emb };
        for (int i = 0; i < blocks; i++)
            layers.Add(new DenseLayer<float>(width, (IActivationFunction<float>)new TanhActivation<float>()));
        layers.Add(new RMSNormalizationLayer<float>(width));
        if (tiedHead)
            layers.Add(new TiedEmbeddingHeadLayer<float>(emb));
        else
            layers.Add(new DenseLayer<float>(vocab, (IActivationFunction<float>)new IdentityActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: width,
            outputSize: vocab,
            layers: layers) { RandomSeed = seed };
        var net = new NeuralNetwork<float>(arch);

        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 100);
        int batch = 4, seq = 8;
        var tokens = new Tensor<float>(new[] { batch, seq });
        var targets = new Tensor<float>(new[] { batch, seq });
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seq; s++)
            {
                tokens[b, s] = rng.Next(0, vocab);
                targets[b, s] = ((int)tokens[b, s] + 1) % vocab;
            }
        }
        // Initialize lazy layers once before counting fused calls.
        _ = net.Predict(tokens);
        return (net, tokens, targets);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void StandardBackpropComputeGradients_FusesReadoutHeadWithoutMaterializingLogits(bool tiedHead)
    {
        var (net, tokens, targets) = BuildSequenceLm(blocks: 2, width: 16, vocab: 48, tiedHead: tiedHead, seed: 53);
        var loss = new CrossEntropyWithLogitsLoss<float>();

        DenseLayer<float>.FusedReadoutLossCount = 0;
        var flat = net.ComputeGradients(tokens, targets, loss);
        Assert.Equal(1L, DenseLayer<float>.FusedReadoutLossCount);
        Assert.True(flat.All(v => !float.IsNaN(v) && !float.IsInfinity(v)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TwoLayerReadout_RmsNormPlusHead_FusesTerminalAndAllHiddenReadoutLosses(bool tiedHead)
    {
        var (net, tokens, targets) = BuildSequenceLm(blocks: 3, width: 16, vocab: 48, tiedHead: tiedHead, seed: 17);
        var loss = new CrossEntropyWithLogitsLoss<float>();
        net.SetCreditRule(new TiedReadoutLocalErrorCreditRule<float>(readoutLayerCount: 2));

        DenseLayer<float>.FusedReadoutLossCount = 0;
        var flat = net.ComputeGradients(tokens, targets, loss);
        // 4 trainable hidden layers (Embedding + 3 blocks): 1 terminal fused loss + 1 depth-stacked batched intermediate fused loss = 2.
        Assert.Equal(2L, DenseLayer<float>.FusedReadoutLossCount);
        Assert.True(flat.All(v => !float.IsNaN(v) && !float.IsInfinity(v)));

        // Verify exact gradient parity of the depth-stacked single-GEMM readout against manual per-layer autodiff.
        var got = new Dictionary<Tensor<float>, float[]>(TensorReferenceComparer<Tensor<float>>.Instance);
        int off = 0;
        foreach (var chunk in net.GetParameterChunks())
        {
            if (chunk is null || chunk.Length == 0) continue;
            got[chunk] = Enumerable.Range(off, chunk.Length).Select(i => flat[i]).ToArray();
            off += chunk.Length;
        }

        net.SetTrainingMode(true);
        var L = net.Layers;
        Tensor<float> DetachF(Tensor<float> t) => new Tensor<float>(t.Shape.ToArray(), t.ToVector());
        Tensor<float> ScalarF(Tensor<float> t) => t.Shape.Length == 0
            ? t
            : AiDotNetEngine.Current.ReduceSum(t, Enumerable.Range(0, t.Shape.Length).ToArray(), keepDims: false);

        var allParams = got.Keys.ToList();
        Dictionary<Tensor<float>, Tensor<float>> manualExpect;
        using (var tape = new GradientTape<float>())
        {
            var o0 = L[0].Forward(tokens);
            var o1 = L[1].Forward(DetachF(o0));
            var o2 = L[2].Forward(DetachF(o1));
            var o3 = L[3].Forward(DetachF(o2));

            var l0 = ScalarF(loss.ComputeTapeLoss(L[5].Forward(L[4].Forward(o0)), targets));
            var l1 = ScalarF(loss.ComputeTapeLoss(L[5].Forward(L[4].Forward(o1)), targets));
            var l2 = ScalarF(loss.ComputeTapeLoss(L[5].Forward(L[4].Forward(o2)), targets));
            var lTerm = ScalarF(loss.ComputeTapeLoss(L[5].Forward(L[4].Forward(o3)), targets));
            var total = AiDotNetEngine.Current.TensorAdd(
                AiDotNetEngine.Current.TensorAdd(l0, l1),
                AiDotNetEngine.Current.TensorAdd(l2, lTerm));
            manualExpect = tape.ComputeGradients(total, sources: allParams);
        }

        foreach (var p in allParams)
        {
            var g = got[p];
            var e = manualExpect[p];
            for (int i = 0; i < g.Length; i++)
            {
                Assert.True(Math.Abs(g[i] - e[i]) <= 1e-4f * Math.Max(1f, Math.Abs(e[i])),
                    $"batched readout grad mismatch (tied={tiedHead}) at [{i}]: got {g[i]}, expected {e[i]}");
            }
        }
    }

    [Fact]
    public void DetachedReadoutOnHiddenLayers_PreventsReadoutCorruption_OnBothFusedAndGeneralReadouts()
    {
        // 1) General non-fused readout (FullyConnectedLayer + Softmax): readout gets ONLY last hidden layer's objective.
        var net = BuildNet(hidden: 3, seed: 19);
        var (x, y, _) = MakeBlobs(20, seed: 5);
        var ce = new CategoricalCrossEntropyLoss<double>();
        net.SetCreditRule(new TiedReadoutLocalErrorCreditRule<double>(
            readoutLayerCount: 1, localWeight: 1.0, detachReadoutOnHiddenLayers: true));
        var flat = net.ComputeGradients(x, y, ce);

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
        var h0 = Detach(L[0].Forward(x));
        var h1 = Detach(L[1].Forward(h0));
        var expectHidden0 = ManualGrads(() => Scalar(ce.ComputeTapeLoss(L[3].Forward(L[0].Forward(x)), y)), Params(L[0]));
        var expectTerminalReadout = ManualGrads(() => Scalar(ce.ComputeTapeLoss(L[3].Forward(L[2].Forward(h1)), y)), Params(L[3]));

        foreach (var p in Params(L[0]))
        {
            var g = got[p]; var e = expectHidden0[p];
            for (int i = 0; i < g.Length; i++)
                Assert.True(Math.Abs(g[i] - e[i]) <= 1e-10 * Math.Max(1.0, Math.Abs(e[i])));
        }
        foreach (var p in Params(L[3]))
        {
            var g = got[p]; var e = expectTerminalReadout[p];
            for (int i = 0; i < g.Length; i++)
                Assert.True(Math.Abs(g[i] - e[i]) <= 1e-10 * Math.Max(1.0, Math.Abs(e[i])),
                    $"readout param leaked intermediate local gradients: got {g[i]:R}, terminal-only {e[i]:R}");
        }

        // 2) Two-layer fused LM readout (RMSNorm + DenseLayer): RMSNorm gamma and Dense weights get terminal-only grads.
        var (lm, tokens, targets) = BuildSequenceLm(blocks: 3, width: 16, vocab: 40, tiedHead: false, seed: 23);
        var lmLoss = new CrossEntropyWithLogitsLoss<float>();
        lm.SetCreditRule(new TiedReadoutLocalErrorCreditRule<float>(
            readoutLayerCount: 2, localWeight: 1.0, detachReadoutOnHiddenLayers: true));
        DenseLayer<float>.FusedReadoutLossCount = 0;
        var lmFlat = lm.ComputeGradients(tokens, targets, lmLoss);
        Assert.Equal(2L, DenseLayer<float>.FusedReadoutLossCount);

        var lmGot = new Dictionary<Tensor<float>, float[]>(TensorReferenceComparer<Tensor<float>>.Instance);
        int lmOff = 0;
        foreach (var chunk in lm.GetParameterChunks())
        {
            if (chunk is null || chunk.Length == 0) continue;
            lmGot[chunk] = Enumerable.Range(lmOff, chunk.Length).Select(i => lmFlat[i]).ToArray();
            lmOff += chunk.Length;
        }

        lm.SetTrainingMode(true);
        var lmLayers = lm.Layers;
        Tensor<float> DetachF(Tensor<float> t) => new Tensor<float>(t.Shape.ToArray(), t.ToVector());
        var s0 = DetachF(lmLayers[0].Forward(tokens));
        var s1 = DetachF(lmLayers[1].Forward(s0));
        var s2 = DetachF(lmLayers[2].Forward(s1));
        var readoutParams = ((ITrainableLayer<float>)lmLayers[4]).GetTrainableParameters()
            .Concat(((ITrainableLayer<float>)lmLayers[5]).GetTrainableParameters()).ToList();

        Dictionary<Tensor<float>, Tensor<float>> termExpect;
        using (var tape = new GradientTape<float>())
        {
            var termPred = lmLayers[5].Forward(lmLayers[4].Forward(lmLayers[3].Forward(s2)));
            var rawTermLoss = lmLoss.ComputeTapeLoss(termPred, targets);
            var termLoss = rawTermLoss.Shape.Length == 0
                ? rawTermLoss
                : AiDotNetEngine.Current.ReduceSum(rawTermLoss, Enumerable.Range(0, rawTermLoss.Shape.Length).ToArray(), keepDims: false);
            termExpect = tape.ComputeGradients(termLoss, sources: readoutParams);
        }

        foreach (var p in readoutParams)
        {
            var g = lmGot[p]; var e = termExpect[p];
            for (int i = 0; i < g.Length; i++)
                Assert.True(Math.Abs(g[i] - e[i]) <= 1e-4f * Math.Max(1f, Math.Abs(e[i])),
                    $"LM readout param leaked hidden gradients at [{i}]: got {g[i]}, expected {e[i]}");
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SampledTiedReadout_ZeroAuxParams_FusesCandidateSubspaceAndLearns(bool tiedHead)
    {
        var (net, tokens, targets) = BuildSequenceLm(blocks: 4, width: 16, vocab: 64, tiedHead: tiedHead, seed: 31);
        var loss = new CrossEntropyWithLogitsLoss<float>();
        var sampled = new SampledTiedReadoutLocalErrorCreditRule<float>(
            readoutLayerCount: 2, maxCandidates: 16, localWeight: 1.0,
            detachReadoutOnHiddenLayers: true, layerStride: 2, useDepthProgressiveWeight: true, seed: 42);
        Assert.Empty(sampled.AuxiliaryParameters);

        net.SetCreditRule(sampled);
        DenseLayer<float>.FusedReadoutLossCount = 0;
        var flat = net.ComputeGradients(tokens, targets, loss);
        // 5 hidden layers (0..4): lastHidden=4 (terminal fused=1) + 1 depth-stacked batched sampled fused call over selected layers {1, 3} = 2.
        Assert.Equal(2L, DenseLayer<float>.FusedReadoutLossCount);
        Assert.True(flat.All(v => !float.IsNaN(v) && !float.IsInfinity(v)));

        var got = new Dictionary<Tensor<float>, float[]>(TensorReferenceComparer<Tensor<float>>.Instance);
        int off = 0;
        foreach (var chunk in net.GetParameterChunks())
        {
            if (chunk is null || chunk.Length == 0) continue;
            got[chunk] = Enumerable.Range(off, chunk.Length).Select(i => flat[i]).ToArray();
            off += chunk.Length;
        }

        net.SetTrainingMode(true);
        var L = net.Layers;
        Tensor<float> DetachF(Tensor<float> t) => new Tensor<float>(t.Shape.ToArray(), t.ToVector());
        Tensor<float> ScalarF(Tensor<float> t) => t.Shape.Length == 0
            ? t
            : AiDotNetEngine.Current.ReduceSum(t, Enumerable.Range(0, t.Shape.Length).ToArray(), keepDims: false);

        // Unselected non-shared block L[2] (k = 2) must receive identically zero gradient.
        foreach (var p in ((ITrainableLayer<float>)L[2]).GetTrainableParameters())
            Assert.All(got[p], v => Assert.Equal(0f, v));

        var flatTargetIds = new Tensor<int>(new[] { targets.Length });
        for (int r = 0; r < targets.Length; r++)
            flatTargetIds.SetFlat(r, (int)targets.GetFlat(r));
        LocalObjectiveGradientComputer<float>.BuildSampledCandidates(
            flatTargetIds, vocabSize: 64, maxCandidates: 16, stepSeed: unchecked(42 + 7919),
            out var candidateIds, out var remappedIds);
        var remappedTargets = new Tensor<float>(targets.Shape.ToArray());
        for (int r = 0; r < remappedIds.Length; r++)
            remappedTargets.SetFlat(r, remappedIds.GetFlat(r));

        var d0 = DetachF(L[0].Forward(tokens));
        var d1 = DetachF(L[1].Forward(d0));
        var d2 = DetachF(L[2].Forward(d1));

        foreach (int k in new[] { 1, 3 })
        {
            double w = ((k + 1.0) / 5.0) * ((k + 1.0) / 5.0);
            var hIn = k == 1 ? d0 : d2;
            var layerParams = ((ITrainableLayer<float>)L[k]).GetTrainableParameters().ToList();
            Dictionary<Tensor<float>, Tensor<float>> expLayer;
            using (var tape = new GradientTape<float>())
            {
                var fullLogits = L[6].Forward(L[5].Forward(L[k].Forward(hIn)));
                var candLogits = AiDotNetEngine.Current.TensorIndexSelect(
                    fullLogits, candidateIds, axis: fullLogits.Rank - 1);
                var lk = AiDotNetEngine.Current.TensorMultiplyScalar(
                    ScalarF(loss.ComputeTapeLoss(candLogits, remappedTargets)), (float)w);
                expLayer = tape.ComputeGradients(lk, sources: layerParams);
            }

            foreach (var p in layerParams)
            {
                var g = got[p];
                var e = expLayer[p];
                for (int i = 0; i < g.Length; i++)
                {
                    Assert.True(Math.Abs(g[i] - e[i]) <= 1e-4f * Math.Max(1f, Math.Abs(e[i])),
                        $"sampled subspace grad mismatch at layer {k} [{i}]: got {g[i]}, expected {e[i]}");
                }
            }
        }
    }

    private sealed class ScaledRMSNormLayer : RMSNormalizationLayer<float>
    {
        private readonly float _scale;
        public ScaledRMSNormLayer(int featureSize, float scale) : base(featureSize) => _scale = scale;
        protected override Tensor<float> ForwardTraced(Tensor<float> input)
            => Engine.TensorMultiplyScalar(base.ForwardTraced(input), _scale);
    }

    [AiDotNet.Attributes.ElementWiseShape]
    private sealed class BatchCenteringLayer : LayerBase<float>
    {
        public BatchCenteringLayer(int width) : base(new[] { width }, new[] { width }) { }
        public override bool SupportsTraining => false;
        public override void ResetState() { }
        public override void UpdateParameters(float learningRate) { }
        protected override Tensor<float> ForwardTraced(Tensor<float> input)
        {
            var sum = Engine.ReduceSum(input, new[] { 0 }, keepDims: true);
            var mean = Engine.TensorDivideScalar(sum, (float)input.Shape[0]);
            return Engine.TensorSubtract(input, mean);
        }
    }

    [Fact]
    public void SubclassedPreHeadNorm_PreservesForwardOverrideAndSampledSubspaceWhenDetached()
    {
        const int width = 12, vocab = 48;
        var layers = new List<ILayer<float>>
        {
            new EmbeddingLayer<float>(vocab, width),
            new DenseLayer<float>(width, (IActivationFunction<float>)new GELUActivation<float>()),
            new DenseLayer<float>(width, (IActivationFunction<float>)new GELUActivation<float>()),
            new ScaledRMSNormLayer(width, scale: 2.5f),
            new DenseLayer<float>(vocab, (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple, inputSize: 6, outputSize: vocab, layers: layers) { RandomSeed = 51 };
        var net = new NeuralNetwork<float>(arch);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(51);
        var tokens = new Tensor<float>(new[] { 2, 6 });
        var targets = new Tensor<float>(new[] { 2, 6 });
        for (int i = 0; i < tokens.Length; i++)
        {
            tokens.SetFlat(i, rng.Next(vocab));
            targets.SetFlat(i, rng.Next(vocab));
        }
        net.Predict(tokens);
        var loss = new CrossEntropyWithLogitsLoss<float>();

        net.SetCreditRule(new SampledTiedReadoutLocalErrorCreditRule<float>(
            readoutLayerCount: 2, maxCandidates: 12, localWeight: 1.0,
            detachReadoutOnHiddenLayers: true, seed: 19));
        var flat = net.ComputeGradients(tokens, targets, loss);

        var got = new Dictionary<Tensor<float>, float[]>(TensorReferenceComparer<Tensor<float>>.Instance);
        int off = 0;
        foreach (var chunk in net.GetParameterChunks())
        {
            if (chunk is null || chunk.Length == 0) continue;
            got[chunk] = Enumerable.Range(off, chunk.Length).Select(i => flat[i]).ToArray();
            off += chunk.Length;
        }

        net.SetTrainingMode(true);
        var L = net.Layers;
        Tensor<float> DetachF(Tensor<float> t) => new Tensor<float>(t.Shape.ToArray(), t.ToVector());
        Tensor<float> ScalarF(Tensor<float> t) => t.Shape.Length == 0
            ? t
            : AiDotNetEngine.Current.ReduceSum(t, Enumerable.Range(0, t.Shape.Length).ToArray(), keepDims: false);

        var flatIds = new Tensor<int>(new[] { targets.Length });
        for (int r = 0; r < targets.Length; r++) flatIds.SetFlat(r, (int)targets.GetFlat(r));
        LocalObjectiveGradientComputer<float>.BuildSampledCandidates(
            flatIds, vocabSize: vocab, maxCandidates: 12, stepSeed: unchecked(19 + 7919),
            out var candIds, out var remappedIds);
        var remappedTargets = new Tensor<float>(targets.Shape.ToArray());
        for (int r = 0; r < remappedIds.Length; r++) remappedTargets.SetFlat(r, remappedIds.GetFlat(r));

        var d0 = DetachF(L[0].Forward(tokens));
        var d1 = DetachF(L[1].Forward(d0));

        // Hidden block L[1] must reflect BOTH the 2.5x ScaledRMSNorm override AND the U=12 sampled candidate set.
        var l1Params = ((ITrainableLayer<float>)L[1]).GetTrainableParameters().ToList();
        Dictionary<Tensor<float>, Tensor<float>> expL1;
        using (var tape = new GradientTape<float>())
        {
            var fullLogits = L[4].Forward(L[3].Forward(L[1].Forward(d0)));
            var candLogits = AiDotNetEngine.Current.TensorIndexSelect(fullLogits, candIds, axis: fullLogits.Rank - 1);
            expL1 = tape.ComputeGradients(ScalarF(loss.ComputeTapeLoss(candLogits, remappedTargets)), sources: l1Params);
        }
        foreach (var p in l1Params)
        {
            var g = got[p]; var e = expL1[p];
            for (int i = 0; i < g.Length; i++)
                Assert.True(Math.Abs(g[i] - e[i]) <= 1e-4f * Math.Max(1f, Math.Abs(e[i])),
                    $"subclassed norm sampled grad mismatch [{i}]: got {g[i]}, expected {e[i]}");
        }

        // Readout params (L[3] norm + L[4] head) must receive terminal-only gradients (zero leak from hidden layers).
        var readoutParams = ((ITrainableLayer<float>)L[3]).GetTrainableParameters()
            .Concat(((ITrainableLayer<float>)L[4]).GetTrainableParameters()).ToList();
        Dictionary<Tensor<float>, Tensor<float>> expReadout;
        using (var tape = new GradientTape<float>())
        {
            var termLogits = L[4].Forward(L[3].Forward(L[2].Forward(d1)));
            expReadout = tape.ComputeGradients(ScalarF(loss.ComputeTapeLoss(termLogits, targets)), sources: readoutParams);
        }
        foreach (var p in readoutParams)
        {
            var g = got[p]; var e = expReadout[p];
            for (int i = 0; i < g.Length; i++)
                Assert.True(Math.Abs(g[i] - e[i]) <= 1e-4f * Math.Max(1f, Math.Abs(e[i])),
                    $"subclassed norm readout param leaked hidden grads [{i}]: got {g[i]}, expected {e[i]}");
        }
    }

    [Fact]
    public void BatchCoupledPreHeadLayer_SkipsDepthStackingAndMatchesPerLayerGradients()
    {
        const int width = 12, vocab = 32;
        var layers = new List<ILayer<float>>
        {
            new EmbeddingLayer<float>(vocab, width),
            new DenseLayer<float>(width, (IActivationFunction<float>)new GELUActivation<float>()),
            new DenseLayer<float>(width, (IActivationFunction<float>)new GELUActivation<float>()),
            new BatchCenteringLayer(width),
            new DenseLayer<float>(vocab, (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple, inputSize: 5, outputSize: vocab, layers: layers) { RandomSeed = 61 };
        var net = new NeuralNetwork<float>(arch);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(61);
        var tokens = new Tensor<float>(new[] { 4, 5 });
        var targets = new Tensor<float>(new[] { 4, 5 });
        for (int i = 0; i < tokens.Length; i++)
        {
            tokens.SetFlat(i, rng.Next(vocab));
            targets.SetFlat(i, rng.Next(vocab));
        }
        net.Predict(tokens);
        var loss = new CrossEntropyWithLogitsLoss<float>();

        net.SetCreditRule(new TiedReadoutLocalErrorCreditRule<float>(
            readoutLayerCount: 2, localWeight: 1.0, detachReadoutOnHiddenLayers: false));
        DenseLayer<float>.FusedReadoutLossCount = 0;
        var flat = net.ComputeGradients(tokens, targets, loss);
        // 1 terminal + 2 separate per-layer calls (must NOT depth-stack across BatchCenteringLayer).
        Assert.Equal(3L, DenseLayer<float>.FusedReadoutLossCount);

        var got = new Dictionary<Tensor<float>, float[]>(TensorReferenceComparer<Tensor<float>>.Instance);
        int off = 0;
        foreach (var chunk in net.GetParameterChunks())
        {
            if (chunk is null || chunk.Length == 0) continue;
            got[chunk] = Enumerable.Range(off, chunk.Length).Select(i => flat[i]).ToArray();
            off += chunk.Length;
        }

        net.SetTrainingMode(true);
        var L = net.Layers;
        Tensor<float> DetachF(Tensor<float> t) => new Tensor<float>(t.Shape.ToArray(), t.ToVector());
        Tensor<float> ScalarF(Tensor<float> t) => t.Shape.Length == 0
            ? t
            : AiDotNetEngine.Current.ReduceSum(t, Enumerable.Range(0, t.Shape.Length).ToArray(), keepDims: false);

        var d0 = DetachF(L[0].Forward(tokens));
        var d1 = DetachF(L[1].Forward(d0));
        var allParams = net.Layers.OfType<ITrainableLayer<float>>().SelectMany(l => l.GetTrainableParameters()).ToList();

        Dictionary<Tensor<float>, Tensor<float>> expected;
        using (var tape = new GradientTape<float>())
        {
            var termLoss = ScalarF(loss.ComputeTapeLoss(L[4].Forward(L[3].Forward(L[2].Forward(d1))), targets));
            var l0Loss = ScalarF(loss.ComputeTapeLoss(L[4].Forward(L[3].Forward(L[0].Forward(tokens))), targets));
            var l1Loss = ScalarF(loss.ComputeTapeLoss(L[4].Forward(L[3].Forward(L[1].Forward(d0))), targets));
            var total = AiDotNetEngine.Current.TensorAdd(
                termLoss, AiDotNetEngine.Current.TensorAdd(l0Loss, l1Loss));
            expected = tape.ComputeGradients(total, sources: allParams);
        }

        foreach (var p in allParams)
        {
            var g = got[p]; var e = expected[p];
            for (int i = 0; i < g.Length; i++)
                Assert.True(Math.Abs(g[i] - e[i]) <= 1e-4f * Math.Max(1f, Math.Abs(e[i])),
                    $"batch-coupled pre-head grad mismatch [{i}]: got {g[i]}, expected {e[i]}");
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DetachedProgressiveBatchedReadout_MatchesPerLayerManualAutodiffGradients(bool tiedHead)
    {
        var (net, tokens, targets) = BuildSequenceLm(blocks: 3, width: 16, vocab: 48, tiedHead: tiedHead, seed: 37);
        var loss = new CrossEntropyWithLogitsLoss<float>();
        net.SetCreditRule(new TiedReadoutLocalErrorCreditRule<float>(
            readoutLayerCount: 2, localWeight: 0.75,
            detachReadoutOnHiddenLayers: true, layerStride: 1, useDepthProgressiveWeight: true));

        DenseLayer<float>.FusedReadoutLossCount = 0;
        var flat = net.ComputeGradients(tokens, targets, loss);
        Assert.Equal(2L, DenseLayer<float>.FusedReadoutLossCount);

        var got = new Dictionary<Tensor<float>, float[]>(TensorReferenceComparer<Tensor<float>>.Instance);
        int off = 0;
        foreach (var chunk in net.GetParameterChunks())
        {
            if (chunk is null || chunk.Length == 0) continue;
            got[chunk] = Enumerable.Range(off, chunk.Length).Select(i => flat[i]).ToArray();
            off += chunk.Length;
        }

        net.SetTrainingMode(true);
        var L = net.Layers;
        Tensor<float> DetachF(Tensor<float> t) => new Tensor<float>(t.Shape.ToArray(), t.ToVector());
        Tensor<float> ScalarF(Tensor<float> t) => t.Shape.Length == 0
            ? t
            : AiDotNetEngine.Current.ReduceSum(t, Enumerable.Range(0, t.Shape.Length).ToArray(), keepDims: false);

        var d0 = DetachF(L[0].Forward(tokens));
        var d1 = DetachF(L[1].Forward(d0));
        var d2 = DetachF(L[2].Forward(d1));

        for (int k = 0; k < 3; k++)
        {
            double w = 0.75 * ((k + 1.0) / 4.0) * ((k + 1.0) / 4.0);
            var hIn = k == 0 ? tokens : (k == 1 ? d0 : d1);
            var layerParams = ((ITrainableLayer<float>)L[k]).GetTrainableParameters().ToList();
            Dictionary<Tensor<float>, Tensor<float>> expLayer;
            using (var tape = new GradientTape<float>())
            {
                var pred = L[5].Forward(L[4].Forward(L[k].Forward(hIn)));
                var lk = AiDotNetEngine.Current.TensorMultiplyScalar(ScalarF(loss.ComputeTapeLoss(pred, targets)), (float)w);
                expLayer = tape.ComputeGradients(lk, sources: layerParams);
            }
            foreach (var p in layerParams)
            {
                // When tiedHead is true, L[0] (Embedding) also receives terminal readout gradient; check L[1]..L[2] directly.
                if (tiedHead && k == 0) continue;
                var g = got[p];
                var e = expLayer[p];
                for (int i = 0; i < g.Length; i++)
                {
                    Assert.True(Math.Abs(g[i] - e[i]) <= 1e-4f * Math.Max(1f, Math.Abs(e[i])),
                        $"progressive detached grad mismatch layer {k} [{i}]: got {g[i]}, expected {e[i]}");
                }
            }
        }
    }
}
