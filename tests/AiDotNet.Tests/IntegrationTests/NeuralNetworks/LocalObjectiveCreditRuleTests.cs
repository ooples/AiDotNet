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
public class LocalObjectiveCreditRuleTests
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
