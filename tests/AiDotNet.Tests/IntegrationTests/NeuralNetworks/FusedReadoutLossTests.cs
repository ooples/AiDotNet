using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.CreditAssignment;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// <see cref="IFusedReadoutLossLayer{T}"/>: a local-objective rule's per-layer readout loss goes through the readout's
/// fused loss (an LM head fused with cross-entropy) and yields the same gradients as the decomposed
/// <c>loss(readout(h), target)</c> path.
/// </summary>
public class FusedReadoutLossTests
{
    private const int Dim = 6, Width = 8, Vocab = 11, Rows = 20;

    /// <summary>Linear LM head <c>logits = h @ W</c> (W [Width, Vocab]) that can fuse itself with cross-entropy.</summary>
    private sealed class FusableHead : LayerBase<double>, IFusedReadoutLossLayer<double>
    {
        private Tensor<double> _w;
        private readonly Tensor<double> _zeroBias = new(new[] { Vocab });
        public bool Fuse = true;
        public int FusedCalls;

        public FusableHead(int seed) : base(new[] { Width }, new[] { Vocab })
        {
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
            _w = new Tensor<double>(new[] { Width, Vocab });
            for (int i = 0; i < _w.Length; i++) _w.SetFlat(i, (rng.NextDouble() * 2 - 1) * 0.5);
            RegisterTrainableParameter(_w, AiDotNet.Tensors.Engines.PersistentTensorRole.Weights);
        }

        public override bool SupportsTraining => true;
        public override void ResetState() { }
        public override IReadOnlyList<Tensor<double>> GetTrainableParameters() => new[] { _w };
        public override void SetTrainableParameters(IReadOnlyList<Tensor<double>> parameters)
        {
            _w = parameters[0];
            ClearRegisteredParameters();
            RegisterTrainableParameter(_w, AiDotNet.Tensors.Engines.PersistentTensorRole.Weights);
        }

        protected override Tensor<double> ForwardTraced(Tensor<double> input) => Engine.TensorMatMul(input, _w);

        public Tensor<double>? TryComputeFusedLoss(Tensor<double> input, Tensor<double> target, ILossFunction<double> loss)
        {
            if (!Fuse || loss is not CrossEntropyWithLogitsLoss<double> || target.Length != input.Shape[0]) return null;
            var ids = new Tensor<int>(new[] { target.Length });
            for (int r = 0; r < target.Length; r++)
            {
                double v = target.GetFlat(r);
                if (v < 0 || v >= Vocab || v != Math.Floor(v)) return null;
                ids.SetFlat(r, (int)v);
            }
            FusedCalls++;
            return Engine.FusedLinearCrossEntropyWithLogits(input, _w, _zeroBias, ids);
        }
    }

    private static (NeuralNetwork<double> net, FusableHead head) Build(int seed)
    {
        var head = new FusableHead(seed + 100);
        var layers = new List<ILayer<double>> { new FullyConnectedLayer<double>(Dim, Width, new TanhActivation<double>()) };
        for (int i = 0; i < 3; i++) layers.Add(new FullyConnectedLayer<double>(Width, Width, new TanhActivation<double>()));
        layers.Add(head);
        var arch = new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification, complexity: NetworkComplexity.Simple,
            inputSize: Dim, outputSize: Vocab, layers: layers) { RandomSeed = seed };
        return (new NeuralNetwork<double>(arch), head);
    }

    private static (Tensor<double> x, Tensor<double> ids) Data(int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var x = new Tensor<double>(new[] { Rows, Dim });
        var ids = new Tensor<double>(new[] { Rows });
        for (int i = 0; i < x.Length; i++) x.SetFlat(i, rng.NextDouble() * 2 - 1);
        for (int r = 0; r < Rows; r++) ids.SetFlat(r, rng.Next(Vocab));
        return (x, ids);
    }

    private static Vector<double> Grad(bool fuse, ICreditRule<double> rule, Tensor<double> x, Tensor<double> ids,
        Vector<double> init, out int fusedCalls)
    {
        var (net, head) = Build(seed: 4);
        net.SetParameters(init);
        head.Fuse = fuse;
        net.SetCreditRule(rule);
        var g = net.ComputeGradients(x, ids, new CrossEntropyWithLogitsLoss<double>());
        fusedCalls = head.FusedCalls;
        return g;
    }

    [Fact]
    public void TiedReadout_FusedReadoutLoss_MatchesDecomposedGradients()
    {
        var (x, ids) = Data(seed: 7);
        var init = Build(seed: 4).net.GetParameters();
        var decomposed = Grad(fuse: false, new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1), x, ids, init, out int callsOff);
        var fused = Grad(fuse: true, new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1), x, ids, init, out int callsOn);

        Assert.Equal(0, callsOff);
        Assert.Equal(3, callsOn); // 4 hidden layers; the last is trained by the output loss, so 3 local readout losses
        Assert.Equal(decomposed.Length, fused.Length);
        double maxAbs = 0, maxDiff = 0;
        for (int i = 0; i < fused.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(decomposed[i]));
            maxDiff = Math.Max(maxDiff, Math.Abs(decomposed[i] - fused[i]));
        }
        Assert.True(maxAbs > 0);
        Assert.True(maxDiff <= 1e-9 * Math.Max(1.0, maxAbs), $"fused vs decomposed gradient max diff {maxDiff:E3} (max |g| {maxAbs:E3})");
    }

    [Fact]
    public void LayerwiseDistillation_TaskTerm_UsesFusedReadoutLoss()
    {
        var (x, ids) = Data(seed: 8);
        var init = Build(seed: 4).net.GetParameters();
        LayerwiseDistillationCreditRule<double> Rule() => new(_ => throw new InvalidOperationException("teacher unused"),
            teacherLogits: null, readoutLayerCount: 1, hiddenWeight: 0.0, kdAlpha: 0.0, seed: 1, localTaskWeight: 0.5);
        var decomposed = Grad(fuse: false, Rule(), x, ids, init, out _);
        var fused = Grad(fuse: true, Rule(), x, ids, init, out int calls);
        Assert.Equal(3, calls);
        double maxDiff = 0;
        for (int i = 0; i < fused.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(decomposed[i] - fused[i]));
        Assert.True(maxDiff <= 1e-9, $"fused vs decomposed gradient max diff {maxDiff:E3}");
    }

    [Fact]
    public void UnsupportedTarget_FallsBackToDecomposedPath()
    {
        var (x, ids) = Data(seed: 9);
        ids.SetFlat(0, Vocab + 3); // out-of-range id: the fused op requires ids in [0, V), so the layer declines
        var init = Build(seed: 4).net.GetParameters();
        var fused = Grad(fuse: true, new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1), x, ids, init, out int calls);
        var decomposed = Grad(fuse: false, new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1), x, ids, init, out _);
        Assert.Equal(0, calls);
        for (int i = 0; i < fused.Length; i++) Assert.Equal(decomposed[i], fused[i]);
    }
}
