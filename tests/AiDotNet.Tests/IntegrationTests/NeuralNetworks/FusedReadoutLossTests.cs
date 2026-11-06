using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
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
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
    private sealed class FusableHead : LayerBase<double>, IFusedReadoutLossLayer<double>, IShapeContract
    {
        public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
            ? new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(Vocab)),
            }
            : null;

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

    // A subclass declines to fuse (its forward may differ), so this head takes the decomposed path with the same math.
    private sealed class UnfusedDense : DenseLayer<double>
    {
        public UnfusedDense() : base(Vocab, (IActivationFunction<double>?)null) { }
    }

    private static NeuralNetwork<double> BuildWithDenseHead(DenseLayer<double> head, int seed)
    {
        var layers = new List<ILayer<double>> { new FullyConnectedLayer<double>(Dim, Width, new TanhActivation<double>()) };
        for (int i = 0; i < 3; i++) layers.Add(new FullyConnectedLayer<double>(Width, Width, new TanhActivation<double>()));
        layers.Add(head);
        var arch = new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification, complexity: NetworkComplexity.Simple,
            inputSize: Dim, outputSize: Vocab, layers: layers) { RandomSeed = seed };
        return new NeuralNetwork<double>(arch);
    }

    [Fact]
    public void DenseHead_FusedLoss_EqualsTheLossOfItsForward()
    {
        var (x, ids) = Data(seed: 10);
        var h = new Tensor<double>(new[] { Rows, Width });
        for (int i = 0; i < h.Length; i++) h.SetFlat(i, x.GetFlat(i % x.Length) * 0.7);
        var head = new DenseLayer<double>(Vocab, (IActivationFunction<double>?)null);
        var logits = head.Forward(h);
        var loss = new CrossEntropyWithLogitsLoss<double>();

        var fused = head.TryComputeFusedLoss(h, ids, loss);
        Assert.NotNull(fused);
        double expected = loss.ComputeTapeLoss(logits, ids).GetFlat(0);
        Assert.Equal(expected, fused.GetFlat(0), 12);
    }

    [Fact]
    public void DenseHead_DeclinesWhenTheFusedOpWouldNotMatch()
    {
        var (_, ids) = Data(seed: 11);
        var h = new Tensor<double>(new[] { Rows, Width });
        var loss = new CrossEntropyWithLogitsLoss<double>();

        var uninitialized = new DenseLayer<double>(Vocab, (IActivationFunction<double>?)null);
        Assert.Null(uninitialized.TryComputeFusedLoss(h, ids, loss));

        var tanhHead = new DenseLayer<double>(Vocab, (IActivationFunction<double>)new TanhActivation<double>());
        tanhHead.Forward(h);
        Assert.Null(tanhHead.TryComputeFusedLoss(h, ids, loss));

        var head = new DenseLayer<double>(Vocab, (IActivationFunction<double>?)null);
        head.Forward(h);
        Assert.NotNull(head.TryComputeFusedLoss(h, ids, loss));
        Assert.Null(head.TryComputeFusedLoss(h, ids, new CrossEntropyWithLogitsLoss<double>(classAxis: 0)));
        Assert.Null(head.TryComputeFusedLoss(h, ids, new MeanSquaredErrorLoss<double>()));
        var ignored = ids.Clone();
        ignored.SetFlat(0, -1); // the loss leaves this row out of its mean; the fused op has no ignored rows
        Assert.Null(head.TryComputeFusedLoss(h, ignored, loss));

        var unfused = new UnfusedDense();
        unfused.Forward(h);
        Assert.Null(unfused.TryComputeFusedLoss(h, ids, loss));
    }

    [Fact]
    public void TiedReadout_DenseHead_FusedMatchesDecomposedGradients()
    {
        var (x, ids) = Data(seed: 12);
        var fusedNet = BuildWithDenseHead(new DenseLayer<double>(Vocab, (IActivationFunction<double>?)null), seed: 5);
        var decomposedNet = BuildWithDenseHead(new UnfusedDense(), seed: 5);
        fusedNet.Predict(x);
        decomposedNet.Predict(x);
        var init = fusedNet.GetParameters();
        decomposedNet.SetParameters(init);
        fusedNet.SetCreditRule(new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1));
        decomposedNet.SetCreditRule(new TiedReadoutLocalErrorCreditRule<double>(readoutLayerCount: 1));
        var fused = fusedNet.ComputeGradients(x, ids, new CrossEntropyWithLogitsLoss<double>());
        var decomposed = decomposedNet.ComputeGradients(x, ids, new CrossEntropyWithLogitsLoss<double>());

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
}
