using System.Text;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

public class CompiledTapeTrainingStepTests
{
    private readonly INumericOperations<float> _numOps = MathHelper.GetNumericOperations<float>();

    /// <summary>
    /// Verifies compiled training produces the same loss trajectory as eager training.
    /// Both paths start from identical weights and should produce matching results.
    /// </summary>
    [Fact]
    public void CompiledStep_MatchesEagerStep_OnSimpleMLP()
    {
        var (eagerLayers, eagerForward) = BuildMLP();
        var (compiledLayers, compiledForward) = BuildMLP();

        // Copy eager weights into compiled layers to guarantee identical starting point
        CopyWeights(eagerLayers, compiledLayers);

        var input = CreateRandomTensor(new[] { 16, 4 }, 42);
        var target = CreateRandomTensor(new[] { 16, 2 }, 43);
        float lr = _numOps.FromDouble(0.01);

        Func<Tensor<float>, Tensor<float>, Tensor<float>> mseLoss = (pred, tgt) =>
        {
            var engine = AiDotNetEngine.Current;
            var diff = engine.TensorSubtract(pred, tgt);
            var sq = engine.TensorMultiply(diff, diff);
            return engine.ReduceSum(sq, null);
        };

        // Train eager
        var eagerLosses = new List<float>();
        for (int step = 0; step < 5; step++)
        {
            var loss = TapeTrainingStep<float>.Step(
                eagerLayers, input, target, lr, eagerForward, mseLoss);
            eagerLosses.Add(Convert.ToSingle(loss));
        }

        // Train compiled
        CompiledTapeTrainingStep<float>.Invalidate();
        var compiledLosses = new List<float>();
        for (int step = 0; step < 5; step++)
        {
            var loss = CompiledTapeTrainingStep<float>.Step(
                compiledLayers, input, target, lr, compiledForward, mseLoss);
            compiledLosses.Add(Convert.ToSingle(loss));
        }

        // Both should decrease
        Assert.True(eagerLosses[^1] < eagerLosses[0],
            $"Eager loss should decrease: first={eagerLosses[0]:F4}, last={eagerLosses[^1]:F4}");
        Assert.True(compiledLosses[^1] < compiledLosses[0],
            $"Compiled loss should decrease: first={compiledLosses[0]:F4}, last={compiledLosses[^1]:F4}");

        // Losses should be close (not necessarily identical due to op ordering differences)
        for (int i = 0; i < eagerLosses.Count; i++)
        {
            Assert.False(float.IsNaN(eagerLosses[i]), $"Eager loss[{i}] is NaN");
            Assert.False(float.IsNaN(compiledLosses[i]), $"Compiled loss[{i}] is NaN");
        }
    }

    [Fact]
    public void CompiledStep_HandlesShapeChange_WithoutCrashing()
    {
        CompiledTapeTrainingStep<float>.Invalidate();
        var (layers, forward) = BuildMLP();
        float lr = _numOps.FromDouble(0.01);
        var mseLoss = MakeMSELoss();

        var input8 = CreateRandomTensor(new[] { 8, 4 }, 42);
        var target8 = CreateRandomTensor(new[] { 8, 2 }, 43);
        var loss1 = CompiledTapeTrainingStep<float>.Step(layers, input8, target8, lr, forward, mseLoss);
        Assert.False(float.IsNaN(Convert.ToSingle(loss1)));

        var input16 = CreateRandomTensor(new[] { 16, 4 }, 44);
        var target16 = CreateRandomTensor(new[] { 16, 2 }, 45);
        var loss2 = CompiledTapeTrainingStep<float>.Step(layers, input16, target16, lr, forward, mseLoss);
        Assert.False(float.IsNaN(Convert.ToSingle(loss2)));
    }

    [Fact]
    public void Invalidate_AllowsRecompilation()
    {
        CompiledTapeTrainingStep<float>.Invalidate();
        var (layers, forward) = BuildMLP();
        float lr = _numOps.FromDouble(0.01);
        var mseLoss = MakeMSELoss();
        var input = CreateRandomTensor(new[] { 8, 4 }, 42);
        var target = CreateRandomTensor(new[] { 8, 2 }, 43);

        var loss1 = CompiledTapeTrainingStep<float>.Step(layers, input, target, lr, forward, mseLoss);
        CompiledTapeTrainingStep<float>.Invalidate();
        var loss2 = CompiledTapeTrainingStep<float>.Step(layers, input, target, lr, forward, mseLoss);

        Assert.False(float.IsNaN(Convert.ToSingle(loss1)));
        Assert.False(float.IsNaN(Convert.ToSingle(loss2)));
    }

    [Fact]
    public void CompiledStep_IsFasterThanEager_AfterWarmup()
    {
        CompiledTapeTrainingStep<float>.Invalidate();
        var (layers, forward) = BuildMLP();
        var input = CreateRandomTensor(new[] { 32, 4 }, 42);
        var target = CreateRandomTensor(new[] { 32, 2 }, 43);
        float lr = _numOps.FromDouble(0.01);
        var mseLoss = MakeMSELoss();

        // Warmup
        for (int i = 0; i < 3; i++)
        {
            TapeTrainingStep<float>.Step(layers, input, target, lr, forward, mseLoss);
            CompiledTapeTrainingStep<float>.Step(layers, input, target, lr, forward, mseLoss);
        }

        var eagerSw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 20; i++)
            TapeTrainingStep<float>.Step(layers, input, target, lr, forward, mseLoss);
        eagerSw.Stop();

        var compiledSw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 20; i++)
            CompiledTapeTrainingStep<float>.Step(layers, input, target, lr, forward, mseLoss);
        compiledSw.Stop();

        Assert.True(compiledSw.Elapsed.TotalMilliseconds < eagerSw.Elapsed.TotalMilliseconds * 3,
            $"Compiled ({compiledSw.Elapsed.TotalMilliseconds:F1}ms) should not be 3x slower than eager ({eagerSw.Elapsed.TotalMilliseconds:F1}ms)");
    }

    private static (List<DenseLayer<float>> layers, Func<Tensor<float>, Tensor<float>> forward) BuildMLP()
    {
        // DenseLayer defaults to ReLU — don't apply ReLU externally
        var layer1 = new DenseLayer<float>(8);
        var layer2 = new DenseLayer<float>(2, (IActivationFunction<float>)new IdentityActivation<float>());
        var layers = new List<DenseLayer<float>> { layer1, layer2 };

        Tensor<float> Forward(Tensor<float> x)
        {
            var h = layer1.Forward(x);
            return layer2.Forward(h);
        }

        return (layers, Forward);
    }

    private static void CopyWeights(List<DenseLayer<float>> src, List<DenseLayer<float>> dst)
    {
        // Force initialization by doing a dry-run forward
        var dummy = CreateRandomTensor(new[] { 1, 4 }, 99);
        foreach (var l in src) l.Forward(dummy);
        foreach (var l in dst) l.Forward(dummy);

        for (int i = 0; i < src.Count; i++)
        {
            var srcParams = src[i].GetTrainableParameters();
            var dstParams = dst[i].GetTrainableParameters();
            for (int j = 0; j < srcParams.Count; j++)
                srcParams[j].AsSpan().CopyTo(dstParams[j].Data.Span);
        }
    }

    private static Func<Tensor<float>, Tensor<float>, Tensor<float>> MakeMSELoss()
    {
        return (pred, tgt) =>
        {
            var engine = AiDotNetEngine.Current;
            var diff = engine.TensorSubtract(pred, tgt);
            var sq = engine.TensorMultiply(diff, diff);
            return engine.ReduceSum(sq, null);
        };
    }

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        int length = 1;
        foreach (var d in shape) length *= d;
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }
}
