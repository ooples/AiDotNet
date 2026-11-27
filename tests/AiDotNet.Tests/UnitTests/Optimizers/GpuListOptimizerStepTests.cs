using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

[Collection("EngineCurrentGlobalState")]
public class GpuListOptimizerStepTests
{
    private static TapeStepContext<float> Context(out Tensor<float>[] parameters)
    {
        parameters = new[] { new Tensor<float>(new[] { 4 }), new Tensor<float>(new[] { 3 }) };
        var grads = new Dictionary<Tensor<float>, Tensor<float>>(TensorReferenceComparer<Tensor<float>>.Instance);
        foreach (var p in parameters)
        {
            var g = new Tensor<float>(p._shape);
            for (int i = 0; i < g.Length; i++) g.SetFlat(i, 1f);
            grads[p] = g;
        }
        return new TapeStepContext<float>(parameters, grads, loss: 0f);
    }

    [Fact]
    public void AStepThePackageDoesNotShip_ResolvesToNull_AndLeavesEveryParameterToTheLoop()
    {
        var ctx = Context(out _);
        Assert.Null(GpuListStepBridge.Resolve("TryNoSuchStepMultiTensor", 0, new object[] { 0.1f }));
        Assert.Null(GpuListOptimizerStep<float>.TryApply(
            ctx, AiDotNetEngine.Current, "TryNoSuchStepMultiTensor", 0, p => Array.Empty<Tensor<float>>(), 0.1f));
    }

    [Fact]
    public void OnACpuEngine_TheListStepDeclines_AndNothingIsUpdated()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var ctx = Context(out var parameters);

            var updated = GpuListOptimizerStep<float>.TryApply(
                ctx, AiDotNetEngine.Current, "TrySgdStepMultiTensor", 0, p => Array.Empty<Tensor<float>>(), 0.5f);

            Assert.Null(updated);
            foreach (var p in parameters)
                for (int i = 0; i < p.Length; i++) Assert.Equal(0f, p.GetFlat(i));
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [Fact]
    public void NonFloatOptimizers_NeverBatch()
    {
        var grads = new Dictionary<Tensor<double>, Tensor<double>>(TensorReferenceComparer<Tensor<double>>.Instance);
        var p = new Tensor<double>(new[] { 2 });
        grads[p] = new Tensor<double>(new[] { 2 });
        var ctx = new TapeStepContext<double>(new[] { p }, grads, loss: 0.0);
        Assert.Null(GpuListOptimizerStep<double>.TryApply(
            ctx, AiDotNetEngine.Current, "TrySgdStepMultiTensor", 0, q => Array.Empty<Tensor<double>>(), 0.5f));
    }

    [Fact]
    public void GetOrAddState_CreatesOnceAndReuses()
    {
        var states = new Dictionary<Tensor<float>, Tensor<float>>(TensorReferenceComparer<Tensor<float>>.Instance);
        var p = new Tensor<float>(new[] { 2 });
        int creations = 0;
        var first = GpuListOptimizerStep<float>.GetOrAddState(states, p, q => { creations++; return new Tensor<float>(q._shape); });
        var second = GpuListOptimizerStep<float>.GetOrAddState(states, p, q => { creations++; return new Tensor<float>(q._shape); });
        Assert.Same(first, second);
        Assert.Equal(1, creations);
    }
}
