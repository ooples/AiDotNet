using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for detection necks — <c>FPN</c>, <c>PANet</c>, <c>BiFPN</c> — which fuse the
/// multi-scale feature maps a backbone produces.
/// </summary>
/// <remarks>
/// <para>
/// A neck cannot be tested through the single-tensor model contract: <c>NeckBase.Predict(Tensor)</c>
/// throws by design, because a neck consumes one feature map PER LEVEL
/// (<c>Forward(List&lt;Tensor&lt;T&gt;&gt;)</c>, highest resolution first). That is why these three
/// models sat uncovered — no generic fixture could feed them — and why this base drives
/// <c>Forward</c> directly.
/// </para>
/// <para>
/// <b>The base owns the channel configuration and passes it to the factory.</b> The generated fixture
/// implements <see cref="CreateNeck"/> as <c>new FPN&lt;double&gt;(inputChannels, outputChannels)</c>,
/// and the feature maps are built from the same <see cref="InputChannels"/>. The neck and its inputs
/// therefore cannot disagree, which is the failure class this generator has kept producing whenever a
/// size was written in two places.
/// </para>
/// <para>
/// <b>No training invariant.</b> A neck is trained by the detector that owns it.
/// </para>
/// </remarks>
public abstract class NeckTestBase<T>
{
    /// <summary>Shared numeric operations, matching the sibling family bases.</summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Converts a numeric value to double for assertion.</summary>
    protected static double ToD(T value) => Convert.ToDouble(value);

    /// <summary>Channels at each backbone level, highest resolution first.</summary>
    protected virtual int[] InputChannels => [8, 16, 32];

    /// <summary>Channels the neck projects every level to.</summary>
    protected virtual int OutputChannels => 16;

    /// <summary>Spatial size of the highest-resolution level; each further level halves it.</summary>
    protected virtual int TopResolution => 16;

    /// <summary>Subclasses construct their neck with exactly the configuration they are handed.</summary>
    protected abstract NeckBase<T> CreateNeck(int[] inputChannels, int outputChannels);

    private NeckBase<T> CreateNeck() => CreateNeck((int[])InputChannels.Clone(), OutputChannels);

    /// <summary>One NCHW map per level, halving resolution level by level; identical every call.</summary>
    private List<Tensor<T>> CreateFeatures()
    {
        var features = new List<Tensor<T>>();
        int size = TopResolution;
        for (int level = 0; level < InputChannels.Length; level++)
        {
            var map = new Tensor<T>([1, InputChannels[level], size, size]);
            for (int i = 0; i < map.Length; i++)
            {
                map[i] = NumOps.FromDouble((((i + level * 7) % 17) / 17.0) - 0.45);
            }

            features.Add(map);
            size = Math.Max(1, size / 2);
        }

        return features;
    }

    [Fact(Timeout = 120000)]
    public async Task Forward_ReturnsOneMapPerLevel_AtTheConfiguredChannelCount()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var neck = CreateNeck();

        var outputs = neck.Forward(CreateFeatures());

        Assert.NotNull(outputs);
        Assert.Equal(InputChannels.Length, outputs.Count);
        foreach (var output in outputs)
        {
            Assert.True(output.Shape.Length >= 2, "A fused feature map lost its channel axis.");
            Assert.Equal(OutputChannels, output.Shape[1]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Forward_ReturnsFiniteValues()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var neck = CreateNeck();

        foreach (var output in neck.Forward(CreateFeatures()))
        {
            for (int i = 0; i < output.Length; i++)
            {
                double value = ToD(output[i]);
                Assert.False(double.IsNaN(value), $"A fused feature value is NaN at {i}.");
                Assert.False(double.IsInfinity(value), $"A fused feature value is infinite at {i}.");
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Forward_IsDeterministic()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var neck = CreateNeck();

        var first = neck.Forward(CreateFeatures());
        var second = neck.Forward(CreateFeatures());

        Assert.Equal(first.Count, second.Count);
        for (int level = 0; level < first.Count; level++)
        {
            Assert.Equal(first[level].Length, second[level].Length);
            for (int i = 0; i < first[level].Length; i++)
            {
                Assert.Equal(ToD(first[level][i]), ToD(second[level][i]), 6);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ParameterCount_IsPositive()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var neck = CreateNeck();

        Assert.True(
            neck.GetParameterCount() > 0,
            "A neck with lateral and fusion convolutions reports no parameters, so nothing downstream "
                + "- optimisation, serialization, clone fidelity - can see its weights.");
    }

    [Fact(Timeout = 120000)]
    public async Task DeepCopy_FusesIdentically()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var neck = CreateNeck();

        var clone = Assert.IsAssignableFrom<NeckBase<T>>(neck.DeepCopy());
        var fromOriginal = neck.Forward(CreateFeatures());
        var fromClone = clone.Forward(CreateFeatures());

        Assert.Equal(fromOriginal.Count, fromClone.Count);
        for (int level = 0; level < fromOriginal.Count; level++)
        {
            for (int i = 0; i < fromOriginal[level].Length; i++)
            {
                Assert.Equal(ToD(fromOriginal[level][i]), ToD(fromClone[level][i]), 6);
            }
        }
    }
}

/// <summary>Double-precision convenience form, matching the other family bases.</summary>
public abstract class NeckTestBase : NeckTestBase<double>
{
}
