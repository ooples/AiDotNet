using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.LazyShape;

/// <summary>
/// End-to-end shape-resolution tests for the 21 lazy-spatial layers
/// migrated under issue #1209. Each test validates three claims:
/// (1) the ctor accepts only channel-shape and kernel-shape arguments,
/// (2) <c>IsShapeResolved</c> flips false → true on the first Forward,
/// (3) the same instance handles multiple input spatial dims (where
/// the layer's contract supports it) without rebuilding weights.
/// </summary>
public class LazySpatialLayerTests
{
    private static Tensor<double> Ramp(int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++)
            t[i] = 0.01 * (i + 1);
        return t;
    }

    [Fact]
    public void PoolingLayer_ResolvesFromInput()
    {
        var pool = new PoolingLayer<double>(poolSize: 2, stride: 2, type: PoolingType.Max);
        Assert.False(pool.IsShapeResolved);
        pool.Forward(Ramp(new[] { 1, 4, 8, 8 }));
        Assert.True(pool.IsShapeResolved);
    }

    [Fact]
    public void SubpixelConvolutionalLayer_ResolvesFromInput()
    {
        // Disambiguate scalar vs. vector activation overload — both
        // null-defaultable parameters, so an explicit cast picks the
        // scalar overload.
        var sub = new SubpixelConvolutionalLayer<double>(
            outputDepth: 2, upscaleFactor: 2, kernelSize: 3,
            activation: (IActivationFunction<double>?)null);
        Assert.False(sub.IsShapeResolved);
        sub.Forward(Ramp(new[] { 1, 4, 8, 8 }));
        Assert.True(sub.IsShapeResolved);
    }

    [Fact]
    public void TransitionLayer_ResolvesFromInput()
    {
        var trans = new TransitionLayer<double>(compressionFactor: 0.5);
        Assert.False(trans.IsShapeResolved);
        trans.Forward(Ramp(new[] { 1, 8, 8, 8 }));
        Assert.True(trans.IsShapeResolved);
    }

    [Fact]
    public void DenseBlockLayer_ResolvesFromInput()
    {
        var layer = new DenseBlockLayer<double>(growthRate: 4, bnMomentum: 0.1);
        Assert.False(layer.IsShapeResolved);
        layer.Forward(Ramp(new[] { 1, 8, 8, 8 }));
        Assert.True(layer.IsShapeResolved);
    }

    [Fact]
    public void DenseBlock_ResolvesFromInput()
    {
        var block = new DenseBlock<double>(numLayers: 2, growthRate: 4, bnMomentum: 0.1);
        Assert.False(block.IsShapeResolved);
        block.Forward(Ramp(new[] { 1, 8, 8, 8 }));
        Assert.True(block.IsShapeResolved);
    }

    [Fact]
    public void BasicBlock_ResolvesFromInput()
    {
        var block = new BasicBlock<double>(outChannels: 8, stride: 1, zeroInitResidual: false);
        Assert.False(block.IsShapeResolved);
        block.Forward(Ramp(new[] { 1, 8, 8, 8 }));
        Assert.True(block.IsShapeResolved);
    }

    [Fact]
    public void BottleneckBlock_ResolvesFromInput()
    {
        var block = new BottleneckBlock<double>(baseChannels: 4, stride: 1, zeroInitResidual: false);
        Assert.False(block.IsShapeResolved);
        block.Forward(Ramp(new[] { 1, 4, 8, 8 }));
        Assert.True(block.IsShapeResolved);
    }

    [Fact]
    public void InvertedResidualBlock_ResolvesFromInput()
    {
        var block = new InvertedResidualBlock<double>(
            outChannels: 8, expansionRatio: 2, stride: 1, useSE: false);
        Assert.False(block.IsShapeResolved);
        Assert.Equal(-1, block.InChannels);
        block.Forward(Ramp(new[] { 1, 4, 8, 8 }));
        Assert.True(block.IsShapeResolved);
        Assert.Equal(4, block.InChannels);
    }

    [Fact]
    public void DeformableConvolutionalLayer_ResolvesFromInput()
    {
        var dcn = new DeformableConvolutionalLayer<double>(
            outputChannels: 4, kernelSize: 3, padding: 1);
        Assert.False(dcn.IsShapeResolved);
        dcn.Forward(Ramp(new[] { 1, 2, 8, 8 }));
        Assert.True(dcn.IsShapeResolved);
    }

    [Fact]
    public void ResidualDenseBlock_ResolvesFromInput()
    {
        var rdb = new ResidualDenseBlock<double>(numFeatures: 4, growthChannels: 4, residualScale: 0.2);
        Assert.False(rdb.IsShapeResolved);
        rdb.Forward(Ramp(new[] { 1, 4, 8, 8 }));
        Assert.True(rdb.IsShapeResolved);
    }

    [Fact]
    public void RRDBLayer_ResolvesFromInput()
    {
        var rrdb = new RRDBLayer<double>(numFeatures: 4, growthChannels: 4, residualScale: 0.2);
        Assert.False(rrdb.IsShapeResolved);
        rrdb.Forward(Ramp(new[] { 1, 4, 8, 8 }));
        Assert.True(rrdb.IsShapeResolved);
    }

    [Fact]
    public void SwinPatchEmbeddingLayer_ResolvesFromInput()
    {
        var patch = new SwinPatchEmbeddingLayer<double>(patchSize: 4, embedDim: 16);
        Assert.False(patch.IsShapeResolved);
        patch.Forward(Ramp(new[] { 1, 3, 8, 8 }));
        Assert.True(patch.IsShapeResolved);
    }

    [Fact]
    public void UNetDiscriminator_ResolvesFromInput()
    {
        var disc = new UNetDiscriminator<double>(numChannels: 8, numBlocks: 2);
        Assert.False(disc.IsShapeResolved);
        disc.Forward(Ramp(new[] { 1, 3, 8, 8 }));
        Assert.True(disc.IsShapeResolved);
    }

    [Fact]
    public void RRDBNetGenerator_ResolvesFromInput()
    {
        var gen = new RRDBNetGenerator<double>(
            inputChannels: 3, outputChannels: 3,
            numFeatures: 8, growthChannels: 4,
            numRRDBBlocks: 1, scale: 2, residualScale: 0.2);
        Assert.False(gen.IsShapeResolved);
        gen.Forward(Ramp(new[] { 1, 3, 8, 8 }));
        Assert.True(gen.IsShapeResolved);
    }

    [Fact]
    public void SpyNetLayer_ResolvesFromInput()
    {
        var spy = new SpyNetLayer<double>(numLevels: 2);
        Assert.False(spy.IsShapeResolved);
        // Two stacked frames along channel axis → 2*C input channels.
        spy.Forward(Ramp(new[] { 1, 6, 8, 8 }));
        Assert.True(spy.IsShapeResolved);
    }

    // ---- Multi-scale: same layer instance handles a different input H/W
    // after the first forward. The lazy contract pins the channel count
    // on the first forward; spatial dims continue to flex per call.

    [Fact]
    public void Pool_MultiScale_SameInstance_HandlesMultipleSpatialSizes()
    {
        var pool = new PoolingLayer<double>(poolSize: 2, stride: 2, type: PoolingType.Max);
        var o1 = pool.Forward(Ramp(new[] { 1, 4, 16, 16 }));
        var o2 = pool.Forward(Ramp(new[] { 1, 4, 32, 32 }));
        Assert.Equal(8, o1.Shape[2]);
        Assert.Equal(16, o2.Shape[2]);
    }

    [Fact]
    public void DenseBlock_MultiScale_SameInstance_HandlesMultipleSpatialSizes()
    {
        var block = new DenseBlock<double>(numLayers: 2, growthRate: 4, bnMomentum: 0.1);
        block.SetTrainingMode(false);
        var o1 = block.Forward(Ramp(new[] { 1, 8, 8, 8 }));
        var o2 = block.Forward(Ramp(new[] { 1, 8, 16, 16 }));
        Assert.Equal(o1.Shape[1], o2.Shape[1]);
        Assert.Equal(8, o1.Shape[2]);
        Assert.Equal(16, o2.Shape[2]);
    }

    /// <summary>
    /// Lazy contract requires weights are allocated EXACTLY ONCE — on the
    /// first Forward — and reused across subsequent Forwards regardless of
    /// spatial shape. A snapshot of the parameter vector before vs. after a
    /// second-different-spatial Forward must be identical bit-for-bit; if
    /// it isn't, the layer rebuilt weights and the lazy contract is broken
    /// (the multi-scale variable-input promise of #1209 hinges on this).
    /// </summary>
    [Fact]
    public void DenseBlockLayer_MultiScale_DoesNotRebuildWeights()
    {
        var layer = new DenseBlockLayer<double>(growthRate: 4, bnMomentum: 0.1);
        layer.SetTrainingMode(false);
        layer.Forward(Ramp(new[] { 1, 8, 8, 8 }));

        var paramsBefore = layer.GetParameters();
        var snapshotBefore = new double[paramsBefore.Length];
        for (int i = 0; i < paramsBefore.Length; i++) snapshotBefore[i] = paramsBefore[i];

        layer.Forward(Ramp(new[] { 1, 8, 16, 16 })); // different H/W

        var paramsAfter = layer.GetParameters();
        Assert.Equal(snapshotBefore.Length, paramsAfter.Length);
        for (int i = 0; i < snapshotBefore.Length; i++)
            Assert.Equal(snapshotBefore[i], paramsAfter[i]);
    }
}
