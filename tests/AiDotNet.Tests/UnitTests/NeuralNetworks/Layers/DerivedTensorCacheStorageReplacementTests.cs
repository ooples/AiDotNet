using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public sealed class DerivedTensorCacheStorageReplacementTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PublicStorageReplacementRebuildsDerivedWeightsEvenWhenEpochsMatch(bool inferenceMode)
    {
        var engine = new CpuEngine();
        var cache = new DerivedTensorCache<double>();
        var source = new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 2 });
        var replacement = new Tensor<double>(new[] { 7.0, 8.0 }, new[] { 2 });
        var originalStorage = source.StorageIdentity;
        var first = cache.GetOrBuild(engine, DoubleWeights, source);
        int epoch = source.StorageMutationVersion;
        int version = source.Version;

        // set_ preserves the source tensor object but adopts fresh storage. Its lazy mutation epoch can equal
        // the old storage's epoch; inference mode also deliberately preserves the per-object autograd version.
        using (inferenceMode ? new InferenceModeScope<double>() : null)
            engine.TensorSetInPlace(source, replacement);

        Assert.NotSame(originalStorage, source.StorageIdentity);
        Assert.Equal(epoch, source.StorageMutationVersion);
        if (inferenceMode) Assert.Equal(version, source.Version);
        var current = cache.GetOrBuild(engine, DoubleWeights, source);
        Assert.NotSame(first, current);
        Assert.Equal(new[] { 2.0, 4.0 }, first.ToArray());
        Assert.Equal(new[] { 14.0, 16.0 }, current.ToArray());
        Assert.Same(current, cache.GetOrBuild(engine, DoubleWeights, source));
    }

    private static Tensor<double> DoubleWeights(IEngine engine, Tensor<double>[] sources)
        => engine.TensorMultiplyScalar(sources[0], 2.0);
}
