using System;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

[Trait("category", "unit")]
public sealed class ParameterChunkMutationContractTests
{
    public ParameterChunkMutationContractTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void Chunk_restore_advances_each_live_tensor_version_without_replacing_storage()
    {
        using var model = CreateModel();
        var live = model.GetParameterChunks().ToArray();
        var arrays = live.Select(t => t.GetDataArray()).ToArray();
        var versions = live.Select(t => t.Version).ToArray();
        var replacement = live.Select(t => t.Clone()).ToArray();
        foreach (var tensor in replacement) tensor.SetFlat(0, tensor.GetFlat(0) + 1);
        model.SetParameterChunks(replacement);
        var updated = model.GetParameterChunks().ToArray();
        for (int i = 0; i < live.Length; i++)
        {
            Assert.Same(live[i], updated[i]);
            Assert.Same(arrays[i], updated[i].GetDataArray());
            Assert.Equal(replacement[i].ToArray(), updated[i].ToArray());
            Assert.True(updated[i].Version > versions[i], $"Chunk {i} did not publish its mutation.");
        }
    }

    [Fact]
    public void Chunk_restore_invalidates_warmed_packed_inference_weights()
    {
        const int m = 8, k = 512, n = 64;
        using var model = CreateModel();
        var weight = model.GetParameterChunks().Single(t => t.Length == k * n);
        var initial = weight.ToArray();
        for (int i = 0; i < initial.Length; i++) initial[i] = 2;
        weight.CopyFromArray(initial);
        var backing = weight.GetDataArray();
        var input = Enumerable.Repeat(1f, m * k).ToArray();
        var output = new float[m * n];
        SimdGemm.SgemmWithCachedB(input, backing, output, m, k, n);
        Assert.All(output, x => Assert.Equal(1024f, x));
        var replacements = model.GetParameterChunks().Select(t => t.Clone()).ToArray();
        var nextWeight = replacements.Single(t => t.Length == k * n);
        // Change one interior element. Whole-array changes can be noticed by the
        // cache's sampled content fingerprint even when the writer omits invalidation.
        var next = (float[])initial.Clone();
        next[1] = 3;
        nextWeight.CopyFromArray(next);
        model.SetParameterChunks(replacements);
        Assert.Same(backing, weight.GetDataArray());
        SimdGemm.SgemmWithCachedB(input, backing, output, m, k, n);
        for (int row = 0; row < m; row++)
            for (int column = 0; column < n; column++)
                Assert.Equal(column == 1 ? 1025f : 1024f, output[row * n + column]);
    }

    [Fact]
    public void Invalid_chunk_stream_changes_neither_parameters_nor_mutation_versions()
    {
        using var model = CreateModel();
        var live = model.GetParameterChunks().ToArray();
        var values = model.GetParameters().ToArray();
        var versions = live.Select(t => t.Version).ToArray();
        var malformed = live.Select(t => t.Clone()).ToArray();
        malformed[0].SetFlat(0, 100);
        malformed[malformed.Length - 1] = new Tensor<float>(new[] { 1 });
        Assert.Throws<ArgumentException>(() => model.SetParameterChunks(malformed));
        Assert.Equal(values, model.GetParameters().ToArray());
        Assert.Equal(versions, live.Select(t => t.Version).ToArray());
    }

    private static NeuralNetwork<float> CreateModel()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 512, outputSize: 64) { RandomSeed = 51 };
        architecture.Layers.Add(new DenseLayer<float>(64, (IActivationFunction<float>)new IdentityActivation<float>()));
        var model = new NeuralNetwork<float>(architecture);
        model.Predict(new Tensor<float>(new[] { 1, 512 }));
        return model;
    }
}
