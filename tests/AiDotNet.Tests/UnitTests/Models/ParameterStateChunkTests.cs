using AiDotNet.AnomalyDetection.Statistical;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Policies;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Locks in the role-aware state-chunk contract used by the generated parameter manifest.
/// </summary>
public sealed class ParameterStateChunkTests
{
    [Fact]
    public async Task NeuralStateChunks_IncludeBuffersWithoutMarkingThemTrainable()
    {
        await Task.Yield();

        using var model = new EchoStateNetwork<double>();
        var chunks = model.GetParameterStateChunks().ToArray();

        Assert.Equal(model.ParameterCount, chunks.Sum(chunk => (long)chunk.Tensor.Length));
        AssertFlatParity(model.GetParameters(), chunks);
        Assert.Contains(chunks, chunk => chunk.Role == ParameterSlotRole.Trainable);
        Assert.Contains(chunks, chunk => chunk.Role == ParameterSlotRole.LearnedState);
        Assert.True(
            chunks.Where(chunk => chunk.Role == ParameterSlotRole.Trainable)
                .Sum(chunk => (long)chunk.Tensor.Length) < model.ParameterCount,
            "Persistent reservoir state must be checkpointed but excluded from optimization.");
        Assert.Equal(chunks.Length, chunks.Select(chunk => chunk.StableId).Distinct().Count());
    }

    [Fact]
    public async Task SparseStateChunks_UseStoredValuesRatherThanDenseLogicalExtent()
    {
        await Task.Yield();

        using var model = new SparseNeuralNetwork<double>();
        var chunks = model.GetParameterStateChunks().ToArray();

        Assert.Equal(model.ParameterCount, chunks.Sum(chunk => (long)chunk.Tensor.Length));
        Assert.Equal(model.GetParameters().Length, chunks.Sum(chunk => chunk.Tensor.Length));
        AssertFlatParity(model.GetParameters(), chunks);
    }

    [Fact]
    public async Task GeneratedClassicalAndComponentModels_InheritChunkParity()
    {
        await Task.Yield();

        using var detector = new ZScoreDetector<double>();
        using var policy = new BetaPolicy<double>();

        Assert.Equal(detector.ParameterCount, detector.GetParameterChunks().Sum(chunk => (long)chunk.Length));
        Assert.Equal(policy.ParameterCount, policy.GetParameterChunks().Sum(chunk => (long)chunk.Length));
        Assert.NotEmpty(detector.GetParameterStateChunks());
        Assert.NotEmpty(policy.GetParameterStateChunks());
    }

    private static void AssertFlatParity(
        AiDotNet.Tensors.LinearAlgebra.Vector<double> flat,
        IReadOnlyList<ParameterChunk<double>> chunks)
    {
        int offset = 0;
        for (int c = 0; c < chunks.Count; c++)
        {
            var tensor = chunks[c].Tensor;
            for (int i = 0; i < tensor.Length; i++)
                Assert.Equal(flat[offset++], tensor[i]);
        }
        Assert.Equal(flat.Length, offset);
    }
}
