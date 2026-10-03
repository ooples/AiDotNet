using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// TransformerEncoderBlock accepts an unbatched [time, features] sequence, as PyTorch's TransformerEncoderLayer
/// accepts an unbatched (S, E) input, and computes on it exactly what it computes on the same sequence as a batch of
/// one. Eight flow-matching TTS models (MatchaTTS, E3TTS, Voicebox, ...) declare a [time, features] input and, when
/// their hidden size equals their mel channel count, start directly at this block.
/// </summary>
public class TransformerEncoderBlockUnbatchedTests
{
    [Fact]
    public void UnbatchedInput_MatchesTheSameSequenceAsABatchOfOne()
    {
        var block = new TransformerEncoderBlock<double>(16, 4, 32, 0.0);
        block.SetTrainingMode(false);
        var rng = RandomHelper.CreateSeededRandom(2284);
        var unbatched = new Tensor<double>([5, 16]);
        for (int i = 0; i < unbatched.Length; i++) unbatched[i] = rng.NextDouble() * 2 - 1;
        var batched = new Tensor<double>([1, 5, 16]);
        for (int i = 0; i < unbatched.Length; i++) batched[i] = unbatched[i];

        var fromUnbatched = block.Forward(unbatched);
        var fromBatched = block.Forward(batched);

        Assert.Equal(new[] { 5, 16 }, fromUnbatched.Shape.ToArray());
        for (int i = 0; i < fromUnbatched.Length; i++)
            Assert.Equal(fromBatched[i], fromUnbatched[i], 12);
    }
}
