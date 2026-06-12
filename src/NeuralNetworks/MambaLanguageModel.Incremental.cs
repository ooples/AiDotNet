using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers.SSM;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Per-token (KV-cached) decoding state for a <see cref="MambaLanguageModel{T}"/>: one
/// <see cref="MambaStepState{T}"/> per Mamba block, in layer order. All other layers (embedding, norm, LM
/// head) are position-wise and need no carried state.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class MambaModelState<T>
{
    internal MambaModelState(List<MambaStepState<T>> blockStates) => BlockStates = blockStates;

    internal List<MambaStepState<T>> BlockStates { get; }
}

public partial class MambaLanguageModel<T>
{
    /// <summary>
    /// Creates fresh decoding state for incremental, O(n) token-by-token generation.
    /// </summary>
    /// <param name="batchSize">The batch size used during stepping (typically 1). Default 1.</param>
    /// <returns>A new <see cref="MambaModelState{T}"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="batchSize"/> is not positive.</exception>
    public MambaModelState<T> CreateStepState(int batchSize = 1)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
        }

        var blockStates = new List<MambaStepState<T>>();
        foreach (var layer in Layers)
        {
            if (layer is MambaBlock<T> block)
            {
                blockStates.Add(block.CreateStepState(batchSize));
            }
        }

        return new MambaModelState<T>(blockStates);
    }

    /// <summary>
    /// Processes a single token using carried KV-cache state, returning the same logits the full-sequence
    /// <see cref="Predict"/> would produce at that position — but in O(1) work per token rather than
    /// reprocessing the prefix. Mamba blocks advance their recurrent state; position-wise layers run normally.
    /// </summary>
    /// <param name="tokenInput">The single-token input, shape [batch, 1, vocab] (one-hot or embeddings).</param>
    /// <param name="state">The decoding state from <see cref="CreateStepState"/>, advanced in place.</param>
    /// <returns>The next-token logits for this position, shape [batch, 1, vocab].</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="tokenInput"/> or <paramref name="state"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="tokenInput"/> is not single-token rank-3, or when <paramref name="state"/>
    /// does not carry one block state per Mamba block (e.g. it came from a different model).
    /// </exception>
    public Tensor<T> Step(Tensor<T> tokenInput, MambaModelState<T> state)
    {
        // Reject contract violations up front: a null/mismatched state or a
        // multi-token input would otherwise surface as NullReference/IndexOutOfRange
        // deep inside generation with no hint of the cause.
        if (tokenInput is null)
        {
            throw new ArgumentNullException(nameof(tokenInput));
        }

        if (state is null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        if (tokenInput.Shape.Length != 3 || tokenInput.Shape[1] != 1)
        {
            throw new ArgumentException(
                "tokenInput must be a single token of shape [batch, 1, vocab].", nameof(tokenInput));
        }

        var expectedBlockStates = 0;
        foreach (var layer in Layers)
        {
            if (layer is MambaBlock<T>)
            {
                expectedBlockStates++;
            }
        }

        if (state.BlockStates.Count != expectedBlockStates)
        {
            throw new ArgumentException(
                $"The decoding state carries {state.BlockStates.Count} block states but this model has " +
                $"{expectedBlockStates} Mamba blocks; use CreateStepState() from the same model.", nameof(state));
        }

        SetTrainingMode(false);
        var x = tokenInput;
        var blockIndex = 0;
        foreach (var layer in Layers)
        {
            if (layer is MambaBlock<T> block)
            {
                x = block.Step(x, state.BlockStates[blockIndex]);
                blockIndex++;
            }
            else
            {
                x = layer.Forward(x);
            }
        }

        return x;
    }
}
