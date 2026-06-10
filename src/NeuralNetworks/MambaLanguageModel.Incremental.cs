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
    public MambaModelState<T> CreateStepState(int batchSize = 1)
    {
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
    public Tensor<T> Step(Tensor<T> tokenInput, MambaModelState<T> state)
    {
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
