using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Adapts a trained AiDotNet <see cref="NeuralNetworkBase{T}"/> language model (e.g.,
/// <c>MambaLanguageModel</c>, <c>GLALanguageModel</c>, or a Transformer LM head) to the
/// <see cref="ICausalLanguageModel{T}"/> seam, so <see cref="LocalEngineChatClient{T}"/> can run real,
/// fully in-process generation over the library's own networks.
/// </summary>
/// <typeparam name="T">The tensor element type of the network (e.g., <see cref="float"/> or <see cref="double"/>).</typeparam>
/// <remarks>
/// <para>
/// AiDotNet's language models accept a one-hot input tensor of shape <c>[1, sequence, vocab]</c> and return
/// logits of the same leading shape. This adapter encodes the context as that one-hot tensor, runs a forward
/// pass, and returns the final position's logits. It re-feeds the full context each step (no KV-cache yet)
/// and calls <see cref="NeuralNetworkBase{T}.ResetState"/> before each pass so recurrent models (Mamba/GLA)
/// start fresh — correct, if not yet optimal. A KV-cached fast path is a planned follow-up.
/// </para>
/// <para><b>For Beginners:</b> This is the bridge that lets the local chat engine talk to a real AiDotNet
/// network. It turns the running list of tokens into the exact tensor shape the network expects, asks the
/// network for its prediction, and hands back the scores for the next token — which the engine then samples
/// from. The result: a chatbot powered entirely by an in-house model, no external service.
/// </para>
/// </remarks>
public sealed class NeuralNetworkCausalLanguageModel<T> : ICausalLanguageModel<T>
{
    private static readonly T One = (T)Convert.ChangeType(1.0, typeof(T));

    private readonly NeuralNetworkBase<T> _network;
    private readonly int _maxContextTokens;

    /// <summary>
    /// Initializes a new adapter over a network language model.
    /// </summary>
    /// <param name="network">The trained network whose <c>Predict</c> maps one-hot tokens to vocab logits.</param>
    /// <param name="vocabularySize">The model's vocabulary size (the width of the one-hot input and logits).</param>
    /// <param name="maxContextTokens">
    /// The maximum number of most-recent tokens to feed the network (typically its max sequence length).
    /// <c>null</c> or non-positive feeds the entire context.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="network"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="vocabularySize"/> is not positive.</exception>
    public NeuralNetworkCausalLanguageModel(NeuralNetworkBase<T> network, int vocabularySize, int? maxContextTokens = null)
    {
        Guard.NotNull(network);
        Guard.Positive(vocabularySize);
        _network = network;
        VocabularySize = vocabularySize;
        _maxContextTokens = maxContextTokens is { } configured && configured > 0 ? configured : int.MaxValue;
    }

    /// <inheritdoc/>
    public int VocabularySize { get; }

    /// <inheritdoc/>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="tokenIds"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="tokenIds"/> is empty.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when a token id is outside the vocabulary.</exception>
    public Vector<T> NextTokenLogits(IReadOnlyList<int> tokenIds)
    {
        Guard.NotNull(tokenIds);
        if (tokenIds.Count == 0)
        {
            throw new ArgumentException("The context must contain at least one token.", nameof(tokenIds));
        }

        var start = _maxContextTokens == int.MaxValue ? 0 : Math.Max(0, tokenIds.Count - _maxContextTokens);
        var sequenceLength = tokenIds.Count - start;

        var input = new Tensor<T>(new[] { 1, sequenceLength, VocabularySize });
        for (var position = 0; position < sequenceLength; position++)
        {
            var id = tokenIds[start + position];
            if (id < 0 || id >= VocabularySize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tokenIds), id, $"Token id is outside the vocabulary [0, {VocabularySize}).");
            }

            input[new[] { 0, position, id }] = One;
        }

        _network.ResetState();
        var output = _network.Predict(input);
        return ExtractLastPositionLogits(output);
    }

    private Vector<T> ExtractLastPositionLogits(Tensor<T> output)
    {
        var shape = output.Shape.ToArray();
        var vocab = shape[shape.Length - 1];
        if (vocab != VocabularySize)
        {
            throw new InvalidOperationException(
                $"Model produced logits of width {vocab}, but vocabulary size is {VocabularySize}.");
        }

        var logits = new T[vocab];
        if (shape.Length == 3)
        {
            var lastPosition = shape[1] - 1;
            for (var v = 0; v < vocab; v++)
            {
                logits[v] = output[new[] { 0, lastPosition, v }];
            }
        }
        else if (shape.Length == 2)
        {
            var lastPosition = shape[0] - 1;
            for (var v = 0; v < vocab; v++)
            {
                logits[v] = output[new[] { lastPosition, v }];
            }
        }
        else
        {
            throw new InvalidOperationException(
                $"Expected logits of rank 2 or 3, but the model returned rank {shape.Length}.");
        }

        return new Vector<T>(logits);
    }
}
