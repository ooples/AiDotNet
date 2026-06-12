using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// A KV-cached <see cref="IIncrementalCausalLanguageModel{T}"/> adapter over <see cref="MambaLanguageModel{T}"/>.
/// It drives the model's per-token <see cref="MambaLanguageModel{T}.Step"/> fast path, carrying the recurrent
/// state (causal-conv window + selective-scan hidden state) so each new token costs O(1) instead of
/// reprocessing the whole context.
/// </summary>
/// <typeparam name="T">The tensor element type (e.g., <see cref="float"/> or <see cref="double"/>).</typeparam>
/// <remarks>
/// <para>
/// The incremental path is mathematically equivalent to the full-sequence forward (Gu &amp; Dao 2023): feeding
/// tokens one at a time while carrying state reproduces <see cref="MambaLanguageModel{T}.Predict"/>'s logits at
/// every position. This is verified by unit tests that assert step-by-step output matches the parallel scan.
/// The base <see cref="NextTokenLogits"/> still runs a full forward (used when a caller does not opt into the
/// cache); the fast path is <see cref="StartSequence"/> + <see cref="AppendToken"/>.
/// </para>
/// <para><b>For Beginners:</b> A plain forward pass re-reads the entire conversation every time it predicts a
/// word, which gets slower as the text grows. Mamba can instead remember a small summary of everything so far
/// and update it with just the new word — same answer, far less work. This adapter exposes that
/// "remember-as-you-go" ability to the chat engine.
/// </para>
/// </remarks>
public sealed class MambaCausalLanguageModel<T> : IIncrementalCausalLanguageModel<T>
{
    private static readonly T One = (T)Convert.ChangeType(1.0, typeof(T));

    private readonly MambaLanguageModel<T> _network;
    private MambaModelState<T>? _state;

    /// <summary>
    /// Initializes a new KV-cached adapter over a Mamba language model.
    /// </summary>
    /// <param name="network">The trained Mamba model whose <c>Step</c> advances one token at a time.</param>
    /// <param name="vocabularySize">The model's vocabulary size (the width of the one-hot input and logits).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="network"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="vocabularySize"/> is not positive.</exception>
    public MambaCausalLanguageModel(MambaLanguageModel<T> network, int vocabularySize)
    {
        Guard.NotNull(network);
        Guard.Positive(vocabularySize);
        _network = network;
        VocabularySize = vocabularySize;
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

        var input = new Tensor<T>(new[] { 1, tokenIds.Count, VocabularySize });
        for (var position = 0; position < tokenIds.Count; position++)
        {
            input[new[] { 0, position, ValidateToken(tokenIds[position]) }] = One;
        }

        // The full forward resets the underlying network, so any incremental
        // cache primed earlier no longer matches the network's state —
        // invalidate it rather than letting a later AppendToken continue from
        // a stale sequence.
        _state = null;
        _network.ResetState();
        var output = _network.Predict(input);
        return ExtractLastPositionLogits(output);
    }

    /// <inheritdoc/>
    public void ResetCache() => _state = null;

    /// <inheritdoc/>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="promptTokenIds"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="promptTokenIds"/> is empty.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when a token id is outside the vocabulary.</exception>
    public Vector<T> StartSequence(IReadOnlyList<int> promptTokenIds)
    {
        Guard.NotNull(promptTokenIds);
        if (promptTokenIds.Count == 0)
        {
            throw new ArgumentException("The prompt must contain at least one token.", nameof(promptTokenIds));
        }

        // Drop any previous cache up front, and only publish the new state
        // once the whole prompt has been processed — if a later token throws
        // (e.g. out of vocabulary), AppendToken must not continue from a
        // half-primed sequence.
        _state = null;
        _network.ResetState();
        var state = _network.CreateStepState(batchSize: 1);

        Vector<T>? logits = null;
        for (var i = 0; i < promptTokenIds.Count; i++)
        {
            logits = StepOne(promptTokenIds[i], state);
        }

        _state = state;

        // Non-null: the loop runs at least once because the prompt is non-empty.
        return logits ?? throw new InvalidOperationException("Prompt produced no logits.");
    }

    /// <inheritdoc/>
    /// <exception cref="InvalidOperationException">Thrown when called before <see cref="StartSequence"/>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="tokenId"/> is outside the vocabulary.</exception>
    public Vector<T> AppendToken(int tokenId)
    {
        var state = _state ?? throw new InvalidOperationException(
            $"{nameof(StartSequence)} must be called before {nameof(AppendToken)} to prime the cache.");

        return StepOne(tokenId, state);
    }

    private Vector<T> StepOne(int tokenId, MambaModelState<T> state)
    {
        var token = new Tensor<T>(new[] { 1, 1, VocabularySize });
        token[new[] { 0, 0, ValidateToken(tokenId) }] = One;
        var output = _network.Step(token, state);
        return ExtractLastPositionLogits(output);
    }

    private int ValidateToken(int id)
    {
        if (id < 0 || id >= VocabularySize)
        {
            throw new ArgumentOutOfRangeException(
                nameof(id), id, $"Token id is outside the vocabulary [0, {VocabularySize}).");
        }

        return id;
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
