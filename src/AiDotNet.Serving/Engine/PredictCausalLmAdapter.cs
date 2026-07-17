using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// Presents any token-in / logits-out forward function as an <see cref="AiDotNet.Interfaces.ICausalLmModel{T}"/>
/// so the serving engine can drive a model that produces vocabulary-width predictions but has not implemented
/// the capability interface directly. This is the widest-coverage fallback: give it the model's forward pass
/// and its vocabulary size, and it serves.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> some models already output a score for every possible next token but don't
/// formally declare themselves a "language model" to the serving layer. This wrapper bridges that gap — you
/// tell it the model's forward function and how many tokens are in the vocabulary, and it can be served like
/// any other generator.</para>
/// </remarks>
/// <typeparam name="T">The model's numeric type.</typeparam>
public sealed class PredictCausalLmAdapter<T> : AiDotNet.Interfaces.ICausalLmModel<T>
{
    private readonly Func<Tensor<T>, Tensor<T>> _forward;

    /// <summary>Creates the adapter.</summary>
    /// <param name="forward">The model's forward pass: token ids <c>[1, seq]</c> → logits (<c>[seq, vocab]</c>
    /// or <c>[1, seq, vocab]</c>). Typically the model's <c>Predict</c>.</param>
    /// <param name="vocabularySize">The model's vocabulary size.</param>
    /// <param name="eosTokenId">The EOS token id, if any.</param>
    public PredictCausalLmAdapter(Func<Tensor<T>, Tensor<T>> forward, int vocabularySize, int? eosTokenId = null)
    {
        _forward = forward ?? throw new ArgumentNullException(nameof(forward));
        if (vocabularySize < 1) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        VocabularySize = vocabularySize;
        EosTokenId = eosTokenId;
    }

    /// <inheritdoc/>
    public int VocabularySize { get; }

    /// <inheritdoc/>
    public int? EosTokenId { get; }

    /// <inheritdoc/>
    public Tensor<T> ForwardLogits(Tensor<T> tokenIds) => _forward(tokenIds);
}
