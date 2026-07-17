using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// A model that can act as a causal (autoregressive) language model for text generation: given a sequence of
/// token ids it returns next-token logits over its vocabulary. Implementing this lets the serving engine drive
/// the model with a correctness-guaranteed recompute path (no special KV plumbing required); models that also
/// want the fast paged path additionally implement the serving engine's incremental runner capability.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> a "causal language model" is one that predicts the next word from the words so
/// far — the family GPT-style text generators belong to. This small interface is the promise a model makes to
/// the serving engine: "give me the tokens read so far and I will score every possible next token." That is all
/// the engine needs to generate text with it. A model that already produces vocabulary-sized logits (like
/// <c>Transformer</c>) implements this almost for free.</para>
/// </remarks>
/// <typeparam name="T">The model's numeric type.</typeparam>
public interface ICausalLmModel<T>
{
    /// <summary>Size of the token vocabulary (the length of each next-token logits row).</summary>
    int VocabularySize { get; }

    /// <summary>
    /// The end-of-sequence token id, if the model defines one. Generating it ends a sequence unless the caller
    /// opts out. Null when the model has no distinguished EOS token.
    /// </summary>
    int? EosTokenId { get; }

    /// <summary>
    /// Computes next-token logits for a token sequence. Given token ids shaped <c>[sequenceLength]</c> or
    /// <c>[1, sequenceLength]</c>, returns logits shaped <c>[sequenceLength, vocabularySize]</c> (or with a
    /// leading batch dimension of 1); the serving engine reads the final position to score the next token.
    /// </summary>
    Tensor<T> ForwardLogits(Tensor<T> tokenIds);
}
