namespace AiDotNet.Enums;

/// <summary>
/// Strategy for collapsing a transformer encoder's <c>[batch, seq, dim]</c>
/// hidden states into a single <c>[batch, dim]</c> vector before the
/// classification head, when the task is single-label per sequence.
/// </summary>
/// <remarks>
/// <para>
/// Picking the wrong mode silently destroys the position-specific signal
/// the model needs to learn. The default <see cref="LastToken"/> matches
/// canonical autoregressive language modelling (GPT-style next-token
/// prediction): the last position has attended to every preceding
/// position via causal self-attention and is therefore the natural
/// summary of the prefix.
/// </para>
/// <para>
/// <see cref="MeanPool"/> averages over all positions and is appropriate
/// for non-causal sequence-classification (e.g. document sentiment
/// where the whole sequence is observed at once and word order matters
/// less than the overall content). Using MeanPool for next-token LM
/// produced the flat-softmax convergence failure tracked in
/// AiDotNet#1232: every context mapped to roughly the same averaged
/// hidden state, the model couldn't learn position-conditioned outputs,
/// and softmax converged to <c>~uniform / V</c>.
/// </para>
/// <para>
/// <see cref="ClsToken"/> matches BERT-style models that prepend a
/// dedicated <c>[CLS]</c> token and use its final-layer hidden state
/// as the sequence summary. <see cref="None"/> keeps the full
/// <c>[batch, seq, dim]</c> shape — the right choice when the loss is
/// applied per-token (token-classification, masked-LM with parallel
/// position prediction, sequence-to-sequence training).
/// </para>
/// </remarks>
public enum SequencePoolingMode
{
    /// <summary>
    /// Take the LAST position's hidden state. Canonical for
    /// autoregressive language modelling and next-token prediction —
    /// matches GPT / Llama / Mistral output-head conventions.
    /// </summary>
    LastToken,

    /// <summary>
    /// Average all positions. Appropriate for non-causal sequence
    /// classification where the whole sequence is observed and order
    /// matters less than overall content.
    /// </summary>
    MeanPool,

    /// <summary>
    /// Use the FIRST position's hidden state, treated as a prepended
    /// <c>[CLS]</c> summary token (BERT-style).
    /// </summary>
    ClsToken,

    /// <summary>
    /// Skip pooling entirely. Output stays
    /// <c>[batch, seq, dim]</c> → after the classification head,
    /// <c>[batch, seq, V]</c>. Use this when the loss is applied
    /// per-position (token classification, masked-LM, sequence-to-
    /// sequence training).
    /// </summary>
    None,
}
