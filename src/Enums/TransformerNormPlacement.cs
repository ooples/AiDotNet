namespace AiDotNet.Enums;

/// <summary>
/// Where a transformer block applies layer normalization relative to each residual connection.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Each transformer sublayer (attention, feed-forward) adds its output back to
/// its input. Pre-norm normalizes the sublayer's input and leaves the running sum un-normalized;
/// post-norm normalizes the sum after the addition, as the original Transformer did.
/// </para>
/// </remarks>
public enum TransformerNormPlacement
{
    /// <summary>
    /// <c>y = x + Sublayer(LayerNorm(x))</c> (Xiong et al. 2020). Trains stably without warm-up; used by
    /// GPT-2 onward and by wav2vec 2.0's "stable layer norm" large variants.
    /// </summary>
    PreNorm,

    /// <summary>
    /// <c>y = LayerNorm(x + Sublayer(x))</c> (Vaswani et al. 2017). Used by BERT and by wav2vec 2.0 BASE.
    /// </summary>
    PostNorm
}
