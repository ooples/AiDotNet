namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Controls how the next token is chosen from the model's logits: temperature, top-k, top-p (nucleus), and
/// an optional seed for reproducibility.
/// </summary>
/// <remarks>
/// <para>
/// All values are nullable with sensible behavior when unset. A temperature of <c>0</c> (or less) selects
/// greedy decoding (always the highest-scoring token); higher temperatures increase randomness. Top-k and
/// top-p restrict sampling to the most likely tokens. These mirror the knobs exposed on
/// <see cref="ChatOptions"/>, which override these per request.
/// </para>
/// <para><b>For Beginners:</b> After the model says how likely each next word-piece is, these settings decide
/// how to pick one. Low temperature = safe and repetitive; higher = more creative. Top-k ("only consider the
/// best k options") and top-p ("only consider the most likely options that together cover p% of the
/// probability") keep the choice from wandering into unlikely tokens.
/// </para>
/// </remarks>
public sealed class LocalSamplingOptions
{
    /// <summary>
    /// Gets or sets the sampling temperature. <c>null</c> uses <c>1.0</c>; <c>0</c> or less selects greedy
    /// (deterministic argmax) decoding.
    /// </summary>
    public double? Temperature { get; set; }

    /// <summary>
    /// Gets or sets the number of highest-probability tokens to consider. <c>null</c> or a non-positive
    /// value disables the top-k restriction.
    /// </summary>
    public int? TopK { get; set; }

    /// <summary>
    /// Gets or sets the nucleus-sampling probability mass (0–1): only the most likely tokens whose
    /// cumulative probability reaches this value are considered. <c>null</c> or a value outside (0,1)
    /// disables the top-p restriction.
    /// </summary>
    public double? TopP { get; set; }

    /// <summary>
    /// Gets or sets the random seed for reproducible sampling. <c>null</c> uses a non-deterministic seed.
    /// Ignored under greedy decoding (which is already deterministic).
    /// </summary>
    public int? Seed { get; set; }
}
