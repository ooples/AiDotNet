namespace AiDotNet.Configuration;

/// <summary>
/// Tuning knobs for speculative decoding — the technique that speeds up autoregressive (GPT-style) text
/// generation by letting a small, fast "draft" model guess the next few tokens and having the main model
/// verify all of them in a single pass. These options control <i>how</i> speculation runs; <i>which</i> draft
/// model does the guessing is chosen separately (see the builder's <c>ConfigureSpeculativeDecoding</c> overloads,
/// which accept an <see cref="AiDotNet.Inference.SpeculativeDecoding.IDraftModel{T}"/> or wrap a whole model).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Generating text one token at a time is slow because the big model runs once per
/// token. Speculative decoding uses a cheap "guesser" to propose several tokens at once, then the big model
/// checks them all in one shot. Guesses that match are kept for free; wrong guesses are simply discarded, so the
/// output is <b>identical</b> to normal generation — just faster (often 1.5–3×). These settings let you tune the
/// trade-off between how far ahead to guess and how aggressive to be. The defaults are safe; leave them alone if
/// you're unsure.</para>
/// <para>
/// When left at its defaults, speculation uses a zero-cost N-gram / prompt-lookup draft (no extra model needed),
/// so simply enabling it via <c>SpeculativeDecoding.Enabled = true</c> already helps repetitive text.
/// </para>
/// </remarks>
public sealed class SpeculativeDecodingOptions
{
    /// <summary>
    /// Gets or sets whether speculative decoding is enabled.
    /// </summary>
    /// <value>True to enable speculative decoding (default: false).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Turn this on to speed up text generation. It only helps autoregressive models
    /// (ones that emit tokens one after another, like language models); it has no effect on single-pass models.
    /// There is no quality loss — the big model verifies every guess — so the only downside is a little wasted
    /// work when the guesser is wrong.</para>
    /// </remarks>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the speculation depth (number of tokens the draft model guesses ahead each step).
    /// </summary>
    /// <value>Speculation depth (default: 4).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many tokens to guess at once.
    ///
    /// Guidelines:
    /// - 3–4: Conservative, most guesses accepted (default: 4)
    /// - 5–6: Balanced
    /// - 7+: Aggressive — bigger potential speedup but more wasted work when guesses miss
    ///
    /// Higher depth means more speedup <i>if</i> the guesser is good, but more discarded work when it isn't.</para>
    /// </remarks>
    public int SpeculationDepth { get; set; } = 4;

    /// <summary>
    /// Gets or sets the policy for when speculative decoding should actually run at request time.
    /// </summary>
    /// <value>Speculation policy (default: <see cref="SpeculationPolicy.Auto"/>).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Even when speculation is enabled, it isn't always worth it. Under heavy load
    /// (large batches) the extra guessing can hurt overall throughput. <see cref="SpeculationPolicy.Auto"/>
    /// (recommended) turns speculation off automatically when the system is busy and on when latency matters.</para>
    /// </remarks>
    public SpeculationPolicy SpeculationPolicy { get; set; } = SpeculationPolicy.Auto;

    /// <summary>
    /// Gets or sets the speculative decoding method (the "style" of speculation).
    /// </summary>
    /// <value>Speculative method (default: <see cref="SpeculativeMethod.Auto"/>, which selects the classic draft-model style today).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This picks the algorithm used to make and verify guesses.
    /// <see cref="SpeculativeMethod.Auto"/> chooses a sensible default for you (classic draft-model speculation).
    /// The other options are advanced tree-based variants.</para>
    /// </remarks>
    public SpeculativeMethod SpeculativeMethod { get; set; } = SpeculativeMethod.Auto;

    /// <summary>
    /// Gets or sets whether to use tree-structured speculation (multiple candidate branches per step).
    /// </summary>
    /// <value>True to enable tree speculation (default: false).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of guessing one straight-line sequence of tokens, tree speculation
    /// guesses several branching possibilities at once. This can get more tokens accepted per step but uses more
    /// memory and compute. Leave it off unless you're specifically tuning for it.</para>
    /// </remarks>
    public bool UseTreeSpeculation { get; set; } = false;

    /// <summary>
    /// Returns a copy of these options. Used when a configuration is cloned so the speculation settings are not
    /// shared by reference between the original and the copy.
    /// </summary>
    public SpeculativeDecodingOptions Clone() => (SpeculativeDecodingOptions)MemberwiseClone();
}
