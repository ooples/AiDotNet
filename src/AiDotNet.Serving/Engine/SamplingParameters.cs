namespace AiDotNet.Serving.Engine;

/// <summary>
/// Per-request sampling / decoding parameters for the inference engine. This is the engine-level, numeric-
/// type-agnostic view of "how to turn the model's next-token logits into a token" — it mirrors the knobs
/// exposed by vLLM's <c>SamplingParams</c>, TGI, and the OpenAI API so an OpenAI-compatible front end can map
/// onto it directly.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> a language model outputs, for each step, a score ("logit") for every possible
/// next token. These parameters control how a concrete token is picked from those scores: how random the
/// choice is (<see cref="Temperature"/>), how much of the probability mass to consider
/// (<see cref="TopP"/>/<see cref="TopK"/>), how to discourage repetition
/// (<see cref="RepetitionPenalty"/>/<see cref="PresencePenalty"/>/<see cref="FrequencyPenalty"/>), and when
/// to stop (<see cref="MaxTokens"/>/<see cref="StopTokenIds"/>).</para>
/// <para>This type is deliberately immutable and independent of the model's numeric type &lt;T&gt;: the
/// scheduler and KV-cache manager reason about token ids (ints) and these scalar knobs, while the typed model
/// runner applies them to the actual logits.</para>
/// </remarks>
public sealed class SamplingParameters
{
    /// <summary>Softmax temperature. 0 means greedy (argmax) decoding; higher values are more random. Default 1.0.</summary>
    public double Temperature { get; init; } = 1.0;

    /// <summary>Nucleus (top-p) sampling: keep the smallest set of tokens whose cumulative probability ≥ this. 1.0 disables. Default 1.0.</summary>
    public double TopP { get; init; } = 1.0;

    /// <summary>Top-k sampling: keep only the k highest-probability tokens. 0 (or negative) disables. Default 0.</summary>
    public int TopK { get; init; }

    /// <summary>Min-p sampling: drop tokens whose probability is below this fraction of the max probability. 0 disables. Default 0.</summary>
    public double MinP { get; init; }

    /// <summary>Multiplicative penalty applied to logits of previously seen tokens (&gt;1 discourages repetition). Default 1.0.</summary>
    public double RepetitionPenalty { get; init; } = 1.0;

    /// <summary>Additive penalty subtracted from the logit of any token that has appeared at least once. Default 0.</summary>
    public double PresencePenalty { get; init; }

    /// <summary>Additive penalty scaled by how many times a token has appeared. Default 0.</summary>
    public double FrequencyPenalty { get; init; }

    /// <summary>Maximum number of tokens to GENERATE (excludes the prompt). Required to bound work; default 16.</summary>
    public int MaxTokens { get; init; } = 16;

    /// <summary>Minimum number of tokens to generate before a stop token / EOS can end the sequence. Default 0.</summary>
    public int MinTokens { get; init; }

    /// <summary>Token ids that, when generated, end the sequence (in addition to the model's EOS). Optional.</summary>
    public IReadOnlyList<int>? StopTokenIds { get; init; }

    /// <summary>If false, a generated EOS token does not stop the sequence (it is emitted like any other). Default true.</summary>
    public bool IgnoreEos { get; init; }

    /// <summary>Number of independent output sequences to sample for this request (parallel sampling). Default 1.</summary>
    public int N { get; init; } = 1;

    /// <summary>Optional RNG seed for reproducible sampling. Null = nondeterministic.</summary>
    public int? Seed { get; init; }

    /// <summary>Greedy decoding is used when temperature is ~0 (argmax); otherwise stochastic sampling.</summary>
    public bool IsGreedy => Temperature <= 1e-6;

    /// <summary>Validates the parameters, throwing <see cref="ArgumentException"/> on an out-of-range value.</summary>
    public void Validate()
    {
        if (Temperature < 0) throw new ArgumentException("Temperature must be >= 0.", nameof(Temperature));
        if (TopP is <= 0 or > 1) throw new ArgumentException("TopP must be in (0, 1].", nameof(TopP));
        if (MinP is < 0 or > 1) throw new ArgumentException("MinP must be in [0, 1].", nameof(MinP));
        if (RepetitionPenalty <= 0) throw new ArgumentException("RepetitionPenalty must be > 0.", nameof(RepetitionPenalty));
        if (MaxTokens < 1) throw new ArgumentException("MaxTokens must be >= 1.", nameof(MaxTokens));
        if (MinTokens < 0 || MinTokens > MaxTokens) throw new ArgumentException("MinTokens must be in [0, MaxTokens].", nameof(MinTokens));
        if (N < 1) throw new ArgumentException("N must be >= 1.", nameof(N));
    }
}
