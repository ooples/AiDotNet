namespace AiDotNet.Serving.Models;

/// <summary>
/// Request for text generation with speculative decoding.
/// </summary>
/// <remarks>
/// <para>
/// Speculative decoding accelerates text generation by using a smaller draft model
/// to generate candidate tokens that are then verified by the target model.
/// </para>
/// <para><b>For Beginners:</b> Think of speculative decoding like having a fast assistant
/// who suggests multiple words at once, which you then verify. Instead of generating
/// one token at a time (slow), we generate several candidates quickly and verify them
/// in parallel (fast).
/// </para>
/// </remarks>
public class SpeculativeDecodingRequest
{
    /// <summary>
    /// Gets or sets the input token IDs to continue from.
    /// </summary>
    public int[] InputTokens { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the maximum number of new tokens to generate.
    /// </summary>
    public int MaxNewTokens { get; set; } = 100;

    /// <summary>
    /// Gets or sets the sampling temperature. Higher values make output more random.
    /// Default is 1.0.
    /// </summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the nucleus (top-p) sampling threshold. 1.0 disables nucleus filtering.
    /// Only applied by the streaming generation path.
    /// </summary>
    public double TopP { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the top-k sampling cutoff (keep the k highest-probability tokens). 0 disables top-k.
    /// Only applied by the streaming generation path.
    /// </summary>
    public int TopK { get; set; }

    /// <summary>
    /// Gets or sets the min-p sampling threshold: drop tokens whose probability is below this fraction of the
    /// top token's probability. 0 disables. Only applied by the streaming generation path.
    /// </summary>
    public double MinP { get; set; }

    /// <summary>
    /// Gets or sets the end-of-sequence token ID. Generation stops when this token is produced.
    /// </summary>
    public int? EosTokenId { get; set; }

    /// <summary>
    /// Gets or sets the number of draft tokens to generate per verification step.
    /// Default is 5.
    /// </summary>
    public int NumDraftTokens { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to use tree-based speculation for higher acceptance rates.
    /// Default is false.
    /// </summary>
    public bool UseTreeSpeculation { get; set; } = false;

    /// <summary>
    /// Gets or sets the branching factor for tree speculation.
    /// Only used when UseTreeSpeculation is true. Default is 2.
    /// </summary>
    public int TreeBranchFactor { get; set; } = 2;

    /// <summary>
    /// Gets or sets the maximum tree depth for tree speculation.
    /// Only used when UseTreeSpeculation is true. Default is 4.
    /// </summary>
    public int MaxTreeDepth { get; set; } = 4;

    /// <summary>
    /// Gets or sets an optional request ID for tracking purposes.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Gets or sets an optional RNG seed (OpenAI <c>seed</c>). When set, sampling is reproducible for the
    /// same seed + parameters; null falls back to a per-request seed derived from <see cref="RequestId"/>.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets an optional structured-output constraint (JSON / regex / grammar / choice) that forces
    /// the generated text to conform to a required format. Null = unconstrained. Built from an OpenAI
    /// <c>response_format</c> by the controller. When set, speculative decoding is disabled for the request.
    /// </summary>
    public AiDotNet.Serving.StructuredOutput.ITokenConstraint? Constraint { get; set; }

    /// <summary>
    /// Gets or sets an optional per-token additive logit bias (OpenAI <c>logit_bias</c>): token id -&gt;
    /// bias added before sampling. Null = none.
    /// </summary>
    public IReadOnlyDictionary<int, float>? LogitBias { get; set; }

    /// <summary>Gets or sets the OpenAI <c>frequency_penalty</c> (default 0).</summary>
    public double FrequencyPenalty { get; set; }

    /// <summary>Gets or sets the OpenAI <c>presence_penalty</c> (default 0).</summary>
    public double PresencePenalty { get; set; }

    /// <summary>
    /// Optional multi-LoRA adapter name to serve this request with (S-LoRA-style shared-base serving). Null
    /// uses the base model. Typically parsed from the OpenAI <c>model</c> field as <c>base@adapter</c>.
    /// </summary>
    public string? AdapterId { get; set; }

    /// <summary>Gets or sets whether to return per-token log-probabilities (OpenAI <c>logprobs</c>).</summary>
    public bool Logprobs { get; set; }

    /// <summary>Gets or sets how many top alternatives to return per token (OpenAI <c>top_logprobs</c>, 0-20).</summary>
    public int TopLogprobs { get; set; }

    internal string? Validate()
    {
        if (InputTokens == null || InputTokens.Length == 0)
        {
            return "InputTokens array is required and cannot be empty";
        }

        if (MaxNewTokens <= 0)
        {
            return "MaxNewTokens must be greater than 0";
        }

        if (Temperature <= 0.0)
        {
            return "Temperature must be greater than 0";
        }

        if (NumDraftTokens <= 0)
        {
            return "NumDraftTokens must be greater than 0";
        }

        if (UseTreeSpeculation)
        {
            if (TreeBranchFactor <= 0)
            {
                return "TreeBranchFactor must be greater than 0 when UseTreeSpeculation is true";
            }

            if (MaxTreeDepth <= 0)
            {
                return "MaxTreeDepth must be greater than 0 when UseTreeSpeculation is true";
            }
        }

        return null;
    }
}

