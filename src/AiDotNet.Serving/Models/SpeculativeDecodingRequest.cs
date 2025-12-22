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

