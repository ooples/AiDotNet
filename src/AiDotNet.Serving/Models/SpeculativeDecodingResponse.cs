namespace AiDotNet.Serving.Models;

/// <summary>
/// Response from text generation with speculative decoding.
/// </summary>
public class SpeculativeDecodingResponse
{
    /// <summary>
    /// Gets or sets all tokens including input and generated tokens.
    /// </summary>
    public int[] AllTokens { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets only the newly generated tokens.
    /// </summary>
    public int[] GeneratedTokens { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the number of tokens generated.
    /// </summary>
    public int NumGenerated { get; set; }

    /// <summary>
    /// Gets or sets the acceptance rate (ratio of draft tokens accepted by target model).
    /// </summary>
    public double AcceptanceRate { get; set; }

    /// <summary>
    /// Gets or sets the time taken to process the request in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the request ID that was provided in the request.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Gets or sets any error message if generation failed.
    /// </summary>
    public string? Error { get; set; }
}

