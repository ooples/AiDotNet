namespace AiDotNet.Agentic.Models;

/// <summary>
/// Token accounting for a chat request: how many tokens went in and came out.
/// </summary>
/// <remarks>
/// <para>
/// Usage drives cost tracking and budgeting. Providers report input (prompt) and output (completion)
/// token counts; <see cref="TotalTokens"/> is their sum.
/// </para>
/// <para><b>For Beginners:</b> Models bill by "tokens" (roughly word-pieces). This records how many
/// tokens your prompt used and how many the reply used, so you can measure and control spend.
/// </para>
/// </remarks>
public sealed class ChatUsage
{
    /// <summary>
    /// Initializes a new <see cref="ChatUsage"/>.
    /// </summary>
    /// <param name="inputTokens">Number of input/prompt tokens. Must be non-negative.</param>
    /// <param name="outputTokens">Number of output/completion tokens. Must be non-negative.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when a count is negative.</exception>
    public ChatUsage(int inputTokens, int outputTokens)
    {
        Guard.NonNegative(inputTokens);
        Guard.NonNegative(outputTokens);
        InputTokens = inputTokens;
        OutputTokens = outputTokens;
    }

    /// <summary>
    /// Gets the number of input (prompt) tokens.
    /// </summary>
    public int InputTokens { get; }

    /// <summary>
    /// Gets the number of output (completion) tokens.
    /// </summary>
    public int OutputTokens { get; }

    /// <summary>
    /// Gets the total number of tokens (<see cref="InputTokens"/> + <see cref="OutputTokens"/>).
    /// </summary>
    public int TotalTokens => InputTokens + OutputTokens;

    /// <inheritdoc/>
    public override string ToString() => $"in={InputTokens}, out={OutputTokens}, total={TotalTokens}";
}
