namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// One labeled example in a prompt-optimization eval set: an input to send the agent and the expected answer
/// to score its response against.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A practice question with its answer key. The optimizer runs each candidate
/// prompt against a batch of these to see which prompt gets the most answers right.
/// </para>
/// </remarks>
public sealed class PromptEvalCase
{
    /// <summary>
    /// Initializes a new eval case.
    /// </summary>
    /// <param name="input">The user input to send the agent.</param>
    /// <param name="expected">The expected answer (used by the default substring scorer).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> or <paramref name="expected"/> is <c>null</c>.</exception>
    public PromptEvalCase(string input, string expected)
    {
        Guard.NotNull(input);
        Guard.NotNull(expected);
        Input = input;
        Expected = expected;
    }

    /// <summary>Gets the user input to send the agent.</summary>
    public string Input { get; }

    /// <summary>Gets the expected answer.</summary>
    public string Expected { get; }
}
