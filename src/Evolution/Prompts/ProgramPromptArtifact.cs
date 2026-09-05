using AiDotNet.Validation;

namespace AiDotNet.Evolution.Prompts;

/// <summary>One named piece of output an evaluation produced, offered back to the model as evidence.</summary>
/// <remarks>
/// <para>
/// Artifacts are what a candidate program actually printed when it ran: standard output, an error stream, a
/// profiler summary, a failing assertion. Showing them to the model is the difference between "your program
/// scored 0.2" and "your program scored 0.2 and here is the exception it threw", and it is usually the single
/// most useful thing a prompt can carry.
/// </para>
/// <para>
/// The content is untrusted. It was produced by code a model wrote, running against data the library did not
/// choose, and it may contain terminal escape sequences, credentials scraped from the environment, or megabytes
/// of noise. The prompt builder therefore scrubs and truncates every artifact before it reaches a request; this
/// type stores what was captured, unmodified, and leaves that decision to the point of use.
/// </para>
/// <para><b>For Beginners:</b> When your program runs, it usually prints something — results, warnings, or an
/// error. This holds one of those outputs together with a name such as <c>stderr</c>. Passing it into the next
/// prompt lets the AI see what went wrong instead of guessing, which is by far the fastest way for it to fix a
/// broken program. You do not have to worry about the output being huge or containing secrets: it is trimmed and
/// cleaned before it is sent.</para>
/// </remarks>
public sealed class ProgramPromptArtifact
{
    /// <summary>The longest artifact name accepted, in characters.</summary>
    public const int MaxNameLength = 128;

    /// <summary>Initializes an artifact.</summary>
    /// <param name="name">A short label such as <c>stdout</c> or <c>stderr</c>.</param>
    /// <param name="content">The captured text, exactly as produced.</param>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> or <paramref name="content"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="name"/> is empty, white space, or longer than <see cref="MaxNameLength"/>.</exception>
    public ProgramPromptArtifact(string name, string content)
    {
        Guard.NotNullOrWhiteSpace(name);
        Guard.NotNull(content);
        string trimmed = name.Trim();
        if (trimmed.Length > MaxNameLength)
        {
            throw new ArgumentException($"An artifact name cannot exceed {MaxNameLength} characters.", nameof(name));
        }

        Name = trimmed;
        Content = content;
    }

    /// <summary>Gets the artifact's label.</summary>
    public string Name { get; }

    /// <summary>Gets the captured text exactly as produced, before scrubbing or truncation.</summary>
    public string Content { get; }

    /// <summary>Returns a description that names the artifact and its size but never echoes its content.</summary>
    /// <returns>The name and character length.</returns>
    public override string ToString() => $"ProgramPromptArtifact({Name}, {Content.Length} chars)";
}
