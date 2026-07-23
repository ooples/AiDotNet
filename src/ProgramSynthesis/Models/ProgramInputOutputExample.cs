namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Represents a single input-output example for program synthesis.
/// </summary>
/// <remarks>
/// <para>
/// Examples can be used to guide generation (inductive synthesis) and to validate candidate programs
/// (execution-based evaluation).
/// </para>
/// <para><b>For Beginners:</b> This is one example of what the program should do.
///
/// It says: "When the program gets this input, it should produce this output."
/// </para>
/// </remarks>
public sealed class ProgramInputOutputExample
{
    public string Input { get; set; } = string.Empty;

    public string ExpectedOutput { get; set; } = string.Empty;
}
