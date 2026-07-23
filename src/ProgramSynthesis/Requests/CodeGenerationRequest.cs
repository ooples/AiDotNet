using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for code generation from a description and/or examples.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> You describe what you want, and the system writes code.</para>
/// </remarks>
public sealed class CodeGenerationRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.Generation;

    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Optional examples (input -> expected output) to guide generation.
    /// </summary>
    public List<ProgramInputOutputExample> Examples { get; set; } = new();
}
