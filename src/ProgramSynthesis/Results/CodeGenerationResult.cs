using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeGenerationResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.Generation;

    public string GeneratedCode { get; set; } = string.Empty;
}
