using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeTestGenerationResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.TestGeneration;

    public List<string> Tests { get; set; } = new();
}
