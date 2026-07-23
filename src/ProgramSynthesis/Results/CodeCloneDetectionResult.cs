using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeCloneDetectionResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.CloneDetection;

    public List<CodeCloneGroup> CloneGroups { get; set; } = new();
}
