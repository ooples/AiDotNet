namespace AiDotNet.ProgramSynthesis.Models;

public sealed class CodeDependency
{
    public string Name { get; set; } = string.Empty;

    public string? Kind { get; set; }
}
