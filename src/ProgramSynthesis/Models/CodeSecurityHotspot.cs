namespace AiDotNet.ProgramSynthesis.Models;

public sealed class CodeSecurityHotspot
{
    public string Category { get; set; } = string.Empty;

    public string Summary { get; set; } = string.Empty;

    public CodeLocation Location { get; set; } = new();
}
