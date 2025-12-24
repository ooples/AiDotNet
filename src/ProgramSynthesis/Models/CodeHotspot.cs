namespace AiDotNet.ProgramSynthesis.Models;

public sealed class CodeHotspot
{
    public string SymbolName { get; set; } = string.Empty;

    public string Reason { get; set; } = string.Empty;

    public double Score { get; set; }
}
