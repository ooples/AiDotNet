using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeUnderstandingResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.Understanding;

    public List<CodeSymbol> Symbols { get; set; } = new();

    public List<CodeDependency> Dependencies { get; set; } = new();

    public CodeComplexityMetrics Complexity { get; set; } = new();

    public List<CodeCallGraphEdge> CallGraph { get; set; } = new();

    public List<CodeHotspot> Hotspots { get; set; } = new();

    public List<string> ControlFlowSummaries { get; set; } = new();

    public List<string> DataFlowSummaries { get; set; } = new();

    public List<CodeSecurityHotspot> SecurityHotspots { get; set; } = new();
}
