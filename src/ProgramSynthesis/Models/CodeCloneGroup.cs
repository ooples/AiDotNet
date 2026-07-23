namespace AiDotNet.ProgramSynthesis.Models;

public sealed class CodeCloneGroup
{
    public double Similarity { get; set; }

    public CodeCloneType CloneType { get; set; } = CodeCloneType.Type3;

    public CodeMatchType MatchType { get; set; } = CodeMatchType.Structural;

    public CodeProvenance? Provenance { get; set; }

    public string? NormalizationSummary { get; set; }

    public List<CodeCloneInstance> Instances { get; set; } = new();

    public List<string> RefactorSuggestions { get; set; } = new();
}
