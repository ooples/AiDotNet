using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for clone detection over a corpus.
/// </summary>
public sealed class CodeCloneDetectionRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.CloneDetection;

    public CodeCorpusReference Corpus { get; set; } = new();

    /// <summary>
    /// Optional minimum similarity (0-1). Higher means fewer, more similar clones.
    /// </summary>
    public double MinSimilarity { get; set; } = 0.8;
}
