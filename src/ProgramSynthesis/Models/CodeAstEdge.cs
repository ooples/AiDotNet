namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Represents a relationship between two AST nodes (typically parent/child).
/// </summary>
public sealed class CodeAstEdge
{
    public int ParentNodeId { get; set; }

    public int ChildNodeId { get; set; }
}

