using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Represents a node in an abstract syntax tree (AST) for a piece of source code.
/// </summary>
/// <remarks>
/// <para>
/// This type is intentionally lightweight and is designed for structural inspection and downstream
/// tasks (understanding, review, search) without requiring consumers to depend on a specific parser.
/// </para>
/// <para><b>For Beginners:</b> An AST is a "tree view" of code.
///
/// Code is not just text â€” it has structure (functions contain statements, statements contain expressions).
/// This node represents one item in that tree with a type (Kind) and a location (Span).
/// </para>
/// </remarks>
public sealed class CodeAstNode
{
    public int NodeId { get; set; }

    public int? ParentNodeId { get; set; }

    public ProgramLanguage Language { get; set; } = ProgramLanguage.Generic;

    public string Kind { get; set; } = string.Empty;

    public CodeSpan Span { get; set; } = new();
}

