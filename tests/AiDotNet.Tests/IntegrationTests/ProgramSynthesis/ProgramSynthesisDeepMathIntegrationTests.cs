using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ProgramSynthesis;

/// <summary>
/// Deep integration tests for ProgramSynthesis:
/// Code data models (CodePosition, CodeSpan, CodeAstNode, CodeAstEdge, CodeIssue, CodeCloneGroup,
/// CodeCompletionCandidate, CodeComplexityMetrics, CodeEditOperation, CodeHotspot, CodeFixSuggestion),
/// Enums (SynthesisType, ProgramLanguage, CodeTask, SqlDialect, CodeCloneType, CodeEditOperationType,
/// CodeIssueCategory, CodeIssueSeverity).
/// </summary>
public class ProgramSynthesisDeepMathIntegrationTests
{
    // ============================
    // CodePosition: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodePosition_Defaults()
    {
        var pos = new CodePosition();
        Assert.Equal(1, pos.Line);
        Assert.Equal(1, pos.Column);
        Assert.Equal(0, pos.Offset);
    }

    [Fact(Timeout = 120000)]
    public async Task CodePosition_SetProperties()
    {
        var pos = new CodePosition { Line = 42, Column = 15, Offset = 1024 };
        Assert.Equal(42, pos.Line);
        Assert.Equal(15, pos.Column);
        Assert.Equal(1024, pos.Offset);
    }

    // ============================
    // CodeSpan: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeSpan_Defaults_StartAndEndNotNull()
    {
        var span = new CodeSpan();
        Assert.NotNull(span.Start);
        Assert.NotNull(span.End);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeSpan_SetStartEnd()
    {
        var span = new CodeSpan
        {
            Start = new CodePosition { Line = 10, Column = 1 },
            End = new CodePosition { Line = 20, Column = 80 }
        };
        Assert.Equal(10, span.Start.Line);
        Assert.Equal(1, span.Start.Column);
        Assert.Equal(20, span.End.Line);
        Assert.Equal(80, span.End.Column);
    }

    // ============================
    // CodeAstNode: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeAstNode_Defaults()
    {
        var node = new CodeAstNode();
        Assert.Equal(0, node.NodeId);
        Assert.Null(node.ParentNodeId);
        Assert.Equal(ProgramLanguage.Generic, node.Language);
        Assert.Equal(string.Empty, node.Kind);
        Assert.NotNull(node.Span);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeAstNode_SetProperties()
    {
        var node = new CodeAstNode
        {
            NodeId = 42,
            ParentNodeId = 10,
            Language = ProgramLanguage.CSharp,
            Kind = "MethodDeclaration",
            Span = new CodeSpan
            {
                Start = new CodePosition { Line = 5, Column = 1 },
                End = new CodePosition { Line = 15, Column = 1 }
            }
        };

        Assert.Equal(42, node.NodeId);
        Assert.Equal(10, node.ParentNodeId);
        Assert.Equal(ProgramLanguage.CSharp, node.Language);
        Assert.Equal("MethodDeclaration", node.Kind);
        Assert.Equal(5, node.Span.Start.Line);
    }

    // ============================
    // CodeAstEdge: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeAstEdge_Defaults()
    {
        var edge = new CodeAstEdge();
        Assert.Equal(0, edge.ParentNodeId);
        Assert.Equal(0, edge.ChildNodeId);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeAstEdge_SetProperties()
    {
        var edge = new CodeAstEdge { ParentNodeId = 1, ChildNodeId = 5 };
        Assert.Equal(1, edge.ParentNodeId);
        Assert.Equal(5, edge.ChildNodeId);
    }

    // ============================
    // CodeLocation: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeLocation_Defaults()
    {
        var loc = new CodeLocation();
        Assert.Null(loc.FilePath);
        Assert.NotNull(loc.Span);
        Assert.Null(loc.NodePath);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeLocation_SetProperties()
    {
        var loc = new CodeLocation
        {
            FilePath = "/src/Program.cs",
            Span = new CodeSpan
            {
                Start = new CodePosition { Line = 10 },
                End = new CodePosition { Line = 20 }
            }
        };
        Assert.Equal("/src/Program.cs", loc.FilePath);
        Assert.Equal(10, loc.Span.Start.Line);
    }

    // ============================
    // CodeIssue: Defaults and Properties
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeIssue_Defaults()
    {
        var issue = new CodeIssue();
        Assert.Equal(CodeIssueSeverity.Warning, issue.Severity);
        Assert.Equal(CodeIssueCategory.Other, issue.Category);
        Assert.Equal(string.Empty, issue.Summary);
        Assert.Null(issue.Details);
        Assert.Null(issue.Rationale);
        Assert.Null(issue.FixGuidance);
        Assert.Null(issue.TestGuidance);
        Assert.NotNull(issue.Location);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeIssue_SetAllProperties()
    {
        var issue = new CodeIssue
        {
            Severity = CodeIssueSeverity.Error,
            Category = CodeIssueCategory.Security,
            Summary = "SQL injection vulnerability",
            Details = "User input not sanitized",
            Rationale = "Parameterized queries prevent injection",
            FixGuidance = "Use parameterized queries",
            TestGuidance = "Test with special characters"
        };

        Assert.Equal(CodeIssueSeverity.Error, issue.Severity);
        Assert.Equal(CodeIssueCategory.Security, issue.Category);
        Assert.Equal("SQL injection vulnerability", issue.Summary);
        Assert.Equal("User input not sanitized", issue.Details);
    }

    // ============================
    // CodeCompletionCandidate: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeCompletionCandidate_Defaults()
    {
        var candidate = new CodeCompletionCandidate();
        Assert.Equal(string.Empty, candidate.CompletionText);
        Assert.Equal(0.0, candidate.Score);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCompletionCandidate_SetProperties()
    {
        var candidate = new CodeCompletionCandidate
        {
            CompletionText = "Console.WriteLine(\"Hello\");",
            Score = 0.95
        };
        Assert.Equal("Console.WriteLine(\"Hello\");", candidate.CompletionText);
        Assert.Equal(0.95, candidate.Score, 1e-10);
    }

    // ============================
    // CodeComplexityMetrics: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeComplexityMetrics_Defaults()
    {
        var metrics = new CodeComplexityMetrics();
        Assert.Equal(0, metrics.LineCount);
        Assert.Equal(0, metrics.CharacterCount);
        Assert.Equal(0, metrics.EstimatedCyclomaticComplexity);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeComplexityMetrics_SetProperties()
    {
        var metrics = new CodeComplexityMetrics
        {
            LineCount = 50,
            CharacterCount = 1200,
            EstimatedCyclomaticComplexity = 8
        };
        Assert.Equal(50, metrics.LineCount);
        Assert.Equal(1200, metrics.CharacterCount);
        Assert.Equal(8, metrics.EstimatedCyclomaticComplexity);
    }

    // ============================
    // CodeEditOperation: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeEditOperation_Defaults()
    {
        var op = new CodeEditOperation();
        Assert.Equal(CodeEditOperationType.Insert, op.OperationType);
        Assert.NotNull(op.Span);
        Assert.Null(op.Text);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeEditOperation_SetProperties()
    {
        var op = new CodeEditOperation
        {
            OperationType = CodeEditOperationType.Replace,
            Text = "newCode()",
            Span = new CodeSpan
            {
                Start = new CodePosition { Line = 5, Column = 10 },
                End = new CodePosition { Line = 5, Column = 20 }
            }
        };
        Assert.Equal(CodeEditOperationType.Replace, op.OperationType);
        Assert.Equal("newCode()", op.Text);
    }

    // ============================
    // CodeHotspot: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeHotspot_Defaults()
    {
        var hotspot = new CodeHotspot();
        Assert.Equal(string.Empty, hotspot.SymbolName);
        Assert.Equal(string.Empty, hotspot.Reason);
        Assert.Equal(0.0, hotspot.Score);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeHotspot_SetProperties()
    {
        var hotspot = new CodeHotspot
        {
            SymbolName = "ProcessData",
            Reason = "High cyclomatic complexity",
            Score = 0.85
        };
        Assert.Equal("ProcessData", hotspot.SymbolName);
        Assert.Equal("High cyclomatic complexity", hotspot.Reason);
        Assert.Equal(0.85, hotspot.Score, 1e-10);
    }

    // ============================
    // CodeFixSuggestion: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeFixSuggestion_Defaults()
    {
        var fix = new CodeFixSuggestion();
        Assert.Equal(string.Empty, fix.Summary);
        Assert.Null(fix.Rationale);
        Assert.Null(fix.FixGuidance);
        Assert.Null(fix.TestGuidance);
        Assert.Null(fix.Diff);
    }

    // ============================
    // CodeCloneGroup: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeCloneGroup_Defaults()
    {
        var group = new CodeCloneGroup();
        Assert.Equal(0.0, group.Similarity);
        Assert.Equal(CodeCloneType.Type3, group.CloneType);
        Assert.Null(group.Provenance);
        Assert.Null(group.NormalizationSummary);
        Assert.Empty(group.Instances);
        Assert.Empty(group.RefactorSuggestions);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCloneGroup_SetProperties()
    {
        var group = new CodeCloneGroup
        {
            Similarity = 0.92,
            CloneType = CodeCloneType.Type1,
            NormalizationSummary = "Whitespace normalized",
            Instances = new List<CodeCloneInstance>
            {
                new() { SnippetText = "int x = 1;" },
                new() { SnippetText = "int x = 1;" }
            },
            RefactorSuggestions = new List<string> { "Extract to shared method" }
        };

        Assert.Equal(0.92, group.Similarity, 1e-10);
        Assert.Equal(CodeCloneType.Type1, group.CloneType);
        Assert.Equal(2, group.Instances.Count);
        Assert.Single(group.RefactorSuggestions);
    }

    // ============================
    // CodeCloneInstance: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeCloneInstance_Defaults()
    {
        var instance = new CodeCloneInstance();
        Assert.NotNull(instance.Location);
        Assert.Equal(string.Empty, instance.SnippetText);
    }

    // ============================
    // SynthesisType Enum
    // ============================

    [Fact(Timeout = 120000)]
    public async Task SynthesisType_HasSixValues()
    {
        var values = (((SynthesisType[])Enum.GetValues(typeof(SynthesisType))));
        Assert.Equal(6, values.Length);
    }

    [Theory]
    [InlineData(SynthesisType.Neural)]
    [InlineData(SynthesisType.Symbolic)]
    [InlineData(SynthesisType.Hybrid)]
    [InlineData(SynthesisType.GeneticProgramming)]
    [InlineData(SynthesisType.Inductive)]
    [InlineData(SynthesisType.Deductive)]
    public void SynthesisType_AllValuesValid(SynthesisType type)
    {
        Assert.True(Enum.IsDefined(typeof(SynthesisType), type));
    }

    // ============================
    // ProgramLanguage Enum
    // ============================

    [Fact(Timeout = 120000)]
    public async Task ProgramLanguage_HasElevenValues()
    {
        var values = (((ProgramLanguage[])Enum.GetValues(typeof(ProgramLanguage))));
        Assert.Equal(11, values.Length);
    }

    [Theory]
    [InlineData(ProgramLanguage.Python)]
    [InlineData(ProgramLanguage.CSharp)]
    [InlineData(ProgramLanguage.Java)]
    [InlineData(ProgramLanguage.JavaScript)]
    [InlineData(ProgramLanguage.TypeScript)]
    [InlineData(ProgramLanguage.CPlusPlus)]
    [InlineData(ProgramLanguage.C)]
    [InlineData(ProgramLanguage.Go)]
    [InlineData(ProgramLanguage.Rust)]
    [InlineData(ProgramLanguage.SQL)]
    [InlineData(ProgramLanguage.Generic)]
    public void ProgramLanguage_AllValuesValid(ProgramLanguage lang)
    {
        Assert.True(Enum.IsDefined(typeof(ProgramLanguage), lang));
    }

    // ============================
    // CodeTask Enum
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeTask_HasThirteenValues()
    {
        var values = (((CodeTask[])Enum.GetValues(typeof(CodeTask))));
        Assert.Equal(13, values.Length);
    }

    [Theory]
    [InlineData(CodeTask.Completion)]
    [InlineData(CodeTask.Generation)]
    [InlineData(CodeTask.Translation)]
    [InlineData(CodeTask.Summarization)]
    [InlineData(CodeTask.BugDetection)]
    [InlineData(CodeTask.BugFixing)]
    [InlineData(CodeTask.Refactoring)]
    [InlineData(CodeTask.Understanding)]
    [InlineData(CodeTask.TestGeneration)]
    [InlineData(CodeTask.Documentation)]
    [InlineData(CodeTask.Search)]
    [InlineData(CodeTask.CloneDetection)]
    [InlineData(CodeTask.CodeReview)]
    public void CodeTask_AllValuesValid(CodeTask task)
    {
        Assert.True(Enum.IsDefined(typeof(CodeTask), task));
    }

    // ============================
    // SqlDialect Enum
    // ============================

    [Fact(Timeout = 120000)]
    public async Task SqlDialect_HasThreeValues()
    {
        var values = (((SqlDialect[])Enum.GetValues(typeof(SqlDialect))));
        Assert.Equal(3, values.Length);
    }

    [Theory]
    [InlineData(SqlDialect.SQLite, 0)]
    [InlineData(SqlDialect.Postgres, 1)]
    [InlineData(SqlDialect.MySql, 2)]
    public void SqlDialect_Values(SqlDialect dialect, int expectedValue)
    {
        Assert.Equal(expectedValue, (int)dialect);
    }

    // ============================
    // CodeCloneType Enum
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeCloneType_HasFourValues()
    {
        var values = (((CodeCloneType[])Enum.GetValues(typeof(CodeCloneType))));
        Assert.Equal(4, values.Length);
    }

    // ============================
    // CodeEditOperationType Enum
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeEditOperationType_HasThreeValues()
    {
        var values = (((CodeEditOperationType[])Enum.GetValues(typeof(CodeEditOperationType))));
        Assert.Equal(3, values.Length);
    }

    // ============================
    // CodeIssueCategory Enum
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeIssueCategory_HasElevenValues()
    {
        var values = (((CodeIssueCategory[])Enum.GetValues(typeof(CodeIssueCategory))));
        Assert.Equal(11, values.Length);
    }

    // ============================
    // CodeIssueSeverity Enum
    // ============================

    [Fact(Timeout = 120000)]
    public async Task CodeIssueSeverity_HasFourValues()
    {
        var values = (((CodeIssueSeverity[])Enum.GetValues(typeof(CodeIssueSeverity))));
        Assert.Equal(4, values.Length);
    }
}
