using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Results;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ProgramSynthesis;

/// <summary>
/// Comprehensive integration tests for the ProgramSynthesis module.
/// Tests Enums, Models, Execution, and Results components.
/// </summary>
public class ProgramSynthesisIntegrationTests
{
    #region CodePosition Tests

    [Fact]
    public void CodePosition_DefaultValues()
    {
        // Arrange & Act
        var position = new CodePosition();

        // Assert
        Assert.Equal(1, position.Line);
        Assert.Equal(1, position.Column);
        Assert.Equal(0, position.Offset);
    }

    [Fact]
    public void CodePosition_SetProperties()
    {
        // Arrange & Act
        var position = new CodePosition
        {
            Line = 10,
            Column = 25,
            Offset = 500
        };

        // Assert
        Assert.Equal(10, position.Line);
        Assert.Equal(25, position.Column);
        Assert.Equal(500, position.Offset);
    }

    #endregion

    #region CodeSpan Tests

    [Fact]
    public void CodeSpan_DefaultValues()
    {
        // Arrange & Act
        var span = new CodeSpan();

        // Assert
        Assert.NotNull(span.Start);
        Assert.NotNull(span.End);
        Assert.Equal(1, span.Start.Line);
        Assert.Equal(1, span.End.Line);
    }

    [Fact]
    public void CodeSpan_SetStartAndEnd()
    {
        // Arrange & Act
        var span = new CodeSpan
        {
            Start = new CodePosition { Line = 5, Column = 1, Offset = 100 },
            End = new CodePosition { Line = 10, Column = 30, Offset = 300 }
        };

        // Assert
        Assert.Equal(5, span.Start.Line);
        Assert.Equal(1, span.Start.Column);
        Assert.Equal(100, span.Start.Offset);
        Assert.Equal(10, span.End.Line);
        Assert.Equal(30, span.End.Column);
        Assert.Equal(300, span.End.Offset);
    }

    #endregion

    #region CodeLocation Tests

    [Fact]
    public void CodeLocation_DefaultValues()
    {
        // Arrange & Act
        var location = new CodeLocation();

        // Assert
        Assert.Null(location.FilePath);
        Assert.NotNull(location.Span);
        Assert.Null(location.NodePath);
    }

    [Fact]
    public void CodeLocation_SetProperties()
    {
        // Arrange & Act
        var location = new CodeLocation
        {
            FilePath = "/src/main.cs",
            Span = new CodeSpan
            {
                Start = new CodePosition { Line = 10 },
                End = new CodePosition { Line = 20 }
            }
        };

        // Assert
        Assert.Equal("/src/main.cs", location.FilePath);
        Assert.Equal(10, location.Span.Start.Line);
        Assert.Equal(20, location.Span.End.Line);
    }

    #endregion

    #region CodeIssue Tests

    [Fact]
    public void CodeIssue_DefaultValues()
    {
        // Arrange & Act
        var issue = new CodeIssue();

        // Assert
        Assert.Equal(CodeIssueSeverity.Warning, issue.Severity);
        Assert.Equal(CodeIssueCategory.Other, issue.Category);
        Assert.Equal(string.Empty, issue.Summary);
        Assert.Null(issue.Details);
        Assert.Null(issue.Rationale);
        Assert.Null(issue.FixGuidance);
        Assert.Null(issue.TestGuidance);
        Assert.NotNull(issue.Location);
    }

    [Fact]
    public void CodeIssue_SetProperties()
    {
        // Arrange & Act
        var issue = new CodeIssue
        {
            Severity = CodeIssueSeverity.Error,
            Category = CodeIssueCategory.Security,
            Summary = "SQL Injection vulnerability",
            Details = "User input is not sanitized",
            Rationale = "Attackers can execute arbitrary SQL",
            FixGuidance = "Use parameterized queries",
            TestGuidance = "Test with SQL injection payloads"
        };

        // Assert
        Assert.Equal(CodeIssueSeverity.Error, issue.Severity);
        Assert.Equal(CodeIssueCategory.Security, issue.Category);
        Assert.Equal("SQL Injection vulnerability", issue.Summary);
        Assert.Equal("User input is not sanitized", issue.Details);
        Assert.Equal("Attackers can execute arbitrary SQL", issue.Rationale);
        Assert.Equal("Use parameterized queries", issue.FixGuidance);
        Assert.Equal("Test with SQL injection payloads", issue.TestGuidance);
    }

    #endregion

    #region CodeIssueSeverity Tests

    [Fact]
    public void CodeIssueSeverity_HasCorrectValues()
    {
        // Assert
        Assert.Equal(0, (int)CodeIssueSeverity.Info);
        Assert.Equal(1, (int)CodeIssueSeverity.Warning);
        Assert.Equal(2, (int)CodeIssueSeverity.Error);
        Assert.Equal(3, (int)CodeIssueSeverity.Critical);
    }

    [Fact]
    public void CodeIssueSeverity_AllValuesAreDefined()
    {
        // Assert
        var values = (CodeIssueSeverity[])Enum.GetValues(typeof(CodeIssueSeverity));
        Assert.Equal(4, values.Length);
    }

    #endregion

    #region ProgramLanguage Tests

    [Fact]
    public void ProgramLanguage_HasExpectedValues()
    {
        // Assert
        Assert.Equal(0, (int)ProgramLanguage.Python);
        Assert.Equal(1, (int)ProgramLanguage.CSharp);
        Assert.Equal(2, (int)ProgramLanguage.Java);
        Assert.Equal(3, (int)ProgramLanguage.JavaScript);
        Assert.Equal(4, (int)ProgramLanguage.TypeScript);
        Assert.Equal(5, (int)ProgramLanguage.CPlusPlus);
        Assert.Equal(6, (int)ProgramLanguage.C);
        Assert.Equal(7, (int)ProgramLanguage.Go);
        Assert.Equal(8, (int)ProgramLanguage.Rust);
        Assert.Equal(9, (int)ProgramLanguage.SQL);
        Assert.Equal(10, (int)ProgramLanguage.Generic);
    }

    [Fact]
    public void ProgramLanguage_AllValuesAreDefined()
    {
        // Assert
        var values = (ProgramLanguage[])Enum.GetValues(typeof(ProgramLanguage));
        Assert.Equal(11, values.Length);
    }

    #endregion

    #region CodeTask Tests

    [Fact]
    public void CodeTask_HasExpectedValues()
    {
        // Assert
        Assert.Equal(0, (int)CodeTask.Completion);
        Assert.Equal(1, (int)CodeTask.Generation);
        Assert.Equal(2, (int)CodeTask.Translation);
        Assert.Equal(3, (int)CodeTask.Summarization);
        Assert.Equal(4, (int)CodeTask.BugDetection);
        Assert.Equal(5, (int)CodeTask.BugFixing);
        Assert.Equal(6, (int)CodeTask.Refactoring);
        Assert.Equal(7, (int)CodeTask.Understanding);
        Assert.Equal(8, (int)CodeTask.TestGeneration);
        Assert.Equal(9, (int)CodeTask.Documentation);
        Assert.Equal(10, (int)CodeTask.Search);
        Assert.Equal(11, (int)CodeTask.CloneDetection);
        Assert.Equal(12, (int)CodeTask.CodeReview);
    }

    [Fact]
    public void CodeTask_AllValuesAreDefined()
    {
        // Assert
        var values = (CodeTask[])Enum.GetValues(typeof(CodeTask));
        Assert.Equal(13, values.Length);
    }

    #endregion

    #region Program<T> Tests

    [Fact]
    public void Program_DefaultConstructor()
    {
        // Arrange & Act
        var program = new Program<double>();

        // Assert
        Assert.Equal(string.Empty, program.SourceCode);
        Assert.Equal(ProgramLanguage.Generic, program.Language);
        Assert.False(program.IsValid);
        Assert.Equal(0.0, program.FitnessScore);
        Assert.Equal(0, program.Complexity);
        Assert.Null(program.Encoding);
        Assert.Null(program.ErrorMessage);
        Assert.Null(program.ExecutionTimeMs);
    }

    [Fact]
    public void Program_ParameterizedConstructor()
    {
        // Arrange & Act
        var program = new Program<double>(
            sourceCode: "print('hello')",
            language: ProgramLanguage.Python,
            isValid: true,
            fitnessScore: 0.95,
            complexity: 5);

        // Assert
        Assert.Equal("print('hello')", program.SourceCode);
        Assert.Equal(ProgramLanguage.Python, program.Language);
        Assert.True(program.IsValid);
        Assert.Equal(0.95, program.FitnessScore);
        Assert.Equal(5, program.Complexity);
    }

    [Fact]
    public void Program_SetProperties()
    {
        // Arrange & Act
        var program = new Program<float>
        {
            SourceCode = "Console.WriteLine(\"Hello\");",
            Language = ProgramLanguage.CSharp,
            IsValid = true,
            FitnessScore = 1.0,
            Complexity = 3,
            ErrorMessage = null,
            ExecutionTimeMs = 10.5
        };

        // Assert
        Assert.Equal("Console.WriteLine(\"Hello\");", program.SourceCode);
        Assert.Equal(ProgramLanguage.CSharp, program.Language);
        Assert.True(program.IsValid);
        Assert.Equal(1.0, program.FitnessScore);
        Assert.Equal(3, program.Complexity);
        Assert.Null(program.ErrorMessage);
        Assert.Equal(10.5, program.ExecutionTimeMs);
    }

    [Fact]
    public void Program_ToString_ReturnsFormattedString()
    {
        // Arrange
        var program = new Program<double>(
            sourceCode: "x = 1 + 2",
            language: ProgramLanguage.Python,
            isValid: true,
            fitnessScore: 0.85,
            complexity: 2);

        // Act
        var result = program.ToString();

        // Assert
        Assert.Contains("[Python]", result);
        Assert.Contains("Valid: True", result);
        Assert.Contains("Fitness: 0.85", result);
        Assert.Contains("Complexity: 2", result);
        Assert.Contains("x = 1 + 2", result);
    }

    [Fact]
    public void Program_InvalidProgram_WithErrorMessage()
    {
        // Arrange & Act
        var program = new Program<double>
        {
            SourceCode = "invalid syntax here",
            Language = ProgramLanguage.Python,
            IsValid = false,
            FitnessScore = 0.0,
            ErrorMessage = "SyntaxError: unexpected token"
        };

        // Assert
        Assert.False(program.IsValid);
        Assert.Equal(0.0, program.FitnessScore);
        Assert.Equal("SyntaxError: unexpected token", program.ErrorMessage);
    }

    #endregion

    #region ProgramInputOutputExample Tests

    [Fact]
    public void ProgramInputOutputExample_DefaultValues()
    {
        // Arrange & Act
        var example = new ProgramInputOutputExample();

        // Assert
        Assert.Equal(string.Empty, example.Input);
        Assert.Equal(string.Empty, example.ExpectedOutput);
    }

    [Fact]
    public void ProgramInputOutputExample_SetProperties()
    {
        // Arrange & Act
        var example = new ProgramInputOutputExample
        {
            Input = "[1, 2, 3]",
            ExpectedOutput = "[3, 2, 1]"
        };

        // Assert
        Assert.Equal("[1, 2, 3]", example.Input);
        Assert.Equal("[3, 2, 1]", example.ExpectedOutput);
    }

    #endregion

    #region CodeCompletionCandidate Tests

    [Fact]
    public void CodeCompletionCandidate_DefaultValues()
    {
        // Arrange & Act
        var candidate = new CodeCompletionCandidate();

        // Assert
        Assert.Equal(string.Empty, candidate.CompletionText);
        Assert.Equal(0.0, candidate.Score);
    }

    [Fact]
    public void CodeCompletionCandidate_SetProperties()
    {
        // Arrange & Act
        var candidate = new CodeCompletionCandidate
        {
            CompletionText = "public void DoSomething()",
            Score = 0.92
        };

        // Assert
        Assert.Equal("public void DoSomething()", candidate.CompletionText);
        Assert.Equal(0.92, candidate.Score);
    }

    #endregion

    #region CodeComplexityMetrics Tests

    [Fact]
    public void CodeComplexityMetrics_DefaultValues()
    {
        // Arrange & Act
        var metrics = new CodeComplexityMetrics();

        // Assert
        Assert.Equal(0, metrics.LineCount);
        Assert.Equal(0, metrics.CharacterCount);
        Assert.Equal(0, metrics.EstimatedCyclomaticComplexity);
    }

    [Fact]
    public void CodeComplexityMetrics_SetProperties()
    {
        // Arrange & Act
        var metrics = new CodeComplexityMetrics
        {
            LineCount = 100,
            CharacterCount = 3500,
            EstimatedCyclomaticComplexity = 15
        };

        // Assert
        Assert.Equal(100, metrics.LineCount);
        Assert.Equal(3500, metrics.CharacterCount);
        Assert.Equal(15, metrics.EstimatedCyclomaticComplexity);
    }

    #endregion

    #region CodeSymbol Tests

    [Fact]
    public void CodeSymbol_DefaultValues()
    {
        // Arrange & Act
        var symbol = new CodeSymbol();

        // Assert
        Assert.Equal(string.Empty, symbol.Name);
        Assert.Equal(CodeSymbolKind.Other, symbol.Kind);
        Assert.NotNull(symbol.Location);
    }

    [Fact]
    public void CodeSymbol_SetProperties()
    {
        // Arrange & Act
        var symbol = new CodeSymbol
        {
            Name = "Calculate",
            Kind = CodeSymbolKind.Function,
            Location = new CodeLocation
            {
                FilePath = "/src/math.cs",
                Span = new CodeSpan
                {
                    Start = new CodePosition { Line = 10 },
                    End = new CodePosition { Line = 25 }
                }
            }
        };

        // Assert
        Assert.Equal("Calculate", symbol.Name);
        Assert.Equal(CodeSymbolKind.Function, symbol.Kind);
        Assert.Equal("/src/math.cs", symbol.Location.FilePath);
        Assert.Equal(10, symbol.Location.Span.Start.Line);
    }

    #endregion

    #region CodeGenerationResult Tests

    [Fact]
    public void CodeGenerationResult_DefaultValues()
    {
        // Arrange & Act
        var result = new CodeGenerationResult();

        // Assert
        Assert.Equal(CodeTask.Generation, result.Task);
        Assert.Equal(ProgramLanguage.Generic, result.Language);
        Assert.False(result.Success);
        Assert.Null(result.RequestId);
        Assert.Null(result.Error);
        Assert.Equal(string.Empty, result.GeneratedCode);
        Assert.NotNull(result.Telemetry);
    }

    [Fact]
    public void CodeGenerationResult_SetProperties()
    {
        // Arrange & Act
        var result = new CodeGenerationResult
        {
            Language = ProgramLanguage.Python,
            Success = true,
            RequestId = "req-12345",
            GeneratedCode = "def hello():\n    print('Hello')"
        };

        // Assert
        Assert.Equal(CodeTask.Generation, result.Task);
        Assert.Equal(ProgramLanguage.Python, result.Language);
        Assert.True(result.Success);
        Assert.Equal("req-12345", result.RequestId);
        Assert.Equal("def hello():\n    print('Hello')", result.GeneratedCode);
    }

    #endregion

    #region CompilationDiagnostic Tests

    [Fact]
    public void CompilationDiagnostic_DefaultValues()
    {
        // Arrange & Act
        var diagnostic = new CompilationDiagnostic();

        // Assert
        Assert.Equal(CompilationDiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Null(diagnostic.Code);
        Assert.Equal(string.Empty, diagnostic.Message);
        Assert.Null(diagnostic.Line);
        Assert.Null(diagnostic.Column);
    }

    [Fact]
    public void CompilationDiagnostic_SetProperties()
    {
        // Arrange & Act
        var diagnostic = new CompilationDiagnostic
        {
            Severity = CompilationDiagnosticSeverity.Error,
            Code = "CS1002",
            Message = "Missing semicolon",
            Line = 42,
            Column = 15
        };

        // Assert
        Assert.Equal(CompilationDiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Equal("CS1002", diagnostic.Code);
        Assert.Equal("Missing semicolon", diagnostic.Message);
        Assert.Equal(42, diagnostic.Line);
        Assert.Equal(15, diagnostic.Column);
    }

    #endregion

    #region CompilationDiagnosticSeverity Tests

    [Fact]
    public void CompilationDiagnosticSeverity_HasExpectedValues()
    {
        // Assert
        var values = (CompilationDiagnosticSeverity[])Enum.GetValues(typeof(CompilationDiagnosticSeverity));
        Assert.True(values.Length >= 3); // At least Info, Warning, Error
    }

    #endregion

    #region ProgramExecuteErrorCode Tests

    [Fact]
    public void ProgramExecuteErrorCode_HasValues()
    {
        // Assert
        var values = (ProgramExecuteErrorCode[])Enum.GetValues(typeof(ProgramExecuteErrorCode));
        Assert.NotEmpty(values);
    }

    #endregion

    #region SqlExecuteErrorCode Tests

    [Fact]
    public void SqlExecuteErrorCode_HasValues()
    {
        // Assert
        var values = (SqlExecuteErrorCode[])Enum.GetValues(typeof(SqlExecuteErrorCode));
        Assert.NotEmpty(values);
    }

    #endregion

    #region SqlValueKind Tests

    [Fact]
    public void SqlValueKind_HasValues()
    {
        // Assert
        var values = (SqlValueKind[])Enum.GetValues(typeof(SqlValueKind));
        Assert.NotEmpty(values);
    }

    #endregion

    #region SqlValue Tests

    [Fact]
    public void SqlValue_WithNullKind()
    {
        // Arrange & Act
        var value = new SqlValue { Kind = SqlValueKind.Null };

        // Assert
        Assert.Equal(SqlValueKind.Null, value.Kind);
    }

    [Fact]
    public void SqlValue_SetProperties()
    {
        // Arrange & Act
        var value = new SqlValue
        {
            Kind = SqlValueKind.Integer,
            IntegerValue = 42
        };

        // Assert
        Assert.Equal(SqlValueKind.Integer, value.Kind);
        Assert.Equal(42, value.IntegerValue);
    }

    #endregion

    #region SynthesisType Tests

    [Fact]
    public void SynthesisType_HasValues()
    {
        // Assert
        var values = (SynthesisType[])Enum.GetValues(typeof(SynthesisType));
        Assert.NotEmpty(values);
    }

    #endregion

    #region SqlDialect Tests

    [Fact]
    public void SqlDialect_HasValues()
    {
        // Assert
        var values = (SqlDialect[])Enum.GetValues(typeof(SqlDialect));
        Assert.NotEmpty(values);
    }

    #endregion

    #region CodeIssueCategory Tests

    [Fact]
    public void CodeIssueCategory_HasValues()
    {
        // Assert
        var values = (CodeIssueCategory[])Enum.GetValues(typeof(CodeIssueCategory));
        Assert.NotEmpty(values);
    }

    #endregion

    #region CodeSymbolKind Tests

    [Fact]
    public void CodeSymbolKind_HasValues()
    {
        // Assert
        var values = (CodeSymbolKind[])Enum.GetValues(typeof(CodeSymbolKind));
        Assert.NotEmpty(values);
    }

    #endregion

    #region CodeCloneType Tests

    [Fact]
    public void CodeCloneType_HasValues()
    {
        // Assert
        var values = (CodeCloneType[])Enum.GetValues(typeof(CodeCloneType));
        Assert.NotEmpty(values);
    }

    #endregion

    #region CodeEditOperationType Tests

    [Fact]
    public void CodeEditOperationType_HasValues()
    {
        // Assert
        var values = (CodeEditOperationType[])Enum.GetValues(typeof(CodeEditOperationType));
        Assert.NotEmpty(values);
    }

    #endregion

    #region CodeMatchType Tests

    [Fact]
    public void CodeMatchType_HasValues()
    {
        // Assert
        var values = (CodeMatchType[])Enum.GetValues(typeof(CodeMatchType));
        Assert.NotEmpty(values);
    }

    #endregion

    #region Integration Scenarios

    [Fact]
    public void Integration_CreateProgramWithLocation()
    {
        // Arrange
        var program = new Program<double>(
            sourceCode: "public class MyClass { }",
            language: ProgramLanguage.CSharp,
            isValid: true,
            fitnessScore: 1.0,
            complexity: 1);

        var location = new CodeLocation
        {
            FilePath = "/src/MyClass.cs",
            Span = new CodeSpan
            {
                Start = new CodePosition { Line = 1, Column = 1, Offset = 0 },
                End = new CodePosition { Line = 1, Column = 25, Offset = 24 }
            }
        };

        // Assert
        Assert.True(program.IsValid);
        Assert.Equal(ProgramLanguage.CSharp, program.Language);
        Assert.Equal("/src/MyClass.cs", location.FilePath);
    }

    [Fact]
    public void Integration_CreateCodeIssueWithSymbol()
    {
        // Arrange
        var symbol = new CodeSymbol
        {
            Name = "vulnerableFunction",
            Kind = CodeSymbolKind.Function,
            Location = new CodeLocation
            {
                FilePath = "/src/security.cs",
                Span = new CodeSpan
                {
                    Start = new CodePosition { Line = 50 },
                    End = new CodePosition { Line = 100 }
                }
            }
        };

        var issue = new CodeIssue
        {
            Severity = CodeIssueSeverity.Critical,
            Category = CodeIssueCategory.Security,
            Summary = $"Security vulnerability in {symbol.Name}",
            Location = symbol.Location
        };

        // Assert
        Assert.Equal(CodeIssueSeverity.Critical, issue.Severity);
        Assert.Equal(CodeIssueCategory.Security, issue.Category);
        Assert.Contains("vulnerableFunction", issue.Summary);
        Assert.Equal("/src/security.cs", issue.Location.FilePath);
    }

    [Fact]
    public void Integration_CreateInputOutputExamples()
    {
        // Arrange
        var examples = new List<ProgramInputOutputExample>
        {
            new() { Input = "1", ExpectedOutput = "1" },
            new() { Input = "2", ExpectedOutput = "2" },
            new() { Input = "5", ExpectedOutput = "120" } // factorial
        };

        // Act
        var program = new Program<double>(
            sourceCode: "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            language: ProgramLanguage.Python,
            isValid: true,
            fitnessScore: 1.0);

        // Assert
        Assert.Equal(3, examples.Count);
        Assert.True(program.IsValid);
        Assert.Equal(1.0, program.FitnessScore);
    }

    [Fact]
    public void Integration_CreateGenerationResultWithDiagnostics()
    {
        // Arrange
        var diagnostics = new List<CompilationDiagnostic>
        {
            new() { Severity = CompilationDiagnosticSeverity.Warning, Code = "CS0168", Message = "Variable declared but never used", Line = 10 },
            new() { Severity = CompilationDiagnosticSeverity.Info, Code = "IDE0044", Message = "Consider making field readonly", Line = 5 }
        };

        var result = new CodeGenerationResult
        {
            Language = ProgramLanguage.CSharp,
            Success = true,
            GeneratedCode = "public class Example { private int x; }"
        };

        // Assert
        Assert.True(result.Success);
        Assert.Equal(2, diagnostics.Count);
        Assert.Contains(diagnostics, d => d.Severity == CompilationDiagnosticSeverity.Warning);
    }

    [Fact]
    public void Integration_CreateComplexityMetricsForProgram()
    {
        // Arrange
        var program = new Program<double>(
            sourceCode: @"
public class Calculator {
    public int Add(int a, int b) {
        if (a < 0) return b;
        if (b < 0) return a;
        return a + b;
    }
}",
            language: ProgramLanguage.CSharp,
            isValid: true);

        var metrics = new CodeComplexityMetrics
        {
            LineCount = 8,
            CharacterCount = 150,
            EstimatedCyclomaticComplexity = 3 // 1 base + 2 if statements
        };

        // Assert
        Assert.True(program.IsValid);
        Assert.Equal(8, metrics.LineCount);
        Assert.Equal(3, metrics.EstimatedCyclomaticComplexity);
    }

    [Fact]
    public void Integration_CodeCompletionCandidatesRanking()
    {
        // Arrange
        var candidates = new List<CodeCompletionCandidate>
        {
            new() { CompletionText = "GetHashCode()", Score = 0.75 },
            new() { CompletionText = "GetType()", Score = 0.80 },
            new() { CompletionText = "GetEnumerator()", Score = 0.95 },
            new() { CompletionText = "ToString()", Score = 0.85 }
        };

        // Act
        var ranked = candidates.OrderByDescending(c => c.Score).ToList();

        // Assert
        Assert.Equal("GetEnumerator()", ranked[0].CompletionText);
        Assert.Equal(0.95, ranked[0].Score);
        Assert.Equal("ToString()", ranked[1].CompletionText);
    }

    #endregion
}
