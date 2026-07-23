using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Options;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.ProgramSynthesis.Serving;
using AiDotNet.ProgramSynthesis.Tokenization;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ProgramSynthesis;

/// <summary>
/// Extended integration tests for ProgramSynthesis module covering engines,
/// request/result types, options, execution models, and tokenization.
/// </summary>
public class ProgramSynthesisExtendedIntegrationTests
{
    #region CodeSynthesisArchitecture

    [Fact(Timeout = 120000)]
    public async Task CodeSynthesisArchitecture_DefaultParams()
    {
        var arch = new CodeSynthesisArchitecture<double>(
            SynthesisType.Neural,
            ProgramLanguage.CSharp,
            CodeTask.Generation);

        Assert.Equal(SynthesisType.Neural, arch.SynthesisType);
        Assert.Equal(ProgramLanguage.CSharp, arch.TargetLanguage);
        Assert.Equal(CodeTask.Generation, arch.CodeTaskType);
        Assert.Equal(6, arch.NumEncoderLayers);
        Assert.Equal(0, arch.NumDecoderLayers);
        Assert.Equal(8, arch.NumHeads);
        Assert.Equal(512, arch.ModelDimension);
        Assert.Equal(2048, arch.FeedForwardDimension);
        Assert.Equal(512, arch.MaxSequenceLength);
        Assert.Equal(50000, arch.VocabularySize);
        Assert.Equal(100, arch.MaxProgramLength);
        Assert.Equal(0.1, arch.DropoutRate);
        Assert.True(arch.UsePositionalEncoding);
        Assert.False(arch.UseDataFlow);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeSynthesisArchitecture_CustomParams()
    {
        var arch = new CodeSynthesisArchitecture<double>(
            SynthesisType.Hybrid,
            ProgramLanguage.Python,
            CodeTask.Translation,
            numEncoderLayers: 12,
            numDecoderLayers: 6,
            numHeads: 16,
            modelDimension: 768,
            feedForwardDimension: 3072,
            maxSequenceLength: 1024,
            vocabularySize: 32000,
            maxProgramLength: 200,
            dropoutRate: 0.15,
            usePositionalEncoding: false,
            useDataFlow: true);

        Assert.Equal(SynthesisType.Hybrid, arch.SynthesisType);
        Assert.Equal(ProgramLanguage.Python, arch.TargetLanguage);
        Assert.Equal(CodeTask.Translation, arch.CodeTaskType);
        Assert.Equal(12, arch.NumEncoderLayers);
        Assert.Equal(6, arch.NumDecoderLayers);
        Assert.Equal(16, arch.NumHeads);
        Assert.Equal(768, arch.ModelDimension);
        Assert.Equal(3072, arch.FeedForwardDimension);
        Assert.Equal(1024, arch.MaxSequenceLength);
        Assert.Equal(32000, arch.VocabularySize);
        Assert.Equal(200, arch.MaxProgramLength);
        Assert.Equal(0.15, arch.DropoutRate);
        Assert.False(arch.UsePositionalEncoding);
        Assert.True(arch.UseDataFlow);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeSynthesisArchitecture_InheritsNeuralNetworkArchitecture()
    {
        var arch = new CodeSynthesisArchitecture<double>(
            SynthesisType.Neural,
            ProgramLanguage.Generic,
            CodeTask.Completion);

        Assert.IsAssignableFrom<AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>>(arch);
    }

    #endregion

    #region Engine Construction

    [Fact(Timeout = 120000)]
    public async Task CodeBERT_Construction_DefaultParams()
    {
        var arch = new CodeSynthesisArchitecture<double>(
            SynthesisType.Neural,
            ProgramLanguage.Python,
            CodeTask.Understanding,
            numEncoderLayers: 2,
            modelDimension: 64,
            feedForwardDimension: 128,
            vocabularySize: 1000,
            maxSequenceLength: 64);

        var model = new CodeBERT<double>(arch);
        Assert.NotNull(model);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact(Timeout = 120000)]
    public async Task CodeT5_Construction_DefaultParams()
    {
        var arch = new CodeSynthesisArchitecture<double>(
            SynthesisType.Neural,
            ProgramLanguage.Python,
            CodeTask.Generation,
            numEncoderLayers: 2,
            numDecoderLayers: 2,
            modelDimension: 64,
            feedForwardDimension: 128,
            vocabularySize: 1000,
            maxSequenceLength: 64);

        var model = new CodeT5<double>(arch);
        Assert.NotNull(model);
        Assert.Equal(2, model.NumEncoderLayers);
        Assert.Equal(2, model.NumDecoderLayers);
    }

    [Fact(Timeout = 120000)]
    public async Task GraphCodeBERT_Construction_DefaultParams()
    {
        var arch = new CodeSynthesisArchitecture<double>(
            SynthesisType.Neural,
            ProgramLanguage.Java,
            CodeTask.BugDetection,
            numEncoderLayers: 2,
            modelDimension: 64,
            feedForwardDimension: 128,
            vocabularySize: 1000,
            maxSequenceLength: 64,
            useDataFlow: true);

        var model = new GraphCodeBERT<double>(arch);
        Assert.NotNull(model);
        Assert.True(model.UsesDataFlow);
    }

    [Fact(Timeout = 120000)]
    public async Task GraphCodeBERT_WithoutDataFlow()
    {
        var arch = new CodeSynthesisArchitecture<double>(
            SynthesisType.Neural,
            ProgramLanguage.Generic,
            CodeTask.Understanding,
            numEncoderLayers: 2,
            modelDimension: 64,
            feedForwardDimension: 128,
            vocabularySize: 1000,
            maxSequenceLength: 64,
            useDataFlow: false);

        var model = new GraphCodeBERT<double>(arch);
        Assert.False(model.UsesDataFlow);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeBERT_Metadata_NotNull()
    {
        var arch = new CodeSynthesisArchitecture<double>(
            SynthesisType.Neural,
            ProgramLanguage.Generic,
            CodeTask.Understanding,
            numEncoderLayers: 2,
            modelDimension: 64,
            feedForwardDimension: 128,
            vocabularySize: 1000,
            maxSequenceLength: 64);

        var model = new CodeBERT<double>(arch);
        Assert.NotNull(model.GetModelMetadata());
    }

    #endregion

    #region ProgramSynthesisOptions

    [Fact(Timeout = 120000)]
    public async Task ProgramSynthesisOptions_DefaultValues()
    {
        var opts = new ProgramSynthesisOptions();

        Assert.Equal(ProgramSynthesisModelKind.CodeT5, opts.ModelKind);
        Assert.Equal(ProgramLanguage.Generic, opts.TargetLanguage);
        Assert.Equal(CodeTask.Generation, opts.DefaultTask);
        Assert.Equal(SynthesisType.Neural, opts.SynthesisType);
        Assert.Equal(512, opts.MaxSequenceLength);
        Assert.Equal(50000, opts.VocabularySize);
        Assert.Equal(6, opts.NumEncoderLayers);
        Assert.Equal(6, opts.NumDecoderLayers);
        Assert.Null(opts.Tokenizer);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgramSynthesisOptions_SetCustomValues()
    {
        var opts = new ProgramSynthesisOptions
        {
            ModelKind = ProgramSynthesisModelKind.CodeBERT,
            TargetLanguage = ProgramLanguage.Python,
            DefaultTask = CodeTask.Completion,
            SynthesisType = SynthesisType.Symbolic,
            MaxSequenceLength = 1024,
            VocabularySize = 32000,
            NumEncoderLayers = 12,
            NumDecoderLayers = 0
        };

        Assert.Equal(ProgramSynthesisModelKind.CodeBERT, opts.ModelKind);
        Assert.Equal(ProgramLanguage.Python, opts.TargetLanguage);
        Assert.Equal(CodeTask.Completion, opts.DefaultTask);
        Assert.Equal(SynthesisType.Symbolic, opts.SynthesisType);
        Assert.Equal(1024, opts.MaxSequenceLength);
        Assert.Equal(32000, opts.VocabularySize);
        Assert.Equal(12, opts.NumEncoderLayers);
        Assert.Equal(0, opts.NumDecoderLayers);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgramSynthesisModelKind_HasExpectedValues()
    {
        Assert.Equal(0, (int)ProgramSynthesisModelKind.CodeBERT);
        Assert.Equal(1, (int)ProgramSynthesisModelKind.GraphCodeBERT);
        Assert.Equal(2, (int)ProgramSynthesisModelKind.CodeT5);
    }

    [Fact(Timeout = 120000)]
    public async Task NeuralProgramSynthesizerOptions_IsNeuralNetworkOptions()
    {
        var opts = new NeuralProgramSynthesizerOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    #endregion

    #region Serving Options

    [Fact(Timeout = 120000)]
    public async Task ProgramSynthesisServingClientOptions_DefaultValues()
    {
        var opts = new ProgramSynthesisServingClientOptions();

        Assert.Null(opts.BaseAddress);
        Assert.Null(opts.ApiKey);
        Assert.Null(opts.BearerToken);
        Assert.Equal("X-AiDotNet-Api-Key", opts.ApiKeyHeaderName);
        Assert.Null(opts.HttpClient);
        Assert.Equal(100_000, opts.TimeoutMs);
        Assert.True(opts.PreferServing);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgramSynthesisServingClientOptions_SetCustomValues()
    {
        var opts = new ProgramSynthesisServingClientOptions
        {
            BaseAddress = new Uri("http://localhost:5000/"),
            ApiKey = "test-key",
            BearerToken = "bearer-token",
            ApiKeyHeaderName = "X-Custom-Key",
            TimeoutMs = 30_000,
            PreferServing = false
        };

        Assert.Equal("http://localhost:5000/", opts.BaseAddress.ToString());
        Assert.Equal("test-key", opts.ApiKey);
        Assert.Equal("bearer-token", opts.BearerToken);
        Assert.Equal("X-Custom-Key", opts.ApiKeyHeaderName);
        Assert.Equal(30_000, opts.TimeoutMs);
        Assert.False(opts.PreferServing);
    }

    #endregion

    #region Tokenization Options

    [Fact(Timeout = 120000)]
    public async Task CodeTokenizationPipelineOptions_DefaultValues()
    {
        var opts = new CodeTokenizationPipelineOptions();

        Assert.False(opts.IncludeAst);
        Assert.Equal(10_000, opts.MaxAstNodes);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeTokenizationPipelineOptions_SetCustomValues()
    {
        var opts = new CodeTokenizationPipelineOptions
        {
            IncludeAst = true,
            MaxAstNodes = 5000
        };

        Assert.True(opts.IncludeAst);
        Assert.Equal(5000, opts.MaxAstNodes);
    }

    #endregion

    #region Request Types

    [Fact(Timeout = 120000)]
    public async Task CodeGenerationRequest_DefaultsAndTaskType()
    {
        var req = new CodeGenerationRequest();
        Assert.Equal(CodeTask.Generation, req.Task);
        Assert.Equal(string.Empty, req.Description);
        Assert.NotNull(req.Examples);
        Assert.Empty(req.Examples);
        Assert.Equal(ProgramLanguage.Generic, req.Language);
        Assert.Null(req.RequestId);
        Assert.Null(req.MaxWallClockMilliseconds);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeGenerationRequest_SetProperties()
    {
        var req = new CodeGenerationRequest
        {
            Description = "Sort a list",
            Language = ProgramLanguage.Python,
            RequestId = "req-123",
            MaxWallClockMilliseconds = 5000
        };
        req.Examples.Add(new ProgramInputOutputExample
        {
            Input = "[3,1,2]",
            ExpectedOutput = "[1,2,3]"
        });

        Assert.Equal("Sort a list", req.Description);
        Assert.Equal(ProgramLanguage.Python, req.Language);
        Assert.Equal("req-123", req.RequestId);
        Assert.Equal(5000, req.MaxWallClockMilliseconds);
        Assert.Single(req.Examples);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCompletionRequest_DefaultsAndTaskType()
    {
        var req = new CodeCompletionRequest();
        Assert.Equal(CodeTask.Completion, req.Task);
        Assert.Equal(string.Empty, req.Code);
        Assert.Null(req.CursorOffset);
        Assert.Equal(3, req.MaxCandidates);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCompletionRequest_SetProperties()
    {
        var req = new CodeCompletionRequest
        {
            Code = "def hello():",
            CursorOffset = 12,
            MaxCandidates = 5,
            Language = ProgramLanguage.Python
        };

        Assert.Equal("def hello():", req.Code);
        Assert.Equal(12, req.CursorOffset);
        Assert.Equal(5, req.MaxCandidates);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeBugDetectionRequest_TaskType()
    {
        var req = new CodeBugDetectionRequest();
        Assert.Equal(CodeTask.BugDetection, req.Task);
        Assert.Equal(string.Empty, req.Code);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeBugFixingRequest_TaskType()
    {
        var req = new CodeBugFixingRequest();
        Assert.Equal(CodeTask.BugFixing, req.Task);
        Assert.Equal(string.Empty, req.Code);
        Assert.Null(req.BugDescription);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCloneDetectionRequest_TaskType()
    {
        var req = new CodeCloneDetectionRequest();
        Assert.Equal(CodeTask.CloneDetection, req.Task);
        Assert.NotNull(req.Corpus);
        Assert.Equal(0.8, req.MinSimilarity);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeDocumentationRequest_TaskType()
    {
        var req = new CodeDocumentationRequest();
        Assert.Equal(CodeTask.Documentation, req.Task);
        Assert.Equal(string.Empty, req.Code);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeRefactoringRequest_TaskType()
    {
        var req = new CodeRefactoringRequest();
        Assert.Equal(CodeTask.Refactoring, req.Task);
        Assert.Equal(string.Empty, req.Code);
        Assert.Null(req.Goal);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeReviewRequest_TaskType()
    {
        var req = new CodeReviewRequest();
        Assert.Equal(CodeTask.CodeReview, req.Task);
        Assert.Equal(string.Empty, req.Code);
        Assert.Null(req.FilePath);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeSearchRequest_TaskType()
    {
        var req = new CodeSearchRequest();
        Assert.Equal(CodeTask.Search, req.Task);
        Assert.Equal(string.Empty, req.Query);
        Assert.NotNull(req.Corpus);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeSummarizationRequest_TaskType()
    {
        var req = new CodeSummarizationRequest();
        Assert.Equal(CodeTask.Summarization, req.Task);
        Assert.Equal(string.Empty, req.Code);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeTestGenerationRequest_TaskType()
    {
        var req = new CodeTestGenerationRequest();
        Assert.Equal(CodeTask.TestGeneration, req.Task);
        Assert.Equal(string.Empty, req.Code);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeTranslationRequest_TaskType()
    {
        var req = new CodeTranslationRequest();
        Assert.Equal(CodeTask.Translation, req.Task);
        Assert.Equal(string.Empty, req.Code);
        Assert.Equal(ProgramLanguage.Generic, req.SourceLanguage);
        Assert.Equal(ProgramLanguage.Generic, req.TargetLanguage);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeUnderstandingRequest_TaskType()
    {
        var req = new CodeUnderstandingRequest();
        Assert.Equal(CodeTask.Understanding, req.Task);
        Assert.Equal(string.Empty, req.Code);
        Assert.Null(req.FilePath);
    }

    [Fact(Timeout = 120000)]
    public async Task AllRequests_InheritSqlDialectProperty()
    {
        var req = new CodeGenerationRequest
        {
            SqlDialect = SqlDialect.SQLite,
            Language = ProgramLanguage.SQL
        };

        Assert.Equal(SqlDialect.SQLite, req.SqlDialect);
        Assert.Equal(ProgramLanguage.SQL, req.Language);
    }

    #endregion

    #region Result Types

    [Fact(Timeout = 120000)]
    public async Task CodeGenerationResult_DefaultsAndTaskType()
    {
        var result = new CodeGenerationResult();
        Assert.Equal(CodeTask.Generation, result.Task);
        Assert.Equal(string.Empty, result.GeneratedCode);
        Assert.False(result.Success);
        Assert.Null(result.Error);
        Assert.NotNull(result.Telemetry);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCompletionResult_TaskType()
    {
        var result = new CodeCompletionResult();
        Assert.Equal(CodeTask.Completion, result.Task);
        Assert.NotNull(result.Candidates);
        Assert.Empty(result.Candidates);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeBugDetectionResult_TaskType()
    {
        var result = new CodeBugDetectionResult();
        Assert.Equal(CodeTask.BugDetection, result.Task);
        Assert.NotNull(result.Issues);
        Assert.Empty(result.Issues);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeBugFixingResult_TaskType()
    {
        var result = new CodeBugFixingResult();
        Assert.Equal(CodeTask.BugFixing, result.Task);
        Assert.Equal(string.Empty, result.FixedCode);
        Assert.NotNull(result.Diff);
        Assert.NotNull(result.FixedIssues);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCloneDetectionResult_TaskType()
    {
        var result = new CodeCloneDetectionResult();
        Assert.Equal(CodeTask.CloneDetection, result.Task);
        Assert.NotNull(result.CloneGroups);
        Assert.Empty(result.CloneGroups);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeDocumentationResult_TaskType()
    {
        var result = new CodeDocumentationResult();
        Assert.Equal(CodeTask.Documentation, result.Task);
        Assert.Equal(string.Empty, result.Documentation);
        Assert.Equal(string.Empty, result.UpdatedCode);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeRefactoringResult_TaskType()
    {
        var result = new CodeRefactoringResult();
        Assert.Equal(CodeTask.Refactoring, result.Task);
        Assert.Equal(string.Empty, result.RefactoredCode);
        Assert.NotNull(result.Diff);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeReviewResult_TaskType()
    {
        var result = new CodeReviewResult();
        Assert.Equal(CodeTask.CodeReview, result.Task);
        Assert.NotNull(result.Issues);
        Assert.NotNull(result.FixSuggestions);
        Assert.NotNull(result.PrioritizedPlan);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeSearchResult_TaskType()
    {
        var result = new CodeSearchResult();
        Assert.Equal(CodeTask.Search, result.Task);
        Assert.NotNull(result.FiltersApplied);
        Assert.NotNull(result.Results);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeSummarizationResult_TaskType()
    {
        var result = new CodeSummarizationResult();
        Assert.Equal(CodeTask.Summarization, result.Task);
        Assert.Equal(string.Empty, result.Summary);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeTestGenerationResult_TaskType()
    {
        var result = new CodeTestGenerationResult();
        Assert.Equal(CodeTask.TestGeneration, result.Task);
        Assert.NotNull(result.Tests);
        Assert.Empty(result.Tests);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeTranslationResult_TaskType()
    {
        var result = new CodeTranslationResult();
        Assert.Equal(CodeTask.Translation, result.Task);
        Assert.Equal(ProgramLanguage.Generic, result.SourceLanguage);
        Assert.Equal(ProgramLanguage.Generic, result.TargetLanguage);
        Assert.Equal(string.Empty, result.TranslatedCode);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeUnderstandingResult_TaskType()
    {
        var result = new CodeUnderstandingResult();
        Assert.Equal(CodeTask.Understanding, result.Task);
        Assert.NotNull(result.Symbols);
        Assert.NotNull(result.Dependencies);
        Assert.NotNull(result.Complexity);
        Assert.NotNull(result.CallGraph);
        Assert.NotNull(result.Hotspots);
        Assert.NotNull(result.ControlFlowSummaries);
    }

    [Fact(Timeout = 120000)]
    public async Task AllResults_SetSuccessAndError()
    {
        var result = new CodeGenerationResult
        {
            Success = true,
            GeneratedCode = "Console.WriteLine(\"hello\");",
            Language = ProgramLanguage.CSharp,
            RequestId = "req-456"
        };

        Assert.True(result.Success);
        Assert.Equal("Console.WriteLine(\"hello\");", result.GeneratedCode);
        Assert.Equal(ProgramLanguage.CSharp, result.Language);
        Assert.Equal("req-456", result.RequestId);
    }

    [Fact(Timeout = 120000)]
    public async Task AllResults_SetErrorInfo()
    {
        var result = new CodeBugFixingResult
        {
            Success = false,
            Error = "Timeout exceeded"
        };

        Assert.False(result.Success);
        Assert.Equal("Timeout exceeded", result.Error);
    }

    #endregion

    #region Execution Models

    [Fact(Timeout = 120000)]
    public async Task ProgramExecuteRequest_DefaultValues()
    {
        var req = new ProgramExecuteRequest();

        Assert.Equal(ProgramLanguage.Generic, req.Language);
        Assert.NotNull(req.AllowedLanguages);
        Assert.Empty(req.AllowedLanguages);
        Assert.Null(req.PreferredLanguage);
        Assert.False(req.AllowUndetectedLanguageFallback);
        Assert.Equal(string.Empty, req.SourceCode);
        Assert.Null(req.StdIn);
        Assert.False(req.CompileOnly);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgramExecuteRequest_SetProperties()
    {
        var req = new ProgramExecuteRequest
        {
            Language = ProgramLanguage.Python,
            SourceCode = "print('hello')",
            StdIn = "test input",
            CompileOnly = true,
            AllowUndetectedLanguageFallback = true,
            PreferredLanguage = ProgramLanguage.Python
        };
        req.AllowedLanguages.Add(ProgramLanguage.Python);

        Assert.Equal(ProgramLanguage.Python, req.Language);
        Assert.Equal("print('hello')", req.SourceCode);
        Assert.Equal("test input", req.StdIn);
        Assert.True(req.CompileOnly);
        Assert.True(req.AllowUndetectedLanguageFallback);
        Assert.Single(req.AllowedLanguages);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgramExecuteResponse_Properties()
    {
        var resp = new ProgramExecuteResponse
        {
            Success = true,
            Language = ProgramLanguage.Python,
            ExitCode = 0,
            StdOut = "hello\n",
            CompilationAttempted = false
        };

        Assert.True(resp.Success);
        Assert.Equal(ProgramLanguage.Python, resp.Language);
        Assert.Equal(0, resp.ExitCode);
        Assert.Equal("hello\n", resp.StdOut);
        Assert.Equal(string.Empty, resp.StdErr);
        Assert.False(resp.CompilationAttempted);
        Assert.Null(resp.CompilationSucceeded);
        Assert.NotNull(resp.CompilationDiagnostics);
        Assert.Null(resp.Error);
        Assert.Null(resp.ErrorCode);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgramEvaluateIoRequest_DefaultValues()
    {
        var req = new ProgramEvaluateIoRequest();

        Assert.Equal(ProgramLanguage.Generic, req.Language);
        Assert.NotNull(req.AllowedLanguages);
        Assert.Equal(string.Empty, req.SourceCode);
        Assert.NotNull(req.TestCases);
        Assert.Empty(req.TestCases);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgramEvaluateIoResponse_Properties()
    {
        var resp = new ProgramEvaluateIoResponse
        {
            Success = true,
            Language = ProgramLanguage.CSharp,
            TotalTests = 5,
            PassedTests = 4,
            PassRate = 0.8
        };

        Assert.True(resp.Success);
        Assert.Equal(5, resp.TotalTests);
        Assert.Equal(4, resp.PassedTests);
        Assert.Equal(0.8, resp.PassRate);
        Assert.NotNull(resp.TestResults);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgramEvaluateIoTestResult_Properties()
    {
        var result = new ProgramEvaluateIoTestResult
        {
            Passed = true,
            TestCase = new ProgramInputOutputExample
            {
                Input = "3",
                ExpectedOutput = "9"
            },
            FailureReason = null
        };

        Assert.True(result.Passed);
        Assert.Equal("3", result.TestCase.Input);
        Assert.Equal("9", result.TestCase.ExpectedOutput);
        Assert.Null(result.FailureReason);
    }

    [Fact(Timeout = 120000)]
    public async Task SqlExecuteRequest_DefaultValues()
    {
        var req = new SqlExecuteRequest();

        Assert.Null(req.Dialect);
        Assert.Equal(string.Empty, req.Query);
        Assert.Null(req.SchemaSql);
        Assert.Null(req.SeedSql);
        Assert.Null(req.DbId);
        Assert.Null(req.DatasetId);
    }

    [Fact(Timeout = 120000)]
    public async Task SqlExecuteRequest_SetProperties()
    {
        var req = new SqlExecuteRequest
        {
            Dialect = SqlDialect.SQLite,
            Query = "SELECT * FROM users",
            SchemaSql = "CREATE TABLE users (id INT)",
            SeedSql = "INSERT INTO users VALUES (1)",
            DbId = "db-1",
            DatasetId = "ds-1"
        };

        Assert.Equal(SqlDialect.SQLite, req.Dialect);
        Assert.Equal("SELECT * FROM users", req.Query);
        Assert.Equal("CREATE TABLE users (id INT)", req.SchemaSql);
    }

    [Fact(Timeout = 120000)]
    public async Task SqlExecuteResponse_Properties()
    {
        var resp = new SqlExecuteResponse
        {
            Success = true,
            Dialect = SqlDialect.Postgres
        };

        Assert.True(resp.Success);
        Assert.Equal(SqlDialect.Postgres, resp.Dialect);
        Assert.NotNull(resp.Columns);
        Assert.NotNull(resp.Rows);
        Assert.Null(resp.Error);
        Assert.Null(resp.ErrorCode);
    }

    #endregion

    #region Model Classes (AST, Corpus, etc.)

    [Fact(Timeout = 120000)]
    public async Task CodeAstNode_DefaultValues()
    {
        var node = new CodeAstNode();
        Assert.Equal(0, node.NodeId);
        Assert.Null(node.ParentNodeId);
        Assert.Equal(ProgramLanguage.Generic, node.Language);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeAstNode_SetProperties()
    {
        var node = new CodeAstNode
        {
            NodeId = 42,
            ParentNodeId = 10,
            Language = ProgramLanguage.CSharp
        };

        Assert.Equal(42, node.NodeId);
        Assert.Equal(10, node.ParentNodeId);
        Assert.Equal(ProgramLanguage.CSharp, node.Language);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeAstEdge_SetProperties()
    {
        var edge = new CodeAstEdge
        {
            ParentNodeId = 1,
            ChildNodeId = 5
        };

        Assert.Equal(1, edge.ParentNodeId);
        Assert.Equal(5, edge.ChildNodeId);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCorpusDocument_DefaultValues()
    {
        var doc = new CodeCorpusDocument();
        Assert.Equal(string.Empty, doc.DocumentId);
        Assert.Null(doc.FilePath);
        Assert.Equal(ProgramLanguage.Generic, doc.Language);
        Assert.Equal(string.Empty, doc.Content);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCorpusDocument_SetProperties()
    {
        var doc = new CodeCorpusDocument
        {
            DocumentId = "doc-1",
            FilePath = "/src/main.py",
            Language = ProgramLanguage.Python,
            Content = "print('hello')"
        };

        Assert.Equal("doc-1", doc.DocumentId);
        Assert.Equal("/src/main.py", doc.FilePath);
        Assert.Equal(ProgramLanguage.Python, doc.Language);
        Assert.Equal("print('hello')", doc.Content);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCorpusReference_DefaultValues()
    {
        var corpus = new CodeCorpusReference();
        Assert.NotNull(corpus.Documents);
        Assert.Empty(corpus.Documents);
        Assert.Null(corpus.CorpusId);
        Assert.Null(corpus.IndexId);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCorpusReference_WithDocuments()
    {
        var corpus = new CodeCorpusReference
        {
            CorpusId = "corp-1",
            IndexId = "idx-1"
        };
        corpus.Documents.Add(new CodeCorpusDocument
        {
            DocumentId = "d1",
            Content = "code here"
        });

        Assert.Equal("corp-1", corpus.CorpusId);
        Assert.Single(corpus.Documents);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeTransformDiff_DefaultValues()
    {
        var diff = new CodeTransformDiff();
        Assert.NotNull(diff.Edits);
        Assert.Empty(diff.Edits);
        Assert.Null(diff.UnifiedDiff);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeFixSuggestion_DefaultValues()
    {
        var fix = new CodeFixSuggestion();
        Assert.Equal(string.Empty, fix.Summary);
        Assert.Null(fix.Rationale);
        Assert.Null(fix.FixGuidance);
        Assert.Null(fix.TestGuidance);
        Assert.Null(fix.Diff);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeSearchHit_DefaultValues()
    {
        var hit = new CodeSearchHit();
        Assert.Equal(0.0, hit.Score);
        Assert.Equal(string.Empty, hit.SnippetText);
        Assert.NotNull(hit.Location);
        Assert.Null(hit.Symbol);
        Assert.Equal(CodeMatchType.Lexical, hit.MatchType);
        Assert.Null(hit.Provenance);
        Assert.Null(hit.MatchExplanation);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeSecurityHotspot_DefaultValues()
    {
        var hotspot = new CodeSecurityHotspot();
        Assert.Equal(string.Empty, hotspot.Category);
        Assert.Equal(string.Empty, hotspot.Summary);
        Assert.NotNull(hotspot.Location);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeHotspot_DefaultValues()
    {
        var hotspot = new CodeHotspot();
        Assert.Equal(string.Empty, hotspot.SymbolName);
        Assert.Equal(string.Empty, hotspot.Reason);
        Assert.Equal(0.0, hotspot.Score);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeDependency_DefaultValues()
    {
        var dep = new CodeDependency();
        Assert.Equal(string.Empty, dep.Name);
        Assert.Null(dep.Kind);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCallGraphEdge_DefaultValues()
    {
        var edge = new CodeCallGraphEdge();
        Assert.Equal(string.Empty, edge.Caller);
        Assert.Equal(string.Empty, edge.Callee);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeProvenance_DefaultValues()
    {
        var prov = new CodeProvenance();
        Assert.Null(prov.IndexId);
        Assert.Null(prov.RepoId);
        Assert.Null(prov.CommitOrRef);
        Assert.Null(prov.SourcePath);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeExecutionTelemetry_DefaultValues()
    {
        var tel = new CodeExecutionTelemetry();
        Assert.Null(tel.ExitCode);
        Assert.False(tel.TimedOut);
        Assert.Null(tel.StdoutBytes);
        Assert.Null(tel.StderrBytes);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeTaskTelemetry_DefaultValues()
    {
        var tel = new CodeTaskTelemetry();
        Assert.Equal(0L, tel.ProcessingTimeMs);
        Assert.Null(tel.Execution);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeTaskTelemetry_WithExecution()
    {
        var tel = new CodeTaskTelemetry
        {
            ProcessingTimeMs = 1500,
            Execution = new CodeExecutionTelemetry
            {
                ExitCode = 0,
                StdoutBytes = 256
            }
        };

        Assert.Equal(1500, tel.ProcessingTimeMs);
        Assert.NotNull(tel.Execution);
        Assert.Equal(0, tel.Execution.ExitCode);
        Assert.Equal(256L, tel.Execution.StdoutBytes);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeTokenizationResult_DefaultValues()
    {
        var result = new CodeTokenizationResult();
        Assert.Equal(ProgramLanguage.Generic, result.Language);
        Assert.Null(result.FilePath);
        Assert.NotNull(result.Tokenization);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeAstNodePath_DefaultValues()
    {
        var path = new CodeAstNodePath();
        Assert.NotNull(path.Segments);
        Assert.Empty(path.Segments);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeAstPathSegment_DefaultValues()
    {
        var segment = new CodeAstPathSegment();
        Assert.Equal(string.Empty, segment.Kind);
        Assert.Null(segment.Name);
        Assert.Null(segment.Index);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeEditOperation_DefaultValues()
    {
        var op = new CodeEditOperation();
        Assert.Equal(CodeEditOperationType.Insert, op.OperationType);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCloneGroup_DefaultValues()
    {
        var group = new CodeCloneGroup();
        Assert.NotNull(group.Instances);
        Assert.Empty(group.Instances);
    }

    [Fact(Timeout = 120000)]
    public async Task CodeCloneInstance_DefaultValues()
    {
        var instance = new CodeCloneInstance();
        Assert.NotNull(instance.Location);
    }

    #endregion

    #region Execution Enums

    [Fact(Timeout = 120000)]
    public async Task ProgramExecuteErrorCode_HasValues()
    {
        var values = Enum.GetValues(typeof(ProgramExecuteErrorCode));
        Assert.True(values.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SqlExecuteErrorCode_HasValues()
    {
        var values = Enum.GetValues(typeof(SqlExecuteErrorCode));
        Assert.True(values.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task CompilationDiagnosticSeverity_HasValues()
    {
        var values = Enum.GetValues(typeof(CompilationDiagnosticSeverity));
        Assert.True(values.Length > 0);
    }

    #endregion

    #region Integration Scenarios

    [Fact(Timeout = 120000)]
    public async Task Integration_CreateCodeReviewWithIssuesAndFixes()
    {
        var result = new CodeReviewResult
        {
            Success = true,
            Language = ProgramLanguage.CSharp,
            RequestId = "review-1"
        };
        result.Issues.Add(new CodeIssue
        {
            Summary = "Null reference possible",
            Category = CodeIssueCategory.Correctness,
            Severity = CodeIssueSeverity.Warning
        });
        result.FixSuggestions.Add(new CodeFixSuggestion
        {
            Summary = "Add null check",
            Rationale = "Prevents NullReferenceException"
        });
        result.PrioritizedPlan.Add("Fix null reference in method A");

        Assert.True(result.Success);
        Assert.Single(result.Issues);
        Assert.Single(result.FixSuggestions);
        Assert.Single(result.PrioritizedPlan);
        Assert.Equal(CodeIssueCategory.Correctness, result.Issues[0].Category);
    }

    [Fact(Timeout = 120000)]
    public async Task Integration_CreateCodeTranslationPipeline()
    {
        var req = new CodeTranslationRequest
        {
            Code = "def greet(name): return f'Hello {name}'",
            SourceLanguage = ProgramLanguage.Python,
            TargetLanguage = ProgramLanguage.CSharp,
            Language = ProgramLanguage.Python
        };

        var result = new CodeTranslationResult
        {
            Success = true,
            SourceLanguage = ProgramLanguage.Python,
            TargetLanguage = ProgramLanguage.CSharp,
            TranslatedCode = "string Greet(string name) => $\"Hello {name}\";",
            Language = ProgramLanguage.CSharp
        };

        Assert.Equal(CodeTask.Translation, req.Task);
        Assert.Equal(CodeTask.Translation, result.Task);
        Assert.Equal(req.SourceLanguage, result.SourceLanguage);
        Assert.Equal(req.TargetLanguage, result.TargetLanguage);
        Assert.NotEqual(string.Empty, result.TranslatedCode);
    }

    [Fact(Timeout = 120000)]
    public async Task Integration_CreateBugFixWithDiff()
    {
        var result = new CodeBugFixingResult
        {
            Success = true,
            FixedCode = "if (x != null) { x.DoStuff(); }",
            Language = ProgramLanguage.CSharp
        };
        result.Diff.Edits.Add(new CodeEditOperation
        {
            OperationType = CodeEditOperationType.Replace
        });
        result.FixedIssues.Add(new CodeIssue
        {
            Summary = "NullReferenceException fixed",
            Severity = CodeIssueSeverity.Error
        });

        Assert.True(result.Success);
        Assert.Single(result.Diff.Edits);
        Assert.Single(result.FixedIssues);
        Assert.Equal(CodeEditOperationType.Replace, result.Diff.Edits[0].OperationType);
    }

    [Fact(Timeout = 120000)]
    public async Task Integration_CreateSearchWithProvenance()
    {
        var result = new CodeSearchResult
        {
            Success = true,
            Language = ProgramLanguage.Generic
        };
        result.FiltersApplied.Add("language:python");
        result.Results.Add(new CodeSearchHit
        {
            Score = 0.95,
            SnippetText = "def sort_list(items):",
            MatchType = CodeMatchType.Semantic,
            Provenance = new CodeProvenance
            {
                RepoId = "repo-1",
                SourcePath = "src/utils.py",
                CommitOrRef = "abc123"
            }
        });

        Assert.Single(result.Results);
        Assert.Equal(0.95, result.Results[0].Score);
        Assert.Equal(CodeMatchType.Semantic, result.Results[0].MatchType);
        Assert.NotNull(result.Results[0].Provenance);
        Assert.Equal("repo-1", result.Results[0].Provenance.RepoId);
    }

    [Fact(Timeout = 120000)]
    public async Task Integration_CreateUnderstandingWithFullAnalysis()
    {
        var result = new CodeUnderstandingResult
        {
            Success = true,
            Language = ProgramLanguage.CSharp
        };
        result.Symbols.Add(new CodeSymbol
        {
            Name = "MyClass",
            Kind = CodeSymbolKind.Class
        });
        result.Dependencies.Add(new CodeDependency
        {
            Name = "System.Linq",
            Kind = "Namespace"
        });
        result.CallGraph.Add(new CodeCallGraphEdge
        {
            Caller = "Main",
            Callee = "Process"
        });
        result.Hotspots.Add(new CodeHotspot
        {
            SymbolName = "HeavyMethod",
            Reason = "High cyclomatic complexity",
            Score = 0.9
        });
        result.ControlFlowSummaries.Add("Main -> Process -> Save");

        Assert.Single(result.Symbols);
        Assert.Single(result.Dependencies);
        Assert.Single(result.CallGraph);
        Assert.Single(result.Hotspots);
        Assert.Single(result.ControlFlowSummaries);
    }

    #endregion
}
