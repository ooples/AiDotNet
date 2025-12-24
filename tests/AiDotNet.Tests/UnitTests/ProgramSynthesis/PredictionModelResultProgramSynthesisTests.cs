using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Tokenization;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.UnitTests.ProgramSynthesis.Fakes;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Serving;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public class PredictionModelResultProgramSynthesisTests
{
    [Fact]
    public void ExecuteCodeTask_DelegatesToUnderlyingCodeModel()
    {
        var model = FakeCodeModel.CreateDefault(targetLanguage: ProgramLanguage.CSharp);

        var optimizationResult = new OptimizationResult<double, Tensor<double>, Tensor<double>>
        {
            BestSolution = model
        };

        var options = new PredictionModelResultOptions<double, Tensor<double>, Tensor<double>>
        {
            OptimizationResult = optimizationResult,
            NormalizationInfo = new NormalizationInfo<double, Tensor<double>, Tensor<double>>()
        };

        var modelResult = new PredictionModelResult<double, Tensor<double>, Tensor<double>>(options);

        var request = new CodeSummarizationRequest
        {
            Code = "int Add(int a,int b){return a+b;}",
            Language = ProgramLanguage.CSharp,
            RequestId = "sum"
        };

        var result = modelResult.ExecuteCodeTask(request);
        Assert.True(result.Success);
        Assert.IsType<CodeSummarizationResult>(result);

        var typed = modelResult.SummarizeCode(request);
        Assert.NotNull(typed.Summary);
    }

    [Fact]
    public void TokenizeCode_UsesCanonicalPipelineAndSupportsSql()
    {
        var optimizationResult = new OptimizationResult<double, Tensor<double>, Tensor<double>>
        {
            BestSolution = FakeCodeModel.CreateDefault(targetLanguage: ProgramLanguage.CSharp)
        };

        var options = new PredictionModelResultOptions<double, Tensor<double>, Tensor<double>>
        {
            OptimizationResult = optimizationResult,
            NormalizationInfo = new NormalizationInfo<double, Tensor<double>, Tensor<double>>()
        };

        var modelResult = new PredictionModelResult<double, Tensor<double>, Tensor<double>>(options);

        var sql = modelResult.TokenizeCode(
            code: "SELECT 1;",
            language: ProgramLanguage.SQL,
            options: new EncodingOptions { AddSpecialTokens = false });

        Assert.Equal(ProgramLanguage.SQL, sql.Language);
        Assert.NotEmpty(sql.Tokenization.Tokens);
        Assert.Equal(sql.Tokenization.Tokens.Count, sql.TokenSpans.Count);

        var csharp = modelResult.TokenizeCode(
            code: "public class C { void M() { } }",
            language: ProgramLanguage.CSharp,
            options: new EncodingOptions { AddSpecialTokens = false },
            pipelineOptions: new CodeTokenizationPipelineOptions { IncludeAst = true, MaxAstNodes = 5000 });

        Assert.Equal(ProgramLanguage.CSharp, csharp.Language);
        Assert.NotEmpty(csharp.AstNodes);
        Assert.NotEmpty(csharp.AstEdges);
    }

    [Fact]
    public async Task ExecuteCodeTaskAsync_UsesServingClient_WhenConfiguredAndPreferred()
    {
        var optimizationResult = new OptimizationResult<double, Tensor<double>, Tensor<double>>
        {
            BestSolution = FakeCodeModel.CreateDefault(targetLanguage: ProgramLanguage.CSharp)
        };

        var stubClient = new StubServingClient();

        var options = new PredictionModelResultOptions<double, Tensor<double>, Tensor<double>>
        {
            OptimizationResult = optimizationResult,
            NormalizationInfo = new NormalizationInfo<double, Tensor<double>, Tensor<double>>(),
            ProgramSynthesisServingClient = stubClient,
            ProgramSynthesisServingClientOptions = new ProgramSynthesisServingClientOptions
            {
                PreferServing = true
            }
        };

        var modelResult = new PredictionModelResult<double, Tensor<double>, Tensor<double>>(options);

        var request = new CodeSummarizationRequest
        {
            Code = "int Add(int a,int b){return a+b;}",
            Language = ProgramLanguage.CSharp,
            RequestId = "sum"
        };

        var result = await modelResult.ExecuteCodeTaskAsync(request);
        Assert.True(result.Success);
        Assert.IsType<CodeSummarizationResult>(result);
        Assert.Equal(1, stubClient.CodeTaskCalls);
    }

    [Fact]
    public void ExecuteCodeTask_UsesLocalModel_WhenServingIsPresentButNotPreferred()
    {
        var model = FakeCodeModel.CreateDefault(targetLanguage: ProgramLanguage.CSharp);

        var optimizationResult = new OptimizationResult<double, Tensor<double>, Tensor<double>>
        {
            BestSolution = model
        };

        var stubClient = new StubServingClient();

        var options = new PredictionModelResultOptions<double, Tensor<double>, Tensor<double>>
        {
            OptimizationResult = optimizationResult,
            NormalizationInfo = new NormalizationInfo<double, Tensor<double>, Tensor<double>>(),
            ProgramSynthesisServingClient = stubClient,
            ProgramSynthesisServingClientOptions = new ProgramSynthesisServingClientOptions
            {
                PreferServing = false
            }
        };

        var modelResult = new PredictionModelResult<double, Tensor<double>, Tensor<double>>(options);

        var request = new CodeSummarizationRequest
        {
            Code = "int Add(int a,int b){return a+b;}",
            Language = ProgramLanguage.CSharp,
            RequestId = "sum"
        };

        var result = modelResult.ExecuteCodeTask(request);
        Assert.True(result.Success);
        Assert.IsType<CodeSummarizationResult>(result);
        Assert.Equal(0, stubClient.CodeTaskCalls);
    }

    [Fact]
    public void ExecuteProgramAsync_ThrowsWhenServingNotConfigured()
    {
        var optimizationResult = new OptimizationResult<double, Tensor<double>, Tensor<double>>
        {
            BestSolution = FakeCodeModel.CreateDefault(targetLanguage: ProgramLanguage.CSharp)
        };

        var options = new PredictionModelResultOptions<double, Tensor<double>, Tensor<double>>
        {
            OptimizationResult = optimizationResult,
            NormalizationInfo = new NormalizationInfo<double, Tensor<double>, Tensor<double>>()
        };

        var modelResult = new PredictionModelResult<double, Tensor<double>, Tensor<double>>(options);

        Assert.Throws<InvalidOperationException>(() =>
            modelResult.ExecuteProgramAsync(new ProgramExecuteRequest()).GetAwaiter().GetResult());
    }

    private sealed class StubServingClient : IProgramSynthesisServingClient
    {
        public int CodeTaskCalls { get; private set; }

        public Task<CodeTaskResultBase> ExecuteCodeTaskAsync(CodeTaskRequestBase request, CancellationToken cancellationToken)
        {
            CodeTaskCalls++;
            return Task.FromResult<CodeTaskResultBase>(new CodeSummarizationResult
            {
                Summary = "ok",
                Success = true,
                Language = request.Language,
                RequestId = request.RequestId
            });
        }

        public Task<ProgramExecuteResponse> ExecuteProgramAsync(ProgramExecuteRequest request, CancellationToken cancellationToken)
            => Task.FromResult(new ProgramExecuteResponse
            {
                Success = true,
                Language = request.Language,
                ExitCode = 0
            });

        public Task<ProgramEvaluateIoResponse> EvaluateProgramIoAsync(ProgramEvaluateIoRequest request, CancellationToken cancellationToken)
            => Task.FromResult(new ProgramEvaluateIoResponse
            {
                Success = true,
                Language = request.Language
            });

        public Task<SqlExecuteResponse> ExecuteSqlAsync(SqlExecuteRequest request, CancellationToken cancellationToken)
            => Task.FromResult(new SqlExecuteResponse { Success = true });
    }
}
