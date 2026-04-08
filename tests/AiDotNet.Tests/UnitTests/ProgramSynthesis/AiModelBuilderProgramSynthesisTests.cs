using AiDotNet.Models.Results;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Options;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.ProgramSynthesis.Serving;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public sealed class AiModelBuilderProgramSynthesisTests
{
    [Fact]
    public async Task ConfigureProgramSynthesis_BuildAsync_ProducesResultWithCodeModel()
    {
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureProgramSynthesis(new ProgramSynthesisOptions
            {
                TargetLanguage = ProgramLanguage.CSharp,
                ModelKind = ProgramSynthesisModelKind.CodeBERT,
                MaxSequenceLength = 32,
                VocabularySize = 2048,
                NumEncoderLayers = 1
            });

        var result = await builder.BuildAsync();

        Assert.NotNull(result);

        var summary = result.SummarizeCode(new CodeSummarizationRequest
        {
            Code = "int Add(int a,int b){return a+b;}",
            Language = ProgramLanguage.CSharp
        });

        Assert.True(summary.Success);
        Assert.False(string.IsNullOrWhiteSpace(summary.Summary));
    }

    [Fact]
    public async Task ConfigureProgramSynthesisServing_WithCustomClient_WiresAiModelResult()
    {
        var fakeClient = new FakeServingClient();

        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureProgramSynthesis(new ProgramSynthesisOptions
            {
                TargetLanguage = ProgramLanguage.CSharp,
                ModelKind = ProgramSynthesisModelKind.CodeBERT,
                MaxSequenceLength = 16,
                VocabularySize = 1024,
                NumEncoderLayers = 1
            })
            .ConfigureProgramSynthesisServing(client: fakeClient);

        AiModelResult<double, Tensor<double>, Tensor<double>> result = await builder.BuildAsync();

        var execute = await result.ExecuteProgramAsync(new ProgramExecuteRequest
        {
            Language = ProgramLanguage.CSharp,
            SourceCode = "Console.WriteLine(1);"
        });

        Assert.True(execute.Success);
        Assert.Equal(ProgramLanguage.CSharp, execute.Language);
        Assert.Equal(1, fakeClient.ExecuteCalls);
    }

    private sealed class FakeServingClient : IProgramSynthesisServingClient
    {
        public int ExecuteCalls { get; private set; }

        public Task<CodeTaskResultBase> ExecuteCodeTaskAsync(CodeTaskRequestBase request, CancellationToken cancellationToken) =>
            Task.FromResult<CodeTaskResultBase>(new CodeSummarizationResult { Success = true, Summary = "ok" });

        public Task<ProgramExecuteResponse> ExecuteProgramAsync(ProgramExecuteRequest request, CancellationToken cancellationToken)
        {
            ExecuteCalls++;
            return Task.FromResult(new ProgramExecuteResponse
            {
                Success = true,
                Language = request.Language,
                ExitCode = 0
            });
        }

        public Task<ProgramEvaluateIoResponse> EvaluateProgramIoAsync(ProgramEvaluateIoRequest request, CancellationToken cancellationToken) =>
            Task.FromResult(new ProgramEvaluateIoResponse { Success = true, Language = request.Language });

        public Task<SqlExecuteResponse> ExecuteSqlAsync(SqlExecuteRequest request, CancellationToken cancellationToken) =>
            Task.FromResult(new SqlExecuteResponse { Success = true });
    }
}
