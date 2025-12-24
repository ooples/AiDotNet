using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Security;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace AiDotNet.Serving.Tests.ProgramSynthesis;

public sealed class ServingCodeTaskExecutorTests
{
    [Fact]
    public async Task ExecuteAsync_Summarization_ReturnsSuccessfulResult()
    {
        var executor = new ServingCodeTaskExecutor(NullLogger<ServingCodeTaskExecutor>.Instance);

        var result = await executor.ExecuteAsync(
            new CodeSummarizationRequest { Code = "public class C { }", Language = ProgramLanguage.CSharp },
            new ServingRequestContext { Tier = ServingTier.Free, IsAuthenticated = false },
            CancellationToken.None);

        var typed = Assert.IsType<CodeSummarizationResult>(result);
        Assert.True(typed.Success);
        Assert.Equal(ProgramLanguage.CSharp, typed.Language);
        Assert.False(string.IsNullOrWhiteSpace(typed.Summary));
    }

    [Theory]
    [InlineData(CodeTask.Summarization, typeof(CodeSummarizationResult))]
    [InlineData(CodeTask.BugFixing, typeof(CodeBugFixingResult))]
    [InlineData(CodeTask.CodeReview, typeof(CodeReviewResult))]
    public void CreateFailureResult_MapsTaskToConcreteResult(CodeTask task, Type expectedType)
    {
        var method = typeof(ServingCodeTaskExecutor).GetMethod(
            "CreateFailureResult",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);

        Assert.NotNull(method);

        var result = (CodeTaskResultBase)method!.Invoke(null, new object?[]
        {
            task,
            ProgramLanguage.CSharp,
            "rid",
            "err"
        })!;

        Assert.IsType(expectedType, result);
        Assert.False(result.Success);
        Assert.Equal("err", result.Error);
        Assert.Equal("rid", result.RequestId);
        Assert.Equal(ProgramLanguage.CSharp, result.Language);
    }

    [Fact]
    public async Task ExecuteAsync_Canceled_Throws()
    {
        var executor = new ServingCodeTaskExecutor(NullLogger<ServingCodeTaskExecutor>.Instance);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            executor.ExecuteAsync(
                new CodeSummarizationRequest { Code = "x", Language = ProgramLanguage.CSharp },
                new ServingRequestContext { Tier = ServingTier.Free, IsAuthenticated = false },
                cts.Token));
    }

    [Fact]
    public async Task ExecuteAsync_NullRequest_Throws()
    {
        var executor = new ServingCodeTaskExecutor(NullLogger<ServingCodeTaskExecutor>.Instance);

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            executor.ExecuteAsync(
                null!,
                new ServingRequestContext { Tier = ServingTier.Free, IsAuthenticated = false },
                CancellationToken.None));
    }
}
