using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Controllers.ProgramSynthesis;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Security;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests.ProgramSynthesis;

public sealed class CodeTasksControllerTests
{
    [Fact]
    public async Task Completion_NullRequest_ReturnsBadRequest()
    {
        var controller = CreateController(
            validator: new FakeValidator((CodeTaskRequestBase _, ServingRequestContext _, out string error) =>
            {
                error = "invalid";
                return false;
            }));

        var result = await controller.Completion(null!, CancellationToken.None);

        var badRequest = Assert.IsType<BadRequestObjectResult>(result);
        var payload = Assert.IsType<CodeCompletionResult>(badRequest.Value);
        Assert.False(payload.Success);
        Assert.Equal("Request body is required.", payload.Error);
    }

    [Fact]
    public async Task Completion_InvalidRequest_ReturnsBadRequest()
    {
        var controller = CreateController(
            validator: new FakeValidator((CodeTaskRequestBase _, ServingRequestContext _, out string error) =>
            {
                error = "bad";
                return false;
            }));

        var result = await controller.Completion(new CodeCompletionRequest { Language = ProgramLanguage.CSharp, Code = "x" }, CancellationToken.None);

        var badRequest = Assert.IsType<BadRequestObjectResult>(result);
        var payload = Assert.IsType<CodeCompletionResult>(badRequest.Value);
        Assert.False(payload.Success);
        Assert.Equal("bad", payload.Error);
    }

    [Fact]
    public async Task Completion_Timeout_ReturnsOk_WithFailureEnvelope()
    {
        var controller = CreateController(
            options: new ServingProgramSynthesisOptions
            {
                Free = new ServingProgramSynthesisLimitOptions
                {
                    MaxTaskTimeSeconds = 0,
                    MaxListItems = 10,
                    MaxResultChars = 200
                }
            },
            executor: new FakeExecutor(async (_, _, ct) =>
            {
                await Task.Delay(Timeout.InfiniteTimeSpan, ct);
                return new CodeCompletionResult();
            }));

        var result = await controller.Completion(new CodeCompletionRequest { Language = ProgramLanguage.CSharp, Code = "x" }, CancellationToken.None);

        var ok = Assert.IsType<OkObjectResult>(result);
        var payload = Assert.IsType<CodeCompletionResult>(ok.Value);
        Assert.False(payload.Success);
        Assert.NotNull(payload.Error);
    }

    [Fact]
    public async Task Completion_ExecutorThrows_Returns500()
    {
        var controller = CreateController(
            executor: new FakeExecutor((_, _, _) => throw new InvalidOperationException("boom")));

        var result = await controller.Completion(new CodeCompletionRequest { Language = ProgramLanguage.CSharp, Code = "x" }, CancellationToken.None);

        var objectResult = Assert.IsType<ObjectResult>(result);
        Assert.Equal(500, objectResult.StatusCode);
        var payload = Assert.IsType<CodeCompletionResult>(objectResult.Value);
        Assert.False(payload.Success);
        Assert.Equal("Unhandled task execution error.", payload.Error);
    }

    [Fact]
    public async Task Completion_Success_ReturnsOk()
    {
        var controller = CreateController(
            executor: new FakeExecutor((_, _, _) => Task.FromResult<CodeTaskResultBase>(new CodeCompletionResult
            {
                Language = ProgramLanguage.CSharp,
                Success = true,
                Candidates = new List<CodeCompletionCandidate> { new() { CompletionText = "ok" } }
            })));

        var result = await controller.Completion(new CodeCompletionRequest { Language = ProgramLanguage.CSharp, Code = "x" }, CancellationToken.None);

        var ok = Assert.IsType<OkObjectResult>(result);
        var payload = Assert.IsType<CodeCompletionResult>(ok.Value);
        Assert.True(payload.Success);
        Assert.Single(payload.Candidates);
    }

    private static CodeTasksController CreateController(
        ServingProgramSynthesisOptions? options = null,
        IServingCodeTaskExecutor? executor = null,
        IServingCodeTaskRequestValidator? validator = null,
        IServingCodeTaskResultRedactor? redactor = null,
        IServingRequestContextAccessor? requestContextAccessor = null,
        IServingProgramSynthesisConcurrencyLimiter? concurrencyLimiter = null)
    {
        var effectiveOptions = options ?? new ServingProgramSynthesisOptions
        {
            Free = new ServingProgramSynthesisLimitOptions
            {
                MaxTaskTimeSeconds = 1,
                MaxListItems = 10,
                MaxResultChars = 200
            }
        };

        return new CodeTasksController(
            executor ?? new FakeExecutor((request, _, _) => Task.FromResult<CodeTaskResultBase>(new CodeCompletionResult
            {
                Language = request.Language,
                Success = true
            })),
            validator ?? new FakeValidator((CodeTaskRequestBase _, ServingRequestContext _, out string error) =>
            {
                error = string.Empty;
                return true;
            }),
            redactor ?? new PassThroughRedactor(),
            requestContextAccessor ?? new FakeRequestContextAccessor(),
            concurrencyLimiter ?? new FakeConcurrencyLimiter(),
            Options.Create(effectiveOptions),
            NullLogger<CodeTasksController>.Instance);
    }

    private sealed class FakeRequestContextAccessor : IServingRequestContextAccessor
    {
        public ServingRequestContext? Current { get; set; } = new()
        {
            Tier = ServingTier.Free,
            IsAuthenticated = false
        };
    }

    private sealed class PassThroughRedactor : IServingCodeTaskResultRedactor
    {
        public CodeTaskResultBase Redact(CodeTaskResultBase result, ServingRequestContext requestContext) => result;
    }

    private sealed class FakeConcurrencyLimiter : IServingProgramSynthesisConcurrencyLimiter
    {
        public Task<IDisposable> AcquireAsync(ServingTier tier, CancellationToken cancellationToken = default) =>
            Task.FromResult<IDisposable>(new ReleaseHandle());

        private sealed class ReleaseHandle : IDisposable
        {
            public void Dispose()
            {
            }
        }
    }

    private sealed class FakeExecutor : IServingCodeTaskExecutor
    {
        private readonly Func<CodeTaskRequestBase, ServingRequestContext, CancellationToken, Task<CodeTaskResultBase>> _handler;

        public FakeExecutor(Func<CodeTaskRequestBase, ServingRequestContext, CancellationToken, Task<CodeTaskResultBase>> handler)
        {
            _handler = handler;
        }

        public Task<CodeTaskResultBase> ExecuteAsync(CodeTaskRequestBase request, ServingRequestContext requestContext, CancellationToken cancellationToken) =>
            _handler(request, requestContext, cancellationToken);
    }

    private sealed class FakeValidator : IServingCodeTaskRequestValidator
    {
        private readonly TryValidateDelegate _handler;

        public FakeValidator(TryValidateDelegate handler)
        {
            _handler = handler;
        }

        public bool TryValidate(CodeTaskRequestBase request, ServingRequestContext requestContext, out string error) =>
            _handler(request, requestContext, out error);

        internal delegate bool TryValidateDelegate(CodeTaskRequestBase request, ServingRequestContext requestContext, out string error);
    }
}
