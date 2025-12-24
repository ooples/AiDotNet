using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Serving.Controllers.ProgramSynthesis;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Sandboxing.Execution;
using AiDotNet.Serving.Security;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace AiDotNet.Serving.Tests.ProgramSynthesis;

public sealed class ProgramControllerTests
{
    [Fact]
    public async Task Execute_NullRequest_ReturnsBadRequest()
    {
        var controller = CreateController(
            executor: new FakeProgramSandboxExecutor(_ => Task.FromResult(new ProgramExecuteResponse { Success = true, Language = ProgramLanguage.Generic, ExitCode = 0 })),
            evaluator: new FakeServingProgramEvaluator(_ => Task.FromResult(new ProgramEvaluateIoResponse { Success = true, Language = ProgramLanguage.CSharp })));

        var result = await controller.Execute(null!, CancellationToken.None);

        var badRequest = Assert.IsType<BadRequestObjectResult>(result);
        var payload = Assert.IsType<ProgramExecuteResponse>(badRequest.Value);
        Assert.False(payload.Success);
        Assert.Equal(ProgramExecuteErrorCode.InvalidRequest, payload.ErrorCode);
    }

    [Fact]
    public async Task Execute_EmptySourceCode_ReturnsBadRequest()
    {
        var controller = CreateController(
            executor: new FakeProgramSandboxExecutor(_ => Task.FromResult(new ProgramExecuteResponse { Success = true, Language = ProgramLanguage.Generic, ExitCode = 0 })),
            evaluator: new FakeServingProgramEvaluator(_ => Task.FromResult(new ProgramEvaluateIoResponse { Success = true, Language = ProgramLanguage.CSharp })));

        var result = await controller.Execute(new ProgramExecuteRequest { SourceCode = "   " }, CancellationToken.None);

        var badRequest = Assert.IsType<BadRequestObjectResult>(result);
        var payload = Assert.IsType<ProgramExecuteResponse>(badRequest.Value);
        Assert.False(payload.Success);
        Assert.Equal(ProgramExecuteErrorCode.SourceCodeRequired, payload.ErrorCode);
    }

    [Fact]
    public async Task Execute_Success_ReturnsOk()
    {
        var controller = CreateController(
            executor: new FakeProgramSandboxExecutor(_ => Task.FromResult(new ProgramExecuteResponse
            {
                Success = true,
                Language = ProgramLanguage.CSharp,
                ExitCode = 0,
                StdOut = "hello"
            })),
            evaluator: new FakeServingProgramEvaluator(_ => Task.FromResult(new ProgramEvaluateIoResponse { Success = true, Language = ProgramLanguage.CSharp })));

        var result = await controller.Execute(new ProgramExecuteRequest { Language = ProgramLanguage.CSharp, SourceCode = "Console.WriteLine(\"hi\");" }, CancellationToken.None);

        var ok = Assert.IsType<OkObjectResult>(result);
        var payload = Assert.IsType<ProgramExecuteResponse>(ok.Value);
        Assert.True(payload.Success);
        Assert.Equal(ProgramLanguage.CSharp, payload.Language);
        Assert.Equal("hello", payload.StdOut);
    }

    [Fact]
    public async Task Execute_Timeout_Returns408()
    {
        var controller = CreateController(
            executor: new FakeProgramSandboxExecutor(_ => Task.FromResult(new ProgramExecuteResponse
            {
                Success = false,
                Language = ProgramLanguage.CSharp,
                ExitCode = -1,
                ErrorCode = ProgramExecuteErrorCode.TimeoutOrCanceled
            })),
            evaluator: new FakeServingProgramEvaluator(_ => Task.FromResult(new ProgramEvaluateIoResponse { Success = true, Language = ProgramLanguage.CSharp })));

        var result = await controller.Execute(new ProgramExecuteRequest { Language = ProgramLanguage.CSharp, SourceCode = "x" }, CancellationToken.None);

        var objectResult = Assert.IsType<ObjectResult>(result);
        Assert.Equal(408, objectResult.StatusCode);
        var payload = Assert.IsType<ProgramExecuteResponse>(objectResult.Value);
        Assert.Equal(ProgramExecuteErrorCode.TimeoutOrCanceled, payload.ErrorCode);
    }

    [Fact]
    public async Task EvaluateIo_NullRequest_ReturnsBadRequest()
    {
        var controller = CreateController(
            executor: new FakeProgramSandboxExecutor(_ => Task.FromResult(new ProgramExecuteResponse { Success = true, Language = ProgramLanguage.Generic, ExitCode = 0 })),
            evaluator: new FakeServingProgramEvaluator(_ => Task.FromResult(new ProgramEvaluateIoResponse { Success = true, Language = ProgramLanguage.CSharp })));

        var result = await controller.EvaluateIo(null!, CancellationToken.None);

        var badRequest = Assert.IsType<BadRequestObjectResult>(result);
        var payload = Assert.IsType<ProgramEvaluateIoResponse>(badRequest.Value);
        Assert.False(payload.Success);
    }

    [Fact]
    public async Task EvaluateIo_Success_ReturnsOk()
    {
        var controller = CreateController(
            executor: new FakeProgramSandboxExecutor(_ => Task.FromResult(new ProgramExecuteResponse { Success = true, Language = ProgramLanguage.Generic, ExitCode = 0 })),
            evaluator: new FakeServingProgramEvaluator(_ => Task.FromResult(new ProgramEvaluateIoResponse
            {
                Success = true,
                Language = ProgramLanguage.CSharp,
                TestResults = new List<ProgramEvaluateIoTestResult>()
            })));

        var result = await controller.EvaluateIo(new ProgramEvaluateIoRequest { Language = ProgramLanguage.CSharp, SourceCode = "x", TestCases = new List<ProgramInputOutputExample>() }, CancellationToken.None);

        var ok = Assert.IsType<OkObjectResult>(result);
        var payload = Assert.IsType<ProgramEvaluateIoResponse>(ok.Value);
        Assert.True(payload.Success);
    }

    [Fact]
    public async Task EvaluateIo_Timeout_Returns408()
    {
        var controller = CreateController(
            executor: new FakeProgramSandboxExecutor(_ => Task.FromResult(new ProgramExecuteResponse { Success = true, Language = ProgramLanguage.Generic, ExitCode = 0 })),
            evaluator: new FakeServingProgramEvaluator(_ => throw new OperationCanceledException()));

        var result = await controller.EvaluateIo(new ProgramEvaluateIoRequest { Language = ProgramLanguage.CSharp, SourceCode = "x", TestCases = new List<ProgramInputOutputExample>() }, CancellationToken.None);

        var objectResult = Assert.IsType<ObjectResult>(result);
        Assert.Equal(408, objectResult.StatusCode);
        var payload = Assert.IsType<ProgramEvaluateIoResponse>(objectResult.Value);
        Assert.False(payload.Success);
    }

    private static ProgramController CreateController(IProgramSandboxExecutor executor, IServingProgramEvaluator evaluator)
    {
        var redactor = new ServingProgramExecuteResponseRedactor();
        var evaluateIoRedactor = new ServingProgramEvaluateIoResponseRedactor(redactor);
        var accessor = new ServingRequestContextAccessor { Current = null };
        var logger = NullLogger<ProgramController>.Instance;

        return new ProgramController(
            executor,
            evaluator,
            redactor,
            evaluateIoRedactor,
            accessor,
            logger);
    }

    private sealed class FakeProgramSandboxExecutor : IProgramSandboxExecutor
    {
        private readonly Func<ProgramExecuteRequest, Task<ProgramExecuteResponse>> _handler;

        public FakeProgramSandboxExecutor(Func<ProgramExecuteRequest, Task<ProgramExecuteResponse>> handler)
        {
            _handler = handler;
        }

        public Task<ProgramExecuteResponse> ExecuteAsync(ProgramExecuteRequest request, ServingRequestContext requestContext, CancellationToken cancellationToken)
        {
            return _handler(request);
        }
    }

    private sealed class FakeServingProgramEvaluator : IServingProgramEvaluator
    {
        private readonly Func<ProgramEvaluateIoRequest, Task<ProgramEvaluateIoResponse>> _handler;

        public FakeServingProgramEvaluator(Func<ProgramEvaluateIoRequest, Task<ProgramEvaluateIoResponse>> handler)
        {
            _handler = handler;
        }

        public Task<ProgramEvaluateIoResponse> EvaluateIoAsync(ProgramEvaluateIoRequest request, ServingRequestContext requestContext, CancellationToken cancellationToken)
        {
            return _handler(request);
        }
    }
}
