using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Sandboxing.Docker;
using AiDotNet.Serving.Sandboxing.Execution;
using AiDotNet.Serving.Security;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests.Sandboxing;

public sealed class DockerProgramSandboxExecutorTests
{
    private const string CompileBeginMarker = "AIDOTNET_COMPILE_BEGIN";
    private const string CompileEndMarker = "AIDOTNET_COMPILE_END";
    private const string RuntimeBeginMarker = "AIDOTNET_RUNTIME_BEGIN";

    [Fact]
    public async Task ExecuteAsync_SourceRequired_ReturnsError()
    {
        var executor = CreateExecutor(new FakeDockerRunner(_ => new DockerCommandResult { ExitCode = 0, StdOut = string.Empty, StdErr = string.Empty }));

        var response = await executor.ExecuteAsync(
            new ProgramExecuteRequest { Language = ProgramLanguage.CSharp, SourceCode = "   " },
            new ServingRequestContext { Tier = ServingTier.Free, IsAuthenticated = false },
            CancellationToken.None);

        Assert.False(response.Success);
        Assert.Equal(ProgramExecuteErrorCode.SourceCodeRequired, response.ErrorCode);
    }

    [Fact]
    public async Task ExecuteAsync_CompileSuccess_ExtractsStdOutAndCompilationInfo()
    {
        var docker = new FakeDockerRunner(_ => new DockerCommandResult
        {
            ExitCode = 0,
            StdOut = $"{CompileBeginMarker}\nBuild succeeded.\n{CompileEndMarker}\n{RuntimeBeginMarker}\nhello",
            StdErr = string.Empty
        });

        var executor = CreateExecutor(docker);

        var response = await executor.ExecuteAsync(
            new ProgramExecuteRequest { Language = ProgramLanguage.CSharp, SourceCode = "class C { static void Main(){} }" },
            new ServingRequestContext { Tier = ServingTier.Free, IsAuthenticated = false },
            CancellationToken.None);

        Assert.True(response.Success);
        Assert.True(response.CompilationAttempted);
        Assert.True(response.CompilationSucceeded);
        Assert.Equal("hello", response.StdOut.Trim());
        Assert.NotEmpty(docker.Arguments);
    }

    [Fact]
    public async Task ExecuteAsync_CompileFailure_ReturnsCompilationFailed()
    {
        var docker = new FakeDockerRunner(_ => new DockerCommandResult
        {
            ExitCode = 1,
            StdOut = $"{CompileBeginMarker}\nerror\n{CompileEndMarker}\n",
            StdErr = string.Empty
        });

        var executor = CreateExecutor(docker);

        var response = await executor.ExecuteAsync(
            new ProgramExecuteRequest { Language = ProgramLanguage.CSharp, SourceCode = "bad" },
            new ServingRequestContext { Tier = ServingTier.Free, IsAuthenticated = false },
            CancellationToken.None);

        Assert.False(response.Success);
        Assert.True(response.CompilationAttempted);
        Assert.False(response.CompilationSucceeded);
        Assert.Equal(ProgramExecuteErrorCode.CompilationFailed, response.ErrorCode);
    }

    private static DockerProgramSandboxExecutor CreateExecutor(IDockerRunner dockerRunner)
    {
        var sandboxOptions = new ServingSandboxOptions
        {
            Free = new ServingSandboxLimitOptions
            {
                MaxConcurrentExecutions = 1,
                MaxSourceCodeChars = 100_000,
                MaxStdInChars = 10_000,
                TimeLimitSeconds = 2,
                MemoryLimitMb = 64,
                CpuLimit = 1,
                MaxStdOutChars = 10_000,
                MaxStdErrChars = 10_000
            },
            Premium = new ServingSandboxLimitOptions(),
            Enterprise = new ServingSandboxLimitOptions()
        };

        return new DockerProgramSandboxExecutor(
            Options.Create(sandboxOptions),
            dockerRunner,
            NullLogger<DockerProgramSandboxExecutor>.Instance);
    }

    private sealed class FakeDockerRunner : IDockerRunner
    {
        private readonly Func<string, DockerCommandResult> _handler;

        public FakeDockerRunner(Func<string, DockerCommandResult> handler)
        {
            _handler = handler;
        }

        public List<string> Arguments { get; } = new();

        public Task<DockerCommandResult> RunAsync(
            string arguments,
            string? stdIn,
            TimeSpan timeout,
            int maxStdOutChars,
            int maxStdErrChars,
            CancellationToken cancellationToken)
        {
            Arguments.Add(arguments);
            return Task.FromResult(_handler(arguments));
        }
    }
}
