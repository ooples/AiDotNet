using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Sandboxing.Docker;
using AiDotNet.Serving.Security;
using Microsoft.Extensions.Options;
using AiDotNet.Validation;

namespace AiDotNet.Serving.Sandboxing.Execution;

public sealed class DockerProgramSandboxExecutor : IProgramSandboxExecutor
{
    private const string CompileBeginMarker = "AIDOTNET_COMPILE_BEGIN";
    private const string CompileEndMarker = "AIDOTNET_COMPILE_END";
    private const string RuntimeBeginMarker = "AIDOTNET_RUNTIME_BEGIN";

    private readonly ServingSandboxOptions _sandboxOptions;
    private readonly IDockerRunner _dockerRunner;
    private readonly ILogger<DockerProgramSandboxExecutor> _logger;
    private readonly SemaphoreSlim _freeConcurrency;
    private readonly SemaphoreSlim _premiumConcurrency;
    private readonly SemaphoreSlim _enterpriseConcurrency;

    public DockerProgramSandboxExecutor(
        IOptions<ServingSandboxOptions> sandboxOptions,
        IDockerRunner dockerRunner,
        ILogger<DockerProgramSandboxExecutor> logger)
    {
        Guard.NotNull(sandboxOptions);
        _sandboxOptions = sandboxOptions.Value;
        Guard.NotNull(dockerRunner);
        _dockerRunner = dockerRunner;
        Guard.NotNull(logger);
        _logger = logger;

        _freeConcurrency = new SemaphoreSlim(Math.Max(1, _sandboxOptions.Free.MaxConcurrentExecutions));
        _premiumConcurrency = new SemaphoreSlim(Math.Max(1, _sandboxOptions.Premium.MaxConcurrentExecutions));
        _enterpriseConcurrency = new SemaphoreSlim(Math.Max(1, _sandboxOptions.Enterprise.MaxConcurrentExecutions));
    }

    public async Task<ProgramExecuteResponse> ExecuteAsync(
        ProgramExecuteRequest request,
        ServingRequestContext requestContext,
        CancellationToken cancellationToken)
    {
        if (request is null) throw new ArgumentNullException(nameof(request));
        if (requestContext is null) throw new ArgumentNullException(nameof(requestContext));

        if (string.IsNullOrWhiteSpace(request.SourceCode))
        {
            return new ProgramExecuteResponse
            {
                Success = false,
                Language = request.Language,
                ExitCode = -1,
                Error = "SourceCode is required.",
                ErrorCode = ProgramExecuteErrorCode.SourceCodeRequired
            };
        }

        var allowed = request.AllowedLanguages?.Where(x => x != ProgramLanguage.Generic).Distinct().ToList()
            ?? new List<ProgramLanguage>();

        var limits = GetLimits(requestContext.Tier);

        if (request.SourceCode.Length > limits.MaxSourceCodeChars)
        {
            return new ProgramExecuteResponse
            {
                Success = false,
                Language = request.Language,
                ExitCode = -1,
                Error = "SourceCode exceeds tier limit.",
                ErrorCode = ProgramExecuteErrorCode.SourceCodeTooLarge
            };
        }

        if (request.StdIn is not null && request.StdIn.Length > limits.MaxStdInChars)
        {
            return new ProgramExecuteResponse
            {
                Success = false,
                Language = request.Language,
                ExitCode = -1,
                Error = "StdIn exceeds tier limit.",
                ErrorCode = ProgramExecuteErrorCode.StdInTooLarge
            };
        }

        var concurrency = GetConcurrencySemaphore(requestContext.Tier);
        await concurrency.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            var timeout = TimeSpan.FromSeconds(Math.Max(1, limits.TimeLimitSeconds));

            var resolvedLanguage = request.Language == ProgramLanguage.Generic
                ? ProgramLanguageDetector.Detect(request.SourceCode, allowed, request.PreferredLanguage)
                : request.Language;

            var languagesToTry = resolvedLanguage.HasValue
                ? new List<ProgramLanguage> { resolvedLanguage.Value }
                : request.Language == ProgramLanguage.Generic &&
                  requestContext.Tier == ServingTier.Enterprise &&
                  request.AllowUndetectedLanguageFallback &&
                  allowed.Count > 0
                    ? allowed
                    : new List<ProgramLanguage>();

            if (languagesToTry.Count == 0)
            {
                return new ProgramExecuteResponse
                {
                    Success = false,
                    Language = ProgramLanguage.Generic,
                    ExitCode = -1,
                    Error = "Language must be detected or specified.",
                    ErrorCode = ProgramExecuteErrorCode.LanguageNotDetected
                };
            }

            if (resolvedLanguage is null && request.Language == ProgramLanguage.Generic && request.AllowUndetectedLanguageFallback)
            {
                _logger.LogInformation(
                    "Enterprise undetected-language fallback enabled; attempting {Count} candidate languages.",
                    languagesToTry.Count);
            }

            foreach (var language in languagesToTry)
            {
                if (language == ProgramLanguage.SQL)
                {
                    return new ProgramExecuteResponse
                    {
                        Success = false,
                        Language = ProgramLanguage.SQL,
                        ExitCode = -1,
                        Error = "Use the SQL execution endpoint for ProgramLanguage.SQL.",
                        ErrorCode = ProgramExecuteErrorCode.SqlNotSupported
                    };
                }

                var response = await ExecuteLanguageAsync(
                        language,
                        request.SourceCode,
                        request.StdIn,
                        limits,
                        compileOnly: request.CompileOnly,
                        timeout,
                        cancellationToken)
                    .ConfigureAwait(false);

                if (response.Success)
                {
                    return response;
                }

                if (request.Language != ProgramLanguage.Generic || requestContext.Tier != ServingTier.Enterprise)
                {
                    return response;
                }
            }

            return new ProgramExecuteResponse
            {
                Success = false,
                Language = resolvedLanguage ?? ProgramLanguage.Generic,
                ExitCode = -1,
                Error = "Execution failed for all attempted languages.",
                ErrorCode = ProgramExecuteErrorCode.ExecutionFailed
            };
        }
        finally
        {
            concurrency.Release();
        }
    }

    private async Task<ProgramExecuteResponse> ExecuteLanguageAsync(
        ProgramLanguage language,
        string sourceCode,
        string? stdIn,
        ServingSandboxLimitOptions limits,
        bool compileOnly,
        TimeSpan timeout,
        CancellationToken cancellationToken)
    {
        var hostDir = CreateTempWorkspace();

        try
        {
            var (image, fileName, command) = GetDockerSpec(language, compileOnly: compileOnly);

            var sourcePath = Path.Combine(hostDir, fileName);
            await File.WriteAllTextAsync(sourcePath, sourceCode, cancellationToken).ConfigureAwait(false);

            var mount = $"--mount type=bind,source=\"{hostDir}\",target=/workspace,readonly";
            var stdinFlag = stdIn is null ? string.Empty : "-i ";
            var dockerArgs =
                $"run --rm {stdinFlag}--network none --memory {Math.Max(32, limits.MemoryLimitMb)}m --cpus {Math.Max(0.1, limits.CpuLimit)} {mount} {image} sh -c \"{command}\"";

            var result = await _dockerRunner.RunAsync(
                dockerArgs,
                stdIn,
                timeout,
                maxStdOutChars: Math.Max(0, limits.MaxStdOutChars),
                maxStdErrChars: Math.Max(0, limits.MaxStdErrChars),
                cancellationToken).ConfigureAwait(false);

            var compilationAttempted = IsCompilationLanguage(language);
            var compilationSucceeded = (bool?)null;
            var diagnostics = new List<CompilationDiagnostic>();
            var stdOut = result.StdOut ?? string.Empty;

            if (compilationAttempted)
            {
                var compilePayload = ExtractBetweenMarkers(stdOut, CompileBeginMarker, CompileEndMarker);
                var runtimeStdOut = ExtractAfterMarker(stdOut, RuntimeBeginMarker);
                compilationSucceeded = runtimeStdOut is not null;
                diagnostics = CompilationDiagnosticParser.Parse(language, compilePayload);
                stdOut = runtimeStdOut ?? string.Empty;
            }

            return new ProgramExecuteResponse
            {
                Success = result.ExitCode == 0,
                Language = language,
                CompilationAttempted = compilationAttempted,
                CompilationSucceeded = compilationSucceeded,
                CompilationDiagnostics = diagnostics,
                ExitCode = result.ExitCode,
                StdOut = stdOut,
                StdErr = result.StdErr ?? string.Empty,
                StdOutTruncated = result.StdOutTruncated,
                StdErrTruncated = result.StdErrTruncated,
                Error = result.ExitCode == 0
                    ? null
                    : compilationAttempted && compilationSucceeded == false
                        ? "Compilation failed."
                        : "Execution failed.",
                ErrorCode = result.ExitCode == 0
                    ? null
                    : compilationAttempted && compilationSucceeded == false
                        ? ProgramExecuteErrorCode.CompilationFailed
                        : ProgramExecuteErrorCode.ExecutionFailed
            };
        }
        catch (OperationCanceledException)
        {
            return new ProgramExecuteResponse
            {
                Success = false,
                Language = language,
                ExitCode = -1,
                Error = "Execution timed out or was canceled.",
                ErrorCode = ProgramExecuteErrorCode.TimeoutOrCanceled
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Execution failed (Language={Language}).", language);
            return new ProgramExecuteResponse
            {
                Success = false,
                Language = language,
                ExitCode = -1,
                Error = "Execution failed.",
                ErrorCode = ProgramExecuteErrorCode.ExecutionFailed
            };
        }
        finally
        {
            TryDeleteDirectory(hostDir);
        }
    }

    private static (string Image, string FileName, string Command) GetDockerSpec(ProgramLanguage language, bool compileOnly)
    {
        return language switch
        {
            ProgramLanguage.Python => ("python:3.12-alpine", "main.py", "python /workspace/main.py"),
            ProgramLanguage.JavaScript => ("node:20-alpine", "main.js", "node /workspace/main.js"),
            ProgramLanguage.TypeScript => ("denoland/deno:alpine", "main.ts", "deno run --no-prompt /workspace/main.ts"),
            ProgramLanguage.C => ("gcc:14", "main.c",
                $"gcc /workspace/main.c -O2 -o /tmp/a.out -fdiagnostics-format=json 2> /tmp/compile.json; status=$?; " +
                $"echo {CompileBeginMarker}; cat /tmp/compile.json; echo {CompileEndMarker}; " +
                "if [ $status -ne 0 ]; then exit $status; fi; " +
                (compileOnly
                    ? $"echo {RuntimeBeginMarker}; exit 0"
                    : $"echo {RuntimeBeginMarker}; /tmp/a.out")),
            ProgramLanguage.CPlusPlus => ("gcc:14", "main.cpp",
                $"g++ /workspace/main.cpp -O2 -std=c++17 -o /tmp/a.out -fdiagnostics-format=json 2> /tmp/compile.json; status=$?; " +
                $"echo {CompileBeginMarker}; cat /tmp/compile.json; echo {CompileEndMarker}; " +
                "if [ $status -ne 0 ]; then exit $status; fi; " +
                (compileOnly
                    ? $"echo {RuntimeBeginMarker}; exit 0"
                    : $"echo {RuntimeBeginMarker}; /tmp/a.out")),
            ProgramLanguage.Go => ("golang:1.23", "main.go", "go run /workspace/main.go"),
            ProgramLanguage.Rust => ("rust:1.83", "main.rs",
                $"rustc /workspace/main.rs -O -o /tmp/a.out --error-format=json 2> /tmp/compile.json; status=$?; " +
                $"echo {CompileBeginMarker}; cat /tmp/compile.json; echo {CompileEndMarker}; " +
                "if [ $status -ne 0 ]; then exit $status; fi; " +
                (compileOnly
                    ? $"echo {RuntimeBeginMarker}; exit 0"
                    : $"echo {RuntimeBeginMarker}; /tmp/a.out")),
            ProgramLanguage.Java => ("eclipse-temurin:21-jdk", "Main.java",
                "javac /workspace/Main.java -d /tmp 2> /tmp/compile.txt; status=$?; " +
                $"echo {CompileBeginMarker}; cat /tmp/compile.txt; echo {CompileEndMarker}; " +
                "if [ $status -ne 0 ]; then exit $status; fi; " +
                (compileOnly
                    ? $"echo {RuntimeBeginMarker}; exit 0"
                    : $"echo {RuntimeBeginMarker}; java -cp /tmp Main")),
            ProgramLanguage.CSharp => ("mcr.microsoft.com/dotnet/sdk:8.0", "Program.cs",
                "DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 " +
                "dotnet new console -n App -o /tmp/app -f net8.0 > /dev/null 2>&1; " +
                "cp /workspace/Program.cs /tmp/app/Program.cs; " +
                "dotnet build /tmp/app/App.csproj -c Release -p:RestoreIgnoreFailedSources=true > /tmp/compile.txt 2>&1; status=$?; " +
                $"echo {CompileBeginMarker}; cat /tmp/compile.txt; echo {CompileEndMarker}; " +
                "if [ $status -ne 0 ]; then exit $status; fi; " +
                (compileOnly
                    ? $"echo {RuntimeBeginMarker}; exit 0"
                    : $"echo {RuntimeBeginMarker}; dotnet run --no-build --project /tmp/app/App.csproj -c Release")),
            _ => throw new InvalidOperationException($"Unsupported language: {language}")
        };
    }

    private ServingSandboxLimitOptions GetLimits(ServingTier tier) =>
        tier switch
        {
            ServingTier.Premium => _sandboxOptions.Premium,
            ServingTier.Enterprise => _sandboxOptions.Enterprise,
            _ => _sandboxOptions.Free
        };

    private SemaphoreSlim GetConcurrencySemaphore(ServingTier tier) =>
        tier switch
        {
            ServingTier.Premium => _premiumConcurrency,
            ServingTier.Enterprise => _enterpriseConcurrency,
            _ => _freeConcurrency
        };

    private static string CreateTempWorkspace()
    {
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-sandbox", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void TryDeleteDirectory(string path)
    {
        try
        {
            if (Directory.Exists(path))
            {
                Directory.Delete(path, recursive: true);
            }
        }
        catch
        {
            // Best-effort cleanup.
        }
    }

    private static bool IsCompilationLanguage(ProgramLanguage language) =>
        language is ProgramLanguage.C or ProgramLanguage.CPlusPlus or ProgramLanguage.Rust or ProgramLanguage.Java or ProgramLanguage.CSharp;

    private static string ExtractBetweenMarkers(string stdOut, string beginMarker, string endMarker)
    {
        if (string.IsNullOrEmpty(stdOut))
        {
            return string.Empty;
        }

        var beginIndex = stdOut.IndexOf(beginMarker, StringComparison.Ordinal);
        if (beginIndex < 0)
        {
            return string.Empty;
        }

        beginIndex += beginMarker.Length;
        beginIndex = SkipNewLines(stdOut, beginIndex);

        var endIndex = stdOut.IndexOf(endMarker, beginIndex, StringComparison.Ordinal);
        if (endIndex < 0)
        {
            return string.Empty;
        }

        return stdOut.Substring(beginIndex, endIndex - beginIndex).Trim('\r', '\n');
    }

    private static string? ExtractAfterMarker(string stdOut, string marker)
    {
        if (string.IsNullOrEmpty(stdOut))
        {
            return null;
        }

        var markerIndex = stdOut.IndexOf(marker, StringComparison.Ordinal);
        if (markerIndex < 0)
        {
            return null;
        }

        markerIndex += marker.Length;
        markerIndex = SkipNewLines(stdOut, markerIndex);

        return markerIndex >= stdOut.Length ? string.Empty : stdOut.Substring(markerIndex);
    }

    private static int SkipNewLines(string value, int startIndex)
    {
        var index = startIndex;
        while (index < value.Length && (value[index] == '\r' || value[index] == '\n'))
        {
            index++;
        }

        return index;
    }
}
