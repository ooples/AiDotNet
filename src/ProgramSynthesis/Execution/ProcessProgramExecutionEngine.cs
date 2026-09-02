using System.Diagnostics;
using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ProgramSynthesis.Execution;

/// <summary>Runs an untrusted candidate program in an isolated, resource-limited child process.</summary>
/// <remarks>
/// <para>
/// This is the execution boundary the core library ships: it needs no container runtime, no network, and no extra
/// package, only an interpreter or compiler already present on the machine and named by
/// <see cref="ProgramSandboxOptions.Interpreters"/>. Each execution writes exactly one source file into a fresh
/// workspace directory, launches the configured command with <c>UseShellExecute</c> off and all three standard
/// streams redirected, and deletes the workspace afterwards — including after a timeout, a kill, or a crash.
/// </para>
/// <para>
/// Five limits are enforced rather than advertised. The child inherits a pinned <c>PATH</c> and a scrubbed
/// environment, so it cannot resolve an executable from an attacker-controlled directory and cannot read the API
/// keys, tokens, or connection strings held in the host's environment. Standard output and standard error are
/// drained concurrently into fixed-size buffers, so a program that prints without stopping is truncated and
/// reported as truncated instead of exhausting host memory, and it never blocks on a full pipe. A wall-clock limit
/// cancels the run and kills the whole process tree — <c>Kill(entireProcessTree)</c> where the framework offers it,
/// with a <c>taskkill /T /F</c> fallback on Windows — so a detached grandchild cannot outlive its parent. Memory is
/// capped by a Windows job object with a kill-on-close limit, or by <c>ulimit -v</c> inside a POSIX shell;
/// <see cref="CanEnforceMemoryLimit"/> reports honestly when neither mechanism is available on this machine. A
/// semaphore bounds how many executions run at once, and the wall-clock limit starts only after that semaphore is
/// acquired, so a queued candidate is never charged for time it spent waiting.
/// </para>
/// <para>
/// Ordinary failure is never an exception. A program that will not compile, exits non-zero, times out, prints too
/// much, or names a language with no configured interpreter comes back as a <see cref="ProgramExecuteResponse"/>
/// carrying the matching <see cref="ProgramExecuteErrorCode"/>, because in an evolution run these outcomes are the
/// common case and must simply score badly. A request that sets
/// <see cref="ProgramExecuteRequest.CompileOnly"/> for a language with no separate compile command is refused
/// outright rather than executed, since running code that the caller asked only to type-check is precisely the
/// mistake this class exists to prevent.
/// </para>
/// <para><b>For Beginners:</b> When a language model writes a program, the only way to find out whether it works is
/// to run it — and that is exactly the dangerous part, because nobody reviewed the code. This class runs it the
/// careful way: in a separate process, in a throwaway folder, with a stopwatch, a memory ceiling, a limit on how
/// much it may print, and with your environment variables hidden from it. If the program hangs, it and everything
/// it started are killed. If it misbehaves, you get a result object describing what went wrong instead of an
/// exception, so an evolution run simply scores that candidate poorly and carries on.</para>
/// </remarks>
public sealed class ProcessProgramExecutionEngine : IProgramExecutionEngine, IDisposable
{
    private const int DrainTimeoutMilliseconds = 2000;
    private const int TaskKillTimeoutMilliseconds = 5000;

    private static readonly bool WindowsHost = DetectWindowsHost();
    private const string PosixShellPath = "/bin/sh";

    private readonly ProgramSandboxOptions _options;
    private readonly ProgramSandboxLimitOptions _limits;
    private readonly SemaphoreSlim _concurrency;
    private readonly string _pinnedPath;
    private readonly string _workspaceRoot;
    private bool _disposed;

    /// <summary>Initializes an engine bound to a validated copy of the supplied sandbox options.</summary>
    /// <param name="options">The sandbox mode, resource limits, and per-language commands to use.</param>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// The options are invalid, or <see cref="ProgramSandboxOptions.Mode"/> is not
    /// <see cref="ProgramSandboxMode.OutOfProcessWorker"/>. This engine implements only the out-of-process boundary;
    /// selecting another mode and then using this engine anyway would quietly contradict the configuration.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">A configured limit is impossible.</exception>
    public ProcessProgramExecutionEngine(ProgramSandboxOptions options)
    {
        Guard.NotNull(options);
        options.Validate();
        if (options.Mode != ProgramSandboxMode.OutOfProcessWorker)
        {
            throw new ArgumentException(
                $"ProcessProgramExecutionEngine implements {ProgramSandboxMode.OutOfProcessWorker} only, but the options select {options.Mode}.",
                nameof(options));
        }

        _options = options.Clone();
        _limits = _options.Limits;
        _concurrency = new SemaphoreSlim(_limits.MaxConcurrentExecutions, _limits.MaxConcurrentExecutions);
        _pinnedPath = BuildPinnedPath();
        _workspaceRoot = string.IsNullOrWhiteSpace(_options.WorkingDirectory)
            ? Path.Combine(Path.GetTempPath(), "aidotnet-program-sandbox")
            : _options.WorkingDirectory ?? Path.GetTempPath();
    }

    /// <summary>Gets whether this machine can enforce the configured memory limit on a sandboxed child.</summary>
    /// <remarks>
    /// <c>true</c> on Windows, where a job object applies the cap, and on any system carrying a POSIX shell at
    /// <c>/bin/sh</c>, where <c>ulimit -v</c> does. When this is <c>false</c> the wall-clock limit and the output
    /// caps still apply, but memory is not bounded; the property exists so a caller can say so rather than assume a
    /// protection that is not there.
    /// </remarks>
    public bool CanEnforceMemoryLimit => WindowsHost || File.Exists(PosixShellPath);

    /// <summary>Gets the sandbox options this engine was constructed with.</summary>
    /// <returns>An independent copy, so inspecting the configuration cannot change a running engine.</returns>
    public ProgramSandboxOptions GetOptions() => _options.Clone();

    /// <inheritdoc/>
    public async Task<ProgramExecuteResponse> ExecuteAsync(
        ProgramExecuteRequest request,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(request);
        if (_disposed) throw new ObjectDisposedException(nameof(ProcessProgramExecutionEngine));

        string source = request.SourceCode ?? string.Empty;
        if (string.IsNullOrWhiteSpace(source))
        {
            return Rejected(request.Language, ProgramExecuteErrorCode.SourceCodeRequired, "SourceCode is required.");
        }

        if (source.Length > _limits.MaxSourceCodeChars)
        {
            return Rejected(request.Language, ProgramExecuteErrorCode.SourceCodeTooLarge,
                $"SourceCode exceeds the configured limit of {_limits.MaxSourceCodeChars} characters.");
        }

        if (request.StdIn is not null && request.StdIn.Length > _limits.MaxStdInChars)
        {
            return Rejected(request.Language, ProgramExecuteErrorCode.StdInTooLarge,
                $"StdIn exceeds the configured limit of {_limits.MaxStdInChars} characters.");
        }

        ProgramExecuteResponse? rejection = ResolveLanguage(request, source, out ProgramLanguage language);
        if (rejection is not null) return rejection;

        if (!_options.TryGetInterpreter(language, out ProgramInterpreterSpecification? specification) ||
            specification is null)
        {
            return Rejected(language, ProgramExecuteErrorCode.InvalidRequest,
                $"No interpreter is configured for {language}. Add one to ProgramSandboxOptions.Interpreters.");
        }

        string? template = request.CompileOnly
            ? specification.CompileArgumentTemplate
            : specification.RunArgumentTemplate;
        if (request.CompileOnly && string.IsNullOrWhiteSpace(template))
        {
            return Rejected(language, ProgramExecuteErrorCode.InvalidRequest,
                $"CompileOnly was requested but no compile-only command is configured for {language}. " +
                "The program was not executed.");
        }

        try
        {
            await _concurrency.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            return Rejected(language, ProgramExecuteErrorCode.TimeoutOrCanceled, "Execution was canceled while queued.");
        }

        try
        {
            return await RunAsync(
                    language,
                    specification,
                    template ?? specification.RunArgumentTemplate,
                    source,
                    request.StdIn,
                    request.CompileOnly,
                    cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            _concurrency.Release();
        }
    }

    /// <inheritdoc/>
    public bool TryExecute(
        ProgramLanguage language,
        string sourceCode,
        string input,
        out string output,
        out string? errorMessage,
        CancellationToken cancellationToken = default)
    {
        output = string.Empty;
        errorMessage = null;

        var request = new ProgramExecuteRequest
        {
            Language = language,
            SourceCode = sourceCode ?? string.Empty,
            StdIn = input ?? string.Empty
        };

        ProgramExecuteResponse response;
        try
        {
            // Run the asynchronous path on the thread pool rather than blocking on it directly: a caller holding a
            // synchronization context (a UI or a legacy ASP.NET request) would otherwise deadlock when the awaited
            // continuation tried to resume on the very thread this call is blocking.
            response = Task.Run(() => ExecuteAsync(request, cancellationToken), cancellationToken)
                .GetAwaiter()
                .GetResult();
        }
        catch (OperationCanceledException)
        {
            errorMessage = "Execution was canceled.";
            return false;
        }
        catch (ObjectDisposedException exception)
        {
            errorMessage = exception.Message;
            return false;
        }

        if (response.Success)
        {
            output = response.StdOut;
            return true;
        }

        errorMessage = response.Error;
        return false;
    }

    /// <summary>Releases the concurrency semaphore held by this engine.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _concurrency.Dispose();
    }

    private async Task<ProgramExecuteResponse> RunAsync(
        ProgramLanguage language,
        ProgramInterpreterSpecification specification,
        string template,
        string source,
        string? stdIn,
        bool compileOnly,
        CancellationToken cancellationToken)
    {
        string workspace = CreateWorkspace();
        try
        {
            string sourcePath = Path.Combine(workspace, GetSourceFileName(language));
            await FilePolyfill.WriteAllTextAsync(sourcePath, source, cancellationToken).ConfigureAwait(false);

            var startInfo = new ProcessStartInfo
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                RedirectStandardInput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = workspace,
                StandardOutputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
                StandardErrorEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)
            };

            ConfigureCommand(startInfo, specification, template, sourcePath, workspace);
            ScrubEnvironment(startInfo, workspace);

            return await LaunchAsync(startInfo, language, stdIn, compileOnly, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            return Rejected(language, ProgramExecuteErrorCode.TimeoutOrCanceled, "Execution was canceled.");
        }
        catch (IOException exception)
        {
            return Rejected(language, ProgramExecuteErrorCode.ExecutionFailed,
                "The sandbox workspace could not be prepared: " + exception.GetType().Name + ".");
        }
        catch (UnauthorizedAccessException exception)
        {
            return Rejected(language, ProgramExecuteErrorCode.ExecutionFailed,
                "The sandbox workspace could not be prepared: " + exception.GetType().Name + ".");
        }
        finally
        {
            TryDeleteDirectory(workspace);
        }
    }

    private async Task<ProgramExecuteResponse> LaunchAsync(
        ProcessStartInfo startInfo,
        ProgramLanguage language,
        string? stdIn,
        bool compileOnly,
        CancellationToken cancellationToken)
    {
        using var timeoutSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutSource.CancelAfter(_limits.GetTimeLimit());

        using WindowsJobObject? job = WindowsJobObject.TryCreate(_limits.GetMemoryLimitBytes());
        using var process = new Process { StartInfo = startInfo };

        try
        {
            if (!process.Start())
            {
                return Rejected(language, ProgramExecuteErrorCode.ExecutionFailed,
                    $"The interpreter '{startInfo.FileName}' could not be started.");
            }
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return Rejected(language, ProgramExecuteErrorCode.ExecutionFailed,
                $"The interpreter '{startInfo.FileName}' was not found or could not be started.");
        }
        catch (InvalidOperationException)
        {
            return Rejected(language, ProgramExecuteErrorCode.ExecutionFailed,
                $"The interpreter '{startInfo.FileName}' could not be started.");
        }
        catch (PlatformNotSupportedException)
        {
            return Rejected(language, ProgramExecuteErrorCode.ExecutionFailed,
                "Starting a sandboxed process is not supported on this platform.");
        }

        int processId = TryGetProcessId(process);
        TryAssignToJob(job, process);

        var stdOutReader = new BoundedOutputReader(_limits.MaxStdOutChars);
        var stdErrReader = new BoundedOutputReader(_limits.MaxStdErrChars);
        Task stdOutTask = stdOutReader.PumpAsync(process.StandardOutput, timeoutSource.Token);
        Task stdErrTask = stdErrReader.PumpAsync(process.StandardError, timeoutSource.Token);
        Task stdInTask = WriteStandardInputAsync(process, stdIn);

        bool exited = await WaitForExitAsync(process, timeoutSource.Token).ConfigureAwait(false);
        if (!exited)
        {
            TryKillProcessTree(process, processId);
            await DrainAsync(stdOutTask, stdErrTask, stdInTask).ConfigureAwait(false);

            (string Text, bool Truncated) timedOutStdOut = stdOutReader.Snapshot();
            (string Text, bool Truncated) timedOutStdErr = stdErrReader.Snapshot();
            bool canceledByCaller = cancellationToken.IsCancellationRequested;

            return new ProgramExecuteResponse
            {
                Success = false,
                Language = language,
                CompilationAttempted = compileOnly,
                CompilationSucceeded = compileOnly ? false : null,
                ExitCode = -1,
                StdOut = timedOutStdOut.Text,
                StdErr = timedOutStdErr.Text,
                StdOutTruncated = timedOutStdOut.Truncated,
                StdErrTruncated = timedOutStdErr.Truncated,
                Error = canceledByCaller
                    ? "Execution was canceled."
                    : $"Execution exceeded the {_limits.TimeLimitSeconds.ToString(CultureInfo.InvariantCulture)} second limit and the process tree was terminated.",
                ErrorCode = ProgramExecuteErrorCode.TimeoutOrCanceled
            };
        }

        await DrainAsync(stdOutTask, stdErrTask, stdInTask).ConfigureAwait(false);

        (string Text, bool Truncated) capturedOut = stdOutReader.Snapshot();
        (string Text, bool Truncated) capturedErr = stdErrReader.Snapshot();
        int exitCode = TryGetExitCode(process);
        bool success = exitCode == 0;

        return new ProgramExecuteResponse
        {
            Success = success,
            Language = language,
            CompilationAttempted = compileOnly,
            CompilationSucceeded = compileOnly ? success : null,
            ExitCode = exitCode,
            StdOut = capturedOut.Text,
            StdErr = capturedErr.Text,
            StdOutTruncated = capturedOut.Truncated,
            StdErrTruncated = capturedErr.Truncated,
            Error = success
                ? null
                : compileOnly
                    ? $"Compilation failed with exit code {exitCode.ToString(CultureInfo.InvariantCulture)}."
                    : $"Execution failed with exit code {exitCode.ToString(CultureInfo.InvariantCulture)}.",
            ErrorCode = success
                ? null
                : compileOnly
                    ? ProgramExecuteErrorCode.CompilationFailed
                    : ProgramExecuteErrorCode.ExecutionFailed
        };
    }

    private ProgramExecuteResponse? ResolveLanguage(
        ProgramExecuteRequest request,
        string source,
        out ProgramLanguage language)
    {
        var allowed = new List<ProgramLanguage>();
        if (request.AllowedLanguages is not null)
        {
            foreach (ProgramLanguage candidate in request.AllowedLanguages)
            {
                if (candidate != ProgramLanguage.Generic &&
                    Enum.IsDefined(typeof(ProgramLanguage), candidate) &&
                    !allowed.Contains(candidate))
                {
                    allowed.Add(candidate);
                }
            }
        }

        language = request.Language;
        if (!Enum.IsDefined(typeof(ProgramLanguage), language))
        {
            language = ProgramLanguage.Generic;
            return Rejected(ProgramLanguage.Generic, ProgramExecuteErrorCode.InvalidRequest,
                "Language is not a defined ProgramLanguage value.");
        }

        if (language == ProgramLanguage.Generic)
        {
            if (ProgramLanguageDetector.TryDetect(source, out ProgramLanguage detected) &&
                (allowed.Count == 0 || allowed.Contains(detected)))
            {
                language = detected;
            }
            else if (request.PreferredLanguage.HasValue &&
                     request.PreferredLanguage.Value != ProgramLanguage.Generic &&
                     Enum.IsDefined(typeof(ProgramLanguage), request.PreferredLanguage.Value))
            {
                language = request.PreferredLanguage.Value;
            }
            else if (request.AllowUndetectedLanguageFallback && allowed.Count > 0)
            {
                language = allowed[0];
            }
        }

        if (language == ProgramLanguage.Generic)
        {
            return Rejected(ProgramLanguage.Generic, ProgramExecuteErrorCode.LanguageNotDetected,
                "The program language could not be detected; set Language or PreferredLanguage explicitly.");
        }

        if (language == ProgramLanguage.SQL)
        {
            return Rejected(ProgramLanguage.SQL, ProgramExecuteErrorCode.SqlNotSupported,
                "SQL is executed through a SQL sandbox, not through a program interpreter.");
        }

        if (allowed.Count > 0 && !allowed.Contains(language))
        {
            return Rejected(language, ProgramExecuteErrorCode.InvalidRequest,
                $"{language} is not present in AllowedLanguages.");
        }

        return null;
    }

    private void ConfigureCommand(
        ProcessStartInfo startInfo,
        ProgramInterpreterSpecification specification,
        string template,
        string sourcePath,
        string workspace)
    {
#if NET5_0_OR_GREATER
        if (!WindowsHost && File.Exists(PosixShellPath))
        {
            startInfo.FileName = PosixShellPath;
            startInfo.ArgumentList.Add("-c");
            startInfo.ArgumentList.Add(BuildPosixScript(specification, template, sourcePath, workspace));
            return;
        }
#endif

        startInfo.FileName = specification.Executable;
        startInfo.Arguments = ProgramInterpreterSpecification.Expand(
            template, sourcePath, workspace, quoteWithDoubleQuotes: true);
    }

#if NET5_0_OR_GREATER
    private string BuildPosixScript(
        ProgramInterpreterSpecification specification,
        string template,
        string sourcePath,
        string workspace)
    {
        var script = new StringBuilder();
        long kilobytes = _limits.GetMemoryLimitBytes() / 1024L;
        if (kilobytes > 0)
        {
            // A shell whose ulimit cannot set the requested resource must not abort the script, so every limit is
            // written defensively: the wall-clock limit and the tree kill remain the backstop either way.
            script.Append("ulimit -v ").Append(kilobytes.ToString(CultureInfo.InvariantCulture))
                .Append(" 2>/dev/null || true; ");
        }

        script.Append("ulimit -t ").Append(GetCpuTimeLimitSeconds().ToString(CultureInfo.InvariantCulture))
            .Append(" 2>/dev/null || true; ");
        script.Append("exec ")
            .Append(ProgramInterpreterSpecification.QuoteForPosixShell(specification.Executable))
            .Append(' ')
            .Append(ProgramInterpreterSpecification.Expand(template, sourcePath, workspace, quoteWithDoubleQuotes: false));
        return script.ToString();
    }

    private int GetCpuTimeLimitSeconds()
    {
        double cores = _limits.CpuLimit > 0.0 ? _limits.CpuLimit : 1.0;
        double seconds = Math.Ceiling(_limits.TimeLimitSeconds * cores);
        if (seconds < 1.0) seconds = 1.0;
        if (seconds > int.MaxValue) seconds = int.MaxValue;
        return (int)seconds;
    }
#endif

    private void ScrubEnvironment(ProcessStartInfo startInfo, string workspace)
    {
        // Start from nothing so no API key, token, or connection string held by the host is visible to code the
        // host did not write, then add back only what a child process genuinely needs to start.
        startInfo.Environment.Clear();
        startInfo.Environment["PATH"] = _pinnedPath;

        if (WindowsHost)
        {
            AddIfPresent(startInfo, "SystemRoot", Environment.GetFolderPath(Environment.SpecialFolder.Windows));
            AddIfPresent(startInfo, "windir", Environment.GetFolderPath(Environment.SpecialFolder.Windows));
            AddIfPresent(startInfo, "ComSpec",
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "cmd.exe"));
            startInfo.Environment["PATHEXT"] = ".COM;.EXE;.BAT;.CMD";
            startInfo.Environment["TEMP"] = workspace;
            startInfo.Environment["TMP"] = workspace;
        }
        else
        {
            startInfo.Environment["HOME"] = workspace;
            startInfo.Environment["TMPDIR"] = workspace;
            startInfo.Environment["LANG"] = "C.UTF-8";
        }

        startInfo.Environment["PYTHONIOENCODING"] = "utf-8";
        startInfo.Environment["PYTHONDONTWRITEBYTECODE"] = "1";
        startInfo.Environment["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1";
        startInfo.Environment["DOTNET_NOLOGO"] = "1";
        startInfo.Environment["DOTNET_SKIP_FIRST_TIME_EXPERIENCE"] = "1";
    }

    private static void AddIfPresent(ProcessStartInfo startInfo, string name, string? value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            startInfo.Environment[name] = value ?? string.Empty;
        }
    }

    private string CreateWorkspace()
    {
        string directory = Path.Combine(_workspaceRoot, Guid.NewGuid().ToString("N"));
#if NET7_0_OR_GREATER
        // Written as an inline platform check rather than through the cached flag so the platform-compatibility
        // analyzer can see that the Unix-only overload is unreachable on Windows.
        if (!OperatingSystem.IsWindows())
        {
            Directory.CreateDirectory(
                directory,
                UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            return directory;
        }
#endif
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string GetSourceFileName(ProgramLanguage language) => language switch
    {
        // A Java compiler requires the file name to match the public class, and the C# scripting host prefers a
        // conventional entry-point file name, so those two are named rather than derived.
        ProgramLanguage.Java => "Main.java",
        ProgramLanguage.CSharp => "Program.cs",
        _ => "program" + ProgramLanguageDetector.GetFileExtension(language)
    };

    private static async Task WriteStandardInputAsync(Process process, string? stdIn)
    {
        try
        {
            if (!string.IsNullOrEmpty(stdIn))
            {
                await process.StandardInput.WriteAsync(stdIn).ConfigureAwait(false);
                await process.StandardInput.FlushAsync().ConfigureAwait(false);
            }
        }
        catch (IOException)
        {
            // The child exited before reading its input.
        }
        catch (ObjectDisposedException)
        {
            // The child was killed and the pipe was torn down.
        }
        catch (InvalidOperationException)
        {
            // The stream was closed concurrently with the write.
        }
        finally
        {
            try
            {
                process.StandardInput.Close();
            }
            catch (IOException)
            {
                // Closing a broken pipe is not an error worth reporting.
            }
            catch (ObjectDisposedException)
            {
                // Already torn down.
            }
        }
    }

    private static async Task DrainAsync(Task stdOutTask, Task stdErrTask, Task stdInTask)
    {
        Task all = Task.WhenAll(stdOutTask, stdErrTask, stdInTask);
        Task completed = await Task.WhenAny(all, Task.Delay(DrainTimeoutMilliseconds)).ConfigureAwait(false);
        if (!ReferenceEquals(completed, all))
        {
            // A descendant kept an inherited pipe handle open past the kill. The snapshots are lock-protected, so
            // reading them now is safe; abandoning the pumps costs nothing because the workspace is deleted next.
            return;
        }

        try
        {
            await all.ConfigureAwait(false);
        }
#pragma warning disable CA1031
        catch (Exception)
#pragma warning restore CA1031
        {
            // Every pump already swallows its own stream faults; this guards the aggregate.
        }
    }

    private static async Task<bool> WaitForExitAsync(Process process, CancellationToken cancellationToken)
    {
#if NET5_0_OR_GREATER
        try
        {
            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
            return true;
        }
        catch (OperationCanceledException)
        {
            return false;
        }
#else
        var completion = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        void OnExited(object? sender, EventArgs args) => completion.TrySetResult(true);

        process.EnableRaisingEvents = true;
        process.Exited += OnExited;
        try
        {
            if (process.HasExited)
            {
                return true;
            }

            using (cancellationToken.Register(() => completion.TrySetResult(false)))
            {
                return await completion.Task.ConfigureAwait(false);
            }
        }
        finally
        {
            process.Exited -= OnExited;
        }
#endif
    }

    private static void TryKillProcessTree(Process process, int processId)
    {
        bool killedTree = false;
        try
        {
            if (process.HasExited)
            {
                killedTree = true;
            }
            else
            {
#if NET5_0_OR_GREATER
                process.Kill(entireProcessTree: true);
                killedTree = true;
#else
                // .NET Framework can only terminate the process itself; taskkill below removes its descendants.
                process.Kill();
#endif
            }
        }
        catch (InvalidOperationException)
        {
            killedTree = true;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            // Access denied or the process died mid-kill; fall through to the taskkill fallback.
        }
        catch (NotSupportedException)
        {
            // Remote process handles cannot be killed; fall through.
        }
        catch (AggregateException)
        {
            // Kill(entireProcessTree) reports per-child failures this way; fall through.
        }

        if (killedTree || !WindowsHost || processId <= 0)
        {
            return;
        }

        TryTaskKill(processId);
    }

    private static void TryTaskKill(int processId)
    {
        try
        {
            string taskKill = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.System), "taskkill.exe");
            if (!File.Exists(taskKill))
            {
                return;
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = taskKill,
                Arguments = "/PID " + processId.ToString(CultureInfo.InvariantCulture) + " /T /F",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };

            using Process? killer = Process.Start(startInfo);
            if (killer is null)
            {
                return;
            }

            killer.StandardOutput.ReadToEnd();
            killer.StandardError.ReadToEnd();
            killer.WaitForExit(TaskKillTimeoutMilliseconds);
        }
        catch (System.ComponentModel.Win32Exception)
        {
            // taskkill is unavailable; the primary kill was the best effort possible.
        }
        catch (InvalidOperationException)
        {
            // The helper process could not be started or observed.
        }
        catch (IOException)
        {
            // Reading the helper's output failed; the kill itself may still have succeeded.
        }
    }

    private static int TryGetProcessId(Process process)
    {
        try
        {
            return process.Id;
        }
        catch (InvalidOperationException)
        {
            return 0;
        }
    }

    private static int TryGetExitCode(Process process)
    {
        try
        {
            return process.ExitCode;
        }
        catch (InvalidOperationException)
        {
            return -1;
        }
    }

    private static void TryAssignToJob(WindowsJobObject? job, Process process)
    {
        if (job is null)
        {
            return;
        }

        try
        {
            job.TryAssign(process.Handle);
        }
        catch (InvalidOperationException)
        {
            // The process exited before it could join the job; the memory cap is moot.
        }
        catch (System.ComponentModel.Win32Exception)
        {
            // The handle could not be opened; the wall-clock limit remains the effective bound.
        }
    }

    private static void TryDeleteDirectory(string path)
    {
        if (string.IsNullOrEmpty(path))
        {
            return;
        }

        for (int attempt = 0; attempt < 5; attempt++)
        {
            try
            {
                if (!Directory.Exists(path))
                {
                    return;
                }

                Directory.Delete(path, recursive: true);
                return;
            }
            catch (IOException)
            {
                // A killed child may still hold the workspace as its working directory for a moment on Windows.
            }
            catch (UnauthorizedAccessException)
            {
                // Same as above: the handle is released as the process finishes dying.
            }

            Thread.Sleep(50);
        }
    }

    private static string BuildPinnedPath()
    {
        if (WindowsHost)
        {
            string system = Environment.GetFolderPath(Environment.SpecialFolder.System);
            string windows = Environment.GetFolderPath(Environment.SpecialFolder.Windows);
            string wbem = string.IsNullOrWhiteSpace(system) ? string.Empty : Path.Combine(system, "wbem");
            var segments = new List<string>();
            foreach (string segment in new[] { system, windows, wbem })
            {
                if (!string.IsNullOrWhiteSpace(segment) &&
                    !segments.Contains(segment, StringComparer.OrdinalIgnoreCase))
                {
                    segments.Add(segment);
                }
            }

            return string.Join(";", segments);
        }

        return "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
    }

    private static ProgramExecuteResponse Rejected(
        ProgramLanguage language,
        ProgramExecuteErrorCode errorCode,
        string error) => new()
        {
            Success = false,
            Language = language,
            ExitCode = -1,
            Error = error,
            ErrorCode = errorCode
        };

    private static bool DetectWindowsHost()
    {
#if NET5_0_OR_GREATER
        return OperatingSystem.IsWindows();
#else
        return Environment.OSVersion.Platform == PlatformID.Win32NT;
#endif
    }
}
