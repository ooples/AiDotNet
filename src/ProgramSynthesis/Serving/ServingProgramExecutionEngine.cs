using System.Net.Http;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ProgramSynthesis.Serving;

/// <summary>
/// Program execution engine that delegates sandboxed execution to an AiDotNet.Serving instance.
/// </summary>
/// <remarks>
/// <para>
/// The serving deployment runs each candidate inside a container with no network, a read-only mount, and container
/// CPU and memory limits, so this is the strongest of the execution boundaries — at the cost of needing a reachable
/// deployment. <see cref="ExecuteAsync"/> is asynchronous end to end and passes the whole request through, so the
/// caller sees the exit code, both captured streams, the truncation flags, and any compilation diagnostics the
/// sandbox produced.
/// </para>
/// <para>
/// Transport failures are folded into the response rather than thrown, matching every other engine: a fitness
/// function scoring thousands of candidates must not stop because one HTTP call failed.
/// </para>
/// <para><b>For Beginners:</b> Instead of running generated code on this machine, this engine sends it to an
/// AiDotNet.Serving server that runs it inside a container and sends back what happened. Use it when you want the
/// strongest isolation available, or when the machine running your evolution has no interpreters installed.</para>
/// </remarks>
public sealed class ServingProgramExecutionEngine : IProgramExecutionEngine
{
    private readonly IProgramSynthesisServingClient _client;
    private readonly TimeSpan? _timeout;

    /// <summary>Initializes an engine that uses the client's own timeout.</summary>
    /// <param name="client">The serving client used to reach the sandbox.</param>
    /// <exception cref="ArgumentNullException"><paramref name="client"/> is <c>null</c>.</exception>
    public ServingProgramExecutionEngine(IProgramSynthesisServingClient client)
    {
        Guard.NotNull(client);
        _client = client;
    }

    /// <summary>Initializes an engine that abandons a call after <paramref name="timeout"/>.</summary>
    /// <param name="client">The serving client used to reach the sandbox.</param>
    /// <param name="timeout">The per-call wall-clock limit; must be greater than zero.</param>
    /// <exception cref="ArgumentNullException"><paramref name="client"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="timeout"/> is not positive.</exception>
    public ServingProgramExecutionEngine(IProgramSynthesisServingClient client, TimeSpan timeout)
    {
        if (timeout <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(timeout), "Timeout must be > 0.");
        }

        Guard.NotNull(client);
        _client = client;
        _timeout = timeout;
    }

    /// <inheritdoc/>
    public async Task<ProgramExecuteResponse> ExecuteAsync(
        ProgramExecuteRequest request,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(request);

        using var timeoutSource = _timeout.HasValue ? new CancellationTokenSource(_timeout.Value) : null;
        using var linkedSource =
            timeoutSource is not null && cancellationToken.CanBeCanceled
                ? CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutSource.Token)
                : null;

        CancellationToken effectiveToken = linkedSource?.Token ?? timeoutSource?.Token ?? cancellationToken;

        try
        {
            return await _client.ExecuteProgramAsync(request, effectiveToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            return Failure(
                request.Language,
                ProgramExecuteErrorCode.TimeoutOrCanceled,
                cancellationToken.IsCancellationRequested
                    ? "Execution was canceled."
                    : "The serving call exceeded its timeout.");
        }
        catch (HttpRequestException exception)
        {
            return Failure(request.Language, ProgramExecuteErrorCode.ExecutionFailed, exception.Message);
        }
        catch (InvalidOperationException exception)
        {
            return Failure(request.Language, ProgramExecuteErrorCode.ExecutionFailed, exception.Message);
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
            // Hop to the thread pool before blocking. Awaiting the asynchronous path directly on a thread that
            // carries a synchronization context (a UI thread, or a legacy ASP.NET request) would deadlock, because
            // the continuation would need the very thread this call is blocking.
            response = Task.Run(() => ExecuteAsync(request, cancellationToken), cancellationToken)
                .GetAwaiter()
                .GetResult();
        }
        catch (OperationCanceledException)
        {
            errorMessage = "Execution was canceled.";
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

    private static ProgramExecuteResponse Failure(
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
}
