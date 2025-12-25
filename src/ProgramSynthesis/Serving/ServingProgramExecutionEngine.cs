using System.Net.Http;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;

namespace AiDotNet.ProgramSynthesis.Serving;

/// <summary>
/// Program execution engine that delegates sandboxed execution to an AiDotNet.Serving instance.
/// </summary>
public sealed class ServingProgramExecutionEngine : IProgramExecutionEngine
{
    private readonly IProgramSynthesisServingClient _client;
    private readonly TimeSpan? _timeout;

    public ServingProgramExecutionEngine(IProgramSynthesisServingClient client)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
    }

    public ServingProgramExecutionEngine(IProgramSynthesisServingClient client, TimeSpan timeout)
    {
        if (timeout <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(timeout), "Timeout must be > 0.");
        }

        _client = client ?? throw new ArgumentNullException(nameof(client));
        _timeout = timeout;
    }

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

        using var timeoutCts = _timeout.HasValue ? new CancellationTokenSource(_timeout.Value) : null;
        using var linkedCts =
            timeoutCts is not null && cancellationToken.CanBeCanceled
                ? CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutCts.Token)
                : null;

        try
        {
            var effectiveCancellationToken = linkedCts?.Token ?? timeoutCts?.Token ?? cancellationToken;

            var response = _client.ExecuteProgramAsync(
                    new ProgramExecuteRequest
                    {
                        Language = language,
                        SourceCode = sourceCode ?? string.Empty,
                        StdIn = input ?? string.Empty
                    },
                    effectiveCancellationToken)
                .ConfigureAwait(false)
                .GetAwaiter()
                .GetResult();

            if (response.Success)
            {
                output = response.StdOut ?? string.Empty;
                return true;
            }

            errorMessage = response.Error;
            return false;
        }
        catch (OperationCanceledException)
        {
            errorMessage = "Execution was canceled.";
            return false;
        }
        catch (HttpRequestException ex)
        {
            errorMessage = ex.Message;
            return false;
        }
        catch (InvalidOperationException ex)
        {
            errorMessage = ex.Message;
            return false;
        }
    }
}
