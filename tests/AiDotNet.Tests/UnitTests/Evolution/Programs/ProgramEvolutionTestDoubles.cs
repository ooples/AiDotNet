using System.Collections.ObjectModel;
using AiDotNet.Agentic.Models;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

internal sealed class FakeChatClient : IChatClient<double>
{
    private readonly IReadOnlyList<string> _responses;
    private readonly List<IReadOnlyList<ChatMessage>> _conversations = new();
    private int _calls;

    public FakeChatClient(params string[] responses) => _responses = responses;

    public string ModelId => "fake-model";

    public int Calls => _calls;

    public ChatOptions? LastOptions { get; private set; }

    public IReadOnlyList<IReadOnlyList<ChatMessage>> Conversations => _conversations;

    public Exception? ThrowOnFirstCall { get; set; }

    public ChatUsage? Usage { get; set; }

    public Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _conversations.Add(new ReadOnlyCollection<ChatMessage>(new List<ChatMessage>(messages)));
        LastOptions = options;
        int index = _calls;
        _calls++;

        if (index == 0 && ThrowOnFirstCall is not null) throw ThrowOnFirstCall;

        string text = _responses.Count == 0
            ? string.Empty
            : _responses[Math.Min(index, _responses.Count - 1)];

        return Task.FromResult(new ChatResponse(ChatMessage.Assistant(text), usage: Usage));
    }

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("The fake chat client does not stream.");
}

internal sealed class NullResultProgramFitnessEvaluator : AiDotNet.Interfaces.IProgramFitnessEvaluator
{
    public string Id => "null-result-evaluator";

    public string VersionHash => "null-result-evaluator-v1";

#pragma warning disable CS8625
    public ValueTask<AiDotNet.Evolution.EvolutionTaskResult> EvaluateAsync(
        AiDotNet.Evolution.Programs.ProgramGenome candidate,
        AiDotNet.Evolution.EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default) =>
        new(default(AiDotNet.Evolution.EvolutionTaskResult));
#pragma warning restore CS8625
}

internal sealed class FakeExecutionOutcome
{
    private FakeExecutionOutcome(bool succeeded, string output, string? errorMessage)
    {
        Succeeded = succeeded;
        Output = output;
        ErrorMessage = errorMessage;
    }

    public bool Succeeded { get; }
    public string Output { get; }
    public string? ErrorMessage { get; }

    public static FakeExecutionOutcome Success(string output) => new(true, output, null);
    public static FakeExecutionOutcome Failure(string errorMessage) => new(false, string.Empty, errorMessage);
}

internal sealed class FakeProgramExecutionEngine : IProgramExecutionEngine
{
    private readonly Func<string, string, FakeExecutionOutcome> _handler;
    private int _calls;

    public FakeProgramExecutionEngine(Func<string, string, FakeExecutionOutcome> handler) => _handler = handler;

    public int Calls => _calls;
    public ProgramLanguage? LastLanguage { get; private set; }
    public string? LastStdIn { get; private set; }
    public bool LastCompileOnly { get; private set; }
    public int PeakConcurrency { get; private set; }

    private int _active;
    private readonly object _gate = new();

    public bool TryExecute(
        ProgramLanguage language,
        string sourceCode,
        string input,
        out string output,
        out string? errorMessage,
        CancellationToken cancellationToken = default)
    {
        _calls++;
        LastLanguage = language;
        LastStdIn = input;
        cancellationToken.ThrowIfCancellationRequested();
        FakeExecutionOutcome outcome = _handler(sourceCode, input);
        output = outcome.Output;
        errorMessage = outcome.ErrorMessage;
        return outcome.Succeeded;
    }

    public Task<ProgramExecuteResponse> ExecuteAsync(
        ProgramExecuteRequest request,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        lock (_gate)
        {
            _calls++;
            _active++;
            if (_active > PeakConcurrency) PeakConcurrency = _active;
        }

        try
        {
            LastLanguage = request.Language;
            LastStdIn = request.StdIn;
            LastCompileOnly = request.CompileOnly;
            FakeExecutionOutcome outcome = _handler(request.SourceCode, request.StdIn ?? string.Empty);

            return Task.FromResult(new ProgramExecuteResponse
            {
                Success = outcome.Succeeded,
                Language = request.Language,
                ExitCode = outcome.Succeeded ? 0 : 1,
                StdOut = outcome.Output,
                StdErr = outcome.ErrorMessage ?? string.Empty,
                Error = outcome.ErrorMessage,
                ErrorCode = outcome.Succeeded ? null : ProgramExecuteErrorCode.ExecutionFailed
            });
        }
        finally
        {
            lock (_gate) _active--;
        }
    }
}

internal sealed class ThrowingProgramExecutionEngine : IProgramExecutionEngine
{
    public bool TryExecute(
        ProgramLanguage language,
        string sourceCode,
        string input,
        out string output,
        out string? errorMessage,
        CancellationToken cancellationToken = default) =>
        throw new InvalidOperationException("sandbox unavailable");

    public Task<ProgramExecuteResponse> ExecuteAsync(
        ProgramExecuteRequest request,
        CancellationToken cancellationToken = default) =>
        throw new InvalidOperationException("sandbox unavailable");
}

internal sealed class ScriptedProgramExecutionEngine : IProgramExecutionEngine
{
    private readonly Func<ProgramExecuteRequest, ProgramExecuteResponse> _handler;

    public ScriptedProgramExecutionEngine(Func<ProgramExecuteRequest, ProgramExecuteResponse> handler) =>
        _handler = handler;

    public int Calls { get; private set; }

    public bool TryExecute(
        ProgramLanguage language,
        string sourceCode,
        string input,
        out string output,
        out string? errorMessage,
        CancellationToken cancellationToken = default)
    {
        ProgramExecuteResponse response = ExecuteAsync(
            new ProgramExecuteRequest { Language = language, SourceCode = sourceCode, StdIn = input },
            cancellationToken).GetAwaiter().GetResult();
        output = response.StdOut;
        errorMessage = response.Error;
        return response.Success;
    }

    public Task<ProgramExecuteResponse> ExecuteAsync(
        ProgramExecuteRequest request,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Calls++;
        return Task.FromResult(_handler(request));
    }
}
