using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Agents;

/// <summary>
/// A single-model agent that drives an <see cref="IChatClient{T}"/> in a native tool-calling loop:
/// it calls the model, runs any tools the model requests, feeds the results back, and repeats until the
/// model returns a final answer (or the iteration cap is hit).
/// </summary>
/// <typeparam name="T">The numeric type shared with the underlying <see cref="IChatClient{T}"/>.</typeparam>
/// <remarks>
/// <para>
/// This replaces the legacy prompt-parsed ReAct loop with provider-native function calling: tool requests
/// arrive as structured <see cref="ToolCallContent"/> rather than scraped from prose, and tool results go
/// back as <see cref="ChatRole.Tool"/> messages. It is the leaf <see cref="IAgent{T}"/> that the multi-agent
/// coordinators (supervisor/swarm) compose.
/// </para>
/// <para><b>For Beginners:</b> Give it a model and (optionally) a toolbox. Ask it something. Internally it
/// loops: "model, here's the task and your tools" → if the model asks to use a tool, the executor runs it
/// and tells the model the result → repeat → until the model just answers. You get that final answer back,
/// plus the full transcript of what happened.
/// </para>
/// </remarks>
public sealed class AgentExecutor<T> : IAgent<T>
{
    private readonly IChatClient<T> _client;
    private readonly ToolCollection _tools;
    private readonly AgentExecutorOptions _options;

    /// <summary>
    /// Initializes a new <see cref="AgentExecutor{T}"/>.
    /// </summary>
    /// <param name="client">The chat model the agent drives.</param>
    /// <param name="tools">The tools the agent may use. <c>null</c> means the agent has no tools.</param>
    /// <param name="options">Agent settings. <c>null</c> uses defaults.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="client"/> is <c>null</c>.</exception>
    public AgentExecutor(IChatClient<T> client, ToolCollection? tools = null, AgentExecutorOptions? options = null)
    {
        Guard.NotNull(client);
        _client = client;
        _tools = tools ?? new ToolCollection();
        _options = options ?? new AgentExecutorOptions();
    }

    /// <inheritdoc/>
    public string Name =>
        // Pattern-match form so the compiler narrows nullability on net471
        // (string.IsNullOrWhiteSpace lacks [NotNullWhen(false)] there).
        _options.Name is { } name && !string.IsNullOrWhiteSpace(name) ? name : "agent";

    /// <inheritdoc/>
    public string Description => _options.Description ?? string.Empty;

    /// <summary>
    /// Runs the agent against a single user message.
    /// </summary>
    /// <param name="userMessage">The user's request.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>A task producing the agent's <see cref="AgentRunResult"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="userMessage"/> is <c>null</c>.</exception>
    public Task<AgentRunResult> RunAsync(string userMessage, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(userMessage);
        return RunAsync(new[] { ChatMessage.User(userMessage) }, cancellationToken);
    }

    /// <inheritdoc/>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="messages"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="messages"/> is empty.</exception>
    public async Task<AgentRunResult> RunAsync(
        IReadOnlyList<ChatMessage> messages,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        if (messages.Count == 0)
        {
            throw new ArgumentException("The conversation must contain at least one message.", nameof(messages));
        }

        var transcript = new List<ChatMessage>(messages.Count + 4);
        var systemPrompt = _options.SystemPrompt;
        if (systemPrompt is { } prompt && prompt.Trim().Length > 0)
        {
            transcript.Add(ChatMessage.System(prompt));
        }

        transcript.AddRange(messages);

        // Validate forwarded sampling options at this boundary: they come from
        // public configuration, and NaN/infinite/negative temperatures or
        // non-positive token caps would otherwise fail downstream in
        // provider-specific (and far less diagnosable) ways.
        if (_options.Temperature is { } temperature &&
            (double.IsNaN(temperature) || double.IsInfinity(temperature) || temperature < 0))
        {
            throw new ArgumentOutOfRangeException(
                nameof(AgentExecutorOptions.Temperature), temperature,
                "Temperature must be a finite, non-negative value.");
        }

        if (_options.MaxOutputTokens is { } maxOutputTokens && maxOutputTokens <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(AgentExecutorOptions.MaxOutputTokens), maxOutputTokens,
                "MaxOutputTokens must be positive.");
        }

        var hasTools = _tools.Count > 0;
        var requestOptions = new ChatOptions
        {
            Temperature = _options.Temperature,
            MaxOutputTokens = _options.MaxOutputTokens,
            Tools = hasTools ? _tools.GetDefinitions() : null,
            ToolChoice = hasTools ? (_options.ToolChoice ?? ToolChoiceMode.Auto) : null,
        };

        var maxIterations = _options.MaxIterations is { } configured && configured > 0
            ? configured
            : AgentExecutorOptions.DefaultMaxIterations;

        var inputTokens = 0;
        var outputTokens = 0;
        var sawUsage = false;

        for (var iteration = 1; iteration <= maxIterations; iteration++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var response = await _client.GetResponseAsync(transcript, requestOptions, cancellationToken)
                .ConfigureAwait(false);
            transcript.Add(response.Message);

            if (response.Usage is { } usage)
            {
                sawUsage = true;
                inputTokens += usage.InputTokens;
                outputTokens += usage.OutputTokens;
            }

            var toolCalls = response.Message.ToolCalls;
            var wantsTools = response.FinishReason == ChatFinishReason.ToolCalls || toolCalls.Count > 0;

            // The model asked for tool use that cannot be honored (no tools are
            // registered, or the response advertised ToolCalls with an empty
            // list). Falling through to Finished would mark the run successful
            // even though no final answer was produced and no tool ever ran —
            // report it as a stopped (incomplete) run instead.
            if (wantsTools && (!hasTools || toolCalls.Count == 0))
            {
                return AgentRunResult.Stopped(
                    response.Message.Text,
                    transcript,
                    iteration,
                    sawUsage ? new ChatUsage(inputTokens, outputTokens) : null,
                    Name);
            }

            if (hasTools && wantsTools && toolCalls.Count > 0)
            {
                foreach (var call in toolCalls)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var toolMessage = await _tools.InvokeToToolMessageAsync(call, cancellationToken)
                        .ConfigureAwait(false);
                    transcript.Add(toolMessage);
                }

                continue;
            }

            return AgentRunResult.Finished(
                response.Message.Text,
                transcript,
                iteration,
                sawUsage ? new ChatUsage(inputTokens, outputTokens) : null,
                Name);
        }

        var lastText = transcript.Count > 0 ? transcript[transcript.Count - 1].Text : string.Empty;
        return AgentRunResult.Stopped(
            lastText,
            transcript,
            maxIterations,
            sawUsage ? new ChatUsage(inputTokens, outputTokens) : null,
            Name);
    }
}
