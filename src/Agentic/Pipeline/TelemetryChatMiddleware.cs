using System.Diagnostics;
using AiDotNet.Agentic.Models;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// An <see cref="IChatMiddleware"/> that emits OpenTelemetry GenAI telemetry for each chat call: a client span
/// tagged with the operation, response model, finish reason, and token usage, plus operation-count and
/// token-usage metrics. Drop it into the middleware pipeline to make every model call observable.
/// </summary>
/// <remarks>
/// <para>
/// Spans and metrics are emitted on <see cref="AgenticTelemetry.Source"/>/<see cref="AgenticTelemetry.Meter"/>;
/// when no collector is listening, the overhead is negligible (the span is not even created). Tag names follow
/// the OpenTelemetry GenAI semantic conventions so standard dashboards understand them.
/// </para>
/// <para><b>For Beginners:</b> Add this filter and each model call automatically reports how long it took, how
/// many tokens it used, and why it stopped — to whatever monitoring tool you've connected, with no other code.
/// </para>
/// </remarks>
public sealed class TelemetryChatMiddleware : IChatMiddleware
{
    /// <inheritdoc/>
    public async Task<ChatResponse> InvokeAsync(ChatRequestContext context, ChatPipelineDelegate next, CancellationToken cancellationToken)
    {
        Guard.NotNull(context);
        Guard.NotNull(next);

        using var activity = AgenticTelemetry.Source.StartActivity("chat", ActivityKind.Client);
        activity?.SetTag("gen_ai.operation.name", "chat");

        ChatResponse response;
        try
        {
            response = await next(context, cancellationToken).ConfigureAwait(false);
        }
        catch (System.Exception ex)
        {
            activity?.SetStatus(ActivityStatusCode.Error, ex.Message);
            AgenticTelemetry.OperationCount.Add(1, new KeyValuePair<string, object?>("error", true));
            throw;
        }

        AgenticTelemetry.OperationCount.Add(1, new KeyValuePair<string, object?>("gen_ai.operation.name", "chat"));

        if (response.ModelId is { } modelId)
        {
            activity?.SetTag("gen_ai.response.model", modelId);
        }

        activity?.SetTag("gen_ai.response.finish_reason", response.FinishReason.ToString());

        if (response.Usage is { } usage)
        {
            activity?.SetTag("gen_ai.usage.input_tokens", usage.InputTokens);
            activity?.SetTag("gen_ai.usage.output_tokens", usage.OutputTokens);
            AgenticTelemetry.TokenUsage.Record(usage.InputTokens, new KeyValuePair<string, object?>("gen_ai.token.type", "input"));
            AgenticTelemetry.TokenUsage.Record(usage.OutputTokens, new KeyValuePair<string, object?>("gen_ai.token.type", "output"));
        }

        return response;
    }
}
