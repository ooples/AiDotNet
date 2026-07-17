using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Serving.Engine;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;

namespace AiDotNet.Serving.Engine.Http;

/// <summary>
/// A running, OpenAI-compatible HTTP server for a single model: continuous batching + paged KV under the hood,
/// exposed at <c>/v1/completions</c> (streaming and non-streaming), <c>/v1/models</c>, and <c>/health</c>. This
/// is what <c>model.Serve()</c> returns — a one-line, self-hosted inference endpoint that existing OpenAI
/// clients can call unchanged.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> this turns your model into a web service. Start it and point any OpenAI-style
/// client at the URL in <see cref="Urls"/>; requests are batched together and answered efficiently. Dispose it
/// (or the <c>using</c> block ends) to stop the server and free memory.</para>
/// </remarks>
/// <typeparam name="T">The model's numeric type.</typeparam>
public sealed class InferenceServer<T> : IAsyncDisposable, IDisposable
{
    private readonly WebApplication _app;
    private readonly AsyncEngineHost<T> _host;
    private readonly IGenerationTokenizer _tokenizer;
    private readonly string _modelName;
    private readonly SamplingParameters _defaultSampling;
    private readonly IChatTemplate _chatTemplate;
    private bool _disposed;

    internal InferenceServer(
        AsyncEngineHost<T> host,
        IGenerationTokenizer tokenizer,
        string modelName,
        SamplingParameters defaultSampling,
        string[] urls,
        IChatTemplate? chatTemplate = null)
    {
        _host = host;
        _tokenizer = tokenizer;
        _modelName = modelName;
        _defaultSampling = defaultSampling;
        _chatTemplate = chatTemplate ?? new DefaultChatTemplate();

        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls(urls);
        builder.Logging.ClearProviders(); // quiet by default; hosting apps can add their own
        _app = builder.Build();
        MapEndpoints(_app);
        _app.Start(); // non-blocking; Kestrel resolves any dynamic (:0) ports here
    }

    /// <summary>The URLs the server is listening on (dynamic ports are resolved after start).</summary>
    public IReadOnlyList<string> Urls => _app.Urls.ToArray();

    /// <summary>A snapshot of engine load and KV-cache utilization.</summary>
    public EngineStatistics GetStatistics() => _host.GetStatistics();

    private void MapEndpoints(WebApplication app)
    {
        app.MapGet("/health", () => Results.Json(new { status = "ok" }));

        app.MapGet("/v1/models", () => Results.Json(new
        {
            @object = "list",
            data = new[] { new { id = _modelName, @object = "model", owned_by = "aidotnet" } },
        }));

        app.MapPost("/v1/completions", HandleCompletionAsync);
        app.MapPost("/v1/chat/completions", HandleChatCompletionAsync);
    }

    private async Task HandleChatCompletionAsync(HttpContext context)
    {
        OpenAiChatRequest? request;
        try
        {
            using var reader = new System.IO.StreamReader(context.Request.Body);
            request = JsonConvert.DeserializeObject<OpenAiChatRequest>(await reader.ReadToEndAsync().ConfigureAwait(false));
        }
        catch (JsonException)
        {
            await WriteErrorAsync(context, StatusCodes.Status400BadRequest, "Invalid JSON body.").ConfigureAwait(false);
            return;
        }

        if (request is null || request.Messages is null || request.Messages.Count == 0)
        {
            await WriteErrorAsync(context, StatusCodes.Status400BadRequest, "A non-empty 'messages' array is required.").ConfigureAwait(false);
            return;
        }

        string prompt = _chatTemplate.Render(request.Messages);

        SamplingParameters sampling;
        IReadOnlyList<int> promptIds;
        try
        {
            sampling = ToSamplingParameters(request.Temperature, request.TopP, request.TopK,
                request.PresencePenalty, request.FrequencyPenalty, request.Seed, request.MaxTokens);
            sampling.Validate();
            promptIds = _tokenizer.Encode(prompt);
            if (promptIds.Count == 0) throw new ArgumentException("Rendered chat prompt tokenized to zero tokens.");
        }
        catch (ArgumentException ex)
        {
            await WriteErrorAsync(context, StatusCodes.Status400BadRequest, ex.Message).ConfigureAwait(false);
            return;
        }

        string id = "chatcmpl-" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        if (request.Stream)
            await StreamChatAsync(context, id, created, promptIds, sampling).ConfigureAwait(false);
        else
            await WriteChatAsync(context, id, created, promptIds, sampling).ConfigureAwait(false);
    }

    private async Task WriteChatAsync(
        HttpContext context, string id, long created, IReadOnlyList<int> promptIds, SamplingParameters sampling)
    {
        var generated = await _host.GenerateAsync(promptIds, sampling, context.RequestAborted).ConfigureAwait(false);
        string text = _tokenizer.Decode(generated);

        var response = new OpenAiChatResponse
        {
            Id = id,
            Created = created,
            Model = _modelName,
            Choices = new[]
            {
                new OpenAiChatChoice
                {
                    Index = 0,
                    Message = new ChatChoiceMessage { Role = "assistant", Content = text },
                    FinishReason = "stop",
                },
            },
            Usage = new OpenAiUsage
            {
                PromptTokens = promptIds.Count,
                CompletionTokens = generated.Count,
                TotalTokens = promptIds.Count + generated.Count,
            },
        };

        context.Response.ContentType = "application/json";
        await context.Response.WriteAsync(JsonConvert.SerializeObject(response), context.RequestAborted).ConfigureAwait(false);
    }

    private async Task StreamChatAsync(
        HttpContext context, string id, long created, IReadOnlyList<int> promptIds, SamplingParameters sampling)
    {
        context.Response.ContentType = "text/event-stream";
        context.Response.Headers["Cache-Control"] = "no-cache";

        // First chunk announces the assistant role.
        await WriteChatChunkAsync(context, id, created, new ChatChoiceDelta { Role = "assistant" }, null).ConfigureAwait(false);

        string previousText = string.Empty;
        await foreach (var update in _host.StreamAsync(promptIds, sampling, context.RequestAborted).ConfigureAwait(false))
        {
            string fullText = _tokenizer.Decode(update.TokenIds);
            string delta = fullText.StartsWith(previousText, StringComparison.Ordinal)
                ? fullText.Substring(previousText.Length)
                : fullText;
            previousText = fullText;

            await WriteChatChunkAsync(context, id, created, new ChatChoiceDelta { Content = delta },
                update.IsFinished ? (update.FinishReason ?? "stop") : null).ConfigureAwait(false);
        }

        await context.Response.WriteAsync("data: [DONE]\n\n", context.RequestAborted).ConfigureAwait(false);
        await context.Response.Body.FlushAsync(context.RequestAborted).ConfigureAwait(false);
    }

    private async Task WriteChatChunkAsync(
        HttpContext context, string id, long created, ChatChoiceDelta delta, string? finishReason)
    {
        var chunk = new OpenAiChatResponse
        {
            Id = id,
            Object = "chat.completion.chunk",
            Created = created,
            Model = _modelName,
            Choices = new[] { new OpenAiChatChoice { Index = 0, Delta = delta, FinishReason = finishReason } },
        };
        await context.Response.WriteAsync($"data: {JsonConvert.SerializeObject(chunk)}\n\n", context.RequestAborted).ConfigureAwait(false);
        await context.Response.Body.FlushAsync(context.RequestAborted).ConfigureAwait(false);
    }

    private async Task HandleCompletionAsync(HttpContext context)
    {
        OpenAiCompletionRequest? request;
        try
        {
            using var reader = new System.IO.StreamReader(context.Request.Body);
            string body = await reader.ReadToEndAsync().ConfigureAwait(false);
            request = JsonConvert.DeserializeObject<OpenAiCompletionRequest>(body);
        }
        catch (JsonException)
        {
            await WriteErrorAsync(context, StatusCodes.Status400BadRequest, "Invalid JSON body.").ConfigureAwait(false);
            return;
        }

        if (request is null || string.IsNullOrEmpty(request.Prompt))
        {
            await WriteErrorAsync(context, StatusCodes.Status400BadRequest, "A non-empty 'prompt' is required.").ConfigureAwait(false);
            return;
        }

        SamplingParameters sampling;
        IReadOnlyList<int> promptIds;
        try
        {
            sampling = ToSamplingParameters(request);
            sampling.Validate();
            promptIds = _tokenizer.Encode(request.Prompt);
            if (promptIds.Count == 0)
                throw new ArgumentException("Prompt tokenized to zero tokens.");
        }
        catch (ArgumentException ex)
        {
            await WriteErrorAsync(context, StatusCodes.Status400BadRequest, ex.Message).ConfigureAwait(false);
            return;
        }

        string id = "cmpl-" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        if (request.Stream)
            await StreamCompletionAsync(context, id, created, promptIds, sampling).ConfigureAwait(false);
        else
            await WriteCompletionAsync(context, id, created, promptIds, sampling).ConfigureAwait(false);
    }

    private async Task WriteCompletionAsync(
        HttpContext context, string id, long created, IReadOnlyList<int> promptIds, SamplingParameters sampling)
    {
        var generated = await _host.GenerateAsync(promptIds, sampling, context.RequestAborted).ConfigureAwait(false);
        string text = _tokenizer.Decode(generated);

        var response = new OpenAiCompletionResponse
        {
            Id = id,
            Created = created,
            Model = _modelName,
            Choices = new[]
            {
                new OpenAiCompletionChoice { Text = text, Index = 0, FinishReason = "stop" },
            },
            Usage = new OpenAiUsage
            {
                PromptTokens = promptIds.Count,
                CompletionTokens = generated.Count,
                TotalTokens = promptIds.Count + generated.Count,
            },
        };

        context.Response.ContentType = "application/json";
        await context.Response.WriteAsync(JsonConvert.SerializeObject(response), context.RequestAborted).ConfigureAwait(false);
    }

    private async Task StreamCompletionAsync(
        HttpContext context, string id, long created, IReadOnlyList<int> promptIds, SamplingParameters sampling)
    {
        context.Response.ContentType = "text/event-stream";
        context.Response.Headers["Cache-Control"] = "no-cache";

        string previousText = string.Empty;
        await foreach (var update in _host.StreamAsync(promptIds, sampling, context.RequestAborted).ConfigureAwait(false))
        {
            string fullText = _tokenizer.Decode(update.TokenIds);
            string delta = fullText.StartsWith(previousText, StringComparison.Ordinal)
                ? fullText.Substring(previousText.Length)
                : fullText;
            previousText = fullText;

            var chunk = new OpenAiCompletionResponse
            {
                Id = id,
                Created = created,
                Model = _modelName,
                Choices = new[]
                {
                    new OpenAiCompletionChoice
                    {
                        Text = delta,
                        Index = 0,
                        FinishReason = update.IsFinished ? (update.FinishReason ?? "stop") : null,
                    },
                },
            };

            await context.Response.WriteAsync($"data: {JsonConvert.SerializeObject(chunk)}\n\n", context.RequestAborted).ConfigureAwait(false);
            await context.Response.Body.FlushAsync(context.RequestAborted).ConfigureAwait(false);
        }

        await context.Response.WriteAsync("data: [DONE]\n\n", context.RequestAborted).ConfigureAwait(false);
        await context.Response.Body.FlushAsync(context.RequestAborted).ConfigureAwait(false);
    }

    private SamplingParameters ToSamplingParameters(OpenAiCompletionRequest r)
        => ToSamplingParameters(r.Temperature, r.TopP, r.TopK, r.PresencePenalty, r.FrequencyPenalty, r.Seed, r.MaxTokens);

    private SamplingParameters ToSamplingParameters(
        double temperature, double topP, int topK, double presencePenalty, double frequencyPenalty, int? seed, int maxTokens) => new()
    {
        Temperature = temperature,
        // OpenAI clients may send top_p = 0 meaning "unused"; our engine treats (0,1] as the valid nucleus, so
        // map a non-positive value to the disabled default of 1.0.
        TopP = topP <= 0 ? 1.0 : topP,
        TopK = topK,
        PresencePenalty = presencePenalty,
        FrequencyPenalty = frequencyPenalty,
        Seed = seed,
        MaxTokens = maxTokens < 1 ? _defaultSampling.MaxTokens : maxTokens,
    };

    private static Task WriteErrorAsync(HttpContext context, int status, string message)
    {
        context.Response.StatusCode = status;
        context.Response.ContentType = "application/json";
        var payload = new { error = new { message, type = "invalid_request_error" } };
        return context.Response.WriteAsync(JsonConvert.SerializeObject(payload));
    }

    /// <inheritdoc/>
    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        await _app.StopAsync().ConfigureAwait(false);
        await _app.DisposeAsync().ConfigureAwait(false);
        _host.Dispose();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _app.StopAsync().GetAwaiter().GetResult();
        (_app as IDisposable)?.Dispose();
        _host.Dispose();
    }
}
