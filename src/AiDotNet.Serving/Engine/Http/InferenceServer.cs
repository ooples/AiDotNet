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
    private bool _disposed;

    internal InferenceServer(
        AsyncEngineHost<T> host,
        IGenerationTokenizer tokenizer,
        string modelName,
        SamplingParameters defaultSampling,
        string[] urls)
    {
        _host = host;
        _tokenizer = tokenizer;
        _modelName = modelName;
        _defaultSampling = defaultSampling;

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

    private SamplingParameters ToSamplingParameters(OpenAiCompletionRequest r) => new()
    {
        Temperature = r.Temperature,
        // OpenAI clients may send top_p = 0 meaning "unused"; our engine treats (0,1] as the valid nucleus, so
        // map a non-positive value to the disabled default of 1.0.
        TopP = r.TopP <= 0 ? 1.0 : r.TopP,
        TopK = r.TopK,
        PresencePenalty = r.PresencePenalty,
        FrequencyPenalty = r.FrequencyPenalty,
        Seed = r.Seed,
        MaxTokens = r.MaxTokens < 1 ? _defaultSampling.MaxTokens : r.MaxTokens,
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
