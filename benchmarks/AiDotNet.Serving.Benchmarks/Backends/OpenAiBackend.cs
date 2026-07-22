using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Serving.Benchmarks.Backends;

/// <summary>
/// OpenAI-compatible streaming backend. Drives <c>/v1/chat/completions</c> or <c>/v1/completions</c>
/// with <c>stream=true</c> and parses the Server-Sent-Events token stream, so it works identically
/// against vLLM, TGI (OpenAI route), and AiDotNet once it exposes the OpenAI API.
/// </summary>
public sealed class OpenAiBackend : IServingBackend
{
    private readonly HttpClient _http;
    private readonly BenchmarkOptions _o;
    private readonly bool _chat;

    public OpenAiBackend(HttpClient http, BenchmarkOptions o)
    {
        _http = http;
        _o = o;
        _chat = !string.Equals(o.Mode, "completions", StringComparison.OrdinalIgnoreCase);
    }

    public string Name => _chat ? "openai/chat" : "openai/completions";

    public async Task<RequestResult> ExecuteAsync(RequestSpec spec, double dispatchMs, CancellationToken ct)
    {
        var result = new RequestResult
        {
            Index = spec.Index,
            DispatchMs = dispatchMs,
            PromptTokens = spec.ApproxPromptTokens,
        };
        var sw = Stopwatch.StartNew();

        try
        {
            string path = _chat ? "/v1/chat/completions" : "/v1/completions";
            string body = BuildBody(spec);
            using var req = new HttpRequestMessage(HttpMethod.Post, path)
            {
                Content = new StringContent(body, Encoding.UTF8, "application/json"),
            };

            using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
            if (!resp.IsSuccessStatusCode)
            {
                string err = await SafeReadAsync(resp, ct);
                result.Success = false;
                result.Error = $"HTTP {(int)resp.StatusCode}: {Truncate(err, 200)}";
                result.EndMs = dispatchMs + sw.Elapsed.TotalMilliseconds;
                return result;
            }

            await using var stream = await resp.Content.ReadAsStreamAsync(ct);
            using var reader = new StreamReader(stream, Encoding.UTF8);

            int contentChunks = 0;
            int? usageCompletion = null;
            int? usagePrompt = null;

            string? line;
            while ((line = await reader.ReadLineAsync(ct)) != null)
            {
                if (line.Length == 0) continue;
                if (!line.StartsWith("data:", StringComparison.Ordinal)) continue;

                string payload = line.Substring(5).Trim();
                if (payload == "[DONE]") break;

                using var doc = JsonDocument.Parse(payload);
                var root = doc.RootElement;

                // Token content (chat: choices[0].delta.content, completions: choices[0].text)
                string? piece = ExtractContent(root);
                if (!string.IsNullOrEmpty(piece))
                {
                    result.TokenArrivalsMs.Add(sw.Elapsed.TotalMilliseconds);
                    contentChunks++;
                }

                // Usage (vLLM/OpenAI emit it in the final chunk with stream_options.include_usage)
                if (root.TryGetProperty("usage", out var usage) && usage.ValueKind == JsonValueKind.Object)
                {
                    if (usage.TryGetProperty("completion_tokens", out var ctok) && ctok.TryGetInt32(out int cv)) usageCompletion = cv;
                    if (usage.TryGetProperty("prompt_tokens", out var ptok) && ptok.TryGetInt32(out int pv)) usagePrompt = pv;
                }
            }

            // Trust ONLY authoritative server usage for token counts. SSE content-chunk counts are NOT token
            // counts (a chunk may carry several tokens, and some decoded tokens produce no text), so
            // substituting them would corrupt token-throughput / TPOT. When usage is absent, mark token
            // metrics UNAVAILABLE (excluded from token aggregates) instead of back-filling with chunk counts.
            result.TtftMs = result.TokenArrivalsMs.Count > 0 ? result.TokenArrivalsMs[0] : null;
            result.EndMs = dispatchMs + sw.Elapsed.TotalMilliseconds;

            if (usageCompletion.HasValue)
            {
                result.OutputTokens = usageCompletion.Value;
                if (usagePrompt.HasValue) result.PromptTokens = usagePrompt.Value;
                result.Success = result.OutputTokens > 0 || contentChunks > 0;
                if (!result.Success) result.Error = "no tokens returned";
            }
            else
            {
                // Endpoint emitted no usage object: the request may have produced text (contentChunks),
                // but exact token counts are unknown. Flag it so aggregation excludes it from token metrics.
                result.TokenMetricsUnavailable = true;
                result.Success = contentChunks > 0;
                result.Error = result.Success
                    ? "token usage unavailable (no usage object emitted); token metrics excluded for this request"
                    : "no tokens returned";
            }
            return result;
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.Error = ex.GetType().Name + ": " + ex.Message;
            result.EndMs = dispatchMs + sw.Elapsed.TotalMilliseconds;
            return result;
        }
    }

    private static string? ExtractContent(JsonElement root)
    {
        if (!root.TryGetProperty("choices", out var choices) || choices.ValueKind != JsonValueKind.Array || choices.GetArrayLength() == 0)
            return null;
        var choice = choices[0];

        // chat.completion.chunk
        if (choice.TryGetProperty("delta", out var delta) &&
            delta.TryGetProperty("content", out var content) &&
            content.ValueKind == JsonValueKind.String)
        {
            return content.GetString();
        }

        // text_completion (legacy completions)
        if (choice.TryGetProperty("text", out var text) && text.ValueKind == JsonValueKind.String)
        {
            return text.GetString();
        }

        return null;
    }

    private string BuildBody(RequestSpec spec)
    {
        var streamOptions = new Dictionary<string, object?> { ["include_usage"] = true };
        Dictionary<string, object?> body;
        if (_chat)
        {
            body = new Dictionary<string, object?>
            {
                ["model"] = _o.Model,
                ["messages"] = new object[]
                {
                    new Dictionary<string, object?> { ["role"] = "user", ["content"] = spec.PromptText ?? "" },
                },
                ["max_tokens"] = spec.MaxTokens,
                ["temperature"] = _o.Temperature,
                ["stream"] = true,
                ["stream_options"] = streamOptions,
            };
        }
        else
        {
            body = new Dictionary<string, object?>
            {
                ["model"] = _o.Model,
                ["prompt"] = spec.PromptText ?? "",
                ["max_tokens"] = spec.MaxTokens,
                ["temperature"] = _o.Temperature,
                ["stream"] = true,
                ["stream_options"] = streamOptions,
            };
        }
        return JsonSerializer.Serialize(body);
    }

    private static async Task<string> SafeReadAsync(HttpResponseMessage resp, CancellationToken ct)
    {
        try { return await resp.Content.ReadAsStringAsync(ct); }
        catch { return "<unreadable body>"; }
    }

    private static string Truncate(string s, int n) => s.Length <= n ? s : s.Substring(0, n) + "...";
}
