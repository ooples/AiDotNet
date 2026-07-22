using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Serving.Benchmarks.Backends;

/// <summary>
/// Backend for AiDotNet's current native generation endpoint
/// (<c>POST api/inference/generate/{model}</c>, <see cref="AiDotNet.Serving.Models.SpeculativeDecodingRequest"/>).
/// This endpoint is NON-streaming and token-ID native, so TTFT / inter-token latency are unavailable
/// (they require the OpenAI streaming route). It still measures request throughput, output-token
/// throughput, and end-to-end latency — i.e. the raw engine speed today.
/// </summary>
public sealed class AiDotNetNativeBackend : IServingBackend
{
    private static readonly JsonSerializerOptions ReadOpts = new() { PropertyNameCaseInsensitive = true };

    private readonly HttpClient _http;
    private readonly BenchmarkOptions _o;

    public AiDotNetNativeBackend(HttpClient http, BenchmarkOptions o)
    {
        _http = http;
        _o = o;
    }

    public string Name => "aidotnet-native";

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
            // ASP.NET Core's default System.Text.Json uses camelCase property names.
            var body = new Dictionary<string, object?>
            {
                ["inputTokens"] = spec.PromptTokenIds ?? Array.Empty<int>(),
                ["maxNewTokens"] = spec.MaxTokens,
                // Pass the requested temperature through unchanged. The native endpoint requires > 0; the
                // unsupported greedy (temperature 0) case is rejected up front in BenchmarkOptions.Parse
                // rather than silently substituted here (which would benchmark different semantics).
                ["temperature"] = _o.Temperature,
            };
            string json = JsonSerializer.Serialize(body);

            string path = $"/api/inference/generate/{Uri.EscapeDataString(_o.Model)}";
            using var req = new HttpRequestMessage(HttpMethod.Post, path)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json"),
            };

            using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseContentRead, ct);
            string payload = await resp.Content.ReadAsStringAsync(ct);

            if (!resp.IsSuccessStatusCode)
            {
                result.Success = false;
                result.Error = $"HTTP {(int)resp.StatusCode}: {Truncate(payload, 200)}";
                result.EndMs = dispatchMs + sw.Elapsed.TotalMilliseconds;
                return result;
            }

            var parsed = JsonSerializer.Deserialize<NativeResponse>(payload, ReadOpts);
            if (parsed?.Error is { Length: > 0 })
            {
                result.Success = false;
                result.Error = parsed.Error;
            }
            else
            {
                result.OutputTokens = parsed?.NumGenerated ?? 0;
                result.Success = result.OutputTokens > 0;
                if (!result.Success) result.Error = "no tokens generated";
            }

            result.TtftMs = null; // non-streaming
            result.EndMs = dispatchMs + sw.Elapsed.TotalMilliseconds;
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

    private sealed class NativeResponse
    {
        public int NumGenerated { get; set; }
        public double AcceptanceRate { get; set; }
        public long ProcessingTimeMs { get; set; }
        public string? Error { get; set; }
    }

    private static string Truncate(string s, int n) => s.Length <= n ? s : s.Substring(0, n) + "...";
}
