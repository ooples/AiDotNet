using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Interfaces;
using AiDotNet.Serving.Engine;
using AiDotNet.Serving.Engine.Http;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Integration tests for the OpenAI-compatible <see cref="InferenceServer{T}"/>: they start a real server on a
/// dynamic local port and exercise /health, /v1/models, and /v1/completions (streaming + non-streaming) over
/// HTTP with a deterministic char-level counter model, so exact generated text can be asserted.
/// </summary>
public class InferenceServerHttpTests
{
    private const int Vocab = 128;

    // Char-level counter LM: next token = (last + 1) mod vocab, one-hot at [1, seq, vocab].
    private sealed class CounterLm : ICausalLmModel<double>
    {
        public int VocabularySize => Vocab;
        public int? EosTokenId => null;
        public Tensor<double> ForwardLogits(Tensor<double> tokenIds)
        {
            int n = tokenIds.Shape[tokenIds.Shape.Length - 1];
            int last = (int)Math.Round(Convert.ToDouble(tokenIds[0, n - 1]));
            var t = new Tensor<double>(new[] { 1, n, Vocab });
            t[0, n - 1, (last + 1) % Vocab] = 1.0;
            return t;
        }
    }

    private sealed class CharTokenizer : IGenerationTokenizer
    {
        public int EosTokenId => -1;
        public IReadOnlyList<int> Encode(string text) => text.Select(c => (int)c).ToArray();
        public string Decode(IReadOnlyList<int> tokenIds) => new string(tokenIds.Select(id => (char)id).ToArray());
    }

    private static InferenceServer<double> StartServer()
    {
        var engine = new ContinuousBatchingEngine<double>(new RecomputeModelRunner<double>(new CounterLm()));
        var host = new AsyncEngineHost<double>(engine);
        return new InferenceServer<double>(
            host, new CharTokenizer(), "test-model",
            new SamplingParameters { Temperature = 0.0, MaxTokens = 16 },
            new[] { "http://127.0.0.1:0" });
    }

    [Fact(Timeout = 60000)]
    public async Task Health_ReturnsOk()
    {
        using var server = StartServer();
        using var client = new HttpClient { BaseAddress = new Uri(server.Urls[0]) };
        var json = JObject.Parse(await client.GetStringAsync("/health"));
        Assert.Equal("ok", (string?)json["status"]);
    }

    [Fact(Timeout = 60000)]
    public async Task Models_ListsServedModel()
    {
        using var server = StartServer();
        using var client = new HttpClient { BaseAddress = new Uri(server.Urls[0]) };
        var json = JObject.Parse(await client.GetStringAsync("/v1/models"));
        Assert.Equal("test-model", (string?)json["data"]![0]!["id"]);
    }

    [Fact(Timeout = 60000)]
    public async Task Completions_NonStreaming_ReturnsGeneratedText()
    {
        using var server = StartServer();
        using var client = new HttpClient { BaseAddress = new Uri(server.Urls[0]) };

        var body = new StringContent(
            "{\"prompt\":\"A\",\"max_tokens\":3,\"temperature\":0}", Encoding.UTF8, "application/json");
        var response = await client.PostAsync("/v1/completions", body);
        response.EnsureSuccessStatusCode();

        var json = JObject.Parse(await response.Content.ReadAsStringAsync());
        // 'A'(65) -> 66,67,68 = "BCD"
        Assert.Equal("BCD", (string?)json["choices"]![0]!["text"]);
        Assert.Equal("stop", (string?)json["choices"]![0]!["finish_reason"]);
        Assert.Equal(3, (int)json["usage"]!["completion_tokens"]!);
    }

    [Fact(Timeout = 60000)]
    public async Task Completions_Streaming_EmitsSseChunksEndingDone()
    {
        using var server = StartServer();
        using var client = new HttpClient { BaseAddress = new Uri(server.Urls[0]) };

        var body = new StringContent(
            "{\"prompt\":\"A\",\"max_tokens\":3,\"temperature\":0,\"stream\":true}", Encoding.UTF8, "application/json");
        var response = await client.PostAsync("/v1/completions", body);
        response.EnsureSuccessStatusCode();
        Assert.Contains("text/event-stream", response.Content.Headers.ContentType!.ToString());

        string sse = await response.Content.ReadAsStringAsync();
        Assert.Contains("data: [DONE]", sse);

        // Reassemble the streamed deltas into the full text.
        var text = new StringBuilder();
        foreach (var line in sse.Split('\n'))
        {
            var trimmed = line.Trim();
            if (!trimmed.StartsWith("data: ") || trimmed.Contains("[DONE]")) continue;
            var chunk = JObject.Parse(trimmed.Substring("data: ".Length));
            text.Append((string?)chunk["choices"]![0]!["text"]);
        }
        Assert.Equal("BCD", text.ToString());
    }

    [Fact(Timeout = 60000)]
    public async Task Completions_MissingPrompt_Returns400()
    {
        using var server = StartServer();
        using var client = new HttpClient { BaseAddress = new Uri(server.Urls[0]) };
        var body = new StringContent("{\"max_tokens\":3}", Encoding.UTF8, "application/json");
        var response = await client.PostAsync("/v1/completions", body);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }
}
