using System.Net;
using System.Text;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Models.Anthropic;
using AiDotNet.Serving.Models.OpenAi;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Algorithms;
using Microsoft.Extensions.DependencyInjection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// End-to-end integration tests verifying OpenAI V1 (<c>/v1/models</c>, <c>/v1/chat/completions</c>, <c>/v1/completions</c>)
/// for OpenCode CLI and Anthropic Messages API (<c>/v1/messages</c>, <c>/v1/messages/count_tokens</c>) for Claude Code CLI.
/// </summary>
[Collection("ServingIntegrationTests")]
public sealed class CliServingProtocolIntegrationTests : IClassFixture<ServingTestWebApplicationFactory>, IAsyncLifetime
{
    private const string TestModelName = "hre-cli-test-model";
    private readonly ServingTestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public CliServingProtocolIntegrationTests(ServingTestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = _factory.CreateClient();
    }

    public Task InitializeAsync()
    {
        using var scope = _factory.Services.CreateScope();
        var repo = scope.ServiceProvider.GetRequiredService<IModelRepository>();
        var tokenizers = scope.ServiceProvider.GetRequiredService<ITokenizerRegistry>();

        foreach (var m in repo.GetAllModelInfo())
        {
            repo.UnloadModel(m.Name);
            tokenizers.Remove(m.Name);
        }

        var tokenizer = CharacterTokenizer.CreateAscii();
        repo.LoadModel(TestModelName, CreateServable(TestModelName, tokenizer.VocabularySize));
        tokenizers.Register(TestModelName, tokenizer);
        return Task.CompletedTask;
    }

    // Deterministic generative stub: position p always peaks at token (vocab - 10 - p % 10).
    private static ServableModelWrapper<float> CreateServable(string name, int vocab)
    {
        Func<Tensor<float>, Tensor<float>> forward = input =>
        {
            int seq = input.Shape[input.Shape.Length - 1];
            var logits = new Tensor<float>(new[] { 1, seq, vocab });
            for (int pos = 0; pos < seq; pos++)
            {
                int peak = Math.Clamp(vocab - 10 - (pos % 10), 0, vocab - 1);
                logits[new[] { 0, pos, peak }] = 25f;
            }
            return logits;
        };

        return new ServableModelWrapper<float>(
            name,
            inputDimension: 1,
            outputDimension: vocab,
            predictFunc: v => v,
            generationForward: forward);
    }

    public Task DisposeAsync()
    {
        using var scope = _factory.Services.CreateScope();
        var repo = scope.ServiceProvider.GetRequiredService<IModelRepository>();
        var tokenizers = scope.ServiceProvider.GetRequiredService<ITokenizerRegistry>();
        repo.UnloadModel(TestModelName);
        tokenizers.Remove(TestModelName);
        return Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task OpenAi_ChatCompletions_NonStreamingAndStreaming_WorksForOpenCodeCli()
    {
        var listResp = await _client.GetAsync("/v1/models");
        listResp.EnsureSuccessStatusCode();
        var list = JsonConvert.DeserializeObject<ModelList>(await listResp.Content.ReadAsStringAsync());
        Assert.NotNull(list);
        Assert.Contains(list.Data, m => m.Id == TestModelName);

        // Non-streaming with developer role and gpt-4o CLI alias resolution
        var reqBody = new
        {
            model = "gpt-4o",
            messages = new object[]
            {
                new { role = "developer", content = "You are a concise coding assistant." },
                new { role = "user", content = "Hello" }
            },
            max_tokens = 6,
            temperature = 0.0,
            stream = false
        };
        var resp = await _client.PostAsync(
            "/v1/chat/completions",
            new StringContent(JsonConvert.SerializeObject(reqBody), Encoding.UTF8, "application/json"));
        resp.EnsureSuccessStatusCode();

        var chatResp = JsonConvert.DeserializeObject<ChatCompletionResponse>(await resp.Content.ReadAsStringAsync());
        Assert.NotNull(chatResp);
        Assert.Single(chatResp.Choices);
        Assert.Equal("assistant", chatResp.Choices[0].Message?.Role);
        Assert.False(string.IsNullOrEmpty(chatResp.Choices[0].Message?.Content));
        Assert.NotNull(chatResp.Usage);
        Assert.Equal(6, chatResp.Usage.CompletionTokens);

        // Streaming SSE
        var streamReqBody = new
        {
            model = TestModelName,
            messages = new object[] { new { role = "user", content = "Stream test" } },
            max_tokens = 4,
            temperature = 0.0,
            stream = true
        };
        var streamResp = await _client.PostAsync(
            "/v1/chat/completions",
            new StringContent(JsonConvert.SerializeObject(streamReqBody), Encoding.UTF8, "application/json"));
        streamResp.EnsureSuccessStatusCode();
        Assert.Equal("text/event-stream", streamResp.Content.Headers.ContentType?.MediaType);
        string sse = await streamResp.Content.ReadAsStringAsync();
        Assert.Contains("chat.completion.chunk", sse);
        Assert.Contains("data: [DONE]", sse);
    }

    [Fact(Timeout = 60000)]
    public async Task Anthropic_MessagesAndCountTokens_NonStreamingAndStreaming_WorksForClaudeCodeCli()
    {
        // 1. Token count endpoint (/v1/messages/count_tokens)
        var countPayload = new
        {
            model = "claude-sonnet-4-5-20250929",
            system = "System instructions",
            messages = new object[]
            {
                new
                {
                    role = "user",
                    content = new object[]
                    {
                        new { type = "text", text = "Count my tokens" }
                    }
                }
            }
        };
        var countResp = await _client.PostAsync(
            "/v1/messages/count_tokens",
            new StringContent(JsonConvert.SerializeObject(countPayload), Encoding.UTF8, "application/json"));
        countResp.EnsureSuccessStatusCode();
        var countObj = JsonConvert.DeserializeObject<AnthropicCountTokensResponse>(await countResp.Content.ReadAsStringAsync());
        Assert.NotNull(countObj);
        // Exact tokenizer count of the rendered prompt; the length/4 heuristic fallback would not match it.
        Assert.Equal(PromptTokens(("system", "System instructions"), ("user", "Count my tokens")), countObj.InputTokens);

        // 2. Non-streaming POST /v1/messages with Claude Code CLI model alias
        var msgPayload = new
        {
            model = "claude-sonnet-4-5-20250929",
            max_tokens = 5,
            temperature = 0.0,
            stream = false,
            system = new object[] { new { type = "text", text = "Be brief." } },
            messages = new object[]
            {
                new { role = "user", content = "Write code" }
            }
        };
        var msgResp = await _client.PostAsync(
            "/v1/messages",
            new StringContent(JsonConvert.SerializeObject(msgPayload), Encoding.UTF8, "application/json"));
        msgResp.EnsureSuccessStatusCode();
        var parsedMsg = JsonConvert.DeserializeObject<AnthropicMessageResponse>(await msgResp.Content.ReadAsStringAsync());
        Assert.NotNull(parsedMsg);
        Assert.Equal("message", parsedMsg.Type);
        Assert.Equal("assistant", parsedMsg.Role);
        Assert.Equal("claude-sonnet-4-5-20250929", parsedMsg.Model);
        Assert.Single(parsedMsg.Content);
        Assert.Equal("text", parsedMsg.Content[0].Type);
        Assert.Equal(PromptTokens(("system", "Be brief."), ("user", "Write code")), parsedMsg.Usage.InputTokens);
        Assert.Equal(5, parsedMsg.Usage.OutputTokens);
        Assert.Equal("max_tokens", parsedMsg.StopReason);
        Assert.Equal(5, parsedMsg.Content[0].Text.Length);

        // 3. Streaming SSE POST /v1/messages
        var streamPayload = new
        {
            model = TestModelName,
            max_tokens = 4,
            temperature = 0.0,
            stream = true,
            messages = new object[]
            {
                new { role = "user", content = "Stream Anthropic SSE" }
            }
        };
        var streamResp = await _client.PostAsync(
            "/v1/messages",
            new StringContent(JsonConvert.SerializeObject(streamPayload), Encoding.UTF8, "application/json"));
        streamResp.EnsureSuccessStatusCode();
        Assert.Equal("text/event-stream", streamResp.Content.Headers.ContentType?.MediaType);
        var events = ParseSse(await streamResp.Content.ReadAsStringAsync());
        var names = events.Select(e => e.Event).ToList();
        Assert.Equal(new[] { "message_start", "content_block_start", "ping" }, names.Take(3));
        Assert.Equal(new[] { "content_block_stop", "message_delta", "message_stop" }, names.Skip(names.Count - 3));
        var deltas = events.Where(e => e.Event == "content_block_delta").ToList();
        Assert.NotEmpty(deltas);
        Assert.Equal(names.Count - 6, deltas.Count);
        Assert.All(deltas, d => Assert.Equal("text_delta", (string?)d.Data["delta"]?["type"]));
        string streamedText = string.Concat(deltas.Select(d => (string?)d.Data["delta"]?["text"]));
        Assert.Equal(4, streamedText.Length);
        var messageDelta = events.Single(e => e.Event == "message_delta").Data;
        Assert.Equal("max_tokens", (string?)messageDelta["delta"]?["stop_reason"]);
        Assert.Equal(4, (int?)messageDelta["usage"]?["output_tokens"]);
    }

    [Fact(Timeout = 60000)]
    public async Task Anthropic_UnknownNonCliModel_Returns404()
    {
        var payload = new
        {
            model = "completely-nonexistent-custom-model",
            max_tokens = 4,
            messages = new object[] { new { role = "user", content = "Hi" } }
        };
        var resp = await _client.PostAsync(
            "/v1/messages",
            new StringContent(JsonConvert.SerializeObject(payload), Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.NotFound, resp.StatusCode);
        await AssertAnthropicErrorAsync(resp, "not_found_error");
    }

    [Fact(Timeout = 60000)]
    public async Task Anthropic_StopSequence_StreamingMatchesNonStreaming_AndNeverLeaksThePartialStop()
    {
        // Learn the deterministic (greedy) output, then pick a 2-char stop from its middle.
        string full = await AnthropicTextAsync(new { model = TestModelName, max_tokens = 8, temperature = 0.0, messages = Turn("Stop test") });
        Assert.Equal(8, full.Length);
        // Greedy decoding is deterministic, so the speculative non-streaming path must emit exactly what the
        // non-speculative streaming path does (temperature 0 once divided by zero and emitted "~" per step).
        var plainStream = ParseSse(await (await PostJsonAsync("/v1/messages", new { model = TestModelName, max_tokens = 8, temperature = 0.0, stream = true, messages = Turn("Stop test") })).Content.ReadAsStringAsync());
        Assert.Equal(full, string.Concat(plainStream.Where(e => e.Event == "content_block_delta").Select(e => (string?)e.Data["delta"]?["text"])));
        string stop = full.Substring(3, 2);
        int cut = full.IndexOf(stop, StringComparison.Ordinal);

        var nonStream = await PostJsonAsync("/v1/messages", new { model = TestModelName, max_tokens = 8, temperature = 0.0, stop_sequences = new[] { stop }, messages = Turn("Stop test") });
        nonStream.EnsureSuccessStatusCode();
        var parsed = JsonConvert.DeserializeObject<AnthropicMessageResponse>(await nonStream.Content.ReadAsStringAsync());
        Assert.NotNull(parsed);
        Assert.Equal(full.Substring(0, cut), parsed.Content[0].Text);
        Assert.Equal("stop_sequence", parsed.StopReason);
        Assert.Equal(stop, parsed.StopSequence);

        var stream = await PostJsonAsync("/v1/messages", new { model = TestModelName, max_tokens = 8, temperature = 0.0, stream = true, stop_sequences = new[] { stop }, messages = Turn("Stop test") });
        stream.EnsureSuccessStatusCode();
        var events = ParseSse(await stream.Content.ReadAsStringAsync());
        string streamed = string.Concat(events.Where(e => e.Event == "content_block_delta").Select(e => (string?)e.Data["delta"]?["text"]));
        // Every delta is final once sent, so the concatenation must equal the truncated non-streaming text:
        // a leaked first character of the stop would make it longer.
        Assert.Equal(full.Substring(0, cut), streamed);
        var messageDelta = events.Single(e => e.Event == "message_delta").Data;
        Assert.Equal("stop_sequence", (string?)messageDelta["delta"]?["stop_reason"]);
        Assert.Equal(stop, (string?)messageDelta["delta"]?["stop_sequence"]);
        Assert.Equal("message_stop", events[events.Count - 1].Event);
    }

    [Fact(Timeout = 60000)]
    public async Task Anthropic_EmptyMessages_Returns400WithAnthropicErrorShape()
    {
        foreach (string route in new[] { "/v1/messages", "/v1/messages/count_tokens" })
        {
            var resp = await PostJsonAsync(route, new { model = TestModelName, max_tokens = 4, messages = Array.Empty<object>() });
            Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
            await AssertAnthropicErrorAsync(resp, "invalid_request_error");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Anthropic_MaxTokensAboveServingLimit_Returns400()
    {
        // ServingLimitsOptions defaults to a 4096-token completion ceiling.
        var resp = await PostJsonAsync("/v1/messages", new { model = TestModelName, max_tokens = 1_000_000, messages = Turn("Hi") });

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        await AssertAnthropicErrorAsync(resp, "invalid_request_error");
    }

    [Fact(Timeout = 60000)]
    public async Task Anthropic_CountTokens_UnknownModel_FallsBackToLengthHeuristic()
    {
        var resp = await PostJsonAsync("/v1/messages/count_tokens", new { model = "completely-nonexistent-custom-model", messages = Turn("Count these characters") });
        resp.EnsureSuccessStatusCode();
        var parsed = JsonConvert.DeserializeObject<AnthropicCountTokensResponse>(await resp.Content.ReadAsStringAsync());

        string prompt = ChatTemplate.Render(new[] { ("user", "Count these characters") });
        Assert.NotNull(parsed);
        Assert.Equal((prompt.Length + 3) / 4, parsed.InputTokens);
    }

    [Fact(Timeout = 60000)]
    public async Task ModelLookup_IsCaseInsensitive()
    {
        string text = await AnthropicTextAsync(new { model = TestModelName.ToUpperInvariant(), max_tokens = 3, temperature = 0.0, messages = Turn("Case") });

        Assert.Equal(3, text.Length);
    }

    [Fact(Timeout = 60000)]
    public async Task CliAlias_WithTwoGenerativeModelsLoaded_IsAmbiguousAndReturns404()
    {
        const string secondModel = "hre-cli-test-model-2";
        using (var scope = _factory.Services.CreateScope())
        {
            var tokenizer = CharacterTokenizer.CreateAscii();
            scope.ServiceProvider.GetRequiredService<IModelRepository>().LoadModel(secondModel, CreateServable(secondModel, tokenizer.VocabularySize));
            scope.ServiceProvider.GetRequiredService<ITokenizerRegistry>().Register(secondModel, tokenizer);
        }

        try
        {
            var anthropic = await PostJsonAsync("/v1/messages", new { model = "claude-sonnet-4-5-20250929", max_tokens = 4, messages = Turn("Hi") });
            Assert.Equal(HttpStatusCode.NotFound, anthropic.StatusCode);
            await AssertAnthropicErrorAsync(anthropic, "not_found_error");

            var openAi = await PostJsonAsync("/v1/chat/completions", new { model = "gpt-4o", max_tokens = 4, messages = Turn("Hi") });
            Assert.Equal(HttpStatusCode.NotFound, openAi.StatusCode);

            // An explicitly named model is unaffected by the ambiguity.
            string text = await AnthropicTextAsync(new { model = secondModel, max_tokens = 4, temperature = 0.0, messages = Turn("Hi") });
            Assert.Equal(4, text.Length);
        }
        finally
        {
            using var scope = _factory.Services.CreateScope();
            scope.ServiceProvider.GetRequiredService<IModelRepository>().UnloadModel(secondModel);
            scope.ServiceProvider.GetRequiredService<ITokenizerRegistry>().Remove(secondModel);
        }
    }

    private static object[] Turn(string userText) => new object[] { new { role = "user", content = userText } };

    private static int PromptTokens(params (string Role, string Content)[] turns) =>
        CharacterTokenizer.CreateAscii().Encode(ChatTemplate.Render(turns)).TokenIds.Count;

    private Task<HttpResponseMessage> PostJsonAsync(string route, object payload) =>
        _client.PostAsync(route, new StringContent(JsonConvert.SerializeObject(payload), Encoding.UTF8, "application/json"));

    private async Task<string> AnthropicTextAsync(object payload)
    {
        var resp = await PostJsonAsync("/v1/messages", payload);
        resp.EnsureSuccessStatusCode();
        var parsed = JsonConvert.DeserializeObject<AnthropicMessageResponse>(await resp.Content.ReadAsStringAsync());
        Assert.NotNull(parsed);
        return parsed.Content[0].Text ?? string.Empty;
    }

    private static async Task AssertAnthropicErrorAsync(HttpResponseMessage resp, string expectedErrorType)
    {
        var body = JObject.Parse(await resp.Content.ReadAsStringAsync());
        Assert.Equal("error", (string?)body["type"]);
        Assert.Equal(expectedErrorType, (string?)body["error"]?["type"]);
        Assert.False(string.IsNullOrWhiteSpace((string?)body["error"]?["message"]));
    }

    private static List<(string Event, JObject Data)> ParseSse(string sse)
    {
        var events = new List<(string Event, JObject Data)>();
        foreach (string frame in sse.Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            string? name = null;
            string? data = null;
            foreach (string line in frame.Split('\n'))
            {
                if (line.StartsWith("event: ", StringComparison.Ordinal)) name = line.Substring(7);
                else if (line.StartsWith("data: ", StringComparison.Ordinal)) data = line.Substring(6);
            }
            Assert.False(name is null || data is null, $"Malformed SSE frame: {frame}");
            events.Add((name ?? string.Empty, JObject.Parse(data ?? "{}")));
        }
        return events;
    }
}

