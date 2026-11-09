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
        int vocab = tokenizer.VocabularySize;

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

        var servable = new ServableModelWrapper<float>(
            TestModelName,
            inputDimension: 1,
            outputDimension: vocab,
            predictFunc: v => v,
            generationForward: forward);

        repo.LoadModel(TestModelName, servable);
        tokenizers.Register(TestModelName, tokenizer);
        return Task.CompletedTask;
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
        Assert.Contains(list!.Data, m => m.Id == TestModelName);

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
        Assert.Single(chatResp!.Choices);
        Assert.Equal("assistant", chatResp.Choices[0].Message?.Role);
        Assert.False(string.IsNullOrEmpty(chatResp.Choices[0].Message?.Content));
        Assert.NotNull(chatResp.Usage);
        Assert.Equal(6, chatResp.Usage!.CompletionTokens);

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
        Assert.True(countObj!.InputTokens > 0);

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
        Assert.Equal("message", parsedMsg!.Type);
        Assert.Equal("assistant", parsedMsg.Role);
        Assert.Equal("claude-sonnet-4-5-20250929", parsedMsg.Model);
        Assert.Single(parsedMsg.Content);
        Assert.Equal("text", parsedMsg.Content[0].Type);
        Assert.False(string.IsNullOrEmpty(parsedMsg.Content[0].Text));
        Assert.Equal(5, parsedMsg.Usage.OutputTokens);

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
        string sse = await streamResp.Content.ReadAsStringAsync();
        Assert.Contains("event: message_start", sse);
        Assert.Contains("event: content_block_start", sse);
        Assert.Contains("event: content_block_delta", sse);
        Assert.Contains("\"text_delta\"", sse);
        Assert.Contains("event: content_block_stop", sse);
        Assert.Contains("event: message_delta", sse);
        Assert.Contains("event: message_stop", sse);
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
    }
}
