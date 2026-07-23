using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using Xunit;
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Tests.UnitTests.RetrievalAugmentedGeneration.Generators;

/// <summary>
/// Unit tests for <see cref="ChatClientGenerator{T}"/> — the real-LLM RAG generator that bridges
/// <see cref="IChatClient{T}"/>. A fake in-memory chat client stands in for a provider connector, so
/// these run in CI with no network: they assert prompt/role construction, option propagation, buffered
/// text, and streaming delta concatenation.
/// </summary>
public class ChatClientGeneratorTests
{
    /// <summary>Records the request and returns/streams a canned answer — no HTTP.</summary>
    private sealed class FakeChatClient : IChatClient<double>
    {
        private readonly string _answer;
        public string ModelId => "fake-model";
        public IReadOnlyList<ChatMessage>? LastMessages { get; private set; }
        public ChatOptions? LastOptions { get; private set; }

        public FakeChatClient(string answer) => _answer = answer;

        public Task<ChatResponse> GetResponseAsync(
            IReadOnlyList<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            LastMessages = messages;
            LastOptions = options;
            var msg = new ChatMessage(ChatRole.Assistant, _answer);
            return Task.FromResult(new ChatResponse(msg, ChatFinishReason.Stop, usage: null, modelId: ModelId));
        }

        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IReadOnlyList<ChatMessage> messages, ChatOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            LastMessages = messages;
            LastOptions = options;
            // Split the answer into word-sized deltas to mimic token streaming.
            foreach (var word in _answer.Split(' '))
            {
                cancellationToken.ThrowIfCancellationRequested();
                await Task.Yield();
                yield return ChatResponseUpdate.ForText(word + " ");
            }
        }
    }

    [Fact]
    public void Generate_ReturnsChatClientText_AndSendsSystemThenUser()
    {
        var fake = new FakeChatClient("Transformers use self-attention [1].");
        var gen = new ChatClientGenerator<double>(fake, systemPrompt: "You are terse.", temperature: 0.2, maxOutputTokens: 256);

        var text = gen.Generate("How do transformers work?");

        Assert.Equal("Transformers use self-attention [1].", text);
        Assert.NotNull(fake.LastMessages);
        Assert.Equal(2, fake.LastMessages!.Count);
        Assert.Equal(ChatRole.System, fake.LastMessages[0].Role);
        Assert.Equal("You are terse.", fake.LastMessages[0].Text);
        Assert.Equal(ChatRole.User, fake.LastMessages[1].Role);
        Assert.Equal("How do transformers work?", fake.LastMessages[1].Text);
        Assert.Equal(0.2, fake.LastOptions!.Temperature);
        Assert.Equal(256, fake.LastOptions!.MaxOutputTokens);
    }

    [Fact]
    public void Generate_WithoutSystemPrompt_SendsOnlyUserMessage()
    {
        var fake = new FakeChatClient("answer");
        var gen = new ChatClientGenerator<double>(fake);

        gen.Generate("q");

        Assert.Single(fake.LastMessages!);
        Assert.Equal(ChatRole.User, fake.LastMessages![0].Role);
    }

    [Fact]
    public async Task GenerateStreamAsync_ConcatenatesToFullAnswer()
    {
        var fake = new FakeChatClient("alpha beta gamma");
        var gen = new ChatClientGenerator<double>(fake);

        var parts = new List<string>();
        await foreach (var delta in gen.GenerateStreamAsync("q"))
        {
            parts.Add(delta);
        }

        Assert.True(parts.Count >= 3, "expected multiple streamed deltas");
        Assert.Equal("alpha beta gamma ", string.Concat(parts));
    }

    [Fact]
    public async Task GenerateStreamAsync_RejectsEmptyPrompt()
    {
        var gen = new ChatClientGenerator<double>(new FakeChatClient("x"));
        await Assert.ThrowsAsync<ArgumentException>(async () =>
        {
            await foreach (var _ in gen.GenerateStreamAsync("   ")) { }
        });
    }

    [Fact]
    public void Constructor_NullChatClient_Throws()
        => Assert.Throws<ArgumentNullException>(() => new ChatClientGenerator<double>(null!));
}
