using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Pipeline;

public sealed class JsonFileChatInteractionStoreTests : IDisposable
{
    private static readonly ChatMessage[] Prompt = { ChatMessage.User("improve this program") };

    private readonly string _directory;
    private readonly string _path;

    public JsonFileChatInteractionStoreTests()
    {
        _directory = Path.Combine(Path.GetTempPath(), "aidotnet-chatstore-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_directory);
        _path = Path.Combine(_directory, "session.json");
    }

    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); }
        catch (IOException) { }
    }

    [Fact]
    public async Task ARecordedSessionReplaysFromTheFileWithoutCallingTheModel()
    {
        // This is what makes a model-driven benchmark reproducible; OpenEvolve logs
        // prompts for inspection but never replays them.
        var live = StubChatClient.Text("gpt-test", "the improved program", new ChatUsage(120, 45));
        IChatClient<double> recorder = ChatClientPipelineFactory.Create(
            live,
            new ChatClientOptions
            {
                MaxRetries = 0,
                RecordingMode = ChatClientRecordingMode.Record,
                RecordingPath = _path
            });

        ChatResponse recorded = await recorder.GetResponseAsync(Prompt);
        Assert.Equal("the improved program", recorded.Text);
        Assert.Equal(1, live.Calls);
        Assert.True(File.Exists(_path));

        var offline = StubChatClient.AlwaysThrows("gpt-test", () => new InvalidOperationException("no network"));
        IChatClient<double> replayer = ChatClientPipelineFactory.Create(
            offline,
            new ChatClientOptions
            {
                MaxRetries = 0,
                RecordingMode = ChatClientRecordingMode.Replay,
                RecordingPath = _path
            });

        ChatResponse replayed = await replayer.GetResponseAsync(Prompt);

        Assert.Equal("the improved program", replayed.Text);
        Assert.Equal(120, replayed.Usage?.InputTokens);
        Assert.Equal(45, replayed.Usage?.OutputTokens);
        Assert.Equal(0, offline.Calls);
    }

    [Fact]
    public async Task ReplayIsIdenticalAcrossRepeatedRunsOfTheSameFile()
    {
        var live = StubChatClient.Text("gpt-test", "answer");
        IChatClient<double> recorder = ChatClientPipelineFactory.Create(
            live,
            new ChatClientOptions
            {
                MaxRetries = 0,
                RecordingMode = ChatClientRecordingMode.Record,
                RecordingPath = _path
            });
        await recorder.GetResponseAsync(Prompt, new ChatOptions { Temperature = 0.1, Seed = 7 });

        for (int run = 0; run < 3; run++)
        {
            var offline = StubChatClient.AlwaysThrows("gpt-test", () => new InvalidOperationException("no network"));
            IChatClient<double> replayer = ChatClientPipelineFactory.Create(
                offline,
                new ChatClientOptions
                {
                    MaxRetries = 0,
                    RecordingMode = ChatClientRecordingMode.Replay,
                    RecordingPath = _path
                });

            ChatResponse replayed = await replayer.GetResponseAsync(Prompt, new ChatOptions { Temperature = 0.1, Seed = 7 });
            Assert.Equal("answer", replayed.Text);
            Assert.Equal(0, offline.Calls);
        }
    }

    [Fact]
    public async Task DifferentRequestSettingsAreDifferentRecordings()
    {
        var live = StubChatClient.Text("gpt-test", "answer");
        IChatClient<double> recorder = ChatClientPipelineFactory.Create(
            live,
            new ChatClientOptions
            {
                MaxRetries = 0,
                RecordingMode = ChatClientRecordingMode.Record,
                RecordingPath = _path
            });

        await recorder.GetResponseAsync(Prompt, new ChatOptions { Seed = 1 });

        var offline = StubChatClient.AlwaysThrows("gpt-test", () => new InvalidOperationException("no network"));
        IChatClient<double> replayer = ChatClientPipelineFactory.Create(
            offline,
            new ChatClientOptions
            {
                MaxRetries = 0,
                RecordingMode = ChatClientRecordingMode.Replay,
                RecordingPath = _path
            });

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => replayer.GetResponseAsync(Prompt, new ChatOptions { Seed = 2 }));
    }

    [Fact]
    public async Task ReplayWithFallbackCallsTheModelOnceAndReplaysAfterwards()
    {
        var live = StubChatClient.Text("gpt-test", "fresh answer");
        var options = new ChatClientOptions
        {
            MaxRetries = 0,
            RecordingMode = ChatClientRecordingMode.ReplayWithFallback,
            RecordingPath = _path
        };

        IChatClient<double> client = ChatClientPipelineFactory.Create(live, options);
        Assert.Equal("fresh answer", (await client.GetResponseAsync(Prompt)).Text);
        Assert.Equal(1, live.Calls);

        Assert.Equal("fresh answer", (await client.GetResponseAsync(Prompt)).Text);
        Assert.Equal(1, live.Calls);
    }

    [Fact]
    public void ToolCallsFinishReasonsAndUsageSurviveTheRoundTrip()
    {
        var store = new JsonFileChatInteractionStore(_path);
        var response = new ChatResponse(
            ChatMessage.Assistant(new AiContent[]
            {
                new TextContent("looking that up"),
                new ToolCallContent("call-1", "search", "{\"q\":\"primes\"}")
            }),
            ChatFinishReason.ToolCalls,
            new ChatUsage(31, 17),
            "gpt-test");

        store.Save("key-a", response);

        var reopened = new JsonFileChatInteractionStore(_path);
        Assert.True(reopened.TryGet("key-a", out ChatResponse restored));
        Assert.Equal("looking that up", restored.Text);
        Assert.Equal(ChatFinishReason.ToolCalls, restored.FinishReason);
        Assert.Equal("gpt-test", restored.ModelId);
        Assert.Equal(31, restored.Usage?.InputTokens);
        Assert.Single(restored.Message.ToolCalls);
        Assert.Equal("search", restored.Message.ToolCalls[0].ToolName);
        Assert.Equal("{\"q\":\"primes\"}", restored.Message.ToolCalls[0].ArgumentsJson);
    }

    [Fact]
    public void PromptTextIsNotWrittenToDiskUnlessKeyStorageIsRequested()
    {
        var store = new JsonFileChatInteractionStore(_path);
        store.Save("model:gpt;t:24:a very secret prompt;", new ChatResponse(ChatMessage.Assistant("ok")));

        string contents = File.ReadAllText(_path);
        Assert.DoesNotContain("a very secret prompt", contents, StringComparison.Ordinal);

        string keyedPath = Path.Combine(_directory, "keyed.json");
        var keyed = new JsonFileChatInteractionStore(keyedPath, autoFlush: true, storeRequestKeys: true);
        keyed.Save("model:gpt;t:24:a very secret prompt;", new ChatResponse(ChatMessage.Assistant("ok")));
        Assert.Contains("a very secret prompt", File.ReadAllText(keyedPath), StringComparison.Ordinal);
    }

    [Fact]
    public void DeferredFlushingLeavesTheFileUntouchedUntilAsked()
    {
        var store = new JsonFileChatInteractionStore(_path, autoFlush: false);
        store.Save("key-a", new ChatResponse(ChatMessage.Assistant("ok")));

        Assert.False(File.Exists(_path));
        Assert.True(store.HasUnsavedChanges);

        store.Flush();
        Assert.True(File.Exists(_path));
        Assert.False(store.HasUnsavedChanges);
        Assert.Equal(1, new JsonFileChatInteractionStore(_path).Count);
    }

    [Fact]
    public void WritingASecondTimeKeepsThePreviousFileBesideIt()
    {
        var store = new JsonFileChatInteractionStore(_path);
        store.Save("key-a", new ChatResponse(ChatMessage.Assistant("first")));
        store.Save("key-b", new ChatResponse(ChatMessage.Assistant("second")));

        Assert.True(File.Exists(_path));
        Assert.True(File.Exists(_path + ".previous"));
        Assert.Equal(2, new JsonFileChatInteractionStore(_path).Count);
    }

    [Fact]
    public void ARewriteOfTheSameKeyReplacesTheRecording()
    {
        var store = new JsonFileChatInteractionStore(_path);
        store.Save("key-a", new ChatResponse(ChatMessage.Assistant("first")));
        store.Save("key-a", new ChatResponse(ChatMessage.Assistant("second")));

        Assert.Equal(1, store.Count);
        Assert.True(store.TryGet("key-a", out ChatResponse stored));
        Assert.Equal("second", stored.Text);
    }

    [Fact]
    public void ClearingEmptiesTheStoreAndTheFile()
    {
        var store = new JsonFileChatInteractionStore(_path);
        store.Save("key-a", new ChatResponse(ChatMessage.Assistant("first")));
        store.Clear();

        Assert.Equal(0, store.Count);
        Assert.Equal(0, new JsonFileChatInteractionStore(_path).Count);
    }

    [Fact]
    public void AMissingKeyReportsAMissRatherThanThrowing()
    {
        var store = new JsonFileChatInteractionStore(_path);
        Assert.False(store.TryGet("absent", out ChatResponse response));
        Assert.Equal(string.Empty, response.Text);
    }

    [Fact]
    public void ACorruptFileIsRejectedRatherThanSilentlyIgnored()
    {
        File.WriteAllText(_path, "{ this is not json");
        Assert.Throws<InvalidDataException>(() => new JsonFileChatInteractionStore(_path));
    }

    [Fact]
    public void AFileFromAnotherSchemaVersionIsRejected()
    {
        File.WriteAllText(_path, "{\"SchemaVersion\": 99, \"Entries\": []}");
        Assert.Throws<InvalidDataException>(() => new JsonFileChatInteractionStore(_path));
    }

    [Fact]
    public void AFileWithAnUnknownContentKindIsRejected()
    {
        File.WriteAllText(
            _path,
            "{\"SchemaVersion\": 1, \"Entries\": [{\"Id\": \"abc\", \"FinishReason\": \"Stop\", " +
            "\"Contents\": [{\"Kind\": \"hologram\"}]}]}");
        Assert.Throws<InvalidDataException>(() => new JsonFileChatInteractionStore(_path));
    }

    [Fact]
    public void RecordingWithoutAStoreOrAPathIsRejected()
    {
        var options = new ChatClientOptions { RecordingMode = ChatClientRecordingMode.Record };
        Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Throws<ArgumentException>(
            () => ChatClientPipelineFactory.Create<double>(StubChatClient.Text("m", "x"), options));
    }

    [Fact]
    public async Task AnExplicitInMemoryStoreIsUsedWhenOneIsSupplied()
    {
        var store = new InMemoryChatInteractionStore();
        var live = StubChatClient.Text("m", "answer");
        IChatClient<double> client = ChatClientPipelineFactory.Create(
            live,
            new ChatClientOptions
            {
                MaxRetries = 0,
                RecordingMode = ChatClientRecordingMode.Record,
                InteractionStore = store
            });

        await client.GetResponseAsync(Prompt);
        Assert.Equal(1, store.Count);
        Assert.False(File.Exists(_path));
    }

    [Fact]
    public void TheSameResolvedStoreIsReturnedEveryTime()
    {
        var options = new ChatClientOptions
        {
            RecordingMode = ChatClientRecordingMode.Record,
            RecordingPath = _path
        };

        IChatInteractionStore? first = options.ResolveInteractionStore();
        IChatInteractionStore? second = options.ResolveInteractionStore();
        Assert.NotNull(first);
        Assert.Same(first, second);
    }

    [Fact]
    public void ResolvingWithNoRecordingConfiguredReturnsNothing()
    {
        Assert.Null(new ChatClientOptions().ResolveInteractionStore());
    }
}
