using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Memory;
using AiDotNet.Agentic.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Agents
{
    public class ConversationMemoryTests
    {
        // ---- ThreadedAgent ----

        [Fact(Timeout = 60000)]
        public async Task ThreadedAgent_RemembersPriorTurns_WithinAThread()
        {
            await Task.Yield();

            var client = ScriptedChatClient<double>.Sequence(
                ChatResponses.Text("Hi there."),
                ChatResponses.Text("I remember."));
            var inner = new AgentExecutor<double>(client);
            var store = new InMemoryConversationStore();
            var threaded = new ThreadedAgent<double>(inner, store);

            await threaded.RunAsync("conv-1", "Hello!");
            await threaded.RunAsync("conv-1", "Do you remember me?");

            // The second model call saw the first turn (user + assistant) plus the new user message.
            var secondRequest = client.Requests[1];
            Assert.Contains(secondRequest, m => m.Role == ChatRole.User && m.Text == "Hello!");
            Assert.Contains(secondRequest, m => m.Role == ChatRole.Assistant && m.Text == "Hi there.");
            Assert.Contains(secondRequest, m => m.Role == ChatRole.User && m.Text == "Do you remember me?");

            // The persisted dialogue is the clean user/assistant transcript (2 turns = 4 messages).
            var history = await store.GetAsync("conv-1");
            Assert.Equal(4, history.Count);
            Assert.Equal(ChatRole.User, history[0].Role);
            Assert.Equal(ChatRole.Assistant, history[1].Role);
            Assert.Equal("I remember.", history[3].Text);
        }

        [Fact(Timeout = 60000)]
        public async Task ThreadedAgent_IsolatesSeparateThreads()
        {
            await Task.Yield();

            var client = ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"));
            var threaded = new ThreadedAgent<double>(new AgentExecutor<double>(client), new InMemoryConversationStore());

            await threaded.RunAsync("alice", "I am Alice.");
            await threaded.RunAsync("bob", "I am Bob.");

            // bob's run must not have seen alice's history.
            var bobRequest = client.Requests[1];
            Assert.DoesNotContain(bobRequest, m => m.Text == "I am Alice.");
            Assert.Contains(bobRequest, m => m.Text == "I am Bob.");
        }

        [Fact(Timeout = 60000)]
        public async Task ThreadedAgent_EmptyThreadId_Throws()
        {
            await Task.Yield();

            var threaded = new ThreadedAgent<double>(
                new AgentExecutor<double>(ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"))),
                new InMemoryConversationStore());

            await Assert.ThrowsAsync<ArgumentException>(() => threaded.RunAsync("   ", "hi"));
        }

        // ---- InMemoryConversationStore ----

        [Fact(Timeout = 60000)]
        public async Task InMemoryStore_AppendGetListClear_RoundTrips()
        {
            await Task.Yield();

            var store = new InMemoryConversationStore();
            await store.AppendAsync("t", new[] { ChatMessage.User("a"), ChatMessage.Assistant("b") });
            await store.AppendAsync("t", new[] { ChatMessage.User("c") });

            var history = await store.GetAsync("t");
            Assert.Equal(3, history.Count);
            Assert.Equal("c", history[2].Text);

            Assert.Contains("t", await store.ListThreadsAsync());

            await store.ClearAsync("t");
            Assert.Empty(await store.GetAsync("t"));
        }

        [Fact(Timeout = 60000)]
        public async Task InMemoryStore_UnknownThread_ReturnsEmpty()
        {
            await Task.Yield();

            var store = new InMemoryConversationStore();
            Assert.Empty(await store.GetAsync("nope"));
        }

        // ---- JsonFileConversationStore (durable; both TFMs, no native dependency) ----

        [Fact(Timeout = 60000)]
        public async Task JsonFileStore_PersistsAcrossInstances()
        {
            await Task.Yield();

            var path = Path.GetTempFileName();
            try
            {
                var writer = new JsonFileConversationStore(path);
                await writer.AppendAsync("conv", new[] { ChatMessage.User("hello"), ChatMessage.Assistant("world") });

                // A brand-new instance over the same file reads the persisted history.
                var reader = new JsonFileConversationStore(path);
                var history = await reader.GetAsync("conv");

                Assert.Equal(2, history.Count);
                Assert.Equal(ChatRole.User, history[0].Role);
                Assert.Equal("hello", history[0].Text);
                Assert.Equal(ChatRole.Assistant, history[1].Role);
                Assert.Equal("world", history[1].Text);

                Assert.Contains("conv", await reader.ListThreadsAsync());
            }
            finally
            {
                TryDelete(path);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task JsonFileStore_ThreadedAgent_SurvivesNewStoreInstance()
        {
            await Task.Yield();

            var path = Path.GetTempFileName();
            try
            {
                var client = ScriptedChatClient<double>.Sequence(
                    ChatResponses.Text("First."),
                    ChatResponses.Text("Second."));
                var inner = new AgentExecutor<double>(client);

                var first = new ThreadedAgent<double>(inner, new JsonFileConversationStore(path));
                await first.RunAsync("s", "Turn one.");

                // New ThreadedAgent + new store instance over the same file: the prior turn is restored.
                var second = new ThreadedAgent<double>(inner, new JsonFileConversationStore(path));
                await second.RunAsync("s", "Turn two.");

                var secondRequest = client.Requests[1];
                Assert.Contains(secondRequest, m => m.Role == ChatRole.User && m.Text == "Turn one.");
                Assert.Contains(secondRequest, m => m.Role == ChatRole.Assistant && m.Text == "First.");
            }
            finally
            {
                TryDelete(path);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task JsonFileStore_Clear_RemovesThread()
        {
            await Task.Yield();

            var path = Path.GetTempFileName();
            try
            {
                var store = new JsonFileConversationStore(path);
                await store.AppendAsync("x", new[] { ChatMessage.User("a") });
                await store.ClearAsync("x");
                Assert.Empty(await store.GetAsync("x"));
            }
            finally
            {
                TryDelete(path);
            }
        }

        private static void TryDelete(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
            catch (IOException)
            {
                // Best-effort cleanup of the temp file.
            }
        }
    }
}
