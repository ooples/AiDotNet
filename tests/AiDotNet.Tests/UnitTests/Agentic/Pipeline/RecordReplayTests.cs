using System;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;
using AiDotNetTests.UnitTests.Agentic.Agents;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Pipeline
{
    public class RecordReplayTests
    {
        [Fact(Timeout = 60000)]
        public async Task Record_ThenReplay_ReturnsSameResponse_WithoutCallingModel()
        {
            var store = new InMemoryChatInteractionStore();

            // Record once against the real (scripted) model.
            var realModel = ScriptedChatClient<double>.Sequence(ChatResponses.Text("the recorded answer"));
            var recorder = new RecordingChatClient<double>(realModel, store);
            var recorded = await recorder.GetResponseAsync(new[] { ChatMessage.User("question") });
            Assert.Equal("the recorded answer", recorded.Text);
            Assert.Equal(1, store.Count);
            Assert.Equal(1, realModel.CallCount);

            // Replay with NO fallback — must serve from the store without any model call.
            // Recordings are keyed per model, so pure playback names the recorded model.
            var replay = new ReplayingChatClient<double>(store, modelId: realModel.ModelId);
            var replayed = await replay.GetResponseAsync(new[] { ChatMessage.User("question") });
            Assert.Equal("the recorded answer", replayed.Text);
            Assert.Equal(1, realModel.CallCount); // still 1 — replay did not touch the model
        }

        [Fact(Timeout = 60000)]
        public async Task Replay_IsDeterministic_AcrossManyCalls()
        {
            var store = new InMemoryChatInteractionStore();
            var model = ScriptedChatClient<double>.Sequence(ChatResponses.Text("stable"));
            await new RecordingChatClient<double>(model, store)
                .GetResponseAsync(new[] { ChatMessage.User("q") });

            var replay = new ReplayingChatClient<double>(store, modelId: model.ModelId);
            for (var i = 0; i < 5; i++)
            {
                Assert.Equal("stable", (await replay.GetResponseAsync(new[] { ChatMessage.User("q") })).Text);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task Replay_Miss_WithoutFallback_Throws()
        {
            var replay = new ReplayingChatClient<double>(new InMemoryChatInteractionStore());
            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                replay.GetResponseAsync(new[] { ChatMessage.User("never recorded") }));
        }

        [Fact(Timeout = 60000)]
        public async Task Replay_Miss_WithFallback_CallsModelOnceThenCaches()
        {
            var store = new InMemoryChatInteractionStore();
            var model = ScriptedChatClient<double>.Sequence(ChatResponses.Text("from model"));
            var replay = new ReplayingChatClient<double>(store, fallback: model);

            var first = await replay.GetResponseAsync(new[] { ChatMessage.User("q") });
            var second = await replay.GetResponseAsync(new[] { ChatMessage.User("q") });

            Assert.Equal("from model", first.Text);
            Assert.Equal("from model", second.Text);
            Assert.Equal(1, model.CallCount); // miss called the model once; the second hit was cached
        }

        [Fact(Timeout = 60000)]
        public async Task Key_DistinguishesRequests()
        {
            var a = ChatInteractionKey.For(new[] { ChatMessage.User("hello") }, null);
            var sameAsA = ChatInteractionKey.For(new[] { ChatMessage.User("hello") }, null);
            var differentMessage = ChatInteractionKey.For(new[] { ChatMessage.User("goodbye") }, null);
            var differentOptions = ChatInteractionKey.For(new[] { ChatMessage.User("hello") },
                new ChatOptions { Temperature = 0.9 });

            Assert.Equal(a, sameAsA);
            Assert.NotEqual(a, differentMessage);
            Assert.NotEqual(a, differentOptions);
            await Task.CompletedTask;
        }
    }
}
