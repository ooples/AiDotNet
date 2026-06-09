using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Connectors;
using Xunit;
using Meai = Microsoft.Extensions.AI;

namespace AiDotNetTests.UnitTests.Agentic.Connectors
{
    public class MeaiChatClientTests
    {
        /// <summary>Minimal fake Microsoft.Extensions.AI client returning a fixed response.</summary>
        private sealed class FakeMeaiClient : Meai.IChatClient
        {
            public Task<Meai.ChatResponse> GetResponseAsync(
                IEnumerable<Meai.ChatMessage> messages,
                Meai.ChatOptions? options = null,
                CancellationToken cancellationToken = default)
            {
                var response = new Meai.ChatResponse(new Meai.ChatMessage(Meai.ChatRole.Assistant, "hi from meai"))
                {
                    FinishReason = Meai.ChatFinishReason.Stop,
                    ModelId = "fake-meai-model",
                    Usage = new Meai.UsageDetails { InputTokenCount = 3, OutputTokenCount = 4 }
                };
                return Task.FromResult(response);
            }

            public async IAsyncEnumerable<Meai.ChatResponseUpdate> GetStreamingResponseAsync(
                IEnumerable<Meai.ChatMessage> messages,
                Meai.ChatOptions? options = null,
                [EnumeratorCancellation] CancellationToken cancellationToken = default)
            {
                await Task.CompletedTask;
                yield return new Meai.ChatResponseUpdate(Meai.ChatRole.Assistant, "hi ");
                yield return new Meai.ChatResponseUpdate(Meai.ChatRole.Assistant, "from meai")
                {
                    FinishReason = Meai.ChatFinishReason.Stop
                };
            }

            public object? GetService(Type serviceType, object? serviceKey = null) => null;

            public void Dispose() { }
        }

        [Fact(Timeout = 60000)]
        public async Task WrapsMeaiClient_MapsResponseTextFinishAndUsage()
        {
            IChatClient<double> client = new MeaiChatClient<double>(new FakeMeaiClient(), "fake-meai-model");

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("hi") });

            Assert.Equal("hi from meai", response.Text);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
            Assert.Equal("fake-meai-model", response.ModelId);
            Assert.Equal(7, response.Usage?.TotalTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Streaming_ReconstructsText()
        {
            IChatClient<double> client = new MeaiChatClient<double>(new FakeMeaiClient());

            var text = new System.Text.StringBuilder();
            await foreach (var update in client.GetStreamingResponseAsync(new[] { ChatMessage.User("hi") }))
            {
                if (update.TextDelta != null) text.Append(update.TextDelta);
            }

            Assert.Equal("hi from meai", text.ToString());
        }

        [Fact(Timeout = 60000)]
        public async Task PassingTools_ThrowsNotSupported()
        {
            IChatClient<double> client = new MeaiChatClient<double>(new FakeMeaiClient());
            var options = new ChatOptions { Tools = new[] { new AiToolDefinition("t", "d") } };

            await Assert.ThrowsAsync<NotSupportedException>(
                () => client.GetResponseAsync(new[] { ChatMessage.User("hi") }, options));
        }
    }
}
