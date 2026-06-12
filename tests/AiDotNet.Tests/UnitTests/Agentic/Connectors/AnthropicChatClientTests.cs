using System;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Connectors;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Connectors
{
    public class AnthropicChatClientTests
    {
        private sealed class StubHandler : HttpMessageHandler
        {
            private readonly string _responseBody;
            public string LastRequestBody { get; private set; } = "";

            public StubHandler(string responseBody) => _responseBody = responseBody;

            protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            {
                if (request.Content is not null)
                {
                    LastRequestBody = await request.Content.ReadAsStringAsync().ConfigureAwait(false);
                }

                return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(_responseBody) };
            }
        }

        private static AnthropicChatClient<double> ClientWith(string responseBody, out StubHandler handler)
        {
            handler = new StubHandler(responseBody);
            return new AnthropicChatClient<double>("test-key", "claude-3-5-sonnet-20241022", httpClient: new HttpClient(handler));
        }

        [Fact(Timeout = 60000)]
        public async Task GetResponse_ParsesTextAndToolUse_MapsStopReasonAndUsage()
        {
            await Task.Yield();

            const string responseBody = @"{
                ""model"": ""claude-3-5-sonnet-20241022"",
                ""content"": [
                    { ""type"": ""text"", ""text"": ""Checking"" },
                    { ""type"": ""tool_use"", ""id"": ""t1"", ""name"": ""get_weather"", ""input"": { ""city"": ""Paris"" } }
                ],
                ""stop_reason"": ""tool_use"",
                ""usage"": { ""input_tokens"": 5, ""output_tokens"": 7 }
            }";

            var client = ClientWith(responseBody, out var handler);
            var messages = new[] { ChatMessage.System("be helpful"), ChatMessage.User("weather in Paris?") };
            var options = new ChatOptions { Tools = new[] { new AiToolDefinition("get_weather", "Gets weather") } };

            var response = await client.GetResponseAsync(messages, options);

            Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
            Assert.Equal("Checking", response.Text);
            Assert.Equal(12, response.Usage?.TotalTokens);
            var call = Assert.Single(response.Message.ToolCalls);
            Assert.Equal("get_weather", call.ToolName);
            Assert.Contains("Paris", call.ArgumentsJson);

            // Anthropic carries system as a top-level field, not a message.
            var request = JObject.Parse(handler.LastRequestBody);
            Assert.Equal("be helpful", (string?)request["system"]);
            Assert.NotNull(request["max_tokens"]);
            Assert.Equal("user", (string?)request["messages"]?[0]?["role"]);
            Assert.Equal("get_weather", (string?)request["tools"]?[0]?["name"]);
        }

        [Fact(Timeout = 60000)]
        public async Task GetStreamingResponse_ReconstructsTextFinishAndUsage()
        {
            await Task.Yield();

            var sse = string.Join("\n", new[]
            {
                "event: message_start",
                "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}",
                "",
                "event: content_block_delta",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}",
                "",
                "event: content_block_delta",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}",
                "",
                "event: message_delta",
                "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}",
                "",
                "event: message_stop",
                "data: {\"type\":\"message_stop\"}"
            });

            var client = ClientWith(sse, out _);

            var text = new System.Text.StringBuilder();
            ChatFinishReason? finish = null;
            ChatUsage? usage = null;
            await foreach (var update in client.GetStreamingResponseAsync(new[] { ChatMessage.User("hi") }))
            {
                if (update.TextDelta != null) text.Append(update.TextDelta);
                if (update.FinishReason != null) finish = update.FinishReason;
                if (update.Usage != null) usage = update.Usage;
            }

            Assert.Equal("Hello", text.ToString());
            Assert.Equal(ChatFinishReason.Stop, finish);
            Assert.Equal(5, usage?.TotalTokens);
        }
    }
}
