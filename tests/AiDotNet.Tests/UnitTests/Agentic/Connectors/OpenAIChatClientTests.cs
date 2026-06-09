using System;
using System.Collections.Generic;
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
    public class OpenAIChatClientTests
    {
        /// <summary>Captures the outbound request body and returns a canned response.</summary>
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

                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(_responseBody)
                };
            }
        }

        private static OpenAIChatClient<double> ClientWith(string responseBody, out StubHandler handler)
        {
            handler = new StubHandler(responseBody);
            return new OpenAIChatClient<double>("test-key", "gpt-4o", httpClient: new HttpClient(handler));
        }

        [Fact(Timeout = 60000)]
        public async Task GetResponse_ParsesText_ToolCalls_FinishReason_Usage()
        {
            const string responseBody = @"{
                ""model"": ""gpt-4o-2024"",
                ""choices"": [{
                    ""message"": {
                        ""content"": ""Checking"",
                        ""tool_calls"": [{
                            ""id"": ""call_1"",
                            ""type"": ""function"",
                            ""function"": { ""name"": ""get_weather"", ""arguments"": ""{\""city\"":\""Paris\""}"" }
                        }]
                    },
                    ""finish_reason"": ""tool_calls""
                }],
                ""usage"": { ""prompt_tokens"": 5, ""completion_tokens"": 7 }
            }";

            var client = ClientWith(responseBody, out var handler);
            var options = new ChatOptions { Tools = new[] { new AiToolDefinition("get_weather", "Gets weather") } };

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("weather in Paris?") }, options);

            Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
            Assert.Equal("Checking", response.Text);
            Assert.Equal("gpt-4o-2024", response.ModelId);
            Assert.Equal(12, response.Usage?.TotalTokens);

            var call = Assert.Single(response.Message.ToolCalls);
            Assert.Equal("call_1", call.CallId);
            Assert.Equal("get_weather", call.ToolName);

            // Verify the outbound request was shaped correctly.
            var request = JObject.Parse(handler.LastRequestBody);
            Assert.Equal("gpt-4o", (string?)request["model"]);
            Assert.Equal("user", (string?)request["messages"]?[0]?["role"]);
            Assert.Equal("get_weather", (string?)request["tools"]?[0]?["function"]?["name"]);
            Assert.Equal("auto", (string?)request["tool_choice"]);
        }

        [Fact(Timeout = 60000)]
        public async Task GetResponse_MapsRequiredToolChoice_AndJsonSchemaFormat()
        {
            const string responseBody = @"{""choices"":[{""message"":{""content"":""{}""},""finish_reason"":""stop""}]}";
            var client = ClientWith(responseBody, out var handler);

            var options = new ChatOptions
            {
                Tools = new[] { new AiToolDefinition("get_weather", "Gets weather") },
                ToolChoice = ToolChoiceMode.Required,
                RequiredToolName = "get_weather",
                ResponseFormat = ChatResponseFormatKind.JsonSchema,
                ResponseJsonSchema = new JObject { ["type"] = "object" }
            };

            await client.GetResponseAsync(new[] { ChatMessage.User("hi") }, options);

            var request = JObject.Parse(handler.LastRequestBody);
            Assert.Equal("get_weather", (string?)request["tool_choice"]?["function"]?["name"]);
            Assert.Equal("json_schema", (string?)request["response_format"]?["type"]);
        }

        [Fact(Timeout = 60000)]
        public async Task GetStreamingResponse_ReconstructsText_FinishReason_Usage()
        {
            var sse = string.Join("\n", new[]
            {
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}",
                "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}",
                "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}",
                "data: {\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2},\"choices\":[]}",
                "data: [DONE]"
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

        [Fact(Timeout = 60000)]
        public async Task GenerateTextAsync_Extension_ReturnsAssistantText()
        {
            const string responseBody = @"{""choices"":[{""message"":{""content"":""42""},""finish_reason"":""stop""}]}";
            var client = ClientWith(responseBody, out _);

            var text = await client.GenerateTextAsync("What is 6 times 7?");
            Assert.Equal("42", text);
        }
    }
}
