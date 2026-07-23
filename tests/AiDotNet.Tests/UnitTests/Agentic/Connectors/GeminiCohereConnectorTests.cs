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
    public class GeminiCohereConnectorTests
    {
        private sealed class CapturingHandler : HttpMessageHandler
        {
            private readonly string _body;
            public Uri? LastUri { get; private set; }
            public string LastRequestBody { get; private set; } = "";
            public string? Authorization { get; private set; }

            public CapturingHandler(string body) => _body = body;

            protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            {
                LastUri = request.RequestUri;
                Authorization = request.Headers.Authorization?.ToString();
                if (request.Content is not null)
                {
                    LastRequestBody = await request.Content.ReadAsStringAsync().ConfigureAwait(false);
                }

                return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(_body) };
            }
        }

        // ---- Gemini ----

        [Fact(Timeout = 60000)]
        public async Task Gemini_ParsesText_FinishReason_Usage()
        {
            await Task.Yield();

            const string response = @"{
                ""candidates"": [{
                    ""content"": { ""parts"": [{ ""text"": ""Hello from Gemini"" }], ""role"": ""model"" },
                    ""finishReason"": ""STOP""
                }],
                ""usageMetadata"": { ""promptTokenCount"": 8, ""candidatesTokenCount"": 4 }
            }";
            var handler = new CapturingHandler(response);
            var client = new GeminiChatClient<double>("key", "gemini-1.5-flash", httpClient: new HttpClient(handler));

            var result = await client.GetResponseAsync(new[] { ChatMessage.System("be brief"), ChatMessage.User("hi") });

            Assert.Equal("Hello from Gemini", result.Text);
            Assert.Equal(ChatFinishReason.Stop, result.FinishReason);
            Assert.Equal(12, result.Usage?.TotalTokens);
            Assert.Contains("generateContent?key=key", handler.LastUri?.ToString());
            // System message becomes systemInstruction; user message becomes a 'user' content.
            var sent = JObject.Parse(handler.LastRequestBody);
            Assert.NotNull(sent["systemInstruction"]);
            Assert.Equal("user", (string?)sent["contents"]?[0]?["role"]);
        }

        [Fact(Timeout = 60000)]
        public async Task Gemini_ParsesFunctionCall()
        {
            await Task.Yield();

            const string response = @"{
                ""candidates"": [{
                    ""content"": { ""parts"": [{ ""functionCall"": { ""name"": ""get_weather"", ""args"": { ""city"": ""Paris"" } } }] }
                }]
            }";
            var client = new GeminiChatClient<double>("key", httpClient: new HttpClient(new CapturingHandler(response)));
            var options = new ChatOptions { Tools = new[] { new AiToolDefinition("get_weather", "Gets weather") } };

            var result = await client.GetResponseAsync(new[] { ChatMessage.User("weather?") }, options);

            Assert.Equal(ChatFinishReason.ToolCalls, result.FinishReason);
            var call = Assert.Single(result.Message.ToolCalls);
            Assert.Equal("get_weather", call.ToolName);
            Assert.Contains("Paris", call.ArgumentsJson);
        }

        // ---- Cohere ----

        [Fact(Timeout = 60000)]
        public async Task Cohere_ParsesText_FinishReason_Usage_AndShapesHistory()
        {
            await Task.Yield();

            const string response = @"{
                ""text"": ""Hello from Cohere"",
                ""finish_reason"": ""COMPLETE"",
                ""meta"": { ""tokens"": { ""input_tokens"": 6, ""output_tokens"": 3 } }
            }";
            var handler = new CapturingHandler(response);
            var client = new CohereChatClient<double>("secret", "command-r-plus", httpClient: new HttpClient(handler));

            var result = await client.GetResponseAsync(new[]
            {
                ChatMessage.System("be helpful"),
                ChatMessage.User("first question"),
                ChatMessage.Assistant("first answer"),
                ChatMessage.User("second question"),
            });

            Assert.Equal("Hello from Cohere", result.Text);
            Assert.Equal(ChatFinishReason.Stop, result.FinishReason);
            Assert.Equal(9, result.Usage?.TotalTokens);
            Assert.Contains("secret", handler.Authorization);

            var sent = JObject.Parse(handler.LastRequestBody);
            Assert.Equal("second question", (string?)sent["message"]);        // latest user msg is 'message'
            Assert.Equal("be helpful", (string?)sent["preamble"]);            // system -> preamble
            Assert.Equal(2, ((JArray?)sent["chat_history"])?.Count);          // prior user + assistant
        }

        [Fact(Timeout = 60000)]
        public async Task Cohere_ParsesToolCalls()
        {
            await Task.Yield();

            const string response = @"{
                ""text"": """",
                ""tool_calls"": [{ ""name"": ""lookup"", ""parameters"": { ""id"": 7 } }],
                ""finish_reason"": ""COMPLETE""
            }";
            var client = new CohereChatClient<double>("k", httpClient: new HttpClient(new CapturingHandler(response)));

            var result = await client.GetResponseAsync(new[] { ChatMessage.User("look up 7") });

            Assert.Equal(ChatFinishReason.ToolCalls, result.FinishReason);
            var call = Assert.Single(result.Message.ToolCalls);
            Assert.Equal("lookup", call.ToolName);
        }

        [Fact(Timeout = 60000)]
        public async Task BothAreDropInIChatClients()
        {
            await Task.Yield();

            const string gemini = @"{ ""candidates"": [{ ""content"": { ""parts"": [{ ""text"": ""g"" }] }, ""finishReason"": ""STOP"" }] }";
            const string cohere = @"{ ""text"": ""c"", ""finish_reason"": ""COMPLETE"" }";
            IChatClient<double> g = new GeminiChatClient<double>("k", httpClient: new HttpClient(new CapturingHandler(gemini)));
            IChatClient<double> c = new CohereChatClient<double>("k", httpClient: new HttpClient(new CapturingHandler(cohere)));

            Assert.Equal("g", (await g.GetResponseAsync(new[] { ChatMessage.User("x") })).Text);
            Assert.Equal("c", (await c.GetResponseAsync(new[] { ChatMessage.User("x") })).Text);
        }
    }
}
