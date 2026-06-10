using System;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Connectors;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Connectors
{
    public class OpenAiCompatibleConnectorTests
    {
        private const string OpenAiStyleResponse = @"{
            ""model"": ""local-model"",
            ""choices"": [{ ""message"": { ""content"": ""hi from the model"" }, ""finish_reason"": ""stop"" }],
            ""usage"": { ""prompt_tokens"": 3, ""completion_tokens"": 4 }
        }";

        private sealed class CapturingHandler : HttpMessageHandler
        {
            private readonly string _body;
            public Uri? LastUri { get; private set; }
            public string? Authorization { get; private set; }

            public CapturingHandler(string body) => _body = body;

            protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            {
                LastUri = request.RequestUri;
                Authorization = request.Headers.Authorization?.ToString();
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(_body)
                });
            }
        }

        [Fact(Timeout = 60000)]
        public async Task Ollama_UsesLocalEndpoint_AndParsesOpenAiResponse()
        {
            var handler = new CapturingHandler(OpenAiStyleResponse);
            var client = new OllamaChatClient<double>("llama3.1", httpClient: new HttpClient(handler));

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("hello") });

            Assert.Equal("hi from the model", response.Text);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
            Assert.Equal(7, response.Usage?.TotalTokens);
            Assert.Equal(OllamaChatClient<double>.DefaultEndpoint, handler.LastUri?.ToString());
            Assert.Equal("llama3.1", client.ModelId);
        }

        [Fact(Timeout = 60000)]
        public async Task Mistral_UsesMistralEndpoint_WithBearerKey()
        {
            var handler = new CapturingHandler(OpenAiStyleResponse);
            var client = new MistralChatClient<double>("secret-key", "mistral-large-latest", httpClient: new HttpClient(handler));

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("hello") });

            Assert.Equal("hi from the model", response.Text);
            Assert.Equal(MistralChatClient<double>.DefaultEndpoint, handler.LastUri?.ToString());
            Assert.Contains("secret-key", handler.Authorization);
        }

        [Fact(Timeout = 60000)]
        public async Task Connectors_AreDropInIChatClients()
        {
            // Both expose the uniform IChatClient<T> surface the rest of the stack depends on.
            IChatClient<double> ollama = new OllamaChatClient<double>("llama3.1", httpClient: new HttpClient(new CapturingHandler(OpenAiStyleResponse)));
            IChatClient<double> mistral = new MistralChatClient<double>("k", httpClient: new HttpClient(new CapturingHandler(OpenAiStyleResponse)));

            Assert.Equal("hi from the model", (await ollama.GetResponseAsync(new[] { ChatMessage.User("x") })).Text);
            Assert.Equal("hi from the model", (await mistral.GetResponseAsync(new[] { ChatMessage.User("x") })).Text);
        }

        [Fact(Timeout = 60000)]
        public async Task CustomEndpoint_Override_IsHonored()
        {
            var handler = new CapturingHandler(OpenAiStyleResponse);
            var client = new OllamaChatClient<double>("llama3.1", endpoint: "http://remote-host:11434/v1/chat/completions",
                httpClient: new HttpClient(handler));

            await client.GetResponseAsync(new[] { ChatMessage.User("x") });

            Assert.Equal("http://remote-host:11434/v1/chat/completions", handler.LastUri?.ToString());
        }
    }
}
