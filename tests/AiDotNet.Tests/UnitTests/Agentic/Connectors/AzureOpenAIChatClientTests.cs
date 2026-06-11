using System;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Connectors;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Connectors
{
    public class AzureOpenAIChatClientTests
    {
        private sealed class CapturingHandler : HttpMessageHandler
        {
            private readonly string _responseBody;
            public string RequestUri { get; private set; } = "";
            public string? ApiKeyHeader { get; private set; }
            public bool HasBearer { get; private set; }

            public CapturingHandler(string responseBody) => _responseBody = responseBody;

            protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            {
                RequestUri = request.RequestUri?.ToString() ?? "";
                HasBearer = request.Headers.Authorization is not null;
                ApiKeyHeader = request.Headers.TryGetValues("api-key", out var values) ? values.FirstOrDefault() : null;
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(_responseBody) });
            }
        }

        [Fact(Timeout = 60000)]
        public async Task UsesDeploymentEndpoint_AndApiKeyHeader_NotBearer()
        {
            const string responseBody = @"{""choices"":[{""message"":{""content"":""ok""},""finish_reason"":""stop""}]}";
            var handler = new CapturingHandler(responseBody);
            var client = new AzureOpenAIChatClient<double>(
                apiKey: "azkey",
                deploymentName: "gpt4o-deploy",
                resourceEndpoint: "https://my-res.openai.azure.com/",
                apiVersion: "2024-10-21",
                httpClient: new HttpClient(handler));

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("hi") });

            Assert.Equal("ok", response.Text);
            Assert.Contains("/openai/deployments/gpt4o-deploy/chat/completions", handler.RequestUri);
            Assert.Contains("api-version=2024-10-21", handler.RequestUri);
            Assert.Equal("azkey", handler.ApiKeyHeader);
            Assert.False(handler.HasBearer); // Azure uses api-key header, not Authorization: Bearer
        }
    }
}
