using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Tools
{
    public class StructuredOutputTests
    {
        private sealed class Weather
        {
            public string Name { get; set; } = "";
            public int TempC { get; set; }
        }

        /// <summary>Stub client that records the options it was called with and returns a fixed JSON object.</summary>
        private sealed class JsonStubClient<T> : IChatClient<T>
        {
            private readonly string _json;
            public ChatOptions? LastOptions { get; private set; }

            public JsonStubClient(string json) => _json = json;

            public string ModelId => "stub";

            public Task<ChatResponse> GetResponseAsync(
                IReadOnlyList<ChatMessage> messages,
                ChatOptions? options = null,
                CancellationToken cancellationToken = default)
            {
                LastOptions = options;
                return Task.FromResult(new ChatResponse(ChatMessage.Assistant(_json)));
            }

            public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
                IReadOnlyList<ChatMessage> messages,
                ChatOptions? options = null,
                [EnumeratorCancellation] CancellationToken cancellationToken = default)
            {
                await Task.CompletedTask;
                yield return ChatResponseUpdate.ForText(_json);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task GetStructuredResponse_DeserializesReply_AndAttachesSchema()
        {
            var client = new JsonStubClient<double>("{\"Name\":\"Paris\",\"TempC\":18}");

            var weather = await client.GetStructuredResponseAsync<double, Weather>("Weather in Paris?");

            Assert.Equal("Paris", weather.Name);
            Assert.Equal(18, weather.TempC);

            // The helper must switch the request to schema-constrained JSON output.
            var opts = client.LastOptions;
            Assert.NotNull(opts);
            Assert.Equal(ChatResponseFormatKind.JsonSchema, opts.ResponseFormat);
            Assert.NotNull(opts.ResponseJsonSchema);
            Assert.Equal("object", (string)opts.ResponseJsonSchema["type"]);
            Assert.NotNull(opts.ResponseJsonSchema["properties"]["TempC"]);
        }

        [Fact(Timeout = 60000)]
        public async Task GetStructuredResponse_Throws_OnUnparseableReply()
        {
            var client = new JsonStubClient<double>("not json at all");

            await Assert.ThrowsAsync<System.InvalidOperationException>(
                () => client.GetStructuredResponseAsync<double, Weather>("hi"));
        }
    }
}
