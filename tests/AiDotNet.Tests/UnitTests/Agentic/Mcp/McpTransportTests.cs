using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Mcp;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Mcp
{
    public class McpTransportTests
    {
        private sealed class StubHandler : HttpMessageHandler
        {
            private readonly string _body;
            private readonly HttpStatusCode _status;
            public string LastRequestBody { get; private set; } = "";

            public StubHandler(string body, HttpStatusCode status = HttpStatusCode.OK)
            {
                _body = body;
                _status = status;
            }

            protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            {
                if (request.Content is not null)
                {
                    LastRequestBody = await request.Content.ReadAsStringAsync().ConfigureAwait(false);
                }

                return new HttpResponseMessage(_status) { Content = new StringContent(_body) };
            }
        }

        // ---- HTTP transport ----

        [Fact(Timeout = 60000)]
        public async Task Http_WrapsRequest_AndUnwrapsResult()
        {
            await Task.Yield();

            const string response = @"{""jsonrpc"":""2.0"",""id"":1,""result"":{""tools"":[
                {""name"":""echo"",""description"":""E"",""inputSchema"":{""type"":""object""}}]}}";
            var handler = new StubHandler(response);
            var transport = new HttpMcpTransport("https://mcp.example/rpc", new HttpClient(handler));
            var client = new McpClient(transport);

            var tools = await client.ListToolsAsync();

            Assert.Single(tools);
            Assert.Equal("echo", tools[0].Name);

            // The outbound body is a JSON-RPC 2.0 envelope.
            var sent = JObject.Parse(handler.LastRequestBody);
            Assert.Equal("2.0", (string?)sent["jsonrpc"]);
            Assert.Equal("tools/list", (string?)sent["method"]);
            Assert.NotNull(sent["id"]);
        }

        [Fact(Timeout = 60000)]
        public async Task Http_JsonRpcError_ThrowsMcpException()
        {
            await Task.Yield();

            const string response = @"{""jsonrpc"":""2.0"",""id"":1,""error"":{""code"":-32601,""message"":""Method not found""}}";
            var transport = new HttpMcpTransport("https://mcp.example/rpc", new HttpClient(new StubHandler(response)));

            var ex = await Assert.ThrowsAsync<McpException>(() => transport.SendRequestAsync("bogus", null));
            Assert.Contains("Method not found", ex.Message);
        }

        // ---- Stream (stdio framing) transport ----

        [Fact(Timeout = 60000)]
        public async Task Stream_WritesJsonLine_AndReadsMatchingResponse()
        {
            await Task.Yield();

            const string responseLine = @"{""jsonrpc"":""2.0"",""id"":1,""result"":{""value"":42}}";
            var writer = new StringWriter();
            var reader = new StringReader(responseLine + "\n");
            using var transport = new StreamMcpTransport(reader, writer);

            var result = await transport.SendRequestAsync("ping", new JObject { ["x"] = 1 });

            Assert.Equal(42, (int?)result["value"]);
            // The request was written as a single JSON line.
            var written = writer.ToString().Trim();
            var sent = JObject.Parse(written);
            Assert.Equal("ping", (string?)sent["method"]);
            Assert.Equal(1, (int?)sent["id"]);
        }

        [Fact(Timeout = 60000)]
        public async Task Stream_SkipsNotifications_UntilMatchingId()
        {
            await Task.Yield();

            // A notification (no id) and a mismatched response precede the real response.
            var lines = string.Join("\n", new[]
            {
                @"{""jsonrpc"":""2.0"",""method"":""notifications/log"",""params"":{}}",
                @"{""jsonrpc"":""2.0"",""id"":99,""result"":{""stale"":true}}",
                @"{""jsonrpc"":""2.0"",""id"":1,""result"":{""ok"":true}}",
            });
            using var transport = new StreamMcpTransport(new StringReader(lines + "\n"), new StringWriter());

            var result = await transport.SendRequestAsync("tools/list", null);

            Assert.True((bool)result["ok"]);
        }

        [Fact(Timeout = 60000)]
        public async Task Stream_ClosedBeforeResponse_Throws()
        {
            await Task.Yield();

            using var transport = new StreamMcpTransport(new StringReader(""), new StringWriter());
            await Assert.ThrowsAsync<McpException>(() => transport.SendRequestAsync("ping", null));
        }
    }
}
