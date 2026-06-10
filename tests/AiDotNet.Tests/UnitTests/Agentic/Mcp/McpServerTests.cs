using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Mcp;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using AiDotNetTests.UnitTests.Agentic.Agents;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Mcp
{
    public class McpServerTests
    {
        private static ToolCollection ToolsWith(out RecordingTool tool)
        {
            tool = new RecordingTool("adder", "Adds a and b.", args =>
                ((int)args["a"] + (int)args["b"]).ToString());
            return new ToolCollection().Add(tool);
        }

        [Fact(Timeout = 60000)]
        public async Task Server_ListsRegisteredTools()
        {
            var server = new McpServer(ToolsWith(out _));

            var result = await server.HandleRequestAsync("tools/list", null);

            var tools = (JArray)result["tools"];
            var entry = Assert.Single(tools);
            Assert.Equal("adder", (string?)entry["name"]);
            Assert.Equal("Adds a and b.", (string?)entry["description"]);
            Assert.NotNull(entry["inputSchema"]);
        }

        [Fact(Timeout = 60000)]
        public async Task Server_CallsToolAndReturnsContent()
        {
            var server = new McpServer(ToolsWith(out var tool));

            var parameters = new JObject
            {
                ["name"] = "adder",
                ["arguments"] = new JObject { ["a"] = 2, ["b"] = 3 },
            };
            var result = await server.HandleRequestAsync("tools/call", parameters);

            Assert.Single(tool.Invocations);
            Assert.False((bool)result["isError"]);
            Assert.Equal("5", (string?)result["content"]?[0]?["text"]);
        }

        [Fact(Timeout = 60000)]
        public async Task Server_Initialize_ReportsCapabilities()
        {
            var server = new McpServer(ToolsWith(out _), serverName: "my-server");
            var result = await server.HandleRequestAsync("initialize", null);

            Assert.Equal("my-server", (string?)result["serverInfo"]?["name"]);
            Assert.NotNull(result["capabilities"]?["tools"]);
        }

        [Fact(Timeout = 60000)]
        public async Task ClientServer_RoundTrip_OverInMemoryTransport()
        {
            // Wire AiDotNet's MCP client to AiDotNet's MCP server with no network: the client lists and calls
            // the server's tools end-to-end.
            var server = new McpServer(ToolsWith(out var tool));
            var client = new McpClient(new InMemoryMcpTransport(server));

            var tools = await client.GetToolsAsync();
            Assert.True(tools.Contains("adder"));

            var result = await client.CallToolAsync("adder", new JObject { ["a"] = 4, ["b"] = 5 });
            Assert.Equal("9", result.Content);
            Assert.Single(tool.Invocations);
        }

        [Fact(Timeout = 60000)]
        public async Task AgentExecutor_UsesServerToolsViaClient_EndToEnd()
        {
            // Full loop: an agent's model calls a tool that is actually served by an MCP server in-process.
            var server = new McpServer(ToolsWith(out var tool));
            var mcpTools = await new McpClient(new InMemoryMcpTransport(server)).GetToolsAsync();

            var llm = ScriptedChatClient<double>.Sequence(
                ChatResponses.ToolCall("c1", "adder", "{\"a\":7,\"b\":8}"),
                ChatResponses.Text("The sum is 15."));
            var agent = new AgentExecutor<double>(llm, mcpTools);

            var run = await agent.RunAsync("add 7 and 8");

            Assert.True(run.Completed);
            Assert.Single(tool.Invocations);
            var secondRequest = llm.Requests[1];
            Assert.Contains(secondRequest, m => m.Role == ChatRole.Tool
                && m.Contents.OfType<ToolResultContent>().Any(r => r.Result == "15"));
        }
    }
}
