using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
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
    public class McpClientTests
    {
        // An in-memory MCP server: answers tools/list with one tool and tools/call by echoing the arguments.
        private sealed class FakeMcpTransport : IMcpTransport
        {
            public List<string> Methods { get; } = new();
            public JObject? LastCallParams { get; private set; }

            public Task<JObject> SendRequestAsync(string method, JObject? parameters, CancellationToken cancellationToken = default)
            {
                Methods.Add(method);
                switch (method)
                {
                    case "initialize":
                        return Task.FromResult(new JObject { ["serverInfo"] = new JObject { ["name"] = "fake-mcp" } });

                    case "tools/list":
                        return Task.FromResult(new JObject
                        {
                            ["tools"] = new JArray
                            {
                                new JObject
                                {
                                    ["name"] = "echo",
                                    ["description"] = "Echoes its input.",
                                    ["inputSchema"] = new JObject
                                    {
                                        ["type"] = "object",
                                        ["properties"] = new JObject { ["text"] = new JObject { ["type"] = "string" } },
                                    },
                                },
                            },
                        });

                    case "tools/call":
                        LastCallParams = parameters;
                        var text = (string?)parameters?["arguments"]?["text"] ?? string.Empty;
                        return Task.FromResult(new JObject
                        {
                            ["content"] = new JArray { new JObject { ["type"] = "text", ["text"] = "echo:" + text } },
                        });

                    default:
                        throw new McpException("Unexpected method: " + method);
                }
            }
        }

        [Fact(Timeout = 60000)]
        public async Task ListTools_ReturnsServerToolDescriptors()
        {
            await Task.Yield();

            var client = new McpClient(new FakeMcpTransport());
            var tools = await client.ListToolsAsync();

            var tool = Assert.Single(tools);
            Assert.Equal("echo", tool.Name);
            Assert.Equal("Echoes its input.", tool.Description);
            Assert.Equal("object", (string?)tool.InputSchema["type"]);
        }

        [Fact(Timeout = 60000)]
        public async Task CallTool_ForwardsArguments_AndReturnsResult()
        {
            await Task.Yield();

            var transport = new FakeMcpTransport();
            var client = new McpClient(transport);

            var result = await client.CallToolAsync("echo", new JObject { ["text"] = "hi" });

            Assert.False(result.IsError);
            Assert.Equal("echo:hi", result.Content);
            Assert.Equal("echo", (string?)transport.LastCallParams?["name"]);
        }

        [Fact(Timeout = 60000)]
        public async Task GetTools_ExposesMcpToolsAsAgentTools()
        {
            await Task.Yield();

            var client = new McpClient(new FakeMcpTransport());
            var tools = await client.GetToolsAsync();

            Assert.True(tools.Contains("echo"));
            var definition = tools.GetDefinitions().Single();
            Assert.Equal("echo", definition.Name);
        }

        [Fact(Timeout = 60000)]
        public async Task Initialize_PerformsHandshake()
        {
            await Task.Yield();

            var transport = new FakeMcpTransport();
            var client = new McpClient(transport);

            var info = await client.InitializeAsync();

            Assert.Equal("fake-mcp", (string?)info["serverInfo"]?["name"]);
            Assert.Contains("initialize", transport.Methods);
        }

        [Fact(Timeout = 60000)]
        public async Task McpTool_IsCallableThroughAgentExecutor()
        {
            await Task.Yield();

            // An MCP server tool, used by a normal agent: the model calls "echo", the call goes to the server,
            // and the result flows back into the loop.
            var mcpTools = await new McpClient(new FakeMcpTransport()).GetToolsAsync();

            var llm = ScriptedChatClient<double>.Sequence(
                ChatResponses.ToolCall("c1", "echo", "{\"text\":\"world\"}"),
                ChatResponses.Text("The tool said: echo:world"));
            var agent = new AgentExecutor<double>(llm, mcpTools);

            var run = await agent.RunAsync("use the echo tool");

            Assert.True(run.Completed);
            var secondRequest = llm.Requests[1];
            Assert.Contains(secondRequest, m => m.Role == ChatRole.Tool
                && m.Contents.OfType<ToolResultContent>().Any(r => r.Result == "echo:world"));
        }
    }
}
