using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Tools;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Tools
{
    internal enum ToolColor { Red, Green, Blue }

    /// <summary>Host with [AgentTool] methods; the source generator emits CreateAgentTools() for it.</summary>
    internal sealed class GeneratedToolHost
    {
        [AgentTool("Adds two integers", Name = "add")]
        public int Add([ToolParameter("the first addend")] int a, int b) => a + b;

        [AgentTool("Echoes text in upper case")]
        public async Task<string> EchoAsync(string text, CancellationToken ct)
        {
            await Task.Yield();
            return text.ToUpperInvariant();
        }

        [AgentTool("Names a color")]
        public string Pick(ToolColor color) => color.ToString();
    }

    public class GeneratedAgentToolTests
    {
        [Fact(Timeout = 60000)]
        public async Task Generated_CreateAgentTools_ProducesSchemaAndInvokes()
        {
            await Task.Yield();

            var host = new GeneratedToolHost();

            // CreateAgentTools is the source-generated extension method.
            var tools = host.CreateAgentTools();
            Assert.Equal(3, tools.Count);

            // ---- schema (computed at compile time) ----
            var add = tools.Single(t => t.Name == "add");
            Assert.Equal("Adds two integers", add.Description);
            Assert.Equal("integer", (string?)add.ParametersSchema["properties"]?["a"]?["type"]);
            Assert.Equal("the first addend", (string?)add.ParametersSchema["properties"]?["a"]?["description"]);
            var required = add.ParametersSchema["required"] as JArray;
            Assert.NotNull(required);
            Assert.Contains("a", required.Select(x => (string?)x));
            Assert.Contains("b", required.Select(x => (string?)x));

            // enum parameter → string enum schema; CancellationToken excluded from schema
            var pick = tools.Single(t => t.Name == "Pick");
            Assert.Equal("string", (string?)pick.ParametersSchema["properties"]?["color"]?["type"]);
            var echo = tools.Single(t => t.Name == "EchoAsync");
            Assert.Null(echo.ParametersSchema["properties"]?["ct"]);

            // ---- invocation (reflection-free, typed binding) ----
            var addResult = await add.InvokeAsync(JObject.Parse("{\"a\":2,\"b\":3}"));
            Assert.False(addResult.IsError);
            Assert.Equal("5", addResult.Content);

            var echoResult = await echo.InvokeAsync(JObject.Parse("{\"text\":\"hi\"}"));
            Assert.False(echoResult.IsError);
            Assert.Equal("HI", echoResult.Content);
        }
    }
}
