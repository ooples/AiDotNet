using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Tools
{
    public class AgentToolTests
    {
        private enum Color { Red, Green, Blue }

        private sealed class Address
        {
            public string Street { get; set; } = "";
            public int Zip { get; set; }
        }

        // ---- JsonSchemaGenerator: type mapping ----

        [Fact(Timeout = 60000)]
        public async Task ForType_MapsPrimitivesAndEnumsAndCollections()
        {
            Assert.Equal("string", (string)JsonSchemaGenerator.ForType(typeof(string))["type"]);
            Assert.Equal("integer", (string)JsonSchemaGenerator.ForType(typeof(int))["type"]);
            Assert.Equal("number", (string)JsonSchemaGenerator.ForType(typeof(double))["type"]);
            Assert.Equal("boolean", (string)JsonSchemaGenerator.ForType(typeof(bool))["type"]);

            var enumSchema = JsonSchemaGenerator.ForType(typeof(Color));
            Assert.Equal("string", (string)enumSchema["type"]);
            Assert.Equal(new[] { "Red", "Green", "Blue" }, ((JArray)enumSchema["enum"]).Select(t => (string)t));

            var arraySchema = JsonSchemaGenerator.ForType(typeof(int[]));
            Assert.Equal("array", (string)arraySchema["type"]);
            Assert.Equal("integer", (string)arraySchema["items"]["type"]);

            var objSchema = JsonSchemaGenerator.ForType(typeof(Address));
            Assert.Equal("object", (string)objSchema["type"]);
            Assert.Equal("string", (string)objSchema["properties"]["Street"]["type"]);
            Assert.Equal("integer", (string)objSchema["properties"]["Zip"]["type"]);

            var dictSchema = JsonSchemaGenerator.ForType(typeof(Dictionary<string, int>));
            Assert.Equal("object", (string)dictSchema["type"]);
            Assert.Equal("integer", (string)dictSchema["additionalProperties"]["type"]);
        }

        // ---- JsonSchemaGenerator: parameters / required ----

        private static int Add(int a, int b) => a + b;
        private static int WithDefault(int a, int b = 10) => a + b;
        private static string Describe([ToolParameter("the city")] string city, CancellationToken ct) => city;

        [Fact(Timeout = 60000)]
        public async Task ForParameters_MarksRequired_ExcludesCancellationToken_AppliesDescriptions()
        {
            var addSchema = JsonSchemaGenerator.ForParameters(
                ((Func<int, int, int>)Add).Method.GetParameters());
            Assert.Equal(new[] { "a", "b" }, ((JArray)addSchema["required"]).Select(t => (string)t).OrderBy(x => x));

            var defaultSchema = JsonSchemaGenerator.ForParameters(
                ((Func<int, int, int>)WithDefault).Method.GetParameters());
            var required = ((JArray)defaultSchema["required"]).Select(t => (string)t).ToList();
            Assert.Contains("a", required);
            Assert.DoesNotContain("b", required); // has default -> optional

            var describeSchema = JsonSchemaGenerator.ForParameters(
                ((Func<string, CancellationToken, string>)Describe).Method.GetParameters());
            Assert.NotNull(describeSchema["properties"]["city"]);
            Assert.Null(describeSchema["properties"]["ct"]); // CancellationToken excluded
            Assert.Equal("the city", (string)describeSchema["properties"]["city"]["description"]);
        }

        // ---- DelegateAgentTool: invocation ----

        private static async Task<string> EchoAsync(string text, CancellationToken ct)
        {
            await Task.Yield();
            return text.ToUpperInvariant();
        }

        private static string Boom() => throw new InvalidOperationException("kaboom");

        [Fact(Timeout = 60000)]
        public async Task DelegateTool_Sync_BindsArgsAndReturnsResult()
        {
            var tool = new DelegateAgentTool("add", "adds", (Func<int, int, int>)Add);
            var result = await tool.InvokeAsync(JObject.Parse("{\"a\":2,\"b\":3}"));

            Assert.False(result.IsError);
            Assert.Equal("5", result.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task DelegateTool_Async_AwaitsAndReturnsResult()
        {
            var tool = new DelegateAgentTool("echo", "echoes", (Func<string, CancellationToken, Task<string>>)EchoAsync);
            var result = await tool.InvokeAsync(JObject.Parse("{\"text\":\"hi\"}"));

            Assert.False(result.IsError);
            Assert.Equal("HI", result.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task DelegateTool_UsesDefault_WhenArgMissing()
        {
            var tool = new DelegateAgentTool("wd", "with default", (Func<int, int, int>)WithDefault);
            var result = await tool.InvokeAsync(JObject.Parse("{\"a\":5}"));

            Assert.False(result.IsError);
            Assert.Equal("15", result.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task DelegateTool_MissingRequired_ReturnsError()
        {
            var tool = new DelegateAgentTool("add", "adds", (Func<int, int, int>)Add);
            var result = await tool.InvokeAsync(JObject.Parse("{\"a\":2}"));

            Assert.True(result.IsError);
            Assert.Contains("Missing required parameter 'b'", result.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task DelegateTool_BadArgConversion_ReturnsError()
        {
            var tool = new DelegateAgentTool("add", "adds", (Func<int, int, int>)Add);
            var result = await tool.InvokeAsync(JObject.Parse("{\"a\":\"notanumber\",\"b\":3}"));

            Assert.True(result.IsError);
            Assert.Contains("'a'", result.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task DelegateTool_ThrowingMethod_IsCaughtAsError()
        {
            var tool = new DelegateAgentTool("boom", "throws", (Func<string>)Boom);
            var result = await tool.InvokeAsync(new JObject());

            Assert.True(result.IsError);
            Assert.Contains("kaboom", result.Content);
        }

        // ---- AgentToolFactory: attribute scanning ----

        private sealed class SampleTools
        {
            [AgentTool("Adds two numbers", Name = "add")]
            public int Add(int a, int b) => a + b;

            [AgentTool("Pings")]
            public static string Ping() => "pong";

            public int NotATool(int x) => x;
        }

        [Fact(Timeout = 60000)]
        public async Task ScanInstance_DiscoversAnnotatedMethods_InstanceAndStatic()
        {
            var tools = AgentToolFactory.ScanInstance(new SampleTools());
            var names = tools.Select(t => t.Name).ToArray();

            Assert.Equal(2, names.Length);
            Assert.Contains("add", names);   // instance method, custom name
            Assert.Contains("Ping", names);  // static method, default name
            var add = tools.Single(t => t.Name == "add");
            Assert.Equal("Adds two numbers", add.Description);
        }

        // ---- ToolCollection ----

        [Fact(Timeout = 60000)]
        public async Task ToolCollection_Add_Duplicate_Throws()
        {
            var tools = new ToolCollection();
            tools.AddDelegate("add", "adds", (Func<int, int, int>)Add);
            Assert.Throws<ArgumentException>(() => tools.AddDelegate("add", "again", (Func<int, int, int>)Add));
        }

        [Fact(Timeout = 60000)]
        public async Task ToolCollection_GetDefinitions_ReflectsRegisteredTools()
        {
            var tools = new ToolCollection().AddFrom(new SampleTools());
            var defs = tools.GetDefinitions();

            Assert.Equal(2, defs.Count);
            Assert.Contains(defs, d => d.Name == "add");
        }

        [Fact(Timeout = 60000)]
        public async Task ToolCollection_InvokeToToolMessage_WrapsResult()
        {
            var tools = new ToolCollection().AddDelegate("add", "adds", (Func<int, int, int>)Add);

            var call = new ToolCallContent("call_1", "add", "{\"a\":4,\"b\":6}");
            var message = await tools.InvokeToToolMessageAsync(call);

            Assert.Equal(ChatRole.Tool, message.Role);
            var content = Assert.IsType<ToolResultContent>(Assert.Single(message.Contents));
            Assert.Equal("call_1", content.CallId);
            Assert.Equal("10", content.Result);
            Assert.False(content.IsError);
        }

        [Fact(Timeout = 60000)]
        public async Task ToolCollection_UnknownTool_ReturnsErrorMessage()
        {
            var tools = new ToolCollection();
            var call = new ToolCallContent("call_1", "missing", "{}");
            var message = await tools.InvokeToToolMessageAsync(call);

            var content = Assert.IsType<ToolResultContent>(Assert.Single(message.Contents));
            Assert.True(content.IsError);
            Assert.Contains("Unknown tool 'missing'", content.Result);
        }

        [Fact(Timeout = 60000)]
        public async Task ToolCollection_MalformedArgsJson_ReturnsError()
        {
            var tools = new ToolCollection().AddDelegate("add", "adds", (Func<int, int, int>)Add);
            var call = new ToolCallContent("call_1", "add", "{not valid json");
            var result = await tools.InvokeAsync(call);

            Assert.True(result.IsError);
            Assert.Contains("not valid JSON", result.Content);
        }
    }
}
