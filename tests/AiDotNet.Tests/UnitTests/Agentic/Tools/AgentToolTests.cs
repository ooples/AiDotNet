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
            await Task.Yield();

            Assert.Equal("string", (JsonSchemaGenerator.ForType(typeof(string))["type"] ?? throw new InvalidOperationException("(string)JsonSchemaGenerator.ForType(typeof(string))[\"type\"] is null where it is required.")).ToString());
            Assert.Equal("integer", (JsonSchemaGenerator.ForType(typeof(int))["type"] ?? throw new InvalidOperationException("(string)JsonSchemaGenerator.ForType(typeof(int))[\"type\"] is null where it is required.")).ToString());
            Assert.Equal("number", (JsonSchemaGenerator.ForType(typeof(double))["type"] ?? throw new InvalidOperationException("(string)JsonSchemaGenerator.ForType(typeof(double))[\"type\"] is null where it is required.")).ToString());
            Assert.Equal("boolean", (JsonSchemaGenerator.ForType(typeof(bool))["type"] ?? throw new InvalidOperationException("(string)JsonSchemaGenerator.ForType(typeof(bool))[\"type\"] is null where it is required.")).ToString());

            var enumSchema = JsonSchemaGenerator.ForType(typeof(Color));
            Assert.Equal("string", (enumSchema["type"] ?? throw new InvalidOperationException("(string)enumSchema[\"type\"] is null where it is required.")).ToString());
            Assert.Equal(new[] { "Red", "Green", "Blue" }, ((JArray)(enumSchema["enum"] ?? throw new InvalidOperationException("(JArray)enumSchema[\"enum\"] is null where it is required."))).Select(t => (t ?? throw new InvalidOperationException("(string)t is null where it is required.")).ToString()));

            var arraySchema = JsonSchemaGenerator.ForType(typeof(int[]));
            Assert.Equal("array", (arraySchema["type"] ?? throw new InvalidOperationException("(string)arraySchema[\"type\"] is null where it is required.")).ToString());
            Assert.Equal("integer", (arraySchema["items"]?["type"] ?? throw new InvalidOperationException("(string)arraySchema[\"items\"][\"type\"] is null where it is required.")).ToString());

            var objSchema = JsonSchemaGenerator.ForType(typeof(Address));
            Assert.Equal("object", (objSchema["type"] ?? throw new InvalidOperationException("(string)objSchema[\"type\"] is null where it is required.")).ToString());
            Assert.Equal("string", (objSchema["properties"]?["Street"]?["type"] ?? throw new InvalidOperationException("(string)objSchema[\"properties\"][\"Street\"][\"type\"] is null where it is required.")).ToString());
            Assert.Equal("integer", (objSchema["properties"]?["Zip"]?["type"] ?? throw new InvalidOperationException("(string)objSchema[\"properties\"][\"Zip\"][\"type\"] is null where it is required.")).ToString());

            var dictSchema = JsonSchemaGenerator.ForType(typeof(Dictionary<string, int>));
            Assert.Equal("object", (dictSchema["type"] ?? throw new InvalidOperationException("(string)dictSchema[\"type\"] is null where it is required.")).ToString());
            Assert.Equal("integer", (dictSchema["additionalProperties"]?["type"] ?? throw new InvalidOperationException("(string)dictSchema[\"additionalProperties\"][\"type\"] is null where it is required.")).ToString());
        }

        // ---- JsonSchemaGenerator: parameters / required ----

        private static int Add(int a, int b) => a + b;
        private static int WithDefault(int a, int b = 10) => a + b;
        private static string Describe([ToolParameter("the city")] string city, CancellationToken ct) => city;

        [Fact(Timeout = 60000)]
        public async Task ForParameters_MarksRequired_ExcludesCancellationToken_AppliesDescriptions()
        {
            await Task.Yield();

            var addSchema = JsonSchemaGenerator.ForParameters(
                ((Func<int, int, int>)Add).Method.GetParameters());
            Assert.Equal(new[] { "a", "b" }, ((JArray)(addSchema["required"] ?? throw new InvalidOperationException("(JArray)addSchema[\"required\"] is null where it is required."))).Select(t => (t ?? throw new InvalidOperationException("(string)t is null where it is required.")).ToString()).OrderBy(x => x));

            var defaultSchema = JsonSchemaGenerator.ForParameters(
                ((Func<int, int, int>)WithDefault).Method.GetParameters());
            var required = ((JArray)(defaultSchema["required"] ?? throw new InvalidOperationException("(JArray)defaultSchema[\"required\"] is null where it is required."))).Select(t => (t ?? throw new InvalidOperationException("(string)t is null where it is required.")).ToString()).ToList();
            Assert.Contains("a", required);
            Assert.DoesNotContain("b", required); // has default -> optional

            var describeSchema = JsonSchemaGenerator.ForParameters(
                ((Func<string, CancellationToken, string>)Describe).Method.GetParameters());
            Assert.NotNull((describeSchema["properties"] ?? throw new InvalidOperationException("describeSchema[\"properties\"] is null where it is required."))["city"]);
            Assert.Null((describeSchema["properties"] ?? throw new InvalidOperationException("describeSchema[\"properties\"] is null where it is required."))["ct"]); // CancellationToken excluded
            Assert.Equal("the city", (describeSchema["properties"]?["city"]?["description"] ?? throw new InvalidOperationException("(string)describeSchema[\"properties\"][\"city\"][\"description\"] is null where it is required.")).ToString());
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
            await Task.Yield();

            var tool = new DelegateAgentTool("add", "adds", (Func<int, int, int>)Add);
            var result = await tool.InvokeAsync(JObject.Parse("{\"a\":2,\"b\":3}"));

            Assert.False(result.IsError);
            Assert.Equal("5", result.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task DelegateTool_Async_AwaitsAndReturnsResult()
        {
            await Task.Yield();

            var tool = new DelegateAgentTool("echo", "echoes", (Func<string, CancellationToken, Task<string>>)EchoAsync);
            var result = await tool.InvokeAsync(JObject.Parse("{\"text\":\"hi\"}"));

            Assert.False(result.IsError);
            Assert.Equal("HI", result.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task DelegateTool_UsesDefault_WhenArgMissing()
        {
            await Task.Yield();

            var tool = new DelegateAgentTool("wd", "with default", (Func<int, int, int>)WithDefault);
            var result = await tool.InvokeAsync(JObject.Parse("{\"a\":5}"));

            Assert.False(result.IsError);
            Assert.Equal("15", result.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task DelegateTool_MissingRequired_ReturnsError()
        {
            await Task.Yield();

            var tool = new DelegateAgentTool("add", "adds", (Func<int, int, int>)Add);
            var result = await tool.InvokeAsync(JObject.Parse("{\"a\":2}"));

            Assert.True(result.IsError);
            Assert.Contains("Missing required parameter 'b'", result.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task DelegateTool_BadArgConversion_ReturnsError()
        {
            await Task.Yield();

            var tool = new DelegateAgentTool("add", "adds", (Func<int, int, int>)Add);
            var result = await tool.InvokeAsync(JObject.Parse("{\"a\":\"notanumber\",\"b\":3}"));

            Assert.True(result.IsError);
            Assert.Contains("'a'", result.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task DelegateTool_ThrowingMethod_IsCaughtAsError()
        {
            await Task.Yield();

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
            await Task.Yield();

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
            await Task.Yield();

            var tools = new ToolCollection();
            tools.AddDelegate("add", "adds", (Func<int, int, int>)Add);
            Assert.Throws<ArgumentException>(() => tools.AddDelegate("add", "again", (Func<int, int, int>)Add));
        }

        [Fact(Timeout = 60000)]
        public async Task ToolCollection_GetDefinitions_ReflectsRegisteredTools()
        {
            await Task.Yield();

            var tools = new ToolCollection().AddFrom(new SampleTools());
            var defs = tools.GetDefinitions();

            Assert.Equal(2, defs.Count);
            Assert.Contains(defs, d => d.Name == "add");
        }

        [Fact(Timeout = 60000)]
        public async Task ToolCollection_InvokeToToolMessage_WrapsResult()
        {
            await Task.Yield();

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
            await Task.Yield();

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
            await Task.Yield();

            var tools = new ToolCollection().AddDelegate("add", "adds", (Func<int, int, int>)Add);
            var call = new ToolCallContent("call_1", "add", "{not valid json");
            var result = await tools.InvokeAsync(call);

            Assert.True(result.IsError);
            Assert.Contains("not valid JSON", result.Content);
        }
    }
}
