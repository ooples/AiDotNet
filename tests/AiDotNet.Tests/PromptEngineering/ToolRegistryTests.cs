using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.Tools;
using Newtonsoft.Json.Linq;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.PromptEngineering;

public class ToolRegistryTests
{
    private class MockTool : IFunctionTool
    {
        public string Name { get; }
        public string Description { get; }
        public JObject ParameterSchema { get; }

        public MockTool(string name)
        {
            Name = name;
            Description = $"Mock tool {name}";
            ParameterSchema = JObject.Parse("""
                {
                    "type": "object",
                    "properties": {
                        "input": { "type": "string" }
                    },
                    "required": ["input"]
                }
                """);
        }

        public string Execute(JObject arguments)
        {
            return $"Executed {Name}";
        }

        public bool ValidateArguments(JObject arguments)
        {
            return arguments.TryGetValue("input", out _);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_CreatesEmptyRegistry()
    {
        var registry = new ToolRegistry();

        Assert.Equal(0, registry.Count);
    }

    [Fact(Timeout = 60000)]
    public async Task RegisterTool_AddsTool()
    {
        var registry = new ToolRegistry();
        var tool = new MockTool("test_tool");

        registry.RegisterTool(tool);

        Assert.Equal(1, registry.Count);
        Assert.True(registry.HasTool("test_tool"));
    }

    [Fact(Timeout = 60000)]
    public async Task RegisterTool_WithNullTool_ThrowsArgumentNullException()
    {
        var registry = new ToolRegistry();

        Assert.Throws<ArgumentNullException>(() => registry.RegisterTool(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task RegisterTool_WithDuplicateName_ThrowsArgumentException()
    {
        var registry = new ToolRegistry();
        var tool1 = new MockTool("duplicate");
        var tool2 = new MockTool("duplicate");

        registry.RegisterTool(tool1);

        Assert.Throws<ArgumentException>(() => registry.RegisterTool(tool2));
    }

    [Fact(Timeout = 60000)]
    public async Task GetTool_ReturnsRegisteredTool()
    {
        var registry = new ToolRegistry();
        var tool = new MockTool("test_tool");
        registry.RegisterTool(tool);

        var retrieved = registry.GetTool("test_tool");

        Assert.NotNull(retrieved);
        Assert.Equal("test_tool", retrieved.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task GetTool_WithNonExistentTool_ReturnsNull()
    {
        var registry = new ToolRegistry();

        var retrieved = registry.GetTool("nonexistent");

        Assert.Null(retrieved);
    }

    [Fact(Timeout = 60000)]
    public async Task UnregisterTool_RemovesTool()
    {
        var registry = new ToolRegistry();
        var tool = new MockTool("test_tool");
        registry.RegisterTool(tool);

        var result = registry.UnregisterTool("test_tool");

        Assert.True(result);
        Assert.Equal(0, registry.Count);
        Assert.False(registry.HasTool("test_tool"));
    }

    [Fact(Timeout = 60000)]
    public async Task UnregisterTool_WithNonExistentTool_ReturnsFalse()
    {
        var registry = new ToolRegistry();

        var result = registry.UnregisterTool("nonexistent");

        Assert.False(result);
    }

    [Fact(Timeout = 60000)]
    public async Task GetAllTools_ReturnsAllRegisteredTools()
    {
        var registry = new ToolRegistry();
        registry.RegisterTool(new MockTool("tool1"));
        registry.RegisterTool(new MockTool("tool2"));
        registry.RegisterTool(new MockTool("tool3"));

        var tools = registry.GetAllTools();

        Assert.Equal(3, tools.Count);
    }

    [Fact(Timeout = 60000)]
    public async Task ExecuteTool_ExecutesRegisteredTool()
    {
        var registry = new ToolRegistry();
        var tool = new MockTool("test_tool");
        registry.RegisterTool(tool);

        var args = JObject.Parse("""{"input": "test"}""");
        var result = registry.ExecuteTool("test_tool", args);

        Assert.Equal("Executed test_tool", result);
    }

    [Fact(Timeout = 60000)]
    public async Task ExecuteTool_WithNonExistentTool_ThrowsArgumentException()
    {
        var registry = new ToolRegistry();
        var args = JObject.Parse("""{"input": "test"}""");

        Assert.Throws<ArgumentException>(() => registry.ExecuteTool("nonexistent", args));
    }

    [Fact(Timeout = 60000)]
    public async Task Clear_RemovesAllTools()
    {
        var registry = new ToolRegistry();
        registry.RegisterTool(new MockTool("tool1"));
        registry.RegisterTool(new MockTool("tool2"));

        registry.Clear();

        Assert.Equal(0, registry.Count);
    }

    [Fact(Timeout = 60000)]
    public async Task GenerateToolsDescription_CreatesFormattedDescription()
    {
        var registry = new ToolRegistry();
        registry.RegisterTool(new MockTool("tool1"));

        var description = registry.GenerateToolsDescription();

        Assert.Contains("Available Tools:", description);
        Assert.Contains("tool1", description);
    }
}
