using System.Text;
using System.Text.Json;
using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Tools;

/// <summary>
/// Registry for managing and executing function tools.
/// </summary>
/// <remarks>
/// <para>
/// The ToolRegistry stores available tools and provides methods for tool discovery,
/// execution, and schema generation. It acts as a central hub for tool management.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a toolbox that holds all available tools.
///
/// Example:
/// ```csharp
/// var registry = new ToolRegistry();
///
/// // Add tools
/// registry.RegisterTool(new CalculatorTool());
/// registry.RegisterTool(new WeatherTool());
/// registry.RegisterTool(new SearchTool());
///
/// // List available tools
/// var tools = registry.GetAllTools(); // Returns all 3 tools
///
/// // Execute a tool
/// var args = JsonDocument.Parse("{\"city\": \"Paris\"}");
/// var result = registry.ExecuteTool("get_weather", args);
/// ```
/// </para>
/// </remarks>
public class ToolRegistry
{
    private readonly Dictionary<string, IFunctionTool> _tools;

    /// <summary>
    /// Initializes a new instance of the ToolRegistry class.
    /// </summary>
    public ToolRegistry()
    {
        _tools = new Dictionary<string, IFunctionTool>(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Registers a tool in the registry.
    /// </summary>
    /// <param name="tool">The tool to register.</param>
    /// <exception cref="ArgumentNullException">Thrown when tool is null.</exception>
    /// <exception cref="ArgumentException">Thrown when a tool with the same name already exists.</exception>
    public void RegisterTool(IFunctionTool tool)
    {
        if (tool == null)
        {
            throw new ArgumentNullException(nameof(tool), "Tool cannot be null.");
        }

        if (_tools.ContainsKey(tool.Name))
        {
            throw new ArgumentException($"A tool with the name '{tool.Name}' is already registered.", nameof(tool));
        }

        _tools[tool.Name] = tool;
    }

    /// <summary>
    /// Unregisters a tool from the registry.
    /// </summary>
    /// <param name="toolName">The name of the tool to unregister.</param>
    /// <returns>True if the tool was removed; false if it wasn't found.</returns>
    public bool UnregisterTool(string toolName)
    {
        if (string.IsNullOrWhiteSpace(toolName))
        {
            return false;
        }

        return _tools.Remove(toolName);
    }

    /// <summary>
    /// Gets a tool by name.
    /// </summary>
    /// <param name="toolName">The name of the tool.</param>
    /// <returns>The tool if found; otherwise, null.</returns>
    public IFunctionTool? GetTool(string toolName)
    {
        if (string.IsNullOrWhiteSpace(toolName))
        {
            return null;
        }

        return _tools.TryGetValue(toolName, out var tool) ? tool : null;
    }

    /// <summary>
    /// Gets all registered tools.
    /// </summary>
    /// <returns>A read-only collection of all tools.</returns>
    public IReadOnlyList<IFunctionTool> GetAllTools()
    {
        return _tools.Values.ToList().AsReadOnly();
    }

    /// <summary>
    /// Executes a tool by name with the provided arguments.
    /// </summary>
    /// <param name="toolName">The name of the tool to execute.</param>
    /// <param name="arguments">The arguments for the tool.</param>
    /// <returns>The tool execution result.</returns>
    /// <exception cref="ArgumentException">Thrown when the tool is not found.</exception>
    public string ExecuteTool(string toolName, JsonDocument arguments)
    {
        var tool = GetTool(toolName);
        if (tool == null)
        {
            throw new ArgumentException($"Tool '{toolName}' not found in registry.", nameof(toolName));
        }

        return tool.Execute(arguments);
    }

    /// <summary>
    /// Gets the count of registered tools.
    /// </summary>
    public int Count => _tools.Count;

    /// <summary>
    /// Checks if a tool with the given name is registered.
    /// </summary>
    /// <param name="toolName">The tool name to check.</param>
    /// <returns>True if the tool is registered; otherwise, false.</returns>
    public bool HasTool(string toolName)
    {
        if (string.IsNullOrWhiteSpace(toolName))
        {
            return false;
        }

        return _tools.ContainsKey(toolName);
    }

    /// <summary>
    /// Generates a formatted description of all tools suitable for including in prompts.
    /// </summary>
    /// <returns>A formatted string describing all available tools.</returns>
    public string GenerateToolsDescription()
    {
        var builder = new StringBuilder();
        builder.AppendLine("Available Tools:");
        builder.AppendLine();

        foreach (var tool in _tools.Values)
        {
            builder.AppendLine($"Tool: {tool.Name}");
            builder.AppendLine($"Description: {tool.Description}");
            builder.AppendLine($"Parameters: {tool.ParameterSchema.RootElement.GetRawText()}");
            builder.AppendLine();
        }

        return builder.ToString();
    }

    /// <summary>
    /// Clears all tools from the registry.
    /// </summary>
    public void Clear()
    {
        _tools.Clear();
    }
}
