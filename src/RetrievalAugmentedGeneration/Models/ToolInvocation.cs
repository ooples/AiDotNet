namespace AiDotNet.RetrievalAugmentedGeneration.Models;

/// <summary>
/// Represents a tool invocation during reasoning.
/// </summary>
public class ToolInvocation
{
    /// <summary>
    /// Name of the tool that was invoked.
    /// </summary>
    public string ToolName { get; set; } = string.Empty;

    /// <summary>
    /// Input provided to the tool.
    /// </summary>
    public string Input { get; set; } = string.Empty;

    /// <summary>
    /// Output returned by the tool.
    /// </summary>
    public string Output { get; set; } = string.Empty;

    /// <summary>
    /// Whether the tool invocation was successful.
    /// </summary>
    public bool Success { get; set; }
}
