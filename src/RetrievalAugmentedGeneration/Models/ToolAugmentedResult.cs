using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Models;

/// <summary>
/// Result of tool-augmented reasoning.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class ToolAugmentedResult<T>
{
    /// <summary>
    /// Documents retrieved during tool-augmented reasoning.
    /// </summary>
    public IEnumerable<Document<T>> Documents { get; set; } = new List<Document<T>>();

    /// <summary>
    /// List of tool invocations made during reasoning.
    /// </summary>
    public IReadOnlyList<ToolInvocation> ToolInvocations { get; set; } = new List<ToolInvocation>();

    /// <summary>
    /// Trace of the reasoning and tool usage.
    /// </summary>
    public string ReasoningTrace { get; set; } = string.Empty;
}
