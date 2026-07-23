using System.Reflection;

namespace AiDotNet.Agentic.Tools;

/// <summary>
/// Creates <see cref="IAgentTool"/> instances from delegates or from objects whose methods are annotated
/// with <see cref="AgentToolAttribute"/>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Two easy ways to make tools without writing a tool class:
/// - <see cref="FromDelegate"/>: wrap a lambda or method reference directly.
/// - <see cref="ScanInstance"/>: take any object, find every method you marked with <c>[AgentTool]</c>,
///   and turn each into a tool automatically.
/// </para>
/// </remarks>
public static class AgentToolFactory
{
    /// <summary>
    /// Creates a tool from a delegate (lambda or method group).
    /// </summary>
    /// <param name="name">The tool name exposed to the model.</param>
    /// <param name="description">A description of what the tool does.</param>
    /// <param name="function">The delegate to invoke.</param>
    /// <returns>A new <see cref="IAgentTool"/>.</returns>
    public static IAgentTool FromDelegate(string name, string description, Delegate function) =>
        new DelegateAgentTool(name, description, function);

    /// <summary>
    /// Scans an object for methods annotated with <see cref="AgentToolAttribute"/> and creates a tool for
    /// each (covering both instance and static annotated methods).
    /// </summary>
    /// <param name="target">The object to scan.</param>
    /// <returns>The discovered tools (possibly empty).</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="target"/> is <c>null</c>.</exception>
    public static IReadOnlyList<IAgentTool> ScanInstance(object target)
    {
        Guard.NotNull(target);

        var tools = new List<IAgentTool>();
        foreach (var method in target.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static))
        {
            var attribute = method.GetCustomAttribute<AgentToolAttribute>();
            if (attribute is null)
            {
                continue;
            }

            var attrName = attribute.Name;
            var name = attrName is null || attrName.Trim().Length == 0 ? method.Name : attrName;
            var toolTarget = method.IsStatic ? null : target;
            tools.Add(new DelegateAgentTool(name, attribute.Description, method, toolTarget));
        }

        return tools;
    }
}
