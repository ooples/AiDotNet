using System.Text;
using AiDotNet.Interfaces;

namespace AiDotNet.Agents;

/// <summary>
/// Provides a base implementation for agents that use language models and tools to solve problems.
/// This abstract class handles common agent functionality like tool management and scratchpad tracking.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// This base class is like a template for creating different types of agents. It handles the common
/// parts that all agents need (storing tools, tracking reasoning history), while allowing specific
/// agent types to implement their own reasoning strategies.
///
/// Think of it like a recipe template: all recipes have ingredients and steps, but each specific
/// recipe (chocolate cake, apple pie) implements those steps differently. Similarly, all agents
/// have tools and a reasoning process, but different agent types might implement different
/// reasoning strategies (ReAct, Chain-of-Thought, Tree-of-Thoughts, etc.).
///
/// This class follows the Template Method pattern, where common behavior is implemented here
/// and specific behavior is left for derived classes to implement.
/// </remarks>
public abstract class AgentBase<T> : IAgent<T>
{
    private readonly List<ITool> _tools;
    private readonly StringBuilder _scratchpad;

    /// <summary>
    /// Initializes a new instance of the <see cref="AgentBase{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for reasoning and decision-making.</param>
    /// <param name="tools">The collection of tools available to the agent. Can be null or empty if no tools are needed.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="chatModel"/> is null.</exception>
    /// <remarks>
    /// For Beginners:
    /// This constructor sets up a new agent with a "brain" (the chat model) and a set of tools.
    /// The chat model is required because every agent needs to be able to think and reason.
    /// Tools are optional - some tasks might not need any tools, while others might need many.
    ///
    /// The scratchpad is initialized as an empty notebook where the agent will write down
    /// its thoughts, actions, and observations as it works.
    /// </remarks>
    protected AgentBase(IChatModel<T> chatModel, IEnumerable<ITool>? tools = null)
    {
        ChatModel = chatModel ?? throw new ArgumentNullException(nameof(chatModel));
        _tools = tools?.ToList() ?? new List<ITool>();
        _scratchpad = new StringBuilder();
    }

    /// <inheritdoc/>
    public IChatModel<T> ChatModel { get; }

    /// <inheritdoc/>
    public IReadOnlyList<ITool> Tools => _tools.AsReadOnly();

    /// <inheritdoc/>
    public string Scratchpad => _scratchpad.ToString();

    /// <inheritdoc/>
    public abstract Task<string> RunAsync(string query, int maxIterations = 5);

    /// <summary>
    /// Appends a message to the agent's scratchpad.
    /// </summary>
    /// <param name="message">The message to append.</param>
    /// <remarks>
    /// For Beginners:
    /// This method is like writing in a notebook. Every time the agent has a thought,
    /// takes an action, or observes a result, it writes it down in the scratchpad.
    /// This creates a complete record of the agent's reasoning process.
    /// </remarks>
    protected void AppendToScratchpad(string message)
    {
        _scratchpad.AppendLine(message);
    }

    /// <summary>
    /// Clears the agent's scratchpad, removing all previous reasoning history.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This method is like erasing a whiteboard or starting with a fresh page in a notebook.
    /// It's typically called at the beginning of RunAsync to ensure each query starts with
    /// a clean slate, without being influenced by previous queries.
    /// </remarks>
    protected void ClearScratchpad()
    {
        _scratchpad.Clear();
    }

    /// <summary>
    /// Finds a tool by its name.
    /// </summary>
    /// <param name="toolName">The name of the tool to find.</param>
    /// <returns>The tool with the specified name, or null if no matching tool is found.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method is like searching through a toolbox for a specific tool by name.
    /// When the agent decides it needs to use a tool (like "Calculator"), this method
    /// finds that tool so the agent can execute it.
    ///
    /// The search is case-insensitive, so "calculator", "Calculator", and "CALCULATOR"
    /// all match the same tool.
    /// </remarks>
    protected ITool? FindTool(string toolName)
    {
        return _tools.FirstOrDefault(t =>
            t.Name.Equals(toolName, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Generates a formatted description of all available tools.
    /// This is typically used in prompts to inform the language model about available tools.
    /// </summary>
    /// <returns>A formatted string describing all available tools.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method creates a list of all tools and their descriptions, formatted in a way
    /// that the language model can understand. It's like giving the agent an instruction manual
    /// that says "Here are the tools you have and what each one does."
    ///
    /// Example output:
    /// <code>
    /// Available tools:
    /// - Calculator: Performs mathematical calculations. Input should be a valid expression.
    /// - Search: Searches for information. Input should be a search query.
    /// </code>
    /// </remarks>
    protected string GetToolDescriptions()
    {
        if (!_tools.Any())
        {
            return "No tools available.";
        }

        var descriptions = new StringBuilder("Available tools:\n");
        foreach (var tool in _tools)
        {
            descriptions.AppendLine($"- {tool.Name}: {tool.Description}");
        }
        return descriptions.ToString();
    }

    /// <summary>
    /// Validates that the specified maximum iterations value is valid.
    /// </summary>
    /// <param name="maxIterations">The maximum iterations value to validate.</param>
    /// <exception cref="ArgumentException">Thrown when maxIterations is less than 1.</exception>
    /// <remarks>
    /// For Beginners:
    /// This method checks that the maximum iterations setting makes sense. An agent needs
    /// at least one iteration to do anything useful, so this method ensures the value isn't
    /// zero or negative.
    ///
    /// It's a safety check to catch configuration errors early, before the agent starts running.
    /// </remarks>
    protected void ValidateMaxIterations(int maxIterations)
    {
        if (maxIterations < 1)
        {
            throw new ArgumentException(
                "Maximum iterations must be at least 1.",
                nameof(maxIterations));
        }
    }
}
