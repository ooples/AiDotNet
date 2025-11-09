namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a tool that can be used by an agent to perform specific operations.
/// Tools enable agents to interact with external systems, perform calculations, or access data.
/// </summary>
/// <remarks>
/// For Beginners:
/// Think of a tool as a specialized instrument in a toolbox. Just like a carpenter has different tools
/// for different tasks (hammer for nails, saw for cutting), an AI agent has different tools for different
/// operations (calculator for math, search for finding information, etc.).
///
/// Each tool has three key components:
/// - Name: What the tool is called (e.g., "Calculator")
/// - Description: What the tool does and when to use it
/// - Execute: The actual operation the tool performs
///
/// Example:
/// A CalculatorTool might have:
/// - Name: "Calculator"
/// - Description: "Performs mathematical calculations. Input should be a valid mathematical expression."
/// - Execute: Takes "2 + 2" and returns "4"
/// </remarks>
public interface ITool
{
    /// <summary>
    /// Gets the unique name of the tool.
    /// This name is used by the agent to identify and invoke the tool.
    /// </summary>
    /// <value>A string representing the tool's name (e.g., "Calculator", "Search", "WebScraper").</value>
    string Name { get; }

    /// <summary>
    /// Gets a detailed description of what the tool does and how to use it.
    /// This description helps the agent understand when and how to use the tool effectively.
    /// </summary>
    /// <value>A string describing the tool's functionality, input format, and expected output.</value>
    string Description { get; }

    /// <summary>
    /// Executes the tool's operation with the provided input.
    /// </summary>
    /// <param name="input">The input string required by the tool to perform its operation.
    /// The format and requirements of this input should be described in the Description property.</param>
    /// <returns>A string containing the result of the tool's execution or an error message if execution fails.</returns>
    /// <remarks>
    /// For Beginners:
    /// The Execute method is where the tool does its actual work. When called, it takes an input string,
    /// processes it according to the tool's purpose, and returns a result string.
    ///
    /// For example:
    /// - A Calculator tool might take "5 * 10" as input and return "50"
    /// - A Search tool might take "weather in New York" and return "Sunny, 72°F"
    /// - A FileReader tool might take "/path/to/file.txt" and return the file's contents
    ///
    /// If something goes wrong (invalid input, error during execution), the method should return
    /// a descriptive error message rather than throwing an exception, so the agent can understand
    /// what went wrong and potentially try again with different input.
    /// </remarks>
    string Execute(string input);
}
