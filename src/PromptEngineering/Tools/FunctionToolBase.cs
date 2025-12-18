using System.Text.Json;
using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Tools;

/// <summary>
/// Base class for function tool implementations.
/// </summary>
/// <remarks>
/// <para>
/// This base class provides common functionality for tools including schema management,
/// argument validation, and error handling. Derived classes implement the core execution logic.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for creating tools that LLMs can use.
///
/// When creating a new tool:
/// 1. Inherit from this class
/// 2. Set Name, Description, and ParameterSchema in constructor
/// 3. Implement ExecuteCore method with your tool's logic
///
/// Everything else (validation, error handling) is handled automatically!
/// </para>
/// </remarks>
public abstract class FunctionToolBase : IFunctionTool
{
    /// <summary>
    /// Gets the name of the function tool.
    /// </summary>
    public string Name { get; protected set; }

    /// <summary>
    /// Gets the description of what the function does.
    /// </summary>
    public string Description { get; protected set; }

    /// <summary>
    /// Gets the JSON schema describing the function's parameters.
    /// </summary>
    public JsonDocument ParameterSchema { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the FunctionToolBase class.
    /// </summary>
    /// <param name="name">The function name.</param>
    /// <param name="description">The function description.</param>
    /// <param name="parameterSchema">The parameter schema as a JSON document.</param>
    protected FunctionToolBase(string name, string description, JsonDocument parameterSchema)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Function name cannot be empty.", nameof(name));
        }

        if (parameterSchema == null)
        {
            throw new ArgumentNullException(nameof(parameterSchema), "Parameter schema cannot be null.");
        }

        Name = name;
        Description = description ?? string.Empty;
        ParameterSchema = parameterSchema;
    }

    /// <summary>
    /// Executes the function with the provided arguments.
    /// </summary>
    /// <param name="arguments">The function arguments as a JSON document.</param>
    /// <returns>The function result as a string.</returns>
    public string Execute(JsonDocument arguments)
    {
        if (arguments == null)
        {
            throw new ArgumentNullException(nameof(arguments), "Arguments cannot be null.");
        }

        if (!ValidateArguments(arguments))
        {
            throw new ArgumentException("Arguments do not match the parameter schema.", nameof(arguments));
        }

        try
        {
            return ExecuteCore(arguments);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            return $"Error executing function '{Name}': {ex.Message}";
        }
    }

    /// <summary>
    /// Validates that the provided arguments match the parameter schema.
    /// </summary>
    /// <param name="arguments">The function arguments to validate.</param>
    /// <returns>True if arguments are valid; otherwise, false.</returns>
    public virtual bool ValidateArguments(JsonDocument arguments)
    {
        if (arguments == null)
        {
            return false;
        }

        try
        {
            // Get required fields from schema
            var schema = ParameterSchema.RootElement;
            if (schema.TryGetProperty("required", out var required))
            {
                var requiredFields = required.EnumerateArray()
                    .Select(e => e.GetString())
                    .Where(s => s != null)
                    .ToList();

                var args = arguments.RootElement;

                // Check that all required fields are present using LINQ
                var allFieldsPresent = requiredFields.All(field => args.TryGetProperty(field!, out _));
                if (!allFieldsPresent)
                {
                    return false;
                }
            }

            return true;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            return false;
        }
    }

    /// <summary>
    /// Core execution logic to be implemented by derived classes.
    /// </summary>
    /// <param name="arguments">The validated arguments.</param>
    /// <returns>The function result.</returns>
    protected abstract string ExecuteCore(JsonDocument arguments);
}
