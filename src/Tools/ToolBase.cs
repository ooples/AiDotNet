using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tools;

/// <summary>
/// Provides a base implementation for tools with common functionality such as input validation and error handling.
/// </summary>
/// <remarks>
/// <para>
/// This abstract base class implements the ITool interface and provides standard input validation, JSON parsing,
/// and error handling that is common across all tool implementations. Derived classes only need to implement
/// the core tool-specific logic by overriding the ExecuteCore method. This design follows the Template Method
/// pattern, ensuring consistent behavior and error handling across all tools while eliminating code duplication.
/// </para>
/// <para><b>For Beginners:</b> This base class provides common functionality that all tools need, so individual
/// tool implementations don't have to repeat the same code.
///
/// Why use a base class:
/// - **Eliminates duplication**: All tools need input validation and error handling - write it once here
/// - **Consistency**: All tools handle errors the same way, providing a uniform experience
/// - **Maintainability**: If we need to change error handling, we change it in one place
/// - **Simpler derived classes**: Tool implementations focus only on their specific logic
///
/// The Template Method pattern:
/// This class implements the public Execute method that:
/// 1. Validates the input (checks for null/empty)
/// 2. Tries to execute the tool-specific logic (ExecuteCore)
/// 3. Catches and handles errors gracefully
/// 4. Returns a consistent error message format
///
/// Derived tools only implement ExecuteCore with their specific logic:
/// <code>
/// public class MyTool : ToolBase
/// {
///     public override string Name => "MyTool";
///     public override string Description => "Does something useful";
///
///     protected override string ExecuteCore(string input)
///     {
///         // Tool-specific logic here
///         // Input validation already done by base class
///         // Errors automatically caught and formatted by base class
///         return "Result";
///     }
/// }
/// </code>
///
/// This makes creating new tools much simpler and more consistent.
/// </para>
/// </remarks>
public abstract class ToolBase : ITool
{
    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <inheritdoc/>
    public abstract string Description { get; }

    /// <inheritdoc/>
    public string Execute(string input)
    {
        // Validate input
        if (string.IsNullOrWhiteSpace(input))
        {
            return GetEmptyInputErrorMessage();
        }

        try
        {
            // Delegate to derived class for actual execution
            return ExecuteCore(input);
        }
        catch (JsonReaderException ex)
        {
            return GetJsonErrorMessage(ex);
        }
        catch (JsonSerializationException ex)
        {
            return GetJsonErrorMessage(new JsonReaderException(ex.Message, ex));
        }
        catch (Exception ex)
        {
            return GetGenericErrorMessage(ex);
        }
    }

    /// <summary>
    /// Executes the tool's core logic with validated input.
    /// </summary>
    /// <param name="input">The validated input string (guaranteed to be non-null and non-empty).</param>
    /// <returns>The result of the tool's execution or an error message.</returns>
    /// <remarks>
    /// <para>
    /// Derived classes implement this method to provide their specific tool functionality. The input has already
    /// been validated by the base class, so implementations can assume it's not null or empty. JSON parsing errors
    /// and general exceptions are automatically caught and handled by the base class, so implementations can focus
    /// on their core logic.
    /// </para>
    /// <para><b>For Beginners:</b> This is where you put the actual "work" your tool does.
    ///
    /// When you create a new tool, you override this method:
    /// <code>
    /// protected override string ExecuteCore(string input)
    /// {
    ///     // Parse JSON input
    ///     var json = JObject.Parse(input);
    ///
    ///     // Extract what you need
    ///     string value = json["some_field"]?.ToString() ?? "";
    ///
    ///     // Do your tool's work
    ///     string result = ProcessData(value);
    ///
    ///     // Return the result
    ///     return result;
    /// }
    /// </code>
    ///
    /// Benefits of overriding ExecuteCore instead of Execute:
    /// - Input is already validated (not null/empty)
    /// - Errors are automatically caught and formatted nicely
    /// - You just focus on your tool's specific logic
    /// - Consistent error handling across all tools
    /// </para>
    /// </remarks>
    protected abstract string ExecuteCore(string input);

    /// <summary>
    /// Gets the error message to return when input is null or empty.
    /// </summary>
    /// <returns>An error message indicating that input is required.</returns>
    /// <remarks>
    /// <para>
    /// Derived classes can override this method to provide tool-specific empty input error messages.
    /// The default implementation returns a generic message asking for JSON input.
    /// </para>
    /// <para><b>For Beginners:</b> This defines what error message users see if they don't provide any input.
    ///
    /// You can override this to provide a more helpful message specific to your tool:
    /// <code>
    /// protected override string GetEmptyInputErrorMessage()
    /// {
    ///     return "Error: CalculatorTool requires a mathematical expression. " +
    ///            "Example: { \"expression\": \"2 + 2\" }";
    /// }
    /// </code>
    ///
    /// If you don't override it, users get a generic "input cannot be empty" message.
    /// </para>
    /// </remarks>
    protected virtual string GetEmptyInputErrorMessage()
    {
        return "Error: Input cannot be empty. Please provide tool input in JSON format.";
    }

    /// <summary>
    /// Gets the error message to return when JSON parsing fails.
    /// </summary>
    /// <param name="ex">The JsonReaderException that occurred.</param>
    /// <returns>A formatted error message explaining the JSON parsing failure.</returns>
    /// <remarks>
    /// <para>
    /// Derived classes can override this method to provide tool-specific JSON error messages with
    /// example input formats. The default implementation provides a generic JSON format error message.
    /// </para>
    /// <para><b>For Beginners:</b> This defines what error message users see if their JSON is invalid.
    ///
    /// You can override this to show the expected JSON format for your specific tool:
    /// <code>
    /// protected override string GetJsonErrorMessage(JsonReaderException ex)
    /// {
    ///     return $"Error: Invalid JSON format. {ex.Message}\n" +
    ///            "Expected format: { \"expression\": \"2 + 2\" }";
    /// }
    /// </code>
    ///
    /// This helps users understand exactly what JSON structure your tool expects.
    /// </para>
    /// </remarks>
    protected virtual string GetJsonErrorMessage(JsonReaderException ex)
    {
        return $"Error: Invalid JSON format. {ex.Message}\n" +
               "Please ensure input is valid JSON.";
    }

    /// <summary>
    /// Gets the error message to return when an unexpected exception occurs.
    /// </summary>
    /// <param name="ex">The exception that occurred.</param>
    /// <returns>A formatted error message describing the unexpected error.</returns>
    /// <remarks>
    /// <para>
    /// This method formats unexpected exceptions into user-friendly error messages. The default implementation
    /// includes the exception message. Derived classes can override this to provide tool-specific error handling
    /// or additional context.
    /// </para>
    /// <para><b>For Beginners:</b> This handles any unexpected errors that occur while your tool runs.
    ///
    /// The default implementation works for most cases:
    /// <code>
    /// Error: An unexpected error occurred during [tool name] execution. [Error details]
    /// </code>
    ///
    /// You usually don't need to override this unless you want to:
    /// - Add tool-specific troubleshooting advice
    /// - Log errors differently
    /// - Provide more context about what the tool was doing when it failed
    ///
    /// Example override:
    /// <code>
    /// protected override string GetGenericErrorMessage(Exception ex)
    /// {
    ///     return $"Error: Calculator failed to evaluate expression. {ex.Message}\n" +
    ///            "Tip: Ensure expression uses valid operators (+, -, *, /)";
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    protected virtual string GetGenericErrorMessage(Exception ex)
    {
        return $"Error: An unexpected error occurred during {Name} execution. {ex.Message}";
    }

    /// <summary>
    /// Helper method to safely try to get a property from a JToken.
    /// </summary>
    /// <param name="token">The JToken to query.</param>
    /// <param name="propertyName">The name of the property to retrieve.</param>
    /// <param name="defaultValue">The default value to return if the property doesn't exist or cannot be converted.</param>
    /// <returns>The property value as a string, or the default value if not found.</returns>
    /// <remarks>
    /// <para>
    /// This utility method simplifies JSON property access by handling missing properties gracefully.
    /// It eliminates the need for repetitive null checks in derived classes.
    /// </para>
    /// <para><b>For Beginners:</b> This helps you safely read values from JSON without crashing if they're missing.
    ///
    /// Instead of this verbose code:
    /// <code>
    /// string value;
    /// if (token[propertyName] != null)
    ///     value = token[propertyName].ToString();
    /// else
    ///     value = "default";
    /// </code>
    ///
    /// You can write this:
    /// <code>
    /// string value = TryGetString(token, "name", "default");
    /// </code>
    ///
    /// Much cleaner and handles all the edge cases for you!
    /// </para>
    /// </remarks>
    protected static string TryGetString(JToken token, string propertyName, string defaultValue = "")
    {
        return token[propertyName]?.ToString() ?? defaultValue;
    }

    /// <summary>
    /// Helper method to safely try to get an integer property from a JToken.
    /// </summary>
    /// <param name="token">The JToken to query.</param>
    /// <param name="propertyName">The name of the property to retrieve.</param>
    /// <param name="defaultValue">The default value to return if the property doesn't exist or cannot be converted.</param>
    /// <returns>The property value as an integer, or the default value if not found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Same as TryGetString but for integer numbers.
    ///
    /// Example:
    /// <code>
    /// int count = TryGetInt(token, "n_samples", 1000);
    /// // If JSON has "n_samples": 5000, count will be 5000
    /// // If "n_samples" is missing or invalid, count will be 1000
    /// </code>
    /// </para>
    /// </remarks>
    protected static int TryGetInt(JToken token, string propertyName, int defaultValue = 0)
    {
        var value = token[propertyName];
        if (value == null)
            return defaultValue;

        try
        {
            return value.ToObject<int>();
        }
        catch (JsonSerializationException)
        {
            return defaultValue;
        }
    }

    /// <summary>
    /// Helper method to safely try to get a double property from a JToken.
    /// </summary>
    /// <param name="token">The JToken to query.</param>
    /// <param name="propertyName">The name of the property to retrieve.</param>
    /// <param name="defaultValue">The default value to return if the property doesn't exist or cannot be converted.</param>
    /// <returns>The property value as a double, or the default value if not found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Same as TryGetInt but for decimal numbers.
    ///
    /// Example:
    /// <code>
    /// double rate = TryGetDouble(token, "learning_rate", 0.01);
    /// </code>
    /// </para>
    /// </remarks>
    protected static double TryGetDouble(JToken token, string propertyName, double defaultValue = 0.0)
    {
        var value = token[propertyName];
        if (value == null)
            return defaultValue;

        try
        {
            return value.ToObject<double>();
        }
        catch (JsonSerializationException)
        {
            return defaultValue;
        }
    }

    /// <summary>
    /// Helper method to safely try to get a boolean property from a JToken.
    /// </summary>
    /// <param name="token">The JToken to query.</param>
    /// <param name="propertyName">The name of the property to retrieve.</param>
    /// <param name="defaultValue">The default value to return if the property doesn't exist or cannot be converted.</param>
    /// <returns>The property value as a boolean, or the default value if not found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Safely reads true/false values from JSON.
    ///
    /// Example:
    /// <code>
    /// bool isEnabled = TryGetBool(token, "is_enabled", false);
    /// </code>
    /// </para>
    /// </remarks>
    protected static bool TryGetBool(JToken token, string propertyName, bool defaultValue = false)
    {
        var value = token[propertyName];
        if (value == null)
            return defaultValue;

        try
        {
            return value.ToObject<bool>();
        }
        catch (FormatException)
        {
            return defaultValue;
        }
        catch (ArgumentException)
        {
            return defaultValue;
        }
    }
}
