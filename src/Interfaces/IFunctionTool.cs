using Newtonsoft.Json.Linq;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for tools (functions) that language models can invoke.
/// </summary>
/// <remarks>
/// <para>
/// A function tool represents an external capability that a language model can use to accomplish tasks
/// beyond text generation. Tools provide structured interfaces for actions like searching databases,
/// performing calculations, fetching web content, or executing code.
/// </para>
/// <para><b>For Beginners:</b> A function tool is like giving the AI a superpower or tool to use.
///
/// Think of tools like apps on a smartphone:
/// - The phone (LLM) is powerful on its own
/// - Apps (tools) extend its capabilities
/// - The phone decides which app to use based on what you need
/// - Each app has specific inputs and outputs
///
/// Example tools:
/// - Calculator: Perform complex math
/// - Web Search: Find current information
/// - Database Query: Retrieve stored data
/// - Weather API: Get current weather
/// - Code Executor: Run programming code
///
/// How tools work with LLMs:
/// 1. User: "What's 15% of 2,847?"
/// 2. LLM: "I need to calculate this" → Calls Calculator tool
/// 3. Calculator: Executes calculation → Returns 427.05
/// 4. LLM: Uses result → "15% of 2,847 is 427.05"
///
/// Without tools, the LLM could only estimate. With tools, it gets exact answers.
/// </para>
/// </remarks>
public interface IFunctionTool
{
    /// <summary>
    /// Gets the name of the function tool.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The name should be descriptive and follow standard naming conventions (e.g., snake_case).
    /// This name is used by the language model to identify and invoke the tool.
    /// </para>
    /// <para><b>For Beginners:</b> This is the tool's identifier, like an app name.
    ///
    /// Examples:
    /// - "get_current_weather"
    /// - "search_database"
    /// - "calculate"
    /// - "fetch_web_page"
    ///
    /// Good names are:
    /// - Descriptive: Clearly indicates what the tool does
    /// - Concise: Not too long or complex
    /// - Consistent: Follow a naming pattern (verb_noun)
    /// - Unambiguous: Distinct from other tool names
    /// </para>
    /// </remarks>
    string Name { get; }

    /// <summary>
    /// Gets the description of what the function does.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The description helps the language model understand when and how to use the tool.
    /// It should clearly explain the tool's purpose, capabilities, and limitations.
    /// </para>
    /// <para><b>For Beginners:</b> This explains what the tool does, like an app description.
    ///
    /// Example:
    /// Name: "get_current_weather"
    /// Description: "Retrieves the current weather conditions for a specified location.
    ///              Provides temperature, conditions, humidity, and wind speed.
    ///              Works for cities worldwide."
    ///
    /// The description helps the LLM decide:
    /// - WHEN to use the tool: "User asked about weather? Use this tool!"
    /// - WHAT it returns: "It gives temperature and conditions"
    /// - LIMITATIONS: "Only current weather, not forecasts"
    ///
    /// Good descriptions:
    /// - Clear purpose: What the tool does
    /// - Expected output: What it returns
    /// - Limitations: What it can't do
    /// - Use cases: When to use it
    /// </para>
    /// </remarks>
    string Description { get; }

    /// <summary>
    /// Gets the JSON schema describing the function's parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The schema defines what arguments the function accepts, their types, whether they're required,
    /// and validation rules. This follows the JSON Schema specification to enable structured function calling.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a form that defines what information the tool needs.
    ///
    /// Example for "get_current_weather":
    /// Schema:
    /// {
    ///   "type": "object",
    ///   "properties": {
    ///     "location": {
    ///       "type": "string",
    ///       "description": "The city and state, e.g., San Francisco, CA"
    ///     },
    ///     "unit": {
    ///       "type": "string",
    ///       "enum": ["celsius", "fahrenheit"],
    ///       "description": "The temperature unit"
    ///     }
    ///   },
    ///   "required": ["location"]
    /// }
    ///
    /// This tells the LLM:
    /// - "location" is required (must provide)
    /// - "unit" is optional (has default)
    /// - "unit" must be either "celsius" or "fahrenheit"
    /// - What each parameter means
    ///
    /// The LLM uses this to generate valid function calls:
    /// get_current_weather(location="Paris, France", unit="celsius")
    /// </para>
    /// </remarks>
    JObject ParameterSchema { get; }

    /// <summary>
    /// Executes the function with the provided arguments.
    /// </summary>
    /// <param name="arguments">The function arguments as a JSON document.</param>
    /// <returns>The function result as a string.</returns>
    /// <remarks>
    /// <para>
    /// This method is called when the language model invokes the tool. The arguments are provided
    /// as a JSON document matching the parameter schema. The result should be a string that the
    /// language model can interpret and use in its response.
    /// </para>
    /// <para><b>For Beginners:</b> This is what actually runs when the tool is used.
    ///
    /// Flow:
    /// 1. LLM decides to use the tool
    /// 2. LLM generates arguments: {"location": "Tokyo, Japan", "unit": "celsius"}
    /// 3. Execute() is called with these arguments
    /// 4. Your code runs (e.g., calls weather API)
    /// 5. Your code returns result: "Temperature: 22°C, Condition: Partly cloudy"
    /// 6. LLM receives result and uses it in response
    ///
    /// Example implementation:
    /// ```csharp
    /// public string Execute(JObject arguments)
    /// {
    ///     var location = arguments.Value<string>("location") ?? "";
    ///     var unit = arguments.Value<string>("unit") ?? "fahrenheit";
    ///
    ///     // Call weather API
    ///     var weather = WeatherAPI.GetCurrent(location, unit);
    ///
    ///     // Return formatted result
    ///     return $"Temperature: {weather.Temp}°{unit}, Condition: {weather.Condition}";
    /// }
    /// ```
    ///
    /// Error handling:
    /// - If arguments are invalid, throw an exception with a clear message
    /// - If the tool fails, return an error description (don't crash)
    /// - The LLM can handle error messages and try alternatives
    /// </para>
    /// </remarks>
    string Execute(JObject arguments);

    /// <summary>
    /// Validates that the provided arguments match the parameter schema.
    /// </summary>
    /// <param name="arguments">The function arguments to validate.</param>
    /// <returns>True if arguments are valid; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Checks whether the provided arguments conform to the parameter schema,
    /// including type validation, required field presence, and constraint checking.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if the arguments are correct before executing.
    ///
    /// Example:
    /// Schema requires: location (required string), unit (optional, must be "celsius" or "fahrenheit")
    ///
    /// Arguments: {"location": "Paris"}
    /// Validate() → True (location present, unit optional)
    ///
    /// Arguments: {"unit": "celsius"}
    /// Validate() → False (missing required "location")
    ///
    /// Arguments: {"location": "Paris", "unit": "kelvin"}
    /// Validate() → False ("kelvin" not in allowed values)
    ///
    /// This helps:
    /// - Catch errors early
    /// - Provide clear error messages
    /// - Prevent invalid operations
    /// - Ensure data quality
    /// </para>
    /// </remarks>
    bool ValidateArguments(JObject arguments);
}
