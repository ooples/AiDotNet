using AiDotNet.Enums;

namespace AiDotNet.PromptEngineering.Templates;

/// <summary>
/// Simple prompt template with basic variable substitution.
/// </summary>
/// <remarks>
/// <para>
/// This template provides straightforward variable substitution without any complex formatting
/// or structure. It's the simplest and fastest template type, suitable for basic prompts.
/// </para>
/// <para><b>For Beginners:</b> The most basic template - just fill in the blanks.
///
/// Example:
/// ```csharp
/// var template = new SimplePromptTemplate(
/// "Translate the following {source_lang} text to {target_lang}: {text}"
/// );
///
/// var prompt = template.Format(new Dictionary<string, string>
/// {
///     ["source_lang"] = "English",
///     ["target_lang"] = "Spanish",
///     ["text"] = "Hello, how are you?"
/// });
///
/// // Result: "Translate the following English text to Spanish: Hello, how are you?"
/// ```
///
/// Use this when:
/// - You have a simple prompt structure
/// - No special formatting needed
/// - Performance is important (fastest option)
/// - Prompt logic is straightforward
/// </para>
/// </remarks>
public class SimplePromptTemplate : PromptTemplateBase
{
    /// <summary>
    /// Initializes a new instance of the SimplePromptTemplate class.
    /// </summary>
    /// <param name="template">The template string with variable placeholders in {variable_name} format.</param>
    public SimplePromptTemplate(string template) : base(template)
    {
    }

    /// <summary>
    /// Creates a SimplePromptTemplate from a template string.
    /// </summary>
    /// <param name="template">The template string.</param>
    /// <returns>A new SimplePromptTemplate instance.</returns>
    public static SimplePromptTemplate FromTemplate(string template)
    {
        return new SimplePromptTemplate(template);
    }
}
