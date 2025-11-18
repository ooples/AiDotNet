namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for prompt templates used in language model interactions.
/// </summary>
/// <remarks>
/// <para>
/// A prompt template provides a structured way to create prompts for language models by combining
/// a template string with runtime variables. Templates support variable substitution, formatting,
/// and composition of complex prompts from reusable components.
/// </para>
/// <para><b>For Beginners:</b> A prompt template is like a form with blanks to fill in.
///
/// Think of it like a mad lib:
/// - Template: "The {adjective} {noun} {verb} over the {place}."
/// - Variables: adjective="quick", noun="fox", verb="jumped", place="fence"
/// - Result: "The quick fox jumped over the fence."
///
/// In LLM applications:
/// - Template: "Translate the following {source_lang} to {target_lang}: {text}"
/// - Variables: source_lang="English", target_lang="French", text="Hello"
/// - Result: "Translate the following English to French: Hello"
///
/// Benefits of using templates:
/// - Reusability: Write the template once, use it many times
/// - Consistency: All prompts have the same structure
/// - Maintainability: Change the template in one place
/// - Safety: Validate inputs before insertion
/// - Clarity: Separate prompt logic from data
/// </para>
/// </remarks>
public interface IPromptTemplate
{
    /// <summary>
    /// Formats the template with the provided variables to create a complete prompt.
    /// </summary>
    /// <param name="variables">Dictionary of variable names and their values.</param>
    /// <returns>The formatted prompt string.</returns>
    /// <remarks>
    /// <para>
    /// This method takes a dictionary of variables and substitutes them into the template
    /// to produce the final prompt string. Variable names in the template are typically
    /// enclosed in curly braces like {variable_name}.
    /// </para>
    /// <para><b>For Beginners:</b> This fills in the blanks in your template.
    ///
    /// Example:
    /// Template: "Summarize this article about {topic} in {num_sentences} sentences: {article_text}"
    ///
    /// Variables:
    /// - topic: "climate change"
    /// - num_sentences: "3"
    /// - article_text: "Global temperatures are rising..."
    ///
    /// Result:
    /// "Summarize this article about climate change in 3 sentences: Global temperatures are rising..."
    ///
    /// Error handling:
    /// - If a required variable is missing, an exception is thrown
    /// - If extra variables are provided, they're typically ignored
    /// - Variable values are safely inserted (no injection attacks)
    /// </para>
    /// </remarks>
    string Format(Dictionary<string, string> variables);

    /// <summary>
    /// Gets the list of variable names that this template expects.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns the names of all variables that should be provided when formatting the template.
    /// This allows validation and introspection of template requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you what blanks need to be filled in.
    ///
    /// Example:
    /// Template: "Convert {amount} {from_currency} to {to_currency}"
    /// InputVariables: ["amount", "from_currency", "to_currency"]
    ///
    /// This is helpful for:
    /// - Validation: Check if you have all required variables before formatting
    /// - Documentation: See what inputs a template needs
    /// - UI generation: Automatically create input forms
    /// - Error prevention: Catch missing variables early
    /// </para>
    /// </remarks>
    IReadOnlyList<string> InputVariables { get; }

    /// <summary>
    /// Gets the raw template string before variable substitution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The template string defines the structure of the prompt with placeholders for variables.
    /// Placeholders are typically denoted with curly braces, e.g., {variable_name}.
    /// </para>
    /// <para><b>For Beginners:</b> This is the original template with the placeholders.
    ///
    /// Example:
    /// Template: "Explain {concept} to a {audience} in {style} language."
    ///
    /// You can inspect the template to:
    /// - See the prompt structure
    /// - Debug issues
    /// - Create variations
    /// - Document your system
    /// </para>
    /// </remarks>
    string Template { get; }

    /// <summary>
    /// Validates that the provided variables match the template's requirements.
    /// </summary>
    /// <param name="variables">Dictionary of variable names and their values.</param>
    /// <returns>True if all required variables are present and valid; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Checks whether the provided variables satisfy the template's requirements,
    /// including presence of all required variables and validity of their values.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if you have all the required pieces before formatting.
    ///
    /// Example:
    /// Template needs: ["question", "context"]
    ///
    /// Variables provided: {"question": "What is AI?"}
    /// Validate() → False (missing "context")
    ///
    /// Variables provided: {"question": "What is AI?", "context": "AI stands for..."}
    /// Validate() → True (all required variables present)
    ///
    /// This helps you catch errors before sending prompts to the model:
    /// - Prevents runtime errors
    /// - Gives clear error messages
    /// - Ensures prompt quality
    /// </para>
    /// </remarks>
    bool Validate(Dictionary<string, string> variables);
}
