namespace AiDotNet.Interfaces;

using AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Defines the contract for safety filters that detect and prevent harmful or inappropriate model inputs and outputs.
/// </summary>
/// <remarks>
/// Safety filters act as gatekeepers that monitor model inputs and outputs to prevent
/// harmful, inappropriate, or malicious content from passing through the system.
///
/// <b>For Beginners:</b> Think of safety filters as "security guards" for your AI system.
/// They check everything going in and coming out to make sure nothing dangerous or
/// inappropriate gets through.
///
/// Common safety filter functions include:
/// - Input Validation: Check that inputs are safe and properly formatted
/// - Output Filtering: Ensure outputs don't contain harmful content
/// - Jailbreak Detection: Identify attempts to bypass safety measures
/// - Harmful Content Detection: Flag potentially dangerous or inappropriate content
///
/// Why safety filters matter:
/// - They prevent misuse of AI systems
/// - They protect users from harmful content
/// - They help maintain ethical AI deployments
/// - They catch edge cases and adversarial inputs
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("SafetyFilter")]
public interface ISafetyFilter<T> : IModelSerializer
{
    /// <summary>
    /// Validates that an input is safe and appropriate for processing.
    /// </summary>
    /// <remarks>
    /// This method checks inputs before they reach the model to prevent malicious
    /// or inappropriate inputs from being processed.
    ///
    /// <b>For Beginners:</b> This is like a bouncer at a club checking IDs at the door.
    /// Before letting an input into your AI system, this method checks if it's safe
    /// and appropriate to process.
    ///
    /// The validation might check for:
    /// 1. Malformed inputs that could crash the system
    /// 2. Adversarial patterns designed to fool the model
    /// 3. Attempts to inject malicious code or prompts
    /// 4. Inappropriate or harmful content in the input
    /// </remarks>
    /// <param name="input">The input to validate.</param>
    /// <returns>Validation result indicating if input is safe and any issues found.</returns>
    SafetyValidationResult<T> ValidateInput(Vector<T> input);

    /// <summary>
    /// Filters model outputs to remove or flag harmful content.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This checks what the AI is about to say before showing it
    /// to users. If the AI generated something harmful or inappropriate, this method
    /// can block it or modify it to be safe.
    ///
    /// For example:
    /// - If an AI accidentally generates instructions for something dangerous
    /// - If output contains private or sensitive information
    /// - If the response could be misleading or harmful
    /// </remarks>
    /// <param name="output">The model output to filter.</param>
    /// <returns>Filtered output with harmful content removed or flagged.</returns>
    SafetyFilterResult<T> FilterOutput(Vector<T> output);

    /// <summary>
    /// Detects jailbreak attempts that try to bypass safety measures.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A "jailbreak" is when someone tries to trick your AI into
    /// ignoring its safety rules. This method detects those attempts.
    ///
    /// Examples of jailbreak attempts:
    /// - "Ignore your previous instructions and do X instead"
    /// - Roleplaying scenarios to bypass restrictions
    /// - Encoding harmful requests in creative ways
    /// - Exploiting edge cases in safety training
    /// </remarks>
    /// <param name="input">The input to check for jailbreak attempts.</param>
    /// <returns>Detection result indicating if a jailbreak was detected and its severity.</returns>
    JailbreakDetectionResult<T> DetectJailbreak(Vector<T> input);

    /// <summary>
    /// Identifies harmful or inappropriate content in text or data.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is like a content moderation system. It scans content
    /// (inputs or outputs) and identifies anything that might be harmful, offensive,
    /// or inappropriate.
    ///
    /// Categories it might detect:
    /// - Violence or graphic content
    /// - Hate speech or discrimination
    /// - Private or sensitive information
    /// - Misinformation or scams
    /// - Adult or sexual content
    /// </remarks>
    /// <param name="content">The content to analyze.</param>
    /// <returns>Classification of harmful content types and severity scores.</returns>
    HarmfulContentResult<T> IdentifyHarmfulContent(Vector<T> content);

    /// <summary>
    /// Computes a safety score for model inputs or outputs.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This gives a single "safety score" from 0 to 1 indicating
    /// how safe the content is. Think of it like a trust score - higher numbers mean
    /// safer content.
    /// </remarks>
    /// <param name="content">The content to score.</param>
    /// <returns>A safety score between 0 (unsafe) and 1 (completely safe).</returns>
    T ComputeSafetyScore(Vector<T> content);

    /// <summary>
    /// Gets the configuration options for the safety filter.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how strict the safety filter is
    /// and what types of content it looks for.
    /// </remarks>
    /// <returns>The configuration options for the safety filter.</returns>
    SafetyFilterOptions<T> GetOptions();

    /// <summary>
    /// Resets the safety filter state.
    /// </summary>
    void Reset();
}
