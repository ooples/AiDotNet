namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for safety filtering mechanisms.
/// </summary>
/// <remarks>
/// <para>
/// These options control how inputs and outputs are validated and filtered to prevent
/// harmful or inappropriate content from passing through the AI system.
/// </para>
/// <para><b>For Beginners:</b> These settings control how strict your "security guards" are.
/// You can adjust sensitivity thresholds, what types of content to filter, and how thoroughly
/// to check for problems.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class SafetyFilterOptions<T>
{
    /// <summary>
    /// Gets or sets the safety threshold for content filtering.
    /// </summary>
    /// <value>The threshold (0-1), defaulting to 0.8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Content with safety scores below this threshold is flagged.
    /// Higher thresholds (closer to 1) are more strict and filter more content.</para>
    /// </remarks>
    public double SafetyThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the jailbreak detection sensitivity.
    /// </summary>
    /// <value>The sensitivity (0-1), defaulting to 0.7.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How sensitive the system is to jailbreak attempts.
    /// Higher values catch more jailbreaks but might have false positives.</para>
    /// </remarks>
    public double JailbreakSensitivity { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets whether to enable input validation.
    /// </summary>
    /// <value>True to validate inputs, false otherwise (default: true).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Input validation checks requests before processing them,
    /// catching malicious or malformed inputs early.</para>
    /// </remarks>
    public bool EnableInputValidation { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable output filtering.
    /// </summary>
    /// <value>True to filter outputs, false otherwise (default: true).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Output filtering checks responses before showing them to users,
    /// ensuring no harmful content gets through.</para>
    /// </remarks>
    public bool EnableOutputFiltering { get; set; } = true;

    /// <summary>
    /// Gets or sets the harmful content categories to check for.
    /// </summary>
    /// <value>Array of category names, defaulting to common harmful categories.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the types of harmful content to watch for,
    /// like violence, hate speech, adult content, etc.</para>
    /// </remarks>
    public string[] HarmfulContentCategories { get; set; } = new[]
    {
        "Violence",
        "HateSpeech",
        "AdultContent",
        "PrivateInformation",
        "Misinformation"
    };

    /// <summary>
    /// Gets or sets whether to use a classifier for harmful content detection.
    /// </summary>
    /// <value>True to use classifier, false for rule-based detection (default: true).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Classifiers use machine learning to detect harmful content
    /// and are generally more accurate than simple rules.</para>
    /// </remarks>
    public bool UseClassifier { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum input length to process.
    /// </summary>
    /// <value>The maximum length, defaulting to 10000.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Inputs longer than this are rejected to prevent abuse
    /// and ensure processing efficiency.</para>
    /// </remarks>
    public int MaxInputLength { get; set; } = 10000;

    /// <summary>
    /// Gets or sets whether to log filtered content for review.
    /// </summary>
    /// <value>True to log filtered content, false otherwise (default: true).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Logging helps you review what's being filtered and
    /// improve the filtering over time.</para>
    /// </remarks>
    public bool LogFilteredContent { get; set; } = true;

    /// <summary>
    /// Gets or sets the file path used when logging filtered content.
    /// </summary>
    /// <remarks>
    /// When null or empty, a default relative path is used.
    /// Prefer an absolute path in production deployments or integrate with a logging framework.
    /// </remarks>
    public string? LogFilePath { get; set; }
}
