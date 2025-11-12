namespace AiDotNet.Reasoning.Models;

/// <summary>
/// Represents a single step in a reasoning chain, capturing the thought process and evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring and calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Think of a reasoning step like a single line in showing your work on a math problem.
/// When you solve "What is 15% of 240?", your steps might be:
/// - Step 1: "Convert 15% to decimal: 15/100 = 0.15"
/// - Step 2: "Multiply by 240: 0.15 × 240 = 36"
/// - Step 3: "Therefore, the answer is 36"
///
/// Each ReasoningStep captures:
/// - What you thought (the reasoning text)
/// - How confident you are (the score)
/// - Whether this step was verified/checked
/// - Any feedback or corrections made
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// var step1 = new ReasoningStep&lt;double&gt;
/// {
///     StepNumber = 1,
///     Content = "Convert 15% to decimal: 15/100 = 0.15",
///     Score = 1.0,  // High confidence - this is a straightforward conversion
///     IsVerified = true
/// };
///
/// var step2 = new ReasoningStep&lt;double&gt;
/// {
///     StepNumber = 2,
///     Content = "Multiply by 240: 0.15 × 240 = 36",
///     Score = 0.95,
///     IsVerified = true,
///     ExternalVerificationResult = "Calculator confirms: 0.15 * 240 = 36"
/// };
/// </code>
/// </para>
/// </remarks>
public class ReasoningStep<T>
{
    /// <summary>
    /// The sequential number of this step in the reasoning chain (starting from 1).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like the step number when showing your work:
    /// Step 1, Step 2, Step 3, etc. It helps keep track of the order of reasoning.
    /// </para>
    /// </remarks>
    public int StepNumber { get; set; }

    /// <summary>
    /// The actual reasoning content or thought for this step.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is what you're actually thinking or doing in this step.
    /// It should be clear and specific, like "Calculate the area by multiplying length × width"
    /// rather than vague like "Do the math."
    /// </para>
    /// </remarks>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Original content before any refinement (if the step was refined).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you made a mistake and corrected it, this stores your original
    /// wrong answer so you can see what was changed. It's like keeping track of your eraser marks.
    /// </para>
    /// </remarks>
    public string? OriginalContent { get; set; }

    /// <summary>
    /// Confidence or quality score for this step (typically 0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how confident the AI is that this step is correct.
    /// Think of it as a percentage:
    /// - 1.0 = 100% confident (absolutely certain)
    /// - 0.8 = 80% confident (pretty sure)
    /// - 0.5 = 50% confident (unsure, might be wrong)
    /// - 0.0 = 0% confident (probably incorrect)
    ///
    /// Higher scores mean the step is more trustworthy.
    /// </para>
    /// </remarks>
    public T Score { get; set; } = default!;

    /// <summary>
    /// Whether this step has been verified by a critic model or external tool.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like having someone check your work. If true, this step
    /// has been reviewed and confirmed to be correct. If false, it hasn't been checked yet.
    /// </para>
    /// </remarks>
    public bool IsVerified { get; set; }

    /// <summary>
    /// Feedback from critic model if verification was performed.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like comments from a teacher reviewing your work.
    /// It might say things like:
    /// - "Good! The logic is sound and well-explained."
    /// - "This step needs more justification."
    /// - "Error: 15% should be 0.15, not 1.5"
    /// </para>
    /// </remarks>
    public string? CriticFeedback { get; set; }

    /// <summary>
    /// Number of times this step was refined/corrected.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This counts how many times you had to revise this step to get it right.
    /// It's like counting how many erasers you used on one problem:
    /// - 0 = Got it right the first time
    /// - 1 = Had to fix it once
    /// - 3 = Took three attempts to get it right
    /// </para>
    /// </remarks>
    public int RefinementCount { get; set; }

    /// <summary>
    /// Tool or method used to verify this step externally (e.g., "Calculator", "CodeExecution").
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If external tools were used to check this step, this records which tool.
    /// For example:
    /// - "Calculator" - Mathematical calculation was verified
    /// - "PythonInterpreter" - Code execution was tested
    /// - "WolframAlpha" - Complex math was double-checked
    /// </para>
    /// </remarks>
    public string? VerificationMethod { get; set; }

    /// <summary>
    /// Result from external verification tool if applicable.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This stores the output from external verification tools.
    /// For example, if a calculator was used, this might contain "36" or "Calculation correct: 36".
    /// </para>
    /// </remarks>
    public string? ExternalVerificationResult { get; set; }

    /// <summary>
    /// Timestamp when this step was created.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Records when this reasoning step happened. Useful for:
    /// - Performance analysis (how long did reasoning take?)
    /// - Debugging (what order did things happen?)
    /// - Audit trails (when was this decision made?)
    /// </para>
    /// </remarks>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Additional metadata or context specific to this step.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a flexible container for any extra information about this step
    /// that doesn't fit in the other properties. You can store things like:
    /// - Which sub-problem this step addresses
    /// - References to external sources
    /// - Alternative approaches considered
    /// </para>
    /// </remarks>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Returns a string representation of this reasoning step.
    /// </summary>
    /// <returns>A formatted string showing the step number and content.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a human-readable version of the step,
    /// useful for logging or debugging.
    /// </para>
    /// </remarks>
    public override string ToString()
    {
        return $"Step {StepNumber}: {Content}";
    }
}
