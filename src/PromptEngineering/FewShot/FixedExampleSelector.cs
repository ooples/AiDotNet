using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.FewShot;

/// <summary>
/// Selects examples in a fixed, predetermined order.
/// </summary>
/// <typeparam name="T">The type of numeric data used for scoring.</typeparam>
/// <remarks>
/// <para>
/// This selector returns examples in the order they were added, always selecting the same examples
/// regardless of the query. Useful when you have carefully curated examples in a specific order.
/// </para>
/// <para><b>For Beginners:</b> Always returns the same examples in the same order.
///
/// Example:
/// ```csharp
/// var selector = new FixedExampleSelector<double>();
///
/// // Add examples in desired order
/// selector.AddExample(new FewShotExample { Input = "Easy example", Output = "Result 1" });
/// selector.AddExample(new FewShotExample { Input = "Medium example", Output = "Result 2" });
/// selector.AddExample(new FewShotExample { Input = "Hard example", Output = "Result 3" });
///
/// // Always returns first 2 examples
/// var examples = selector.SelectExamples("Any query", 2);
/// // Always returns: ["Easy example" → "Result 1", "Medium example" → "Result 2"]
/// ```
///
/// Use this when:
/// - You've carefully curated the best examples
/// - Order matters (simple to complex progression)
/// - Consistency is important
/// - All queries are similar enough
/// </para>
/// </remarks>
public class FixedExampleSelector<T> : FewShotExampleSelectorBase<T>
{
    /// <summary>
    /// Initializes a new instance of the FixedExampleSelector class.
    /// </summary>
    public FixedExampleSelector()
    {
    }

    /// <summary>
    /// Selects the first N examples in order.
    /// </summary>
    protected override IReadOnlyList<FewShotExample> SelectExamplesCore(string query, int count)
    {
        return Examples.Take(count).ToList().AsReadOnly();
    }
}
