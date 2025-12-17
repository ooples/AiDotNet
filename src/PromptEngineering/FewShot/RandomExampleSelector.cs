using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PromptEngineering.FewShot;

/// <summary>
/// Selects examples randomly from the available pool.
/// </summary>
/// <typeparam name="T">The type of numeric data used for scoring.</typeparam>
/// <remarks>
/// <para>
/// This selector picks examples randomly without considering relevance to the query.
/// It's the simplest and fastest selection strategy.
/// </para>
/// <para><b>For Beginners:</b> Picks random examples, like drawing names from a hat.
///
/// Example:
/// ```csharp
/// var selector = new RandomExampleSelector<double>();
///
/// // Add examples
/// selector.AddExample(new FewShotExample { Input = "Hello", Output = "Hola" });
/// selector.AddExample(new FewShotExample { Input = "Goodbye", Output = "Adiós" });
/// selector.AddExample(new FewShotExample { Input = "Thank you", Output = "Gracias" });
/// selector.AddExample(new FewShotExample { Input = "Please", Output = "Por favor" });
///
/// // Select 2 random examples
/// var examples = selector.SelectExamples("Good morning", 2);
/// // Might return: ["Hello" → "Hola", "Please" → "Por favor"]
/// // Or any other random pair
/// ```
///
/// Use this when:
/// - All examples are roughly equally useful
/// - Simplicity and speed are important
/// - You don't have semantic similarity infrastructure
/// </para>
/// </remarks>
public class RandomExampleSelector<T> : FewShotExampleSelectorBase<T>
{
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the RandomExampleSelector class.
    /// </summary>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public RandomExampleSelector(int? seed = null)
    {
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Selects random examples from the pool.
    /// </summary>
    protected override IReadOnlyList<FewShotExample> SelectExamplesCore(string query, int count)
    {
        // Shuffle and take the first 'count' examples
        var shuffled = Examples.OrderBy(_ => _random.Next()).ToList();
        return shuffled.Take(count).ToList().AsReadOnly();
    }
}
