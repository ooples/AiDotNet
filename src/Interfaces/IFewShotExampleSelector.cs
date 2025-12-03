namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for selecting few-shot examples to include in prompts.
/// </summary>
/// <typeparam name="T">The type of numeric data used for similarity scoring.</typeparam>
/// <remarks>
/// <para>
/// A few-shot example selector chooses which examples to include in a prompt to guide the language model's behavior.
/// Different selection strategies (random, semantic similarity, diversity, MMR) optimize for different goals.
/// </para>
/// <para><b>For Beginners:</b> An example selector picks which examples to show the language model.
///
/// Think of it like a teacher choosing practice problems:
/// - You have 100 practice problems in the textbook
/// - You can only show 3-5 examples in class
/// - Which ones do you choose?
///
/// Different strategies:
/// - Random: Pick any 3 problems
/// - Similar to homework: Pick problems like tonight's homework
/// - Diverse: Pick problems covering different concepts
/// - Best of both: Pick relevant problems that are also diverse
///
/// The examples you choose significantly affect how well students (or the LLM) learn!
///
/// Example - Sentiment classification:
/// Available examples: 1,000 labeled movie reviews
/// Current query: "This movie was fantastic, loved every minute!"
///
/// Selector's job:
/// 1. Look at the query
/// 2. Choose 3-5 most helpful examples
/// 3. Return them to include in the prompt
///
/// Good examples → Better LLM performance
/// </para>
/// </remarks>
public interface IFewShotExampleSelector<T>
{
    /// <summary>
    /// Selects the most appropriate examples for the given query.
    /// </summary>
    /// <param name="query">The input query to select examples for.</param>
    /// <param name="count">The number of examples to select.</param>
    /// <returns>A list of selected examples.</returns>
    /// <remarks>
    /// <para>
    /// Chooses 'count' examples from the available pool that are most appropriate for the given query.
    /// The selection strategy depends on the implementation (random, semantic similarity, diversity, etc.).
    /// </para>
    /// <para><b>For Beginners:</b> This picks the best examples to show for a specific query.
    ///
    /// Example - Code generation:
    /// Query: "Write a function to sort a list of numbers"
    ///
    /// Available examples:
    /// 1. "Sort a list of strings" → def sort_strings(lst): return sorted(lst)
    /// 2. "Reverse a list" → def reverse(lst): return lst[::-1]
    /// 3. "Filter even numbers" → def evens(lst): return [x for x in lst if x % 2 == 0]
    /// 4. "Find max in list" → def find_max(lst): return max(lst)
    /// 5. "Sort numbers descending" → def sort_desc(lst): return sorted(lst, reverse=True)
    ///
    /// SelectExamples(query, count=2) might return:
    /// - Example 1: "Sort a list of strings" (very similar operation)
    /// - Example 5: "Sort numbers descending" (exact same task, different order)
    ///
    /// These are most relevant for teaching the model how to sort numbers.
    ///
    /// The selection adapts to each query:
    /// - Different query → Different examples selected
    /// - More relevant examples → Better results
    /// </para>
    /// </remarks>
    IReadOnlyList<FewShotExample> SelectExamples(string query, int count);

    /// <summary>
    /// Adds an example to the selector's pool of available examples.
    /// </summary>
    /// <param name="example">The example to add.</param>
    /// <remarks>
    /// <para>
    /// Adds a new example to the pool that the selector can choose from.
    /// Examples typically consist of an input and corresponding output.
    /// </para>
    /// <para><b>For Beginners:</b> This adds a new example to the pool of available examples.
    ///
    /// Example - Building a translation example pool:
    /// ```csharp
    /// var selector = new SemanticSimilaritySelector();
    ///
    /// // Add examples
    /// selector.AddExample(new FewShotExample
    /// {
    ///     Input = "Hello",
    ///     Output = "Hola"
    /// });
    ///
    /// selector.AddExample(new FewShotExample
    /// {
    ///     Input = "Goodbye",
    ///     Output = "Adiós"
    /// });
    ///
    /// // Now selector can choose from these examples
    /// ```
    ///
    /// Over time, you build up a comprehensive example library.
    /// </para>
    /// </remarks>
    void AddExample(FewShotExample example);

    /// <summary>
    /// Removes an example from the selector's pool.
    /// </summary>
    /// <param name="example">The example to remove.</param>
    /// <returns>True if the example was removed; false if it wasn't found.</returns>
    /// <remarks>
    /// <para>
    /// Removes an example from the pool. Useful for removing outdated or incorrect examples.
    /// </para>
    /// <para><b>For Beginners:</b> This removes an example from the pool.
    ///
    /// Use this when:
    /// - An example is incorrect
    /// - An example is outdated
    /// - You're refining your example set
    /// - An example causes problems
    ///
    /// Example:
    /// ```csharp
    /// // Remove a bad example
    /// var badExample = new FewShotExample
    /// {
    ///     Input = "Test",
    ///     Output = "Wrong answer"
    /// };
    ///
    /// bool removed = selector.RemoveExample(badExample);
    /// // removed = true if it was in the pool, false otherwise
    /// ```
    /// </para>
    /// </remarks>
    bool RemoveExample(FewShotExample example);

    /// <summary>
    /// Gets all examples currently in the selector's pool.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns all examples that the selector can choose from. Useful for inspection,
    /// debugging, and understanding the selector's behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This shows all available examples in the pool.
    ///
    /// Use this to:
    /// - See what examples are available
    /// - Debug selection issues
    /// - Export examples for documentation
    /// - Analyze example coverage
    ///
    /// Example:
    /// ```csharp
    /// var allExamples = selector.GetAllExamples();
    /// Console.WriteLine($"Pool has {allExamples.Count} examples");
    ///
    /// foreach (var ex in allExamples)
    /// {
    ///     Console.WriteLine($"Input: {ex.Input} → Output: {ex.Output}");
    /// }
    /// ```
    /// </para>
    /// </remarks>
    IReadOnlyList<FewShotExample> GetAllExamples();

    /// <summary>
    /// Gets the total number of examples in the pool.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns the count of examples available for selection.
    /// </para>
    /// <para><b>For Beginners:</b> How many examples are in the pool.
    ///
    /// Useful for:
    /// - Validation: Ensure you have enough examples
    /// - Logging: Track pool size over time
    /// - Debugging: Verify examples were added correctly
    ///
    /// Example:
    /// ```csharp
    /// if (selector.ExampleCount < 5)
    /// {
    ///     Console.WriteLine("Warning: Less than 5 examples available");
    /// }
    /// ```
    /// </para>
    /// </remarks>
    int ExampleCount { get; }
}

/// <summary>
/// Represents a single few-shot example with input and output.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A few-shot example is an input-output pair shown to the model.
///
/// Think of it like a flashcard:
/// - Front (Input): The question or task
/// - Back (Output): The correct answer or response
///
/// Example - Math tutoring:
/// Input: "What is 5 + 3?"
/// Output: "5 + 3 = 8"
///
/// Example - Code generation:
/// Input: "Write a function to add two numbers"
/// Output: "def add(a, b): return a + b"
///
/// Example - Translation:
/// Input: "Good morning"
/// Output: "Buenos días"
///
/// The model learns the pattern from these examples and applies it to new inputs.
/// </para>
/// </remarks>
public class FewShotExample
{
    /// <summary>
    /// Gets or sets the input part of the example.
    /// </summary>
    public string Input { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the output part of the example.
    /// </summary>
    public string Output { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets optional metadata about the example.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Metadata can include information like creation date, source, category,
    /// quality score, or any other relevant information.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
