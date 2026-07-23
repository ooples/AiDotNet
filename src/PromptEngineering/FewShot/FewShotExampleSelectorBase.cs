using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.FewShot;

/// <summary>
/// Base class for few-shot example selector implementations.
/// </summary>
/// <typeparam name="T">The type of numeric data used for similarity scoring.</typeparam>
/// <remarks>
/// <para>
/// This base class provides common functionality for example selectors including example storage
/// and basic validation. Derived classes implement the selection strategy.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for all example selectors.
///
/// It handles:
/// - Storing examples
/// - Adding/removing examples
/// - Basic validation
///
/// Derived classes just need to implement how to SELECT examples!
/// </para>
/// </remarks>
public abstract class FewShotExampleSelectorBase<T> : IFewShotExampleSelector<T>
{
    /// <summary>
    /// Numeric operations for type <typeparamref name="T"/>.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The pool of available examples.
    /// </summary>
    protected readonly List<FewShotExample> Examples;

    /// <summary>
    /// Initializes a new instance of the FewShotExampleSelectorBase class.
    /// </summary>
    protected FewShotExampleSelectorBase()
    {
        Examples = new List<FewShotExample>();
    }

    /// <summary>
    /// Selects the most appropriate examples for the given query.
    /// </summary>
    /// <param name="query">The input query to select examples for.</param>
    /// <param name="count">The number of examples to select.</param>
    /// <returns>A list of selected examples.</returns>
    public IReadOnlyList<FewShotExample> SelectExamples(string query, int count)
    {
        if (string.IsNullOrWhiteSpace(query))
        {
            throw new ArgumentException("Query cannot be empty.", nameof(query));
        }

        if (count <= 0)
        {
            throw new ArgumentException("Count must be positive.", nameof(count));
        }

        if (Examples.Count == 0)
        {
            return new List<FewShotExample>().AsReadOnly();
        }

        // Limit count to available examples
        var actualCount = Math.Min(count, Examples.Count);

        return SelectExamplesCore(query, actualCount);
    }

    /// <summary>
    /// Adds an example to the selector's pool.
    /// </summary>
    /// <param name="example">The example to add.</param>
    public void AddExample(FewShotExample example)
    {
        if (example == null)
        {
            throw new ArgumentNullException(nameof(example), "Example cannot be null.");
        }

        if (string.IsNullOrWhiteSpace(example.Input))
        {
            throw new ArgumentException("Example input cannot be empty.", nameof(example));
        }

        if (string.IsNullOrWhiteSpace(example.Output))
        {
            throw new ArgumentException("Example output cannot be empty.", nameof(example));
        }

        Examples.Add(example);
        OnExampleAdded(example);
    }

    /// <summary>
    /// Removes an example from the selector's pool.
    /// </summary>
    /// <param name="example">The example to remove.</param>
    /// <returns>True if the example was removed; false if it wasn't found.</returns>
    public bool RemoveExample(FewShotExample example)
    {
        if (example == null)
        {
            return false;
        }

        var removed = Examples.Remove(example);
        if (removed)
        {
            OnExampleRemoved(example);
        }

        return removed;
    }

    /// <summary>
    /// Gets all examples currently in the selector's pool.
    /// </summary>
    public IReadOnlyList<FewShotExample> GetAllExamples()
    {
        return Examples.AsReadOnly();
    }

    /// <summary>
    /// Gets the total number of examples in the pool.
    /// </summary>
    public int ExampleCount => Examples.Count;

    /// <summary>
    /// Core selection logic to be implemented by derived classes.
    /// </summary>
    /// <param name="query">The query to select examples for.</param>
    /// <param name="count">The number of examples to select (already validated).</param>
    /// <returns>Selected examples.</returns>
    protected abstract IReadOnlyList<FewShotExample> SelectExamplesCore(string query, int count);

    /// <summary>
    /// Clamps a value to the [0, 1] interval.
    /// </summary>
    protected static T ClampToUnitInterval(T value)
    {
        if (NumOps.LessThan(value, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        if (NumOps.GreaterThan(value, NumOps.One))
        {
            return NumOps.One;
        }

        return value;
    }

    /// <summary>
    /// Calculates cosine similarity between two vectors.
    /// </summary>
    protected static T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vectors must have the same length.");
        }

        T dotProduct = NumOps.Zero;
        T magnitudeA = NumOps.Zero;
        T magnitudeB = NumOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(a[i], b[i]));
            magnitudeA = NumOps.Add(magnitudeA, NumOps.Multiply(a[i], a[i]));
            magnitudeB = NumOps.Add(magnitudeB, NumOps.Multiply(b[i], b[i]));
        }

        var denom = NumOps.Multiply(NumOps.Sqrt(magnitudeA), NumOps.Sqrt(magnitudeB));
        return NumOps.GreaterThan(denom, NumOps.Zero) ? NumOps.Divide(dotProduct, denom) : NumOps.Zero;
    }

    /// <summary>
    /// Calculates squared Euclidean distance between two vectors.
    /// </summary>
    protected static T EuclideanDistanceSquared(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vectors must have the same length.");
        }

        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return sum;
    }

    /// <summary>
    /// Compares two values in descending order using <see cref="INumericOperations{T}"/> comparisons.
    /// </summary>
    protected static int CompareDescending(T left, T right)
    {
        if (NumOps.GreaterThan(left, right))
        {
            return -1;
        }

        if (NumOps.LessThan(left, right))
        {
            return 1;
        }

        return 0;
    }

    /// <summary>
    /// Called when an example is added. Override for custom behavior.
    /// </summary>
    /// <param name="example">The added example.</param>
    protected virtual void OnExampleAdded(FewShotExample example)
    {
        // Override in derived classes if needed
    }

    /// <summary>
    /// Called when an example is removed. Override for custom behavior.
    /// </summary>
    /// <param name="example">The removed example.</param>
    protected virtual void OnExampleRemoved(FewShotExample example)
    {
        // Override in derived classes if needed
    }
}
