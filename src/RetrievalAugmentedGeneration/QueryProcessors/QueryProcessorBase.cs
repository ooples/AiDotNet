namespace AiDotNet.RetrievalAugmentedGeneration.QueryProcessors;

/// <summary>
/// Base class for query processor implementations with common validation logic.
/// </summary>
/// <remarks>
/// <para>
/// This base class provides standard validation for query processors and defines
/// the template for implementing custom query processing logic.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for all query processors.
/// 
/// It handles the boring stuff so you can focus on the interesting parts:
/// - Checking that the query isn't empty or null
/// - Providing a clean structure for your processing logic
/// - Ensuring consistent error handling
/// 
/// When you create a new query processor, you just need to:
/// 1. Inherit from this class
/// 2. Implement ProcessQueryCore with your custom logic
/// 3. Everything else is handled for you
/// </para>
/// </remarks>
public abstract class QueryProcessorBase : IQueryProcessor
{
    /// <summary>
    /// Processes the query with validation.
    /// </summary>
    /// <param name="query">The original user query.</param>
    /// <returns>The processed query string.</returns>
    /// <exception cref="ArgumentNullException">Thrown when query is null.</exception>
    /// <exception cref="ArgumentException">Thrown when query is empty or whitespace.</exception>
    public string ProcessQuery(string query)
    {
        if (query == null)
        {
            throw new ArgumentNullException(nameof(query), "Query cannot be null.");
        }

        if (string.IsNullOrWhiteSpace(query))
        {
            throw new ArgumentException("Query cannot be empty or whitespace.", nameof(query));
        }

        return ProcessQueryCore(query);
    }

    /// <summary>
    /// Core query processing logic to be implemented by derived classes.
    /// </summary>
    /// <param name="query">The validated query string.</param>
    /// <returns>The processed query string.</returns>
    protected abstract string ProcessQueryCore(string query);
}
