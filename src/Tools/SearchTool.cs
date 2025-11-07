using AiDotNet.Interfaces;

namespace AiDotNet.Tools;

/// <summary>
/// A mock tool that simulates searching for information.
/// This is a demonstration/testing tool that returns predefined responses rather than performing real searches.
/// </summary>
/// <remarks>
/// For Beginners:
/// The SearchTool is a simulated search engine. Unlike a real search tool that would query Google, Bing,
/// or a database, this mock tool returns predefined answers for demonstration and testing purposes.
///
/// Think of it like a practice version of a search engine - it has a small set of "canned" answers
/// for specific queries, which is perfect for:
/// - Testing agent behavior without needing internet access
/// - Demonstrating how agents use tools
/// - Unit testing where you need predictable, reproducible results
///
/// In a production system, you would replace this with a real search implementation that:
/// - Calls a real search API (Google Custom Search, Bing Search API, etc.)
/// - Queries a database
/// - Accesses a knowledge base or documentation system
///
/// Example usage:
/// <code>
/// var searchTool = new SearchTool();
/// string result1 = searchTool.Execute("capital of France");
/// // Returns: "The capital of France is Paris."
///
/// string result2 = searchTool.Execute("random topic");
/// // Returns: "Search results for 'random topic': [Mock search results - no specific information available]"
/// </code>
/// </remarks>
public class SearchTool : ITool
{
    private readonly Dictionary<string, string> _mockResults;

    /// <summary>
    /// Initializes a new instance of the <see cref="SearchTool"/> class with default mock data.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This constructor sets up the mock search tool with some predefined question-answer pairs.
    /// These are used to simulate search results without actually searching anything.
    /// </remarks>
    public SearchTool()
    {
        _mockResults = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["capital of France"] = "The capital of France is Paris.",
            ["capital of Japan"] = "The capital of Japan is Tokyo.",
            ["capital of USA"] = "The capital of the United States is Washington, D.C.",
            ["population of Earth"] = "The estimated population of Earth is approximately 8 billion people as of 2024.",
            ["speed of light"] = "The speed of light in vacuum is approximately 299,792,458 meters per second (or about 186,282 miles per second).",
            ["largest ocean"] = "The largest ocean on Earth is the Pacific Ocean, covering approximately 165 million square kilometers.",
            ["highest mountain"] = "The highest mountain on Earth is Mount Everest, with a peak at 8,849 meters (29,032 feet) above sea level.",
            ["Python programming language"] = "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            ["machine learning"] = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can analyze data and make predictions.",
            ["photosynthesis"] = "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy stored in glucose. It primarily occurs in the chloroplasts of plant cells using sunlight, water, and carbon dioxide."
        };
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SearchTool"/> class with custom mock data.
    /// </summary>
    /// <param name="mockResults">A dictionary of query-result pairs for the mock search.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor allows you to provide your own set of mock search results, which is useful
    /// for testing specific scenarios or customizing the tool for particular use cases.
    /// </remarks>
    public SearchTool(Dictionary<string, string> mockResults)
    {
        _mockResults = new Dictionary<string, string>(mockResults, StringComparer.OrdinalIgnoreCase);
    }

    /// <inheritdoc/>
    public string Name => "Search";

    /// <inheritdoc/>
    public string Description =>
        "Searches for information on a given topic. " +
        "Input should be a search query or question. " +
        "Returns relevant information about the query. " +
        "Example inputs: 'capital of France', 'what is machine learning', 'population of Earth'.";

    /// <inheritdoc/>
    public string Execute(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return "Error: Search query cannot be empty.";
        }

        string query = input.Trim();

        // Try to find an exact match first
        if (_mockResults.TryGetValue(query, out var exactResult))
        {
            return exactResult;
        }

        // Try to find a partial match (if the query contains any of our known topics)
        foreach (var kvp in _mockResults)
        {
            if (query.Contains(kvp.Key, StringComparison.OrdinalIgnoreCase))
            {
                return kvp.Value;
            }
        }

        // No match found - return a generic response
        return $"Search results for '{query}': [Mock search results - no specific information available for this query. " +
               "In a real implementation, this would search actual data sources.]";
    }

    /// <summary>
    /// Adds or updates a mock search result.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="result">The result to return for this query.</param>
    /// <remarks>
    /// For Beginners:
    /// This method allows you to add new mock data to the search tool or update existing entries.
    /// This is useful for testing different scenarios without creating a new SearchTool instance.
    ///
    /// Example:
    /// <code>
    /// var searchTool = new SearchTool();
    /// searchTool.AddMockResult("capital of Canada", "The capital of Canada is Ottawa.");
    /// string result = searchTool.Execute("capital of Canada");
    /// // Returns: "The capital of Canada is Ottawa."
    /// </code>
    /// </remarks>
    public void AddMockResult(string query, string result)
    {
        _mockResults[query] = result;
    }

    /// <summary>
    /// Removes a mock search result.
    /// </summary>
    /// <param name="query">The query to remove.</param>
    /// <returns>True if the query was found and removed; otherwise, false.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method removes a mock result from the tool. Useful for testing scenarios where
    /// certain information should not be available.
    /// </remarks>
    public bool RemoveMockResult(string query)
    {
        return _mockResults.Remove(query);
    }

    /// <summary>
    /// Clears all mock search results.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This method removes all mock data from the search tool, leaving it empty.
    /// Useful for testing error handling or resetting the tool to a clean state.
    /// </remarks>
    public void ClearMockResults()
    {
        _mockResults.Clear();
    }
}
