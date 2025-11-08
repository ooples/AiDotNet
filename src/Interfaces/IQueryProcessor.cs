namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for processing and transforming user queries before retrieval.
/// </summary>
/// <remarks>
/// <para>
/// Query processors enhance retrieval quality by transforming the user's input query
/// in various ways such as expansion, reformulation, or keyword extraction.
/// </para>
/// <para><b>For Beginners:</b> Query processors improve search results by refining your question.
/// 
/// Think of it like asking a librarian for help:
/// - You say: "cars"
/// - Librarian suggests: "Did you mean automobiles, vehicles, or transportation?"
/// - You get better results with the expanded search
/// 
/// Common query transformations:
/// - Expansion: Add related terms ("solar" → "solar, photovoltaic, renewable energy")
/// - Reformulation: Rephrase for clarity ("how r cars made" → "how are cars manufactured")
/// - Keyword extraction: Focus on important terms ("What is the capital of France?" → "capital France")
/// </para>
/// </remarks>
public interface IQueryProcessor
{
    /// <summary>
    /// Processes and transforms the input query.
    /// </summary>
    /// <param name="query">The original user query.</param>
    /// <returns>The processed query string.</returns>
    string ProcessQuery(string query);
}
