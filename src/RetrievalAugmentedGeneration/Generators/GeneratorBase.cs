using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Generators;

/// <summary>
/// Base class for generator implementations providing common functionality and validation.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// This base class provides standard validation, prompt construction, and citation handling
/// for generator implementations. It defines the template for implementing custom generation logic.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for all text generators in RAG systems.
/// 
/// It handles common tasks so you don't have to repeat them:
/// - Checking that inputs aren't null or empty
/// - Building prompts that combine the query and retrieved documents
/// - Extracting citations from generated text
/// - Creating properly formatted answers
/// 
/// When you create a new generator (like OpenAIGenerator or OnnxGenerator):
/// 1. Inherit from this class
/// 2. Set MaxContextTokens and MaxGenerationTokens in the constructor
/// 3. Implement GenerateCore with your specific generation logic
/// 4. Everything else (validation, prompt formatting, citations) is handled automatically
/// </para>
/// </remarks>
public abstract class GeneratorBase<T> : IGenerator<T>
{

    /// <summary>
    /// Gets the maximum number of tokens this generator can process in a single request.
    /// </summary>
    public int MaxContextTokens { get; protected set; }

    /// <summary>
    /// Gets the maximum number of tokens this generator can generate in a response.
    /// </summary>
    public int MaxGenerationTokens { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the GeneratorBase class.
    /// </summary>
    /// <param name="maxContextTokens">The maximum context window size in tokens.</param>
    /// <param name="maxGenerationTokens">The maximum number of tokens to generate.</param>
    protected GeneratorBase(int maxContextTokens, int maxGenerationTokens)
    {
        if (maxContextTokens <= 0)
        {
            throw new ArgumentException("Maximum context tokens must be positive.", nameof(maxContextTokens));
        }

        if (maxGenerationTokens <= 0)
        {
            throw new ArgumentException("Maximum generation tokens must be positive.", nameof(maxGenerationTokens));
        }

        MaxContextTokens = maxContextTokens;
        MaxGenerationTokens = maxGenerationTokens;
    }

    /// <summary>
    /// Generates a text response based on a prompt with validation.
    /// </summary>
    /// <param name="prompt">The input prompt or question.</param>
    /// <returns>The generated text response.</returns>
    /// <exception cref="ArgumentNullException">Thrown when prompt is null.</exception>
    /// <exception cref="ArgumentException">Thrown when prompt is empty or whitespace.</exception>
    public string Generate(string prompt)
    {
        if (prompt == null)
        {
            throw new ArgumentNullException(nameof(prompt), "Prompt cannot be null.");
        }

        if (string.IsNullOrWhiteSpace(prompt))
        {
            throw new ArgumentException("Prompt cannot be empty or whitespace.", nameof(prompt));
        }

        return GenerateCore(prompt);
    }

    /// <summary>
    /// Generates a grounded answer using provided context documents.
    /// </summary>
    /// <param name="query">The user's original query or question.</param>
    /// <param name="context">The retrieved documents providing context for the answer.</param>
    /// <returns>A grounded answer with the generated text, source documents, and extracted citations.</returns>
    /// <exception cref="ArgumentNullException">Thrown when query or context is null.</exception>
    /// <exception cref="ArgumentException">Thrown when query is empty or context has no documents.</exception>
    public GroundedAnswer<T> GenerateGrounded(string query, IEnumerable<Document<T>> context)
    {
        if (query == null)
        {
            throw new ArgumentNullException(nameof(query), "Query cannot be null.");
        }

        if (string.IsNullOrWhiteSpace(query))
        {
            throw new ArgumentException("Query cannot be empty or whitespace.", nameof(query));
        }

        if (context == null)
        {
            throw new ArgumentNullException(nameof(context), "Context cannot be null.");
        }

        var contextList = context.ToList();
        if (contextList.Count == 0)
        {
            throw new ArgumentException("Context must contain at least one document.", nameof(context));
        }

        // Build the prompt with context
        var prompt = BuildPromptWithContext(query, contextList);

        // Generate the answer
        var generatedText = GenerateCore(prompt);

        // Extract citations from the generated text
        var citations = ExtractCitations(generatedText, contextList);

        return new GroundedAnswer<T>
        {
            Answer = generatedText,
            SourceDocuments = contextList.AsReadOnly(),
            Citations = citations.Values.Select(d => d.Id).ToList().AsReadOnly()
        };
    }

    /// <summary>
    /// Core generation logic to be implemented by derived classes.
    /// </summary>
    /// <param name="prompt">The validated prompt string.</param>
    /// <returns>The generated text response.</returns>
    protected abstract string GenerateCore(string prompt);

    /// <summary>
    /// Builds a prompt that incorporates the query and retrieved context documents.
    /// </summary>
    /// <param name="query">The user's query.</param>
    /// <param name="context">The retrieved context documents.</param>
    /// <returns>A formatted prompt string.</returns>
    /// <remarks>
    /// <para>
    /// This method can be overridden to customize prompt formatting. The default
    /// implementation creates a structured prompt with numbered context documents
    /// followed by the user's question.
    /// </para>
    /// </remarks>
    protected virtual string BuildPromptWithContext(string query, List<Document<T>> context)
    {
        var promptBuilder = new System.Text.StringBuilder();
        promptBuilder.AppendLine("Answer the following question based on the provided context. Include citations using [1], [2], etc. to reference the source documents.");
        promptBuilder.AppendLine();
        promptBuilder.AppendLine("Context:");

        for (int i = 0; i < context.Count; i++)
        {
            promptBuilder.AppendLine($"[{i + 1}] {context[i].Content}");
            promptBuilder.AppendLine();
        }

        promptBuilder.AppendLine("Question:");
        promptBuilder.AppendLine(query);
        promptBuilder.AppendLine();
        promptBuilder.Append("Answer:");

        return promptBuilder.ToString();
    }

    /// <summary>
    /// Extracts citation markers from the generated text and maps them to source documents.
    /// </summary>
    /// <param name="generatedText">The generated text containing citations.</param>
    /// <param name="sourceDocuments">The source documents that were used for generation.</param>
    /// <returns>A dictionary mapping citation indices to documents.</returns>
    /// <remarks>
    /// <para>
    /// This method can be overridden to customize citation extraction logic. The default
    /// implementation looks for patterns like [1], [2], etc. in the generated text.
    /// </para>
    /// </remarks>
    protected virtual Dictionary<int, Document<T>> ExtractCitations(string generatedText, List<Document<T>> sourceDocuments)
    {
        var citations = new Dictionary<int, Document<T>>();
        var citationPattern = RegexHelper.Create(@"\[(\d+)\]");
        var matches = citationPattern.Matches(generatedText);

        foreach (System.Text.RegularExpressions.Match match in matches.Cast<System.Text.RegularExpressions.Match>())
        {
            if (int.TryParse(match.Groups[1].Value, out int citationIndex))
            {
                var docIndex = citationIndex - 1; // Convert to 0-based index
                if (docIndex >= 0 && docIndex < sourceDocuments.Count)
                {
                    citations[citationIndex] = sourceDocuments[docIndex];
                }
            }
        }

        return citations;
    }
}



