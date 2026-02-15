namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the base contract for language models that can generate text responses.
/// This interface unifies both synchronous and asynchronous text generation capabilities.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// A language model is an AI that understands and generates human-like text. Think of it as
/// a very sophisticated autocomplete that can:
/// - Answer questions
/// - Write essays or code
/// - Translate languages
/// - Summarize documents
/// - Have conversations
///
/// This interface is the foundation for all language models in AiDotNet, whether they:
/// - Run in the cloud (OpenAI, Anthropic, Azure)
/// - Run locally on your machine (Ollama, ONNX)
/// - Are used for chat applications (IChatModel)
/// - Are used for RAG systems (IGenerator)
///
/// The interface provides both synchronous and asynchronous methods:
/// - Async methods (GenerateAsync): Better for web apps, don't block the UI
/// - Sync methods (Generate): Simpler for scripts and batch processing
///
/// Example usage:
/// <code>
/// ILanguageModel&lt;double&gt; model = new OpenAIChatModel&lt;double&gt;("your-api-key");
///
/// // Async usage (recommended for most applications)
/// string response = await model.GenerateAsync("Explain quantum computing");
///
/// // Sync usage (for simple scripts)
/// string response = model.Generate("Explain quantum computing");
/// </code>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("LanguageModel")]
public interface ILanguageModel<T>
{
    /// <summary>
    /// Gets the name or identifier of the language model.
    /// </summary>
    /// <value>A string representing the model's name (e.g., "gpt-4", "claude-3-opus", "llama-2-7b").</value>
    /// <remarks>
    /// For Beginners:
    /// This identifies which specific model you're using. Different models have different:
    /// - Capabilities (some are better at code, others at creative writing)
    /// - Costs (GPT-4 is more expensive than GPT-3.5)
    /// - Speed (smaller models are faster)
    /// - Context windows (how much text they can process at once)
    ///
    /// Examples:
    /// - "gpt-4" or "gpt-3.5-turbo" (OpenAI)
    /// - "claude-3-opus-20240229" or "claude-3-sonnet-20240229" (Anthropic)
    /// - "llama-2-7b" or "mixtral-8x7b" (Open source models)
    /// </remarks>
    string ModelName { get; }

    /// <summary>
    /// Gets the maximum number of tokens this model can process in a single request (context window).
    /// </summary>
    /// <value>The context window size in tokens.</value>
    /// <remarks>
    /// For Beginners:
    /// This is how much text the model can "remember" or process at once. Think of it as
    /// the model's working memory.
    ///
    /// Token counts (approximate):
    /// - 1 token ≈ 0.75 words
    /// - 100 tokens ≈ 75 words ≈ 1 paragraph
    /// - 1000 tokens ≈ 750 words ≈ 1 page
    /// - 8000 tokens ≈ 6000 words ≈ 8-10 pages
    ///
    /// Common context window sizes:
    /// - GPT-3.5-turbo: 4,096 tokens (3 pages)
    /// - GPT-4: 8,192 tokens (6 pages) or 32,768 tokens (24 pages)
    /// - Claude 3: 200,000 tokens (150 pages)
    /// - Gemini 1.5: 1,000,000 tokens (750 pages!)
    ///
    /// Why it matters:
    /// - If your prompt + desired response exceeds this, the request will fail
    /// - Larger contexts let you provide more information but may be slower/costlier
    /// </remarks>
    int MaxContextTokens { get; }

    /// <summary>
    /// Gets the maximum number of tokens this model can generate in a single response.
    /// </summary>
    /// <value>The maximum generation length in tokens.</value>
    /// <remarks>
    /// For Beginners:
    /// This limits how long the model's response can be. It's usually smaller than
    /// MaxContextTokens because you need room for your input prompt too.
    ///
    /// Typical values:
    /// - 512 tokens ≈ 384 words ≈ short answer (1-2 paragraphs)
    /// - 2048 tokens ≈ 1536 words ≈ medium answer (1 page)
    /// - 4096 tokens ≈ 3072 words ≈ long answer (2-3 pages)
    ///
    /// You can usually configure this when creating the model to balance:
    /// - Shorter responses: Faster and cheaper
    /// - Longer responses: More detailed but slower and more expensive
    /// </remarks>
    int MaxGenerationTokens { get; }

    /// <summary>
    /// Generates a text response to the given prompt asynchronously.
    /// </summary>
    /// <param name="prompt">The input text prompt to send to the language model.
    /// This can be a question, instruction, or any text that requires a response.</param>
    /// <param name="cancellationToken">Optional cancellation token to cancel the generation operation.
    /// Use this to implement timeouts or allow users to cancel long-running requests.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains
    /// the model's generated response as a string.</returns>
    /// <remarks>
    /// For Beginners:
    /// This is the main method for getting responses from the language model. It's asynchronous
    /// (uses async/await) which means your application won't freeze while waiting for the response.
    ///
    /// The method is asynchronous because:
    /// - API calls to cloud models can take 1-10 seconds
    /// - Local models might need time to process
    /// - Your UI stays responsive while waiting
    ///
    /// Example:
    /// <code>
    /// string prompt = "Write a haiku about programming";
    /// string response = await model.GenerateAsync(prompt);
    /// Console.WriteLine(response);
    /// // Output: "Code flows like water
    /// //          Bugs emerge then disappear
    /// //          Peace in the logic"
    /// </code>
    ///
    /// Best practices:
    /// - Always use try-catch to handle errors (API failures, rate limits, etc.)
    /// - Consider retry logic for transient failures
    /// - Monitor token usage to control costs
    /// - Use CancellationToken to implement timeouts and allow cancellation
    ///   (e.g., var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30)))
    /// </remarks>
    Task<string> GenerateAsync(string prompt, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates a text response to the given prompt synchronously.
    /// </summary>
    /// <param name="prompt">The input text prompt to send to the language model.</param>
    /// <returns>The model's generated response as a string.</returns>
    /// <remarks>
    /// For Beginners:
    /// This is a synchronous version of GenerateAsync - it blocks until the response is ready.
    ///
    /// When to use this:
    /// - Simple command-line scripts
    /// - Batch processing where you process one request at a time
    /// - When you can't use async/await for some reason
    ///
    /// When NOT to use this:
    /// - Web applications (will block request threads)
    /// - UI applications (will freeze the interface)
    /// - When processing multiple requests (use GenerateAsync and Task.WhenAll)
    ///
    /// Example:
    /// <code>
    /// // Simple script usage
    /// string response = model.Generate("What is 2 + 2?");
    /// Console.WriteLine(response); // "2 + 2 equals 4."
    /// </code>
    ///
    /// Note: Many implementations just call GenerateAsync().GetAwaiter().GetResult()
    /// internally, so the async version is usually the "real" implementation.
    /// </remarks>
    string Generate(string prompt);
}
