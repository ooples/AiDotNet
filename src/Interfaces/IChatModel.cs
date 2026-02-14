namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for chat-based language models that can generate responses to prompts.
/// This interface abstracts the underlying implementation, allowing agents to work with different LLM providers.
/// Extends ILanguageModel to provide unified language model capabilities across the AiDotNet ecosystem.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// A chat model is like having a conversation partner that can understand and respond to text.
/// Think of it as the "brain" of an AI agent - it reads prompts (questions or instructions) and
/// generates intelligent responses.
///
/// This interface allows the AiDotNet library to work with different language model providers
/// (like OpenAI's GPT, Anthropic's Claude, or local models) without changing the agent code.
/// The generic type parameter T allows the model to work with different numeric precision levels.
///
/// This interface extends ILanguageModel, which means it inherits:
/// - GenerateAsync() - Primary async method for text generation
/// - Generate() - Sync version for simple scripts
/// - ModelName - Model identifier
/// - MaxContextTokens - Context window size
/// - MaxGenerationTokens - Maximum response length
///
/// Example usage:
/// <code>
/// IChatModel&lt;double&gt; model = new OpenAIChatModel&lt;double&gt;(apiKey);
///
/// // Using the base ILanguageModel method
/// string response1 = await model.GenerateAsync("What is 2 + 2?");
///
/// // Using the IChatModel alias method (same as GenerateAsync)
/// string response2 = await model.GenerateResponseAsync("What is 2 + 2?");
///
/// // Both produce: "2 + 2 equals 4."
/// </code>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ChatModel")]
public interface IChatModel<T> : ILanguageModel<T>
{
    /// <summary>
    /// Generates a text response to the given prompt asynchronously.
    /// This is an alias for GenerateAsync() provided for backward compatibility and clarity in chat contexts.
    /// </summary>
    /// <param name="prompt">The input text prompt to send to the language model.
    /// This can be a question, instruction, or any text that requires a response.</param>
    /// <param name="cancellationToken">Optional cancellation token to cancel the operation.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains
    /// the model's generated response as a string.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method is the core of how we communicate with a language model. You give it a prompt
    /// (what you want to ask or tell the model), and it generates a response.
    ///
    /// This method is identical to GenerateAsync() from ILanguageModel - it's provided as an
    /// alias because "GenerateResponseAsync" is clearer in chat/agent contexts.
    ///
    /// The method is asynchronous (uses async/await) because communicating with language models
    /// can take time, especially when calling external APIs. This prevents your application from
    /// freezing while waiting for a response.
    ///
    /// Example:
    /// <code>
    /// string prompt = "Explain what a neural network is in simple terms.";
    /// string response = await chatModel.GenerateResponseAsync(prompt);
    /// Console.WriteLine(response);
    /// </code>
    ///
    /// Implementation notes for developers:
    /// - Most implementations just call GenerateAsync() internally
    /// - Handle API errors gracefully and return meaningful error messages
    /// - Consider implementing retry logic for transient failures
    /// - Respect rate limits and timeouts
    /// - Sanitize sensitive information from logs
    /// </remarks>
    Task<string> GenerateResponseAsync(string prompt, CancellationToken cancellationToken = default);
}
