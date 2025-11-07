namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for chat-based language models that can generate responses to prompts.
/// This interface abstracts the underlying implementation, allowing agents to work with different LLM providers.
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
/// Example usage:
/// <code>
/// IChatModel&lt;double&gt; model = new OpenAIChatModel&lt;double&gt;(apiKey);
/// string response = await model.GenerateResponseAsync("What is 2 + 2?");
/// // response might be: "2 + 2 equals 4."
/// </code>
/// </remarks>
public interface IChatModel<T>
{
    /// <summary>
    /// Generates a text response to the given prompt asynchronously.
    /// </summary>
    /// <param name="prompt">The input text prompt to send to the language model.
    /// This can be a question, instruction, or any text that requires a response.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains
    /// the model's generated response as a string.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method is the core of how we communicate with a language model. You give it a prompt
    /// (what you want to ask or tell the model), and it generates a response.
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
    /// - Handle API errors gracefully and return meaningful error messages
    /// - Consider implementing retry logic for transient failures
    /// - Respect rate limits and timeouts
    /// - Sanitize sensitive information from logs
    /// </remarks>
    Task<string> GenerateResponseAsync(string prompt);

    /// <summary>
    /// Gets the name or identifier of the chat model.
    /// </summary>
    /// <value>A string representing the model's name (e.g., "gpt-4", "claude-2", "local-llama-7b").</value>
    /// <remarks>
    /// For Beginners:
    /// This property helps identify which language model is being used. Different models have
    /// different capabilities, costs, and performance characteristics. Knowing which model
    /// you're using is important for debugging and optimization.
    /// </remarks>
    string ModelName { get; }
}
