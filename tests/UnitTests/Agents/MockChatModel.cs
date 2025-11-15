using AiDotNet.Interfaces;

namespace AiDotNetTests.UnitTests.Agents;

/// <summary>
/// A mock implementation of IChatModel for testing purposes.
/// This allows tests to control and verify agent behavior without calling real language models.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class MockChatModel<T> : IChatModel<T>
{
    private readonly Queue<string> _responses;
    private readonly List<string> _receivedPrompts;

    /// <summary>
    /// Initializes a new instance of the <see cref="MockChatModel{T}"/> class.
    /// </summary>
    /// <param name="responses">Predefined responses to return in sequence.</param>
    public MockChatModel(params string[] responses)
    {
        _responses = new Queue<string>(responses);
        _receivedPrompts = new List<string>();
        ModelName = "MockModel";
    }

    /// <inheritdoc/>
    public string ModelName { get; set; }

    /// <inheritdoc/>
    public int MaxContextTokens { get; set; } = 4096;

    /// <inheritdoc/>
    public int MaxGenerationTokens { get; set; } = 2048;

    /// <summary>
    /// Gets the list of all prompts that were sent to this mock model.
    /// Useful for verifying that the agent is sending the correct prompts.
    /// </summary>
    public IReadOnlyList<string> ReceivedPrompts => _receivedPrompts.AsReadOnly();

    /// <inheritdoc/>
    public Task<string> GenerateAsync(string prompt)
    {
        _receivedPrompts.Add(prompt);

        if (_responses.Count == 0)
        {
            throw new InvalidOperationException(
                "Mock ChatModel ran out of predefined responses. " +
                "Make sure to provide enough responses for all expected calls.");
        }

        var response = _responses.Dequeue();
        return Task.FromResult(response);
    }

    /// <inheritdoc/>
    public string Generate(string prompt)
    {
        return GenerateAsync(prompt).GetAwaiter().GetResult();
    }

    /// <inheritdoc/>
    public Task<string> GenerateResponseAsync(string prompt, CancellationToken cancellationToken = default)
    {
        // Alias for GenerateAsync
        return GenerateAsync(prompt);
    }

    /// <summary>
    /// Adds a new response to the queue.
    /// </summary>
    /// <param name="response">The response to add.</param>
    public void AddResponse(string response)
    {
        _responses.Enqueue(response);
    }

    /// <summary>
    /// Gets the number of remaining predefined responses.
    /// </summary>
    public int RemainingResponseCount => _responses.Count;
}
