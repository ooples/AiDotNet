using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for domain-specific reasoning models that solve problems using
/// LLM-based reasoning strategies (chain-of-thought, tree search, consensus).
/// </summary>
/// <typeparam name="T">The numeric type used for scoring.</typeparam>
public interface IDomainReasoner<T>
{
    /// <summary>
    /// Solves a problem using the domain-specific reasoning strategy.
    /// </summary>
    /// <param name="problem">The problem or question to solve.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A reasoning result with the answer and reasoning steps.</returns>
    Task<ReasoningResult<T>> SolveAsync(string problem, CancellationToken cancellationToken = default);
}
