using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// A routing policy over candidate agents, trained by REINFORCE: it keeps a learnable preference weight per
/// agent (optionally per context), selects via a softmax over those weights, and updates them with a
/// policy-gradient step from observed rewards. Compared with the simpler mean-reward bandit
/// (<see cref="LearnedAgentRouter{T}"/>), this learns a stochastic policy that handles non-stationary rewards
/// and keeps exploring in proportion to its current confidence.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// The update for an observed (agent, reward) is the REINFORCE rule with a running-mean baseline:
/// <c>w[a'] += lr · (reward − baseline) · (1[a'=chosen] − π(a'))</c> for every candidate <c>a'</c>. As a
/// result, agents that earn above-baseline reward gain probability mass. As an <see cref="IAgent{T}"/> it
/// composes with tracing, closing the loop.
/// </para>
/// <para><b>For Beginners:</b> A dispatcher that keeps a "preference score" for each worker and nudges those
/// scores up or down based on how well the chosen worker did versus the recent average. Over time it sends
/// more work to the workers that do better, while still occasionally trying others.
/// </para>
/// </remarks>
public sealed class SoftmaxPolicyRouter<T> : IAgent<T>
{
    private readonly Dictionary<string, IAgent<T>> _candidates;
    private readonly Dictionary<string, Dictionary<string, double>> _weights = new(StringComparer.Ordinal);
    private readonly double _learningRate;
    private readonly Func<IReadOnlyList<ChatMessage>, string>? _contextKey;
    private readonly Random _random;
    // Serializes access to the mutable policy state (_weights, _baseline,
    // _baselineCount, _random): concurrent RunAsync/Update/LearnFrom calls
    // would otherwise race on the nested dictionaries, corrupt the running
    // baseline, and hit the (non-thread-safe) Random from multiple threads.
    private readonly object _gate = new();
    private double _baseline;
    private int _baselineCount;

    /// <summary>
    /// Initializes a new policy router.
    /// </summary>
    /// <param name="candidates">The candidate agents. Must be non-empty with unique names.</param>
    /// <param name="learningRate">The REINFORCE step size. Must be positive. Default 0.5.</param>
    /// <param name="contextKey">Optional function mapping a request to a context key (per-context policy). <c>null</c> learns globally.</param>
    /// <param name="seed">Optional RNG seed for reproducible sampling.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="candidates"/> (or any candidate) is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when candidates is empty or has duplicate names.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="learningRate"/> is not positive.</exception>
    public SoftmaxPolicyRouter(
        IReadOnlyList<IAgent<T>> candidates,
        double learningRate = 0.5,
        Func<IReadOnlyList<ChatMessage>, string>? contextKey = null,
        int? seed = null)
    {
        Guard.NotNull(candidates);
        if (candidates.Count == 0)
        {
            throw new ArgumentException("At least one candidate agent is required.", nameof(candidates));
        }

        if (learningRate <= 0 || double.IsNaN(learningRate))
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        _candidates = new Dictionary<string, IAgent<T>>(StringComparer.Ordinal);
        foreach (var candidate in candidates)
        {
            Guard.NotNull(candidate);
            if (_candidates.ContainsKey(candidate.Name))
            {
                throw new ArgumentException($"Duplicate candidate agent name '{candidate.Name}'.", nameof(candidates));
            }

            _candidates.Add(candidate.Name, candidate);
        }

        _learningRate = learningRate;
        _contextKey = contextKey;
        _random = seed is { } value
            ? RandomHelper.CreateSeededRandom(value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public string Name => "policy-router";

    /// <inheritdoc/>
    public string Description => "Routes to a candidate agent via a REINFORCE-trained softmax policy.";

    /// <summary>
    /// Applies a single REINFORCE update for an observed (agent, reward) in the given context.
    /// </summary>
    /// <param name="context">The request context (used for the context key), or <c>null</c>.</param>
    /// <param name="agentName">The agent that produced the reward; ignored when not a candidate.</param>
    /// <param name="reward">The observed reward.</param>
    public void Update(IReadOnlyList<ChatMessage>? context, string agentName, double reward)
    {
        Guard.NotNull(agentName);
        if (!_candidates.ContainsKey(agentName))
        {
            return;
        }

        // Compute the (potentially caller-supplied) context key outside the
        // lock so external code never runs while the gate is held.
        var key = ContextKey(context);
        lock (_gate)
        {
            _baselineCount++;
            _baseline += (reward - _baseline) / _baselineCount;
            var advantage = reward - _baseline;

            var weights = WeightsFor(key);
            var probabilities = Softmax(weights);

            foreach (var candidate in _candidates.Keys)
            {
                var indicator = string.Equals(candidate, agentName, StringComparison.Ordinal) ? 1.0 : 0.0;
                weights[candidate] += _learningRate * advantage * (indicator - probabilities[candidate]);
            }
        }
    }

    /// <summary>
    /// Trains the policy from graded trajectories (each whose agent is a candidate and whose reward is set).
    /// </summary>
    /// <param name="trajectories">The graded trajectories.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="trajectories"/> is <c>null</c>.</exception>
    public void LearnFrom(IEnumerable<AgentTrajectory> trajectories)
    {
        Guard.NotNull(trajectories);
        foreach (var trajectory in trajectories)
        {
            if (trajectory.Reward is { } reward)
            {
                Update(trajectory.Messages, trajectory.AgentName, reward);
            }
        }
    }

    /// <summary>Gets the current policy probability of choosing an agent for a context.</summary>
    /// <param name="agentName">The candidate agent name.</param>
    /// <param name="context">The request context, or <c>null</c>.</param>
    public double ProbabilityOf(string agentName, IReadOnlyList<ChatMessage>? context = null)
    {
        Guard.NotNull(agentName);
        var key = ContextKey(context);
        lock (_gate)
        {
            var probabilities = Softmax(WeightsFor(key));
            return probabilities.TryGetValue(agentName, out var p) ? p : 0.0;
        }
    }

    /// <summary>Returns the highest-probability agent for a context (deterministic, the exploit choice).</summary>
    /// <param name="context">The request context, or <c>null</c>.</param>
    public string SelectBestAgentName(IReadOnlyList<ChatMessage>? context = null)
    {
        var key = ContextKey(context);
        lock (_gate)
        {
            var weights = WeightsFor(key);
            var best = _candidates.Keys.First();
            var bestWeight = double.NegativeInfinity;
            foreach (var candidate in _candidates.Keys)
            {
                if (weights[candidate] > bestWeight)
                {
                    bestWeight = weights[candidate];
                    best = candidate;
                }
            }

            return best;
        }
    }

    /// <inheritdoc/>
    public Task<AgentRunResult> RunAsync(IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        var name = SampleAgentName(messages);
        return _candidates[name].RunAsync(messages, cancellationToken);
    }

    private string SampleAgentName(IReadOnlyList<ChatMessage> messages)
    {
        var key = ContextKey(messages);
        lock (_gate)
        {
            var probabilities = Softmax(WeightsFor(key));
            var threshold = _random.NextDouble();
            var cumulative = 0.0;
            foreach (var candidate in _candidates.Keys)
            {
                cumulative += probabilities[candidate];
                if (threshold <= cumulative)
                {
                    return candidate;
                }
            }

            return _candidates.Keys.Last();
        }
    }

    private string ContextKey(IReadOnlyList<ChatMessage>? context) =>
        _contextKey is null || context is null ? string.Empty : _contextKey(context);

    private Dictionary<string, double> WeightsFor(string contextKey)
    {
        if (!_weights.TryGetValue(contextKey, out var weights))
        {
            weights = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (var candidate in _candidates.Keys)
            {
                weights[candidate] = 0.0;
            }

            _weights[contextKey] = weights;
        }

        return weights;
    }

    private Dictionary<string, double> Softmax(Dictionary<string, double> weights)
    {
        var max = double.NegativeInfinity;
        foreach (var value in weights.Values)
        {
            if (value > max)
            {
                max = value;
            }
        }

        var sum = 0.0;
        var exponentials = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var pair in weights)
        {
            var e = Math.Exp(pair.Value - max);
            exponentials[pair.Key] = e;
            sum += e;
        }

        var probabilities = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var pair in exponentials)
        {
            probabilities[pair.Key] = pair.Value / sum;
        }

        return probabilities;
    }
}
