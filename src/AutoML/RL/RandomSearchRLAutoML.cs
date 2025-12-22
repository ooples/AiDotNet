using System.Globalization;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.ReinforcementLearning.Agents.A2C;
using AiDotNet.ReinforcementLearning.Agents.DDPG;
using AiDotNet.ReinforcementLearning.Agents.DQN;
using AiDotNet.ReinforcementLearning.Agents.PPO;
using AiDotNet.ReinforcementLearning.Agents.SAC;

namespace AiDotNet.AutoML.RL;

internal sealed class RandomSearchRLAutoML<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly IEnvironment<T> _environment;
    private readonly RLAutoMLOptions<T> _options;
    private readonly TimeSpan _timeLimit;
    private readonly int _trialLimit;
    private readonly int _maxStepsPerEpisode;
    private readonly Random _random;

    public RandomSearchRLAutoML(
        IEnvironment<T> environment,
        RLAutoMLOptions<T> options,
        TimeSpan timeLimit,
        int trialLimit,
        int maxStepsPerEpisode,
        int? seed = null)
    {
        _environment = environment ?? throw new ArgumentNullException(nameof(environment));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        _timeLimit = timeLimit;
        _trialLimit = trialLimit;
        _maxStepsPerEpisode = Math.Max(1, maxStepsPerEpisode);

        _numOps = MathHelper.GetNumericOperations<T>();
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    public (IRLAgent<T> BestAgent, AutoMLRunSummary Summary) Search()
    {
        var startedUtc = DateTimeOffset.UtcNow;
        var summary = new AutoMLRunSummary
        {
            SearchStrategy = AutoMLSearchStrategy.RandomSearch,
            TimeLimit = _timeLimit,
            TrialLimit = _trialLimit,
            OptimizationMetric = MetricType.AverageEpisodeReward,
            MaximizeMetric = true,
            BestScore = double.NegativeInfinity,
            SearchStartedUtc = startedUtc
        };

        var candidateAgents = ResolveCandidateAgents();
        if (candidateAgents.Count == 0)
        {
            throw new InvalidOperationException("No candidate RL agents are configured for AutoML.");
        }

        var deadline = startedUtc.UtcDateTime.Add(_timeLimit);
        Dictionary<string, object>? bestParams = null;
        RLAutoMLAgentType bestAgentType = candidateAgents[0];

        int trialId = 0;
        while (DateTime.UtcNow < deadline && trialId < _trialLimit)
        {
            trialId++;
            var trialStart = DateTime.UtcNow;

            try
            {
                var agentType = candidateAgents[_random.Next(candidateAgents.Count)];
                var searchSpace = GetSearchSpace(agentType);
                var parameters = AutoMLParameterSampler.Sample(_random, searchSpace);

                // Track agent type internally (not emitted in the facade summary).
                parameters["AgentType"] = agentType.ToString();

                // Train briefly, then evaluate without exploration.
                var trainedAgent = CreateAgent(agentType, parameters);
                TrainEpisodes(trainedAgent, Math.Max(1, _options.TrainingEpisodesPerTrial));
                var score = EvaluateEpisodes(trainedAgent, Math.Max(1, _options.EvaluationEpisodesPerTrial));

                var duration = DateTime.UtcNow - trialStart;
                summary.Trials.Add(new AutoMLTrialSummary
                {
                    TrialId = trialId,
                    Score = score,
                    Duration = duration,
                    CompletedUtc = DateTime.UtcNow,
                    Success = true
                });

                if (score > summary.BestScore)
                {
                    summary.BestScore = score;
                    bestParams = parameters;
                    bestAgentType = agentType;
                }
            }
            catch (ArgumentException ex)
            {
                RecordFailedTrial(summary, trialId, trialStart, ex);
            }
            catch (InvalidOperationException ex)
            {
                RecordFailedTrial(summary, trialId, trialStart, ex);
            }
            catch (NotSupportedException ex)
            {
                RecordFailedTrial(summary, trialId, trialStart, ex);
            }
            catch (ArithmeticException ex)
            {
                RecordFailedTrial(summary, trialId, trialStart, ex);
            }
        }

        if (bestParams is null)
        {
            throw new InvalidOperationException("RL AutoML failed to find a valid agent configuration within the given budget.");
        }

        summary.SearchEndedUtc = DateTimeOffset.UtcNow;

        // Create a fresh agent instance for full training using the best discovered parameters.
        var bestAgent = CreateAgent(bestAgentType, bestParams);
        return (bestAgent, summary);
    }

    private static void RecordFailedTrial(AutoMLRunSummary summary, int trialId, DateTime trialStartUtc, Exception exception)
    {
        if (summary is null)
        {
            throw new ArgumentNullException(nameof(summary));
        }

        var completedUtc = DateTime.UtcNow;
        var duration = completedUtc - trialStartUtc;

        summary.Trials.Add(new AutoMLTrialSummary
        {
            TrialId = trialId,
            Score = double.NegativeInfinity,
            Duration = duration,
            CompletedUtc = completedUtc,
            Success = false,
            ErrorMessage = exception.Message
        });
    }

    private List<RLAutoMLAgentType> ResolveCandidateAgents()
    {
        if (_options.CandidateAgents is { Count: > 0 })
        {
            return _options.CandidateAgents.ToList();
        }

        if (_environment.IsContinuousActionSpace)
        {
            return new List<RLAutoMLAgentType> { RLAutoMLAgentType.PPO, RLAutoMLAgentType.A2C, RLAutoMLAgentType.DDPG, RLAutoMLAgentType.SAC };
        }

        return new List<RLAutoMLAgentType> { RLAutoMLAgentType.DQN, RLAutoMLAgentType.PPO, RLAutoMLAgentType.A2C };
    }

    private Dictionary<string, ParameterRange> GetSearchSpace(RLAutoMLAgentType agentType)
    {
        Dictionary<string, ParameterRange> defaults = agentType switch
        {
            RLAutoMLAgentType.DQN => GetDqnSearchSpace(),
            RLAutoMLAgentType.PPO => GetPpoSearchSpace(),
            RLAutoMLAgentType.A2C => GetA2cSearchSpace(),
            RLAutoMLAgentType.DDPG => GetDdpGSearchSpace(),
            RLAutoMLAgentType.SAC => GetSacSearchSpace(),
            _ => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        };

        if (_options.SearchSpaceOverrides.Count == 0)
        {
            return defaults;
        }

        var merged = new Dictionary<string, ParameterRange>(defaults, StringComparer.Ordinal);
        foreach (var (key, value) in _options.SearchSpaceOverrides)
        {
            merged[key] = (ParameterRange)value.Clone();
        }

        return merged;
    }

    private Dictionary<string, ParameterRange> GetDqnSearchSpace()
    {
        return new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["LearningRate"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1e-5, MaxValue = 1e-2, UseLogScale = true, DefaultValue = 0.001 },
            ["DiscountFactor"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.90, MaxValue = 0.999, Step = 0.001, DefaultValue = 0.99 },
            ["EpsilonStart"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.5, MaxValue = 1.0, Step = 0.05, DefaultValue = 1.0 },
            ["EpsilonEnd"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.001, MaxValue = 0.1, UseLogScale = true, DefaultValue = 0.01 },
            ["EpsilonDecay"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.90, MaxValue = 0.9999, Step = 0.0005, DefaultValue = 0.995 },
            ["BatchSize"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 32, MaxValue = 256, Step = 32, DefaultValue = 64 },
            ["ReplayBufferSize"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 50000, MaxValue = 1000000, Step = 50000, DefaultValue = 100000 },
            ["TargetUpdateFrequency"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 100, MaxValue = 5000, Step = 100, DefaultValue = 1000 }
        };
    }

    private Dictionary<string, ParameterRange> GetPpoSearchSpace()
    {
        return new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["PolicyLearningRate"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1e-5, MaxValue = 1e-3, UseLogScale = true, DefaultValue = 0.0003 },
            ["ValueLearningRate"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1e-5, MaxValue = 1e-3, UseLogScale = true, DefaultValue = 0.001 },
            ["DiscountFactor"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.90, MaxValue = 0.999, Step = 0.001, DefaultValue = 0.99 },
            ["GaeLambda"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.90, MaxValue = 0.99, Step = 0.01, DefaultValue = 0.95 },
            ["ClipEpsilon"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.10, MaxValue = 0.30, Step = 0.01, DefaultValue = 0.20 },
            ["EntropyCoefficient"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 0.05, Step = 0.005, DefaultValue = 0.01 },
            ["ValueLossCoefficient"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.25, MaxValue = 1.0, Step = 0.05, DefaultValue = 0.5 },
            ["StepsPerUpdate"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 128, MaxValue = 2048, Step = 128, DefaultValue = 2048 },
            ["MiniBatchSize"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 32, MaxValue = 256, Step = 32, DefaultValue = 64 },
            ["TrainingEpochs"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 3, MaxValue = 15, Step = 1, DefaultValue = 10 }
        };
    }

    private Dictionary<string, ParameterRange> GetA2cSearchSpace()
    {
        return new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["PolicyLearningRate"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1e-5, MaxValue = 1e-3, UseLogScale = true, DefaultValue = 0.0007 },
            ["ValueLearningRate"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1e-5, MaxValue = 1e-3, UseLogScale = true, DefaultValue = 0.001 },
            ["DiscountFactor"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.90, MaxValue = 0.999, Step = 0.001, DefaultValue = 0.99 },
            ["EntropyCoefficient"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 0.05, Step = 0.005, DefaultValue = 0.01 },
            ["ValueLossCoefficient"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.25, MaxValue = 1.0, Step = 0.05, DefaultValue = 0.5 },
            ["StepsPerUpdate"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 1, MaxValue = 20, Step = 1, DefaultValue = 5 }
        };
    }

    private Dictionary<string, ParameterRange> GetDdpGSearchSpace()
    {
        return new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["ActorLearningRate"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1e-5, MaxValue = 1e-3, UseLogScale = true, DefaultValue = 0.0001 },
            ["CriticLearningRate"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1e-5, MaxValue = 1e-3, UseLogScale = true, DefaultValue = 0.001 },
            ["DiscountFactor"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.90, MaxValue = 0.999, Step = 0.001, DefaultValue = 0.99 },
            ["TargetUpdateTau"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0005, MaxValue = 0.02, UseLogScale = true, DefaultValue = 0.001 },
            ["ExplorationNoise"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.01, MaxValue = 0.5, UseLogScale = true, DefaultValue = 0.1 },
            ["BatchSize"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 32, MaxValue = 256, Step = 32, DefaultValue = 64 },
            ["ReplayBufferSize"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 50000, MaxValue = 1000000, Step = 50000, DefaultValue = 100000 },
            ["WarmupSteps"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 0, MaxValue = 20000, Step = 500, DefaultValue = 1000 }
        };
    }

    private Dictionary<string, ParameterRange> GetSacSearchSpace()
    {
        return new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["PolicyLearningRate"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1e-5, MaxValue = 1e-3, UseLogScale = true, DefaultValue = 0.0003 },
            ["QLearningRate"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1e-5, MaxValue = 1e-3, UseLogScale = true, DefaultValue = 0.0003 },
            ["AlphaLearningRate"] = new ParameterRange { Type = ParameterType.Float, MinValue = 1e-5, MaxValue = 1e-3, UseLogScale = true, DefaultValue = 0.0003 },
            ["DiscountFactor"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.90, MaxValue = 0.999, Step = 0.001, DefaultValue = 0.99 },
            ["TargetUpdateTau"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.001, MaxValue = 0.02, UseLogScale = true, DefaultValue = 0.005 },
            ["InitialTemperature"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.05, MaxValue = 1.0, UseLogScale = true, DefaultValue = 0.2 },
            ["BatchSize"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 64, MaxValue = 512, Step = 64, DefaultValue = 256 },
            ["ReplayBufferSize"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 50000, MaxValue = 1000000, Step = 50000, DefaultValue = 1000000 },
            ["WarmupSteps"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 0, MaxValue = 50000, Step = 1000, DefaultValue = 10000 }
        };
    }

    private IRLAgent<T> CreateAgent(RLAutoMLAgentType agentType, IReadOnlyDictionary<string, object> parameters)
    {
        int stateSize = _environment.ObservationSpaceDimension;
        int actionSize = _environment.ActionSpaceSize;
        int? seed = TryGetSeed(parameters);

        switch (agentType)
        {
            case RLAutoMLAgentType.DQN:
            {
                if (_environment.IsContinuousActionSpace)
                {
                    throw new NotSupportedException("DQN is not supported for continuous action spaces.");
                }

                var options = new DQNOptions<T>
                {
                    StateSize = stateSize,
                    ActionSize = actionSize,
                    Seed = seed
                };
                AutoMLHyperparameterApplicator.ApplyToOptions(options, parameters);
                return new DQNAgent<T>(options);
            }

            case RLAutoMLAgentType.PPO:
            {
                var options = new PPOOptions<T>
                {
                    StateSize = stateSize,
                    ActionSize = actionSize,
                    IsContinuous = _environment.IsContinuousActionSpace,
                    Seed = seed
                };
                AutoMLHyperparameterApplicator.ApplyToOptions(options, parameters);
                return new PPOAgent<T>(options);
            }

            case RLAutoMLAgentType.A2C:
            {
                var options = new A2COptions<T>
                {
                    StateSize = stateSize,
                    ActionSize = actionSize,
                    IsContinuous = _environment.IsContinuousActionSpace,
                    Seed = seed
                };
                AutoMLHyperparameterApplicator.ApplyToOptions(options, parameters);
                return new A2CAgent<T>(options);
            }

            case RLAutoMLAgentType.DDPG:
            {
                if (!_environment.IsContinuousActionSpace)
                {
                    throw new NotSupportedException("DDPG is intended for continuous action spaces.");
                }

                var options = new DDPGOptions<T>
                {
                    StateSize = stateSize,
                    ActionSize = actionSize,
                    Seed = seed
                };
                AutoMLHyperparameterApplicator.ApplyToOptions(options, parameters);
                return new DDPGAgent<T>(options);
            }

            case RLAutoMLAgentType.SAC:
            {
                if (!_environment.IsContinuousActionSpace)
                {
                    throw new NotSupportedException("SAC is intended for continuous action spaces.");
                }

                var options = new SACOptions<T>
                {
                    StateSize = stateSize,
                    ActionSize = actionSize,
                    Seed = seed
                };
                AutoMLHyperparameterApplicator.ApplyToOptions(options, parameters);
                return new SACAgent<T>(options);
            }
        }

        throw new NotSupportedException($"Unsupported RL agent type: {agentType}");
    }

    private int? TryGetSeed(IReadOnlyDictionary<string, object> parameters)
    {
        if (parameters is null
            || !parameters.TryGetValue("Seed", out var seedValue)
            || seedValue is null)
        {
            return null;
        }

        if (seedValue is int seed)
        {
            return seed;
        }

        switch (seedValue)
        {
            case short value:
                return value;
            case ushort value:
                return value;
            case byte value:
                return value;
            case sbyte value:
                return value;
            case long value:
                return value >= int.MinValue && value <= int.MaxValue ? (int)value : null;
            case uint value:
                return value <= int.MaxValue ? (int)value : null;
            case ulong value:
                return value <= (ulong)int.MaxValue ? (int)value : null;
        }

        if (seedValue is string seedText)
        {
            return int.TryParse(seedText, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
                ? parsed
                : null;
        }

        if (seedValue is double doubleValue)
        {
            if (double.IsNaN(doubleValue) || double.IsInfinity(doubleValue))
            {
                return null;
            }

            double rounded = Math.Round(doubleValue);
            const double integerEpsilon = 1e-8;
            if (Math.Abs(doubleValue - rounded) > integerEpsilon)
            {
                return null;
            }

            return rounded >= int.MinValue && rounded <= int.MaxValue ? (int)rounded : null;
        }

        if (seedValue is float floatValue)
        {
            double asDouble = floatValue;
            if (double.IsNaN(asDouble) || double.IsInfinity(asDouble))
            {
                return null;
            }

            double rounded = Math.Round(asDouble);
            const double integerEpsilon = 1e-8;
            if (Math.Abs(asDouble - rounded) > integerEpsilon)
            {
                return null;
            }

            return rounded >= int.MinValue && rounded <= int.MaxValue ? (int)rounded : null;
        }

        if (seedValue is decimal decimalValue)
        {
            if (decimalValue != decimal.Truncate(decimalValue))
            {
                return null;
            }

            return decimalValue >= int.MinValue && decimalValue <= int.MaxValue ? (int)decimalValue : null;
        }

        try
        {
            return Convert.ToInt32(seedValue, CultureInfo.InvariantCulture);
        }
        catch (Exception ex) when (ex is FormatException or InvalidCastException or OverflowException)
        {
            return null;
        }
    }

    private void TrainEpisodes(IRLAgent<T> agent, int episodes)
    {
        _ = RunEpisodes(agent, episodes, explore: true, train: true);
    }

    private double EvaluateEpisodes(IRLAgent<T> agent, int episodes)
    {
        return RunEpisodes(agent, episodes, explore: false, train: false);
    }

    private double RunEpisodes(IRLAgent<T> agent, int episodes, bool explore, bool train)
    {
        if (episodes <= 0)
        {
            return double.NegativeInfinity;
        }

        double totalReward = 0.0;
        for (int episode = 0; episode < episodes; episode++)
        {
            var state = _environment.Reset();
            agent.ResetEpisode();

            double episodeReward = 0.0;
            bool done = false;
            int steps = 0;

            while (!done && steps < _maxStepsPerEpisode)
            {
                var action = agent.SelectAction(state, explore: explore);
                var (nextState, reward, isDone, _) = _environment.Step(action);

                if (train)
                {
                    agent.StoreExperience(state, action, reward, nextState, isDone);
                    _ = agent.Train();
                }

                episodeReward += _numOps.ToDouble(reward);
                state = nextState;
                done = isDone;
                steps++;
            }

            totalReward += episodeReward;
        }

        return totalReward / episodes;
    }
}
