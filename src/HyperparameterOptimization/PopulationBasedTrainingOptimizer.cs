using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.HyperparameterOptimization;

/// <summary>
/// Implements Population-based Training (PBT) for hyperparameter optimization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Population-based Training is an evolutionary approach that:
/// - Trains a population of models simultaneously
/// - Periodically checks each model's performance
/// - Poor performers "exploit" better performers (copy their weights/hyperparameters)
/// - Copies are then "explored" (slightly mutated hyperparameters)
/// - This creates an online hyperparameter schedule adapted during training
///
/// Key concepts:
/// - Population: Multiple models training in parallel
/// - Exploit: Copy weights and hyperparameters from a better performer
/// - Explore: Perturb hyperparameters to discover better values
/// - Ready: A model is "ready" to exploit/explore after training for some steps
///
/// PBT can discover hyperparameter schedules that vary during training,
/// something that grid/random/Bayesian search cannot do.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for models.</typeparam>
/// <typeparam name="TOutput">The output data type for models.</typeparam>
public class PopulationBasedTrainingOptimizer<T, TInput, TOutput> : HyperparameterOptimizerBase<T, TInput, TOutput>
{
    private readonly Random _random;
    private readonly INumericOperations<T> _numOps;
    private readonly int _populationSize;
    private readonly int _readyInterval;
    private readonly double _exploitFraction;
    private readonly double _perturbFactor;
    private readonly ExploitStrategy _exploitStrategy;
    private readonly ExploreStrategy _exploreStrategy;

    // Population state
    private List<PopulationMember>? _population;

    /// <summary>
    /// Initializes a new instance of the PopulationBasedTrainingOptimizer class.
    /// </summary>
    /// <param name="maximize">Whether to maximize the objective (true) or minimize it (false).</param>
    /// <param name="populationSize">Number of models in the population.</param>
    /// <param name="readyInterval">Training steps between exploit/explore cycles.</param>
    /// <param name="exploitFraction">Fraction of bottom performers that will exploit (default 0.2).</param>
    /// <param name="perturbFactor">Factor for perturbing continuous hyperparameters (default 0.2).</param>
    /// <param name="exploitStrategy">Strategy for selecting which member to copy from.</param>
    /// <param name="exploreStrategy">Strategy for perturbing hyperparameters.</param>
    /// <param name="seed">Random seed for reproducibility. If null, uses a random seed.</param>
    public PopulationBasedTrainingOptimizer(
        bool maximize = true,
        int populationSize = 10,
        int readyInterval = 10,
        double exploitFraction = 0.2,
        double perturbFactor = 0.2,
        ExploitStrategy exploitStrategy = ExploitStrategy.Truncation,
        ExploreStrategy exploreStrategy = ExploreStrategy.Perturb,
        int? seed = null) : base(maximize)
    {
        if (populationSize < 2)
            throw new ArgumentException("populationSize must be at least 2.", nameof(populationSize));
        if (readyInterval < 1)
            throw new ArgumentException("readyInterval must be at least 1.", nameof(readyInterval));
        if (exploitFraction <= 0 || exploitFraction > 0.5)
            throw new ArgumentException("exploitFraction must be in (0, 0.5].", nameof(exploitFraction));
        if (perturbFactor <= 0 || perturbFactor > 1)
            throw new ArgumentException("perturbFactor must be in (0, 1].", nameof(perturbFactor));

        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        _numOps = MathHelper.GetNumericOperations<T>();
        _populationSize = populationSize;
        _readyInterval = readyInterval;
        _exploitFraction = exploitFraction;
        _perturbFactor = perturbFactor;
        _exploitStrategy = exploitStrategy;
        _exploreStrategy = exploreStrategy;
    }

    /// <summary>
    /// Runs Population-based Training optimization.
    /// </summary>
    /// <param name="objectiveFunction">
    /// Function that trains and evaluates a configuration.
    /// Should accept "step" and "member_id" in addition to hyperparameters.
    /// </param>
    /// <param name="searchSpace">The hyperparameter search space.</param>
    /// <param name="nTrials">Total number of training steps across all population members.</param>
    public override HyperparameterOptimizationResult<T> Optimize(
        Func<Dictionary<string, object>, T> objectiveFunction,
        HyperparameterSearchSpace searchSpace,
        int nTrials)
    {
        ValidateOptimizationInputs(objectiveFunction, searchSpace, nTrials);

        SearchSpace = searchSpace;
        Trials.Clear();
        var startTime = DateTime.UtcNow;

        lock (SyncLock)
        {
            // Initialize population
            InitializePopulation(searchSpace);

            int trialId = 0;
            int totalSteps = nTrials;
            int stepsPerGeneration = _populationSize;
            int generations = (int)Math.Ceiling((double)totalSteps / stepsPerGeneration);

            for (int gen = 0; gen < generations && trialId < totalSteps; gen++)
            {
                // Train each member for one step
                for (int i = 0; i < _populationSize && trialId < totalSteps; i++)
                {
                    var member = _population![i];
                    member.StepCount++;

                    var trial = new HyperparameterTrial<T>(trialId++);
                    var config = new Dictionary<string, object>(member.Configuration)
                    {
                        ["step"] = member.StepCount,
                        ["member_id"] = member.MemberId,
                        ["generation"] = gen
                    };

                    EvaluateTrialSafely(trial, objectiveFunction, config);
                    Trials.Add(trial);

                    if (trial.Status == TrialStatus.Complete && trial.ObjectiveValue != null)
                    {
                        member.LastScore = trial.ObjectiveValue;
                        member.Trials.Add(trial);
                    }
                }

                // Exploit and explore at ready intervals
                if ((gen + 1) % _readyInterval == 0)
                {
                    ExploitAndExplore(searchSpace);
                }
            }
        }

        var endTime = DateTime.UtcNow;
        return CreateOptimizationResult(searchSpace, startTime, endTime, Trials.Count);
    }

    /// <summary>
    /// Initializes the population with random configurations.
    /// </summary>
    private void InitializePopulation(HyperparameterSearchSpace searchSpace)
    {
        _population = new List<PopulationMember>();

        for (int i = 0; i < _populationSize; i++)
        {
            var config = SampleRandomConfiguration(searchSpace);
            _population.Add(new PopulationMember(i, config));
        }
    }

    /// <summary>
    /// Performs the exploit and explore phase.
    /// </summary>
    private void ExploitAndExplore(HyperparameterSearchSpace searchSpace)
    {
        if (_population == null) return;

        // Get members with valid scores
        var scoredMembers = _population
            .Where(m => m.LastScore != null)
            .ToList();

        if (scoredMembers.Count < 2) return;

        // Sort by performance
        var sorted = Maximize
            ? scoredMembers.OrderByDescending(m => _numOps.ToDouble(m.LastScore!)).ToList()
            : scoredMembers.OrderBy(m => _numOps.ToDouble(m.LastScore!)).ToList();

        // Determine how many members will exploit
        int numToExploit = Math.Max(1, (int)(_populationSize * _exploitFraction));
        int numTop = Math.Max(1, _populationSize - numToExploit);

        // Bottom performers exploit top performers
        for (int i = sorted.Count - 1; i >= numTop; i--)
        {
            var exploiter = sorted[i];

            // Select a member to copy from
            PopulationMember target = SelectExploitTarget(sorted, numTop);

            // Exploit: copy configuration
            exploiter.Configuration = new Dictionary<string, object>(target.Configuration);

            // Explore: perturb hyperparameters
            ExploreConfiguration(exploiter.Configuration, searchSpace);
        }
    }

    /// <summary>
    /// Selects a member to exploit from.
    /// </summary>
    private PopulationMember SelectExploitTarget(List<PopulationMember> sorted, int numTop)
    {
        return _exploitStrategy switch
        {
            ExploitStrategy.Truncation => sorted[_random.Next(numTop)],
            ExploitStrategy.Binary => sorted[0], // Always copy best
            ExploitStrategy.Probabilistic => SelectProbabilistic(sorted, numTop),
            _ => sorted[_random.Next(numTop)]
        };
    }

    /// <summary>
    /// Selects a target using fitness-proportional selection.
    /// </summary>
    private PopulationMember SelectProbabilistic(List<PopulationMember> sorted, int numTop)
    {
        // Simple rank-based selection
        double[] weights = new double[numTop];
        double total = 0;
        for (int i = 0; i < numTop; i++)
        {
            weights[i] = numTop - i; // Higher rank = higher weight
            total += weights[i];
        }

        double r = _random.NextDouble() * total;
        double cumulative = 0;
        for (int i = 0; i < numTop; i++)
        {
            cumulative += weights[i];
            if (r <= cumulative)
                return sorted[i];
        }

        return sorted[0];
    }

    /// <summary>
    /// Perturbs a configuration's hyperparameters.
    /// </summary>
    private void ExploreConfiguration(Dictionary<string, object> config, HyperparameterSearchSpace searchSpace)
    {
        foreach (var paramName in searchSpace.Parameters.Keys.ToList())
        {
            if (!config.ContainsKey(paramName)) continue;

            var distribution = searchSpace.Parameters[paramName];
            config[paramName] = _exploreStrategy switch
            {
                ExploreStrategy.Perturb => PerturbParameter(config[paramName], distribution),
                ExploreStrategy.Resample => distribution.Sample(_random),
                ExploreStrategy.PerturbOrResample => _random.NextDouble() < 0.8
                    ? PerturbParameter(config[paramName], distribution)
                    : distribution.Sample(_random),
                _ => PerturbParameter(config[paramName], distribution)
            };
        }
    }

    /// <summary>
    /// Perturbs a single parameter value.
    /// </summary>
    private object PerturbParameter(object value, ParameterDistribution distribution)
    {
        return distribution switch
        {
            ContinuousDistribution cont => PerturbContinuous(Convert.ToDouble(value), cont),
            IntegerDistribution intDist => PerturbInteger(Convert.ToInt32(value), intDist),
            CategoricalDistribution cat => PerturbCategorical(value, cat),
            _ => value
        };
    }

    private double PerturbContinuous(double value, ContinuousDistribution dist)
    {
        // Multiply or divide by perturbFactor
        double factor = _random.NextDouble() < 0.5
            ? 1.0 + _perturbFactor
            : 1.0 / (1.0 + _perturbFactor);

        double newValue;
        if (dist.LogScale)
        {
            newValue = value * factor;
        }
        else
        {
            double range = dist.Max - dist.Min;
            double perturbation = range * _perturbFactor * (_random.NextDouble() * 2 - 1);
            newValue = value + perturbation;
        }

        return Math.Max(dist.Min, Math.Min(dist.Max, newValue));
    }

    private int PerturbInteger(int value, IntegerDistribution dist)
    {
        int range = dist.Max - dist.Min;
        int perturbation = (int)Math.Ceiling(range * _perturbFactor);
        int delta = _random.Next(-perturbation, perturbation + 1);
        int newValue = value + delta;

        return Math.Max(dist.Min, Math.Min(dist.Max, newValue));
    }

    private object PerturbCategorical(object value, CategoricalDistribution dist)
    {
        // With probability perturbFactor, pick a random category
        if (_random.NextDouble() < _perturbFactor)
        {
            return dist.Choices[_random.Next(dist.Choices.Count)];
        }
        return value;
    }

    /// <summary>
    /// Suggests the next hyperparameter configuration to try.
    /// </summary>
    public override Dictionary<string, object> SuggestNext(HyperparameterTrial<T> trial)
    {
        if (SearchSpace == null)
            throw new InvalidOperationException("Search space not initialized. Call Optimize() first.");

        return SampleRandomConfiguration(SearchSpace);
    }

    /// <summary>
    /// Samples a random configuration from the search space.
    /// </summary>
    private Dictionary<string, object> SampleRandomConfiguration(HyperparameterSearchSpace searchSpace)
    {
        var config = new Dictionary<string, object>();
        foreach (var param in searchSpace.Parameters)
        {
            config[param.Key] = param.Value.Sample(_random);
        }
        return config;
    }

    /// <summary>
    /// Gets the current population state.
    /// </summary>
    public List<PopulationMemberInfo> GetPopulationState()
    {
        lock (SyncLock)
        {
            if (_population == null)
                return new List<PopulationMemberInfo>();

            return _population.Select(m => new PopulationMemberInfo(
                m.MemberId,
                m.Configuration,
                m.LastScore != null ? _numOps.ToDouble(m.LastScore) : (double?)null,
                m.StepCount,
                m.Trials.Count
            )).ToList();
        }
    }

    /// <summary>
    /// Gets the best member of the population.
    /// </summary>
    public PopulationMemberInfo? GetBestMember()
    {
        lock (SyncLock)
        {
            if (_population == null) return null;

            var scoredMembers = _population.Where(m => m.LastScore != null).ToList();
            if (scoredMembers.Count == 0) return null;

            var best = Maximize
                ? scoredMembers.OrderByDescending(m => _numOps.ToDouble(m.LastScore!)).First()
                : scoredMembers.OrderBy(m => _numOps.ToDouble(m.LastScore!)).First();

            return new PopulationMemberInfo(
                best.MemberId,
                best.Configuration,
                _numOps.ToDouble(best.LastScore!),
                best.StepCount,
                best.Trials.Count
            );
        }
    }

    #region Helper Classes

    private class PopulationMember
    {
        public int MemberId { get; }
        public Dictionary<string, object> Configuration { get; set; }
        public T? LastScore { get; set; }
        public int StepCount { get; set; }
        public List<HyperparameterTrial<T>> Trials { get; }

        public PopulationMember(int memberId, Dictionary<string, object> configuration)
        {
            MemberId = memberId;
            Configuration = configuration;
            Trials = new List<HyperparameterTrial<T>>();
        }
    }

    #endregion
}

/// <summary>
/// Strategy for exploiting better performers in PBT.
/// </summary>
public enum ExploitStrategy
{
    /// <summary>
    /// Randomly sample from top performers (truncation selection).
    /// </summary>
    Truncation,

    /// <summary>
    /// Always copy from the best performer.
    /// </summary>
    Binary,

    /// <summary>
    /// Fitness-proportional selection weighted by rank.
    /// </summary>
    Probabilistic
}

/// <summary>
/// Strategy for exploring hyperparameters in PBT.
/// </summary>
public enum ExploreStrategy
{
    /// <summary>
    /// Perturb each hyperparameter by a factor.
    /// </summary>
    Perturb,

    /// <summary>
    /// Resample hyperparameters from the search space.
    /// </summary>
    Resample,

    /// <summary>
    /// Mix of perturbing and resampling.
    /// </summary>
    PerturbOrResample
}

/// <summary>
/// Information about a population member in PBT.
/// </summary>
public class PopulationMemberInfo
{
    /// <summary>
    /// Unique identifier for this member.
    /// </summary>
    public int MemberId { get; }

    /// <summary>
    /// Current hyperparameter configuration.
    /// </summary>
    public Dictionary<string, object> Configuration { get; }

    /// <summary>
    /// Most recent performance score, if evaluated.
    /// </summary>
    public double? LastScore { get; }

    /// <summary>
    /// Number of training steps completed.
    /// </summary>
    public int StepCount { get; }

    /// <summary>
    /// Total number of evaluations for this member.
    /// </summary>
    public int TrialCount { get; }

    /// <summary>
    /// Initializes a new PopulationMemberInfo.
    /// </summary>
    public PopulationMemberInfo(int memberId, Dictionary<string, object> configuration,
        double? lastScore, int stepCount, int trialCount)
    {
        MemberId = memberId;
        Configuration = configuration;
        LastScore = lastScore;
        StepCount = stepCount;
        TrialCount = trialCount;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        string scoreStr = LastScore.HasValue ? $"{LastScore:F4}" : "N/A";
        return $"Member {MemberId}: score={scoreStr}, steps={StepCount}";
    }
}
