using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.HyperparameterOptimization;

/// <summary>
/// Implements Hyperband optimization for hyperparameter tuning with early stopping.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Hyperband is a smart resource allocation strategy that:
/// - Trains many configurations with minimal resources initially
/// - Progressively eliminates poorly performing configurations
/// - Allocates more resources to promising configurations
/// - Uses "successive halving" to efficiently explore the search space
///
/// Key concepts:
/// - Resource (R): Training budget (e.g., epochs, iterations, data samples)
/// - Configuration: A specific set of hyperparameter values
/// - Bracket: A group of configurations competing via successive halving
/// - Successive Halving: Repeatedly train, evaluate, and keep top half
///
/// Hyperband runs multiple brackets with different exploration/exploitation trade-offs,
/// combining the best properties of random search with aggressive early stopping.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for models.</typeparam>
/// <typeparam name="TOutput">The output data type for models.</typeparam>
public class HyperbandOptimizer<T, TInput, TOutput> : HyperparameterOptimizerBase<T, TInput, TOutput>
{
    private readonly Random _random;
    private readonly INumericOperations<T> _numOps;
    private readonly int _maxResource;
    private readonly int _reductionFactor;
    private readonly int _minResource;

    /// <summary>
    /// Gets the number of brackets in this Hyperband configuration.
    /// </summary>
    public int NumBrackets { get; }

    /// <summary>
    /// Initializes a new instance of the HyperbandOptimizer class.
    /// </summary>
    /// <param name="maximize">Whether to maximize the objective (true) or minimize it (false).</param>
    /// <param name="maxResource">Maximum resource budget per configuration (e.g., max epochs).</param>
    /// <param name="reductionFactor">Factor to reduce configurations by at each round (typically 3).</param>
    /// <param name="minResource">Minimum resource budget per configuration (e.g., min epochs).</param>
    /// <param name="seed">Random seed for reproducibility. If null, uses a random seed.</param>
    public HyperbandOptimizer(
        bool maximize = true,
        int maxResource = 81,
        int reductionFactor = 3,
        int minResource = 1,
        int? seed = null) : base(maximize)
    {
        if (maxResource < 1)
            throw new ArgumentException("maxResource must be at least 1.", nameof(maxResource));
        if (reductionFactor < 2)
            throw new ArgumentException("reductionFactor must be at least 2.", nameof(reductionFactor));
        if (minResource < 1)
            throw new ArgumentException("minResource must be at least 1.", nameof(minResource));
        if (minResource > maxResource)
            throw new ArgumentException("minResource cannot exceed maxResource.", nameof(minResource));

        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        _numOps = MathHelper.GetNumericOperations<T>();
        _maxResource = maxResource;
        _reductionFactor = reductionFactor;
        _minResource = minResource;

        // Calculate number of brackets: s_max = floor(log_eta(R / r))
        NumBrackets = (int)Math.Floor(Math.Log(_maxResource / (double)_minResource) / Math.Log(_reductionFactor)) + 1;
    }

    /// <summary>
    /// Searches for the best hyperparameter configuration using Hyperband.
    /// </summary>
    /// <param name="objectiveFunction">
    /// Function that takes hyperparameters and resource budget, returns performance.
    /// The resource budget is passed as a key "resource" in the dictionary.
    /// </param>
    /// <param name="searchSpace">The hyperparameter search space.</param>
    /// <param name="nTrials">Maximum number of configurations to evaluate (approximate).</param>
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
            int trialId = 0;
            int trialsRemaining = nTrials;

            // Run Hyperband outer loop (over brackets)
            while (trialsRemaining > 0)
            {
                // Iterate over brackets from most aggressive to least
                for (int bracket = NumBrackets - 1; bracket >= 0 && trialsRemaining > 0; bracket--)
                {
                    var bracketResult = RunBracket(
                        bracket,
                        objectiveFunction,
                        searchSpace,
                        ref trialId,
                        ref trialsRemaining);

                    // Track trials from this bracket
                    Trials.AddRange(bracketResult);
                }
            }
        }

        var endTime = DateTime.UtcNow;
        return CreateOptimizationResult(searchSpace, startTime, endTime, Trials.Count);
    }

    /// <summary>
    /// Runs a single Hyperband bracket with successive halving.
    /// </summary>
    private List<HyperparameterTrial<T>> RunBracket(
        int bracket,
        Func<Dictionary<string, object>, T> objectiveFunction,
        HyperparameterSearchSpace searchSpace,
        ref int trialId,
        ref int trialsRemaining)
    {
        var bracketTrials = new List<HyperparameterTrial<T>>();

        // Calculate bracket parameters
        // n: initial number of configurations
        // r: initial resource per configuration
        int sMax = NumBrackets - 1;
        double eta = _reductionFactor;

        // n = ceil((B/R) * (eta^s / (s+1)))
        // where B is approximate budget per bracket
        int n = (int)Math.Ceiling(
            (sMax + 1.0) / (bracket + 1) * Math.Pow(eta, bracket)
        );
        n = Math.Max(1, Math.Min(n, trialsRemaining));

        // r = R * eta^(-s)
        int r = (int)Math.Max(_minResource, _maxResource * Math.Pow(eta, -bracket));

        // Initialize configurations
        var configurations = new List<(Dictionary<string, object> config, T? score, int trialIndex)>();
        for (int i = 0; i < n && trialsRemaining > 0; i++)
        {
            var config = SampleRandomConfiguration(searchSpace);
            configurations.Add((config, default, trialId++));
            trialsRemaining--;
        }

        // Successive halving rounds
        int numRounds = bracket + 1;
        for (int round = 0; round < numRounds && configurations.Count > 0; round++)
        {
            int currentResource = (int)(r * Math.Pow(eta, round));
            currentResource = Math.Min(currentResource, _maxResource);

            // Evaluate all remaining configurations with current resource budget
            var evaluatedConfigs = new List<(Dictionary<string, object> config, T score, int trialIndex)>();

            foreach (var (config, _, trialIndex) in configurations)
            {
                var trial = new HyperparameterTrial<T>(trialIndex);

                // Add resource budget to configuration
                var configWithResource = new Dictionary<string, object>(config)
                {
                    ["resource"] = currentResource
                };

                // Evaluate
                EvaluateTrialSafely(trial, objectiveFunction, configWithResource);
                bracketTrials.Add(trial);

                if (trial.Status == TrialStatus.Complete && trial.ObjectiveValue != null)
                {
                    evaluatedConfigs.Add((config, trial.ObjectiveValue, trialIndex));
                }
            }

            // Select top configurations to continue
            int numToKeep = (int)Math.Max(1, Math.Floor(configurations.Count / eta));

            if (round < numRounds - 1 && evaluatedConfigs.Count > numToKeep)
            {
                // Sort by performance
                var sorted = Maximize
                    ? evaluatedConfigs.OrderByDescending(c => _numOps.ToDouble(c.score)).ToList()
                    : evaluatedConfigs.OrderBy(c => _numOps.ToDouble(c.score)).ToList();

                // Keep top performers
                configurations = sorted.Take(numToKeep)
                    .Select(c => (c.config, (T?)c.score, c.trialIndex))
                    .ToList();
            }
            else
            {
                // Last round or too few configurations - stop
                break;
            }
        }

        return bracketTrials;
    }

    /// <summary>
    /// Suggests the next hyperparameter configuration to try.
    /// </summary>
    /// <remarks>
    /// For Hyperband, this returns a random sample. The actual resource-aware
    /// optimization happens within the Optimize() method.
    /// </remarks>
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
    /// Gets the total number of configurations that would be evaluated in a full Hyperband run.
    /// </summary>
    public int GetTotalConfigurationCount()
    {
        int total = 0;
        double eta = _reductionFactor;
        int sMax = NumBrackets - 1;

        for (int bracket = 0; bracket <= sMax; bracket++)
        {
            int n = (int)Math.Ceiling(
                (sMax + 1.0) / (bracket + 1) * Math.Pow(eta, bracket)
            );
            total += n;
        }

        return total;
    }

    /// <summary>
    /// Gets detailed information about the bracket structure.
    /// </summary>
    public List<BracketInfo> GetBracketInfo()
    {
        var brackets = new List<BracketInfo>();
        double eta = _reductionFactor;
        int sMax = NumBrackets - 1;

        for (int bracket = sMax; bracket >= 0; bracket--)
        {
            int n = (int)Math.Ceiling(
                (sMax + 1.0) / (bracket + 1) * Math.Pow(eta, bracket)
            );
            int r = (int)Math.Max(_minResource, _maxResource * Math.Pow(eta, -bracket));

            var rounds = new List<(int configs, int resource)>();
            int currentConfigs = n;

            for (int round = 0; round <= bracket; round++)
            {
                int currentResource = (int)(r * Math.Pow(eta, round));
                currentResource = Math.Min(currentResource, _maxResource);
                rounds.Add((currentConfigs, currentResource));
                currentConfigs = (int)Math.Max(1, Math.Floor(currentConfigs / eta));
            }

            brackets.Add(new BracketInfo(bracket, n, r, rounds));
        }

        return brackets;
    }
}

/// <summary>
/// Information about a Hyperband bracket.
/// </summary>
public class BracketInfo
{
    /// <summary>
    /// The bracket index (s value in Hyperband paper).
    /// </summary>
    public int BracketIndex { get; }

    /// <summary>
    /// Initial number of configurations in this bracket.
    /// </summary>
    public int InitialConfigurations { get; }

    /// <summary>
    /// Initial resource budget per configuration.
    /// </summary>
    public int InitialResource { get; }

    /// <summary>
    /// Rounds in this bracket with (configurations, resource) at each round.
    /// </summary>
    public List<(int Configurations, int Resource)> Rounds { get; }

    /// <summary>
    /// Total resource units consumed by this bracket.
    /// </summary>
    public int TotalResource => Rounds.Sum(r => r.Configurations * r.Resource);

    /// <summary>
    /// Initializes a new BracketInfo.
    /// </summary>
    public BracketInfo(int bracketIndex, int initialConfigurations, int initialResource,
        List<(int configs, int resource)> rounds)
    {
        BracketIndex = bracketIndex;
        InitialConfigurations = initialConfigurations;
        InitialResource = initialResource;
        Rounds = rounds;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Bracket {BracketIndex}: {InitialConfigurations} configs starting at {InitialResource} resources";
    }
}
