using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Environments;
using AiDotNet.ReinforcementLearning.Environments.Trading;
using AiDotNet.ReinforcementLearning.Tournament.Results;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Tournament
{
    /// <summary>
    /// A system for evaluating and comparing different reinforcement learning models.
    /// </summary>
    /// <typeparam name="TState">The type of environment state.</typeparam>
    /// <typeparam name="TAction">The type of action.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class ModelTournament<TState, TAction, T> 
        where TState : class
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        private readonly List<TournamentEntry<TState, TAction, T>> _entries = default!;
        private readonly IEnvironment<TState, TAction, T> _environment = default!;
        private readonly List<IEvaluationMetric<T>> _metrics = default!;
        private readonly int _numEpisodes;
        private readonly Random _random = default!;
        
        /// <summary>
        /// Gets the list of model entries in the tournament.
        /// </summary>
        public IReadOnlyList<TournamentEntry<TState, TAction, T>> Entries => _entries;
        
        /// <summary>
        /// Gets the tournament results, if the tournament has been run.
        /// </summary>
        public TournamentResult<T>? Results { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelTournament{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="environment">The environment to evaluate models in.</param>
        /// <param name="metrics">The evaluation metrics to use.</param>
        /// <param name="numEpisodes">The number of episodes to run for each model.</param>
        public ModelTournament(
            IEnvironment<TState, TAction, T> environment,
            IEnumerable<IEvaluationMetric<T>> metrics,
            int numEpisodes = 10)
        {
            _environment = environment ?? throw new ArgumentNullException(nameof(environment));
            _metrics = metrics?.ToList() ?? throw new ArgumentNullException(nameof(metrics));
            _numEpisodes = numEpisodes;
            _entries = new List<TournamentEntry<TState, TAction, T>>();
            _random = new Random();
        }

        /// <summary>
        /// Adds a model to the tournament.
        /// </summary>
        /// <param name="model">The model to add.</param>
        /// <param name="name">A name for the model.</param>
        /// <param name="description">A description of the model.</param>
        /// <param name="metadata">Additional metadata about the model.</param>
        public void AddModel(
            IReinforcementLearningModel<T> model,
            string name,
            string description = "",
            Dictionary<string, object>? metadata = null)
        {
            var entry = new TournamentEntry<TState, TAction, T>(model, name, description, metadata ?? new Dictionary<string, object>());
            _entries.Add(entry);
        }

        /// <summary>
        /// Runs the tournament and evaluates all models.
        /// </summary>
        /// <param name="progress">Optional progress callback.</param>
        /// <returns>The tournament results.</returns>
        public TournamentResult<T> RunTournament(IProgress<double>? progress = null)
        {
            if (_entries.Count == 0)
            {
                throw new InvalidOperationException("Cannot run tournament with no entries. Add models first.");
            }
            
            var results = new Dictionary<string, List<ModelEpisodeResult<T>>>();
            var metricValues = new Dictionary<string, Dictionary<string, List<T>>>();
            
            // Initialize metric values dictionary
            foreach (var entry in _entries)
            {
                results[entry.Name] = new List<ModelEpisodeResult<T>>();
                metricValues[entry.Name] = new Dictionary<string, List<T>>();
                
                foreach (var metric in _metrics)
                {
                    metricValues[entry.Name][metric.Name] = new List<T>();
                }
            }
            
            // Run evaluation for each model
            double progressValue = 0;
            double progressIncrement = 1.0 / (_entries.Count * _numEpisodes);
            
            foreach (var entry in _entries)
            {
                // Models are assumed to be in evaluation mode for tournaments
                
                for (int episode = 0; episode < _numEpisodes; episode++)
                {
                    // Reset environment for a new episode
                    TState state = _environment.Reset();
                    bool done = false;
                    T episodeReward = NumOps.Zero;
                    var episodeStates = new List<TState>();
                    var episodeActions = new List<TAction>();
                    var episodeRewards = new List<T>();
                    
                    // Run the episode
                    while (!done)
                    {
                        // Convert to tensor if necessary (for trading environment)
                        TState input = state;
                        
                        // Select action using the model
                        var actionVector = entry.Model.SelectAction(input as Tensor<T> ?? throw new InvalidCastException("State must be Tensor<T>"), false);
                        
                        // Convert Vector<T> to TAction
                        TAction typedAction;
                        if (typeof(TAction) == typeof(Vector<T>))
                        {
                            typedAction = (TAction)(object)actionVector;
                        }
                        else if (typeof(TAction) == typeof(int))
                        {
                            // For discrete actions, take the argmax
                            int maxIndex = 0;
                            T maxValue = actionVector[0];
                            for (int i = 1; i < actionVector.Length; i++)
                            {
                                if (NumOps.GreaterThan(actionVector[i], maxValue))
                                {
                                    maxValue = actionVector[i];
                                    maxIndex = i;
                                }
                            }
                            typedAction = (TAction)(object)maxIndex;
                        }
                        else
                        {
                            throw new NotSupportedException($"Cannot convert Vector<{typeof(T).Name}> to {typeof(TAction).Name}");
                        }
                        
                        // Take action in environment
                        var (nextState, reward, isDone) = _environment.Step(typedAction);
                        
                        // Record step
                        episodeStates.Add(state);
                        episodeActions.Add(typedAction);
                        episodeRewards.Add(reward);
                        
                        // Update state and reward
                        state = nextState;
                        episodeReward = NumOps.Add(episodeReward, reward);
                        done = isDone;
                    }
                    
                    // Record episode results
                    var episodeResult = new ModelEpisodeResult<T>(
                        episodeReward,
                        episodeStates.Cast<object>(),
                        episodeActions.Cast<object>(),
                        episodeRewards);
                    
                    results[entry.Name].Add(episodeResult);
                    
                    // Calculate metrics for this episode
                    foreach (var metric in _metrics)
                    {
                        T metricValue = metric.Calculate(episodeResult);
                        metricValues[entry.Name][metric.Name].Add(metricValue);
                    }
                    
                    // Update progress
                    progressValue += progressIncrement;
                    progress?.Report(progressValue);
                }
            }
            
            // Calculate aggregate metric values
            var aggregateMetrics = new Dictionary<string, Dictionary<string, AggregateMetricValues<T>>>();
            
            foreach (var entry in _entries)
            {
                aggregateMetrics[entry.Name] = new Dictionary<string, AggregateMetricValues<T>>();
                
                foreach (var metric in _metrics)
                {
                    var values = metricValues[entry.Name][metric.Name];
                    
                    // Calculate mean
                    T sum = NumOps.Zero;
                    foreach (var value in values)
                    {
                        sum = NumOps.Add(sum, value);
                    }
                    T mean = NumOps.Divide(sum, NumOps.FromDouble(values.Count));
                    
                    // Calculate standard deviation
                    T sumSquaredDiff = NumOps.Zero;
                    foreach (var value in values)
                    {
                        T diff = NumOps.Subtract(value, mean);
                        sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
                    }
                    T stdDev = NumOps.Sqrt(NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(values.Count)));
                    
                    // Calculate min and max
                    T min = values.Count > 0 ? values[0] : NumOps.Zero;
                    T max = values.Count > 0 ? values[0] : NumOps.Zero;
                    
                    foreach (var value in values)
                    {
                        if (NumOps.LessThan(value, min)) min = value;
                        if (NumOps.GreaterThan(value, max)) max = value;
                    }
                    
                    // Create aggregate metric values
                    var aggregate = new AggregateMetricValues<T>(mean, stdDev, min, max, values);
                    aggregateMetrics[entry.Name][metric.Name] = aggregate;
                }
            }
            
            // Determine rankings for each metric
            var rankings = new Dictionary<string, Dictionary<string, int>>();
            
            foreach (var metric in _metrics)
            {
                var metricRankings = new Dictionary<string, int>();
                
                // Create a list of entries sorted by mean metric value
                var sortedEntries = _entries
                    .Select(e => new { 
                        Name = e.Name, 
                        MetricValue = aggregateMetrics[e.Name][metric.Name].Mean 
                    })
                    .OrderByDescending(e => e.MetricValue)
                    .ToList();
                
                // Assign rankings
                for (int i = 0; i < sortedEntries.Count; i++)
                {
                    metricRankings[sortedEntries[i].Name] = i + 1;
                }
                
                // Add to rankings dictionary
                rankings[metric.Name] = metricRankings;
            }
            
            // Create tournament result
            var tournamentResult = new TournamentResult<T>(
                _entries.Select(e => e.Name).ToList(),
                _metrics.Select(m => m.Name).ToList(),
                results,
                aggregateMetrics,
                rankings);
            
            Results = tournamentResult;
            
            return tournamentResult;
        }

        /// <summary>
        /// Creates a trading-specific tournament.
        /// </summary>
        /// <typeparam name="TNumeric">The numeric type used for calculations.</typeparam>
        /// <param name="environment">The trading environment to use.</param>
        /// <param name="numEpisodes">The number of episodes to run for each model.</param>
        /// <returns>A tournament configured with trading-specific metrics.</returns>
        public static ModelTournament<TradingEnvironmentState<TNumeric>, object, TNumeric> CreateTradingTournament<TNumeric>(
            TradingEnvironment<TNumeric> environment,
            int numEpisodes = 10)
        {
            // Create standard trading metrics
            var metrics = new List<IEvaluationMetric<TNumeric>>
            {
                new TotalReturnMetric<TNumeric>(),
                new SharpeRatioMetric<TNumeric>(),
                new MaxDrawdownMetric<TNumeric>(),
                new CalmarRatioMetric<TNumeric>(),
                new SortinoRatioMetric<TNumeric>(),
                new WinRateMetric<TNumeric>()
            };
            
            return new ModelTournament<TradingEnvironmentState<TNumeric>, object, TNumeric>(
                environment,
                metrics,
                numEpisodes);
        }
    }
}