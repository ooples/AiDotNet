using System.Text;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ReinforcementLearning.Tournament.Results
{
    /// <summary>
    /// Represents the aggregate values for a metric.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class AggregateMetricValues<T>
    {
        /// <summary>
        /// Gets the mean value.
        /// </summary>
        public T Mean { get; }
        
        /// <summary>
        /// Gets the standard deviation.
        /// </summary>
        public T StdDev { get; }
        
        /// <summary>
        /// Gets the minimum value.
        /// </summary>
        public T Min { get; }
        
        /// <summary>
        /// Gets the maximum value.
        /// </summary>
        public T Max { get; }
        
        /// <summary>
        /// Gets all the individual values.
        /// </summary>
        public IReadOnlyList<T> Values { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="AggregateMetricValues{T}"/> class.
        /// </summary>
        /// <param name="mean">The mean value.</param>
        /// <param name="stdDev">The standard deviation.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
        /// <param name="values">All the individual values.</param>
        public AggregateMetricValues(T mean, T stdDev, T min, T max, IEnumerable<T> values)
        {
            Mean = mean;
            StdDev = stdDev;
            Min = min;
            Max = max;
            Values = values?.ToList() ?? throw new ArgumentNullException(nameof(values));
        }
    }

    /// <summary>
    /// Represents the results of a tournament.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class TournamentResult<T>
    {
        /// <summary>
        /// Gets the list of model names in the tournament.
        /// </summary>
        public IReadOnlyList<string> ModelNames { get; }
        
        /// <summary>
        /// Gets the list of metric names used in the tournament.
        /// </summary>
        public IReadOnlyList<string> MetricNames { get; }
        
        /// <summary>
        /// Gets the episode results for each model.
        /// </summary>
        public IReadOnlyDictionary<string, IReadOnlyList<ModelEpisodeResult<T>>> EpisodeResults { get; }
        
        /// <summary>
        /// Gets the aggregate metric values for each model and metric.
        /// </summary>
        public IReadOnlyDictionary<string, IReadOnlyDictionary<string, AggregateMetricValues<T>>> MetricValues { get; }
        
        /// <summary>
        /// Gets the rankings for each model on each metric.
        /// </summary>
        public IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> Rankings { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="TournamentResult{T}"/> class.
        /// </summary>
        /// <param name="modelNames">The list of model names in the tournament.</param>
        /// <param name="metricNames">The list of metric names used in the tournament.</param>
        /// <param name="episodeResults">The episode results for each model.</param>
        /// <param name="metricValues">The aggregate metric values for each model and metric.</param>
        /// <param name="rankings">The rankings for each model on each metric.</param>
        public TournamentResult(
            IReadOnlyList<string> modelNames,
            IReadOnlyList<string> metricNames,
            IReadOnlyDictionary<string, List<ModelEpisodeResult<T>>> episodeResults,
            IReadOnlyDictionary<string, Dictionary<string, AggregateMetricValues<T>>> metricValues,
            IReadOnlyDictionary<string, Dictionary<string, int>> rankings)
        {
            ModelNames = modelNames ?? throw new ArgumentNullException(nameof(modelNames));
            MetricNames = metricNames ?? throw new ArgumentNullException(nameof(metricNames));
            
            // Convert dictionaries to read-only dictionaries to satisfy interface
            EpisodeResults = episodeResults?.ToDictionary(
                kvp => kvp.Key,
                kvp => (IReadOnlyList<ModelEpisodeResult<T>>)kvp.Value) 
                ?? throw new ArgumentNullException(nameof(episodeResults));
            
            MetricValues = metricValues?.ToDictionary(
                kvp => kvp.Key,
                kvp => (IReadOnlyDictionary<string, AggregateMetricValues<T>>)kvp.Value)
                ?? throw new ArgumentNullException(nameof(metricValues));
            
            Rankings = rankings?.ToDictionary(
                kvp => kvp.Key,
                kvp => (IReadOnlyDictionary<string, int>)kvp.Value)
                ?? throw new ArgumentNullException(nameof(rankings));
        }

        /// <summary>
        /// Gets the overall rank for each model across all metrics.
        /// </summary>
        /// <returns>A dictionary mapping model names to their overall rank.</returns>
        public IReadOnlyDictionary<string, int> GetOverallRankings()
        {
            var rankSums = new Dictionary<string, int>();
            
            // Initialize rank sums
            foreach (var model in ModelNames)
            {
                rankSums[model] = 0;
            }
            
            // Sum up ranks across metrics
            foreach (var metricName in MetricNames)
            {
                var metricRankings = Rankings[metricName];
                
                foreach (var model in ModelNames)
                {
                    rankSums[model] += metricRankings[model];
                }
            }
            
            // Sort models by rank sum (lower is better)
            var sortedModels = rankSums
                .OrderBy(kvp => kvp.Value)
                .Select(kvp => kvp.Key)
                .ToList();
            
            // Assign overall ranks
            var overallRankings = new Dictionary<string, int>();
            for (int i = 0; i < sortedModels.Count; i++)
            {
                overallRankings[sortedModels[i]] = i + 1;
            }
            
            return overallRankings;
        }

        /// <summary>
        /// Returns a string representation of the tournament results.
        /// </summary>
        /// <returns>A string representation of the tournament results.</returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            
            sb.AppendLine("Tournament Results");
            sb.AppendLine("=================");
            
            // Overall rankings
            var overallRankings = GetOverallRankings();
            
            sb.AppendLine("\nOverall Rankings:");
            foreach (var model in overallRankings.OrderBy(kvp => kvp.Value).Select(kvp => kvp.Key))
            {
                sb.AppendLine($"{overallRankings[model]}. {model}");
            }
            
            // Metric results
            sb.AppendLine("\nMetric Results:");
            foreach (var metric in MetricNames)
            {
                sb.AppendLine($"\n{metric}:");
                
                // Sort models by rank for this metric
                var metricRankings = Rankings[metric];
                var sortedModels = metricRankings
                    .OrderBy(kvp => kvp.Value)
                    .Select(kvp => kvp.Key)
                    .ToList();
                
                foreach (var model in sortedModels)
                {
                    var values = MetricValues[model][metric];
                    sb.AppendLine($"{metricRankings[model]}. {model}: {Convert.ToDouble(values.Mean):F4} ± {Convert.ToDouble(values.StdDev):F4} (min: {Convert.ToDouble(values.Min):F4}, max: {Convert.ToDouble(values.Max):F4})");
                }
            }
            
            return sb.ToString();
        }
    }
}