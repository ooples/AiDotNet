using AiDotNet.ReinforcementLearning.Tournament.Results;
using AiDotNet.Helpers;
using System.Text;

namespace AiDotNet.ReinforcementLearning.Tournament.Visualization
{
    /// <summary>
    /// Provides visualization tools for tournament results.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class TournamentVisualizer<T>
    {
        private readonly TournamentResult<T> _results = default!;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="TournamentVisualizer{T}"/> class.
        /// </summary>
        /// <param name="results">The tournament results to visualize.</param>
        public TournamentVisualizer(TournamentResult<T> results)
        {
            _results = results ?? throw new ArgumentNullException(nameof(results));
        }

        /// <summary>
        /// Generates a summary table of the tournament results.
        /// </summary>
        /// <param name="includeDetails">Whether to include detailed statistics for each metric.</param>
        /// <returns>A string containing the formatted summary table.</returns>
        public string GenerateSummaryTable(bool includeDetails = false)
        {
            var sb = new StringBuilder();
            
            // Calculate overall rankings
            var overallRankings = _results.GetOverallRankings();
            
            // Generate header row
            sb.Append("Model Name");
            sb.Append("\t| Overall Rank");
            
            foreach (var metric in _results.MetricNames)
            {
                sb.Append($"\t| {metric} Rank");
                if (includeDetails)
                {
                    sb.Append($"\t| {metric} Mean");
                    sb.Append($"\t| {metric} StdDev");
                }
            }
            sb.AppendLine();
            
            // Add separator line
            sb.Append(new string('-', 20));
            sb.Append("\t| ------------");
            
            foreach (var metric in _results.MetricNames)
            {
                sb.Append("\t| -----------");
                if (includeDetails)
                {
                    sb.Append("\t| -----------");
                    sb.Append("\t| -----------");
                }
            }
            sb.AppendLine();
            
            // Generate rows for each model, sorted by overall rank
            foreach (var model in overallRankings.OrderBy(kvp => kvp.Value).Select(kvp => kvp.Key))
            {
                sb.Append(model.PadRight(20).Substring(0, 20));
                sb.Append($"\t| {overallRankings[model]}");
                
                foreach (var metric in _results.MetricNames)
                {
                    var metricRanking = _results.Rankings[metric][model];
                    sb.Append($"\t| {metricRanking}");
                    
                    if (includeDetails)
                    {
                        var metricValues = _results.MetricValues[model][metric];
                        sb.Append($"\t| {Convert.ToDouble(metricValues.Mean):F4}");
                        sb.Append($"\t| {Convert.ToDouble(metricValues.StdDev):F4}");
                    }
                }
                sb.AppendLine();
            }
            
            return sb.ToString();
        }

        /// <summary>
        /// Generates an ASCII bar chart for a specific metric.
        /// </summary>
        /// <param name="metricName">The name of the metric to visualize.</param>
        /// <param name="width">The width of the chart in characters.</param>
        /// <returns>A string containing the formatted bar chart.</returns>
        public string GenerateBarChart(string metricName, int width = 60)
        {
            if (!_results.MetricNames.Contains(metricName))
            {
                throw new ArgumentException($"Metric '{metricName}' not found in tournament results.", nameof(metricName));
            }
            
            var sb = new StringBuilder();
            
            // Get metric values for each model
            var modelValues = new Dictionary<string, double>();
            foreach (var model in _results.ModelNames)
            {
                var metricValue = _results.MetricValues[model][metricName];
                modelValues[model] = Convert.ToDouble(metricValue.Mean);
            }
            
            // Determine min and max values
            double minValue = modelValues.Values.Min();
            double maxValue = modelValues.Values.Max();
            
            // Adjust for negative values if needed
            double range = maxValue - minValue;
            if (range <= 0)
            {
                range = 1; // Avoid division by zero
            }
            
            // Sort models by value (descending for higher is better, ascending for lower is better)
            var metric = _results.MetricNames.First(m => m == metricName);
            var isHigherBetter = true; // Default to higher is better
            
            // Calculate if higher is better based on rankings
            if (_results.Rankings.TryGetValue(metric, out var rankings))
            {
                // Get the model with rank 1
                var bestModel = rankings.OrderBy(r => r.Value).First().Key;
                var bestValue = modelValues[bestModel];
                
                // If the best value is the minimum value, then lower is better
                isHigherBetter = Math.Abs(bestValue - maxValue) < Math.Abs(bestValue - minValue);
            }
            
            // Sort models by value
            IOrderedEnumerable<KeyValuePair<string, double>> sortedModels;
            if (isHigherBetter)
            {
                sortedModels = modelValues.OrderByDescending(kvp => kvp.Value);
            }
            else
            {
                sortedModels = modelValues.OrderBy(kvp => kvp.Value);
            }
            
            // Generate chart title
            sb.AppendLine($"{metricName} Comparison:");
            sb.AppendLine();
            
            // Generate chart bars
            foreach (var kvp in sortedModels)
            {
                string model = kvp.Key;
                double value = kvp.Value;
                
                // Calculate bar length
                int barLength = (int)((value - minValue) / range * (width - 20));
                barLength = Math.Max(1, barLength); // Ensure at least length 1
                
                // Add model name and value
                sb.Append(model.PadRight(20).Substring(0, 20));
                sb.Append(" ");
                
                // Add bar
                sb.Append(new string('#', barLength));
                
                // Add value at end of bar
                sb.Append($" {value:F4}");
                
                sb.AppendLine();
            }
            
            // Add scale
            sb.AppendLine();
            sb.Append("Scale: ".PadRight(21));
            sb.Append(minValue.ToString("F2").PadRight(10));
            sb.Append(new string('-', width - 40));
            sb.Append(maxValue.ToString("F2").PadLeft(10));
            sb.AppendLine();
            
            return sb.ToString();
        }

        /// <summary>
        /// Generates a returns chart for a specific model.
        /// </summary>
        /// <param name="modelName">The name of the model to visualize.</param>
        /// <param name="episodeIndex">The index of the episode to visualize.</param>
        /// <param name="height">The height of the chart in characters.</param>
        /// <param name="width">The width of the chart in characters.</param>
        /// <returns>A string containing the formatted returns chart.</returns>
        public string GenerateReturnsChart(string modelName, int episodeIndex, int height = 20, int width = 80)
        {
            if (!_results.ModelNames.Contains(modelName))
            {
                throw new ArgumentException($"Model '{modelName}' not found in tournament results.", nameof(modelName));
            }
            
            var episodes = _results.EpisodeResults[modelName];
            if (episodeIndex < 0 || episodeIndex >= episodes.Count)
            {
                throw new ArgumentException($"Episode index {episodeIndex} is out of range. Available episodes: 0 to {episodes.Count - 1}.", nameof(episodeIndex));
            }
            
            var episode = episodes[episodeIndex];
            var rewards = episode.Rewards;
            
            if (rewards.Count == 0)
            {
                return "No reward data available for this episode.";
            }
            
            var sb = new StringBuilder();
            
            // Generate chart title
            sb.AppendLine($"Returns Chart for {modelName} (Episode {episodeIndex}):");
            sb.AppendLine();
            
            // Convert rewards to cumulative returns
            var cumulativeReturns = new List<double>();
            double cumulativeReturn = 1.0;
            
            foreach (var reward in rewards)
            {
                cumulativeReturn *= (1.0 + Convert.ToDouble(reward));
                cumulativeReturns.Add(cumulativeReturn);
            }
            
            // Determine min and max values
            double minValue = cumulativeReturns.Min();
            double maxValue = cumulativeReturns.Max();
            double range = maxValue - minValue;
            
            if (range <= 0)
            {
                range = 1; // Avoid division by zero
            }
            
            // Initialize chart
            var chart = new char[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    chart[i, j] = ' ';
                }
            }
            
            // Draw x and y axes
            for (int i = 0; i < height; i++)
            {
                chart[i, 0] = '|';
            }
            
            for (int j = 0; j < width; j++)
            {
                chart[height - 1, j] = '-';
            }
            
            chart[height - 1, 0] = '+';
            
            // Plot the returns
            int numPoints = cumulativeReturns.Count;
            double xScale = (double)(width - 1) / numPoints;
            
            for (int i = 0; i < numPoints; i++)
            {
                int x = (int)(i * xScale);
                int y = height - 1 - (int)((cumulativeReturns[i] - minValue) / range * (height - 2));
                
                y = Math.Max(0, Math.Min(height - 1, y));
                x = Math.Max(0, Math.Min(width - 1, x));
                
                chart[y, x] = '*';
            }
            
            // Add y-axis labels
            sb.Append($"{maxValue:F2} ");
            for (int j = 0; j < width; j++)
            {
                sb.Append(chart[0, j]);
            }
            sb.AppendLine();
            
            for (int i = 1; i < height - 1; i++)
            {
                if (i == height / 2)
                {
                    sb.Append($"{(minValue + maxValue) / 2:F2} ");
                }
                else
                {
                    sb.Append("      ");
                }
                
                for (int j = 0; j < width; j++)
                {
                    sb.Append(chart[i, j]);
                }
                sb.AppendLine();
            }
            
            sb.Append($"{minValue:F2} ");
            for (int j = 0; j < width; j++)
            {
                sb.Append(chart[height - 1, j]);
            }
            sb.AppendLine();
            
            // Add x-axis labels
            sb.Append("      ");
            sb.Append("0".PadRight((int)(width / 3) - 1));
            sb.Append((numPoints / 2).ToString().PadRight((int)(width / 3)));
            sb.Append(numPoints.ToString());
            sb.AppendLine();
            
            // Add final return
            sb.AppendLine();
            sb.AppendLine($"Final Return: {cumulativeReturns.Last():P2}");
            
            return sb.ToString();
        }

        /// <summary>
        /// Generates comparison charts for all metrics.
        /// </summary>
        /// <param name="width">The width of the charts in characters.</param>
        /// <returns>A string containing formatted comparison charts for all metrics.</returns>
        public string GenerateAllMetricCharts(int width = 60)
        {
            var sb = new StringBuilder();
            
            // Generate overall ranking chart
            sb.AppendLine("Overall Rankings:");
            sb.AppendLine();
            
            var overallRankings = _results.GetOverallRankings();
            foreach (var model in overallRankings.OrderBy(kvp => kvp.Value).Select(kvp => kvp.Key))
            {
                int rank = overallRankings[model];
                int barLength = width - 25;
                int filledBar = barLength - ((rank - 1) * barLength / (_results.ModelNames.Count - 1));
                filledBar = Math.Max(1, filledBar);
                
                sb.Append(model.PadRight(20).Substring(0, 20));
                sb.Append($" {rank} ");
                sb.Append(new string('#', filledBar));
                sb.Append(new string('.', barLength - filledBar));
                sb.AppendLine();
            }
            
            sb.AppendLine();
            sb.AppendLine();
            
            // Generate charts for each metric
            foreach (var metric in _results.MetricNames)
            {
                sb.AppendLine(GenerateBarChart(metric, width));
                sb.AppendLine();
            }
            
            return sb.ToString();
        }

        /// <summary>
        /// Exports the tournament results to a CSV file.
        /// </summary>
        /// <param name="filePath">The path to save the CSV file.</param>
        /// <param name="includeEpisodeDetails">Whether to include detailed episode results.</param>
        public void ExportToCsv(string filePath, bool includeEpisodeDetails = false)
        {
            var sb = new StringBuilder();
            
            // Generate header row for main results
            sb.Append("Model");
            sb.Append(",Overall Rank");
            
            foreach (var metric in _results.MetricNames)
            {
                sb.Append($",{metric} Rank");
                sb.Append($",{metric} Mean");
                sb.Append($",{metric} StdDev");
                sb.Append($",{metric} Min");
                sb.Append($",{metric} Max");
            }
            sb.AppendLine();
            
            // Calculate overall rankings
            var overallRankings = _results.GetOverallRankings();
            
            // Generate rows for each model
            foreach (var model in _results.ModelNames)
            {
                sb.Append($"\"{model}\"");
                sb.Append($",{overallRankings[model]}");
                
                foreach (var metric in _results.MetricNames)
                {
                    var metricRanking = _results.Rankings[metric][model];
                    var metricValues = _results.MetricValues[model][metric];
                    
                    sb.Append($",{metricRanking}");
                    sb.Append($",{Convert.ToDouble(metricValues.Mean)}");
                    sb.Append($",{Convert.ToDouble(metricValues.StdDev)}");
                    sb.Append($",{Convert.ToDouble(metricValues.Min)}");
                    sb.Append($",{Convert.ToDouble(metricValues.Max)}");
                }
                sb.AppendLine();
            }
            
            // Add episode details if requested
            if (includeEpisodeDetails)
            {
                sb.AppendLine();
                sb.AppendLine("Episode Details:");
                
                // For each model
                foreach (var model in _results.ModelNames)
                {
                    sb.AppendLine();
                    sb.AppendLine($"Model: {model}");
                    
                    var episodes = _results.EpisodeResults[model];
                    
                    // For each episode
                    for (int e = 0; e < episodes.Count; e++)
                    {
                        sb.AppendLine($"Episode {e}:");
                        sb.Append("Step,Reward");
                        
                        // Add metric values for this episode
                        foreach (var metric in _results.MetricNames)
                        {
                            sb.Append($",{metric}");
                        }
                        sb.AppendLine();
                        
                        var episode = episodes[e];
                        
                        // Calculate metric values for this episode
                        var metricValues = new Dictionary<string, T>();
                        foreach (var metric in _results.MetricNames)
                        {
                            try
                            {
                                // Try to get the existing metric
                                metricValues[metric] = _results.MetricValues[model][metric].Values[e];
                            }
                            catch
                            {
                                // If there's an issue, default to zero
                                metricValues[metric] = MathHelper.GetNumericOperations<T>().Zero;
                            }
                        }
                        
                        // For each step
                        for (int s = 0; s < episode.Rewards.Count; s++)
                        {
                            sb.Append($"{s},{Convert.ToDouble(episode.Rewards[s])}");
                            
                            // Add the same metric values for each step (they're episode-level)
                            foreach (var metric in _results.MetricNames)
                            {
                                sb.Append($",{Convert.ToDouble(metricValues[metric])}");
                            }
                            sb.AppendLine();
                        }
                    }
                }
            }
            
            // Write to file
            File.WriteAllText(filePath, sb.ToString());
        }
    }
}