using AiDotNet.Enums;
using AiDotNet.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Represents a branch in a pipeline that can execute steps conditionally or in parallel
    /// </summary>
    public class PipelineBranch
    {
        private readonly List<IPipelineStep> _steps;
        private readonly Predicate<double[][]>? _condition;
        private BranchMergeStrategy _mergeStrategy;
        private Dictionary<string, object>? _mergeParameters;

        /// <summary>
        /// Gets the unique identifier for this branch
        /// </summary>
        public string Id { get; }

        /// <summary>
        /// Gets or sets the name of this branch
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets the steps in this branch
        /// </summary>
        public IReadOnlyList<IPipelineStep> Steps => _steps.AsReadOnly();

        /// <summary>
        /// Gets whether this branch is conditional
        /// </summary>
        public bool IsConditional => _condition != null;

        /// <summary>
        /// Gets or sets the merge strategy for this branch
        /// </summary>
        public BranchMergeStrategy MergeStrategy
        {
            get => _mergeStrategy;
            set => _mergeStrategy = value;
        }

        /// <summary>
        /// Gets or sets whether this branch should execute in parallel with others
        /// </summary>
        public bool ExecuteInParallel { get; set; }

        /// <summary>
        /// Gets or sets the priority of this branch (higher priority executes first)
        /// </summary>
        public int Priority { get; set; }

        /// <summary>
        /// Gets or sets whether this branch is enabled
        /// </summary>
        public bool IsEnabled { get; set; }

        /// <summary>
        /// Gets execution statistics for this branch
        /// </summary>
        public BranchStatistics Statistics { get; }

        /// <summary>
        /// Branch execution statistics
        /// </summary>
        public class BranchStatistics
        {
            public int ExecutionCount { get; set; }
            public int SkippedCount { get; set; }
            public TimeSpan TotalExecutionTime { get; set; }
            public TimeSpan AverageExecutionTime => ExecutionCount > 0 
                ? TimeSpan.FromMilliseconds(TotalExecutionTime.TotalMilliseconds / ExecutionCount) 
                : TimeSpan.Zero;
            public DateTime? LastExecutedAt { get; set; }
        }

        /// <summary>
        /// Initializes a new instance of the PipelineBranch class
        /// </summary>
        /// <param name="name">Name of the branch</param>
        /// <param name="condition">Optional condition for branch execution</param>
        public PipelineBranch(string name, Predicate<double[][]>? condition = null)
        {
            Id = Guid.NewGuid().ToString();
            Name = name;
            _steps = new List<IPipelineStep>();
            _condition = condition;
            _mergeStrategy = BranchMergeStrategy.Concatenate;
            ExecuteInParallel = false;
            Priority = 0;
            IsEnabled = true;
            Statistics = new BranchStatistics();
        }

        /// <summary>
        /// Adds a step to this branch
        /// </summary>
        /// <param name="step">The pipeline step to add</param>
        public void AddStep(IPipelineStep step)
        {
            if (step == null)
            {
                throw new ArgumentNullException(nameof(step));
            }

            _steps.Add(step);
        }

        /// <summary>
        /// Removes a step from this branch
        /// </summary>
        /// <param name="step">The pipeline step to remove</param>
        /// <returns>True if the step was removed, false otherwise</returns>
        public bool RemoveStep(IPipelineStep step)
        {
            return _steps.Remove(step);
        }

        /// <summary>
        /// Clears all steps from this branch
        /// </summary>
        public void ClearSteps()
        {
            _steps.Clear();
        }

        /// <summary>
        /// Sets the merge parameters for this branch
        /// </summary>
        /// <param name="parameters">Merge parameters</param>
        public void SetMergeParameters(Dictionary<string, object> parameters)
        {
            _mergeParameters = parameters ?? new Dictionary<string, object>();
        }

        /// <summary>
        /// Evaluates whether this branch should execute for the given data
        /// </summary>
        /// <param name="data">Input data</param>
        /// <returns>True if the branch should execute, false otherwise</returns>
        public bool ShouldExecute(double[][] data)
        {
            if (!IsEnabled)
            {
                return false;
            }

            if (_condition == null)
            {
                return true;
            }

            try
            {
                return _condition(data);
            }
            catch
            {
                // If condition evaluation fails, don't execute the branch
                return false;
            }
        }

        /// <summary>
        /// Executes the branch pipeline on the given data
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Optional target data</param>
        /// <returns>Transformed data</returns>
        public async Task<double[][]> ExecuteAsync(double[][] inputs, double[]? targets = null)
        {
            var startTime = DateTime.UtcNow;

            try
            {
                if (!ShouldExecute(inputs))
                {
                    Statistics.SkippedCount++;
                    return inputs;
                }

                var currentData = inputs;

                // Execute each step in sequence
                foreach (var step in _steps)
                {
                    if (!step.IsFitted && targets != null)
                    {
                        await step.FitAsync(currentData, targets).ConfigureAwait(false);
                    }

                    currentData = await step.TransformAsync(currentData).ConfigureAwait(false);
                }

                Statistics.ExecutionCount++;
                Statistics.LastExecutedAt = DateTime.UtcNow;
                return currentData;
            }
            finally
            {
                Statistics.TotalExecutionTime += DateTime.UtcNow - startTime;
            }
        }

        /// <summary>
        /// Fits all steps in the branch
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Optional target data</param>
        public async Task FitAsync(double[][] inputs, double[]? targets = null)
        {
            if (!ShouldExecute(inputs))
            {
                return;
            }

            var currentData = inputs;

            foreach (var step in _steps)
            {
                await step.FitAsync(currentData, targets).ConfigureAwait(false);
                
                // Transform data for next step
                if (_steps.IndexOf(step) < _steps.Count - 1)
                {
                    currentData = await step.TransformAsync(currentData).ConfigureAwait(false);
                }
            }
        }

        /// <summary>
        /// Merges multiple branch results according to the merge strategy
        /// </summary>
        /// <param name="results">Results from multiple branches</param>
        /// <returns>Merged result</returns>
        public static double[][] MergeBranchResults(List<(PipelineBranch branch, double[][] data)> results)
        {
            if (results.Count == 0)
            {
                throw new ArgumentException("No results to merge", nameof(results));
            }

            if (results.Count == 1)
            {
                return results[0].data;
            }

            // Group by merge strategy
            var strategyGroups = results.GroupBy(r => r.branch.MergeStrategy).ToList();

            if (strategyGroups.Count > 1)
            {
                // Multiple strategies - merge each group then combine
                var groupResults = new List<double[][]>();
                
                foreach (var group in strategyGroups)
                {
                    var groupData = group.Select(g => g.data).ToList();
                    var merged = ApplyMergeStrategy(group.Key, groupData, group.First().branch._mergeParameters);
                    groupResults.Add(merged);
                }

                // Final merge using concatenation
                return ApplyMergeStrategy(BranchMergeStrategy.Concatenate, groupResults, null);
            }
            else
            {
                // Single strategy - merge all at once
                var strategy = strategyGroups[0].Key;
                var allData = results.Select(r => r.data).ToList();
                return ApplyMergeStrategy(strategy, allData, results[0].branch._mergeParameters);
            }
        }

        /// <summary>
        /// Applies a specific merge strategy to combine results
        /// </summary>
        private static double[][] ApplyMergeStrategy(BranchMergeStrategy strategy, 
            List<double[][]> results, Dictionary<string, object>? parameters)
        {
            switch (strategy)
            {
                case BranchMergeStrategy.Concatenate:
                    return MergeConcatenate(results);

                case BranchMergeStrategy.Average:
                    return MergeAverage(results);

                case BranchMergeStrategy.WeightedAverage:
                    return MergeWeightedAverage(results, parameters);

                case BranchMergeStrategy.Maximum:
                    return MergeMaximum(results);

                case BranchMergeStrategy.Minimum:
                    return MergeMinimum(results);

                case BranchMergeStrategy.Sum:
                    return MergeSum(results);

                case BranchMergeStrategy.Product:
                    return MergeProduct(results);

                case BranchMergeStrategy.Voting:
                    return MergeVoting(results, parameters);

                case BranchMergeStrategy.FirstCompleted:
                    return results[0]; // Assumes first in list completed first

                case BranchMergeStrategy.BestPerforming:
                    return MergeBestPerforming(results, parameters);

                case BranchMergeStrategy.LogicalAnd:
                    return MergeLogicalAnd(results);

                case BranchMergeStrategy.LogicalOr:
                    return MergeLogicalOr(results);

                default:
                    return MergeConcatenate(results);
            }
        }

        /// <summary>
        /// Concatenates results horizontally (adds features)
        /// </summary>
        private static double[][] MergeConcatenate(List<double[][]> results)
        {
            int numSamples = results[0].Length;
            int totalFeatures = results.Sum(r => r[0].Length);

            var merged = new double[numSamples][];
            
            for (int i = 0; i < numSamples; i++)
            {
                merged[i] = new double[totalFeatures];
                int currentIndex = 0;

                foreach (var result in results)
                {
                    Array.Copy(result[i], 0, merged[i], currentIndex, result[i].Length);
                    currentIndex += result[i].Length;
                }
            }

            return merged;
        }

        /// <summary>
        /// Averages results element-wise
        /// </summary>
        private static double[][] MergeAverage(List<double[][]> results)
        {
            int numSamples = results[0].Length;
            int numFeatures = results[0][0].Length;

            var merged = new double[numSamples][];
            
            for (int i = 0; i < numSamples; i++)
            {
                merged[i] = new double[numFeatures];
                
                for (int j = 0; j < numFeatures; j++)
                {
                    double sum = 0;
                    foreach (var result in results)
                    {
                        sum += result[i][j];
                    }
                    merged[i][j] = sum / results.Count;
                }
            }

            return merged;
        }

        /// <summary>
        /// Weighted average of results
        /// </summary>
        private static double[][] MergeWeightedAverage(List<double[][]> results, Dictionary<string, object>? parameters)
        {
            var weights = parameters?.GetValueOrDefault("Weights") as double[] 
                ?? Enumerable.Repeat(1.0 / results.Count, results.Count).ToArray();

            int numSamples = results[0].Length;
            int numFeatures = results[0][0].Length;

            var merged = new double[numSamples][];
            
            for (int i = 0; i < numSamples; i++)
            {
                merged[i] = new double[numFeatures];
                
                for (int j = 0; j < numFeatures; j++)
                {
                    double weightedSum = 0;
                    for (int k = 0; k < results.Count; k++)
                    {
                        weightedSum += results[k][i][j] * weights[k];
                    }
                    merged[i][j] = weightedSum;
                }
            }

            return merged;
        }

        /// <summary>
        /// Takes maximum value element-wise
        /// </summary>
        private static double[][] MergeMaximum(List<double[][]> results)
        {
            int numSamples = results[0].Length;
            int numFeatures = results[0][0].Length;

            var merged = new double[numSamples][];
            
            for (int i = 0; i < numSamples; i++)
            {
                merged[i] = new double[numFeatures];
                
                for (int j = 0; j < numFeatures; j++)
                {
                    merged[i][j] = results.Max(r => r[i][j]);
                }
            }

            return merged;
        }

        /// <summary>
        /// Takes minimum value element-wise
        /// </summary>
        private static double[][] MergeMinimum(List<double[][]> results)
        {
            int numSamples = results[0].Length;
            int numFeatures = results[0][0].Length;

            var merged = new double[numSamples][];
            
            for (int i = 0; i < numSamples; i++)
            {
                merged[i] = new double[numFeatures];
                
                for (int j = 0; j < numFeatures; j++)
                {
                    merged[i][j] = results.Min(r => r[i][j]);
                }
            }

            return merged;
        }

        /// <summary>
        /// Sums results element-wise
        /// </summary>
        private static double[][] MergeSum(List<double[][]> results)
        {
            int numSamples = results[0].Length;
            int numFeatures = results[0][0].Length;

            var merged = new double[numSamples][];
            
            for (int i = 0; i < numSamples; i++)
            {
                merged[i] = new double[numFeatures];
                
                for (int j = 0; j < numFeatures; j++)
                {
                    merged[i][j] = results.Sum(r => r[i][j]);
                }
            }

            return merged;
        }

        /// <summary>
        /// Multiplies results element-wise
        /// </summary>
        private static double[][] MergeProduct(List<double[][]> results)
        {
            int numSamples = results[0].Length;
            int numFeatures = results[0][0].Length;

            var merged = new double[numSamples][];
            
            for (int i = 0; i < numSamples; i++)
            {
                merged[i] = new double[numFeatures];
                
                for (int j = 0; j < numFeatures; j++)
                {
                    double product = 1.0;
                    foreach (var result in results)
                    {
                        product *= result[i][j];
                    }
                    merged[i][j] = product;
                }
            }

            return merged;
        }

        /// <summary>
        /// Voting-based merge (for classification-like outputs)
        /// </summary>
        private static double[][] MergeVoting(List<double[][]> results, Dictionary<string, object>? parameters)
        {
            int numSamples = results[0].Length;
            int numFeatures = results[0][0].Length;
            var threshold = parameters?.GetValueOrDefault("Threshold") as double? ?? 0.5;

            var merged = new double[numSamples][];
            
            for (int i = 0; i < numSamples; i++)
            {
                merged[i] = new double[numFeatures];
                
                for (int j = 0; j < numFeatures; j++)
                {
                    int votes = results.Count(r => r[i][j] > threshold);
                    merged[i][j] = votes > results.Count / 2 ? 1.0 : 0.0;
                }
            }

            return merged;
        }

        /// <summary>
        /// Selects best performing result based on a metric
        /// </summary>
        private static double[][] MergeBestPerforming(List<double[][]> results, Dictionary<string, object>? parameters)
        {
            // If no metric provided, return first result
            var metricFunc = parameters?.GetValueOrDefault("MetricFunction") as Func<double[][], double>;
            if (metricFunc == null)
            {
                return results[0];
            }

            double bestScore = double.MinValue;
            double[][] bestResult = results[0];

            foreach (var result in results)
            {
                double score = metricFunc(result);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestResult = result;
                }
            }

            return bestResult;
        }

        /// <summary>
        /// Logical AND merge (for binary outputs)
        /// </summary>
        private static double[][] MergeLogicalAnd(List<double[][]> results)
        {
            int numSamples = results[0].Length;
            int numFeatures = results[0][0].Length;

            var merged = new double[numSamples][];
            
            for (int i = 0; i < numSamples; i++)
            {
                merged[i] = new double[numFeatures];
                
                for (int j = 0; j < numFeatures; j++)
                {
                    merged[i][j] = results.All(r => r[i][j] > 0.5) ? 1.0 : 0.0;
                }
            }

            return merged;
        }

        /// <summary>
        /// Logical OR merge (for binary outputs)
        /// </summary>
        private static double[][] MergeLogicalOr(List<double[][]> results)
        {
            int numSamples = results[0].Length;
            int numFeatures = results[0][0].Length;

            var merged = new double[numSamples][];
            
            for (int i = 0; i < numSamples; i++)
            {
                merged[i] = new double[numFeatures];
                
                for (int j = 0; j < numFeatures; j++)
                {
                    merged[i][j] = results.Any(r => r[i][j] > 0.5) ? 1.0 : 0.0;
                }
            }

            return merged;
        }

        /// <summary>
        /// Creates a clone of this branch
        /// </summary>
        /// <returns>A new branch with the same configuration</returns>
        public PipelineBranch Clone()
        {
            var clone = new PipelineBranch(Name, _condition)
            {
                MergeStrategy = MergeStrategy,
                ExecuteInParallel = ExecuteInParallel,
                Priority = Priority,
                IsEnabled = IsEnabled
            };

            foreach (var step in _steps)
            {
                clone.AddStep(step);
            }

            if (_mergeParameters != null)
            {
                clone.SetMergeParameters(new Dictionary<string, object>(_mergeParameters));
            }

            return clone;
        }
    }
}