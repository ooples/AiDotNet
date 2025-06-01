using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Represents the result of a pipeline execution
    /// </summary>
    public class PipelineResult
    {
        /// <summary>
        /// Gets or sets the unique identifier for this pipeline execution
        /// </summary>
        public string PipelineId { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the name of the pipeline
        /// </summary>
        public string PipelineName { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the start time of the pipeline execution
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets the end time of the pipeline execution
        /// </summary>
        public DateTime EndTime { get; set; }

        /// <summary>
        /// Gets the total duration of the pipeline execution
        /// </summary>
        public TimeSpan Duration => EndTime - StartTime;

        /// <summary>
        /// Gets or sets whether the pipeline execution was successful
        /// </summary>
        public bool IsSuccessful { get; set; }

        /// <summary>
        /// Gets or sets the error message if the pipeline failed
        /// </summary>
        public string? ErrorMessage { get; set; }

        /// <summary>
        /// Gets or sets the name of the step that failed (if any)
        /// </summary>
        public string? FailedStep { get; set; }

        /// <summary>
        /// Gets or sets the input data shape [samples, features]
        /// </summary>
        public int[] InputShape { get; set; } = Array.Empty<int>();

        /// <summary>
        /// Gets or sets the output data shape [samples, features]
        /// </summary>
        public int[] OutputShape { get; set; } = Array.Empty<int>();

        /// <summary>
        /// Gets or sets the final output data
        /// </summary>
        public double[][]? OutputData { get; set; }

        /// <summary>
        /// Gets the list of step execution results
        /// </summary>
        public List<StepResult> StepResults { get; } = new List<StepResult>();

        /// <summary>
        /// Gets the list of branch execution results
        /// </summary>
        public List<BranchResult> BranchResults { get; } = new List<BranchResult>();

        /// <summary>
        /// Gets or sets performance metrics for the pipeline
        /// </summary>
        public PerformanceMetrics Performance { get; set; } = new PerformanceMetrics();

        /// <summary>
        /// Gets or sets custom metadata for the pipeline execution
        /// </summary>
        public Dictionary<string, object> Metadata { get; } = new Dictionary<string, object>();

        /// <summary>
        /// Represents the result of a single step execution
        /// </summary>
        public class StepResult
        {
            /// <summary>
            /// Gets or sets the step name
            /// </summary>
            public string StepName { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets the step execution duration
            /// </summary>
            public TimeSpan Duration { get; set; }

            /// <summary>
            /// Gets or sets whether the step was successful
            /// </summary>
            public bool IsSuccessful { get; set; }

            /// <summary>
            /// Gets or sets the error message if the step failed
            /// </summary>
            public string? ErrorMessage { get; set; }

            /// <summary>
            /// Gets or sets the output shape after this step
            /// </summary>
            public int[] OutputShape { get; set; } = Array.Empty<int>();

            /// <summary>
            /// Gets or sets step-specific metrics
            /// </summary>
            public Dictionary<string, double> Metrics { get; } = new Dictionary<string, double>();
        }

        /// <summary>
        /// Represents the result of a branch execution
        /// </summary>
        public class BranchResult
        {
            /// <summary>
            /// Gets or sets the branch name
            /// </summary>
            public string BranchName { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets whether the branch was executed
            /// </summary>
            public bool WasExecuted { get; set; }

            /// <summary>
            /// Gets or sets the branch execution duration
            /// </summary>
            public TimeSpan Duration { get; set; }

            /// <summary>
            /// Gets or sets branch-specific metadata
            /// </summary>
            public Dictionary<string, object> Metadata { get; } = new Dictionary<string, object>();
        }

        /// <summary>
        /// Performance metrics for the pipeline execution
        /// </summary>
        public class PerformanceMetrics
        {
            /// <summary>
            /// Gets or sets the peak memory usage in bytes
            /// </summary>
            public long PeakMemoryUsageBytes { get; set; }

            /// <summary>
            /// Gets or sets the average CPU usage percentage
            /// </summary>
            public double AverageCpuUsage { get; set; }

            /// <summary>
            /// Gets or sets the total number of samples processed
            /// </summary>
            public int TotalSamplesProcessed { get; set; }

            /// <summary>
            /// Gets or sets the throughput (samples per second)
            /// </summary>
            public double Throughput { get; set; }

            /// <summary>
            /// Gets or sets the number of cache hits (if caching was used)
            /// </summary>
            public int CacheHits { get; set; }

            /// <summary>
            /// Gets or sets the number of cache misses (if caching was used)
            /// </summary>
            public int CacheMisses { get; set; }

            /// <summary>
            /// Gets the cache hit rate
            /// </summary>
            public double CacheHitRate => 
                (CacheHits + CacheMisses) > 0 ? (double)CacheHits / (CacheHits + CacheMisses) : 0;

            /// <summary>
            /// Gets or sets custom performance metrics
            /// </summary>
            public Dictionary<string, double> CustomMetrics { get; } = new Dictionary<string, double>();
        }

        /// <summary>
        /// Gets a summary of the pipeline execution
        /// </summary>
        /// <returns>A string summary of the execution</returns>
        public string GetSummary()
        {
            var summary = new StringBuilder();
            summary.AppendLine($"Pipeline Execution Summary - {PipelineName}");
            summary.AppendLine(new string('=', 60));
            summary.AppendLine($"Execution ID: {PipelineId}");
            summary.AppendLine($"Status: {(IsSuccessful ? "✓ Success" : "✗ Failed")}");
            summary.AppendLine($"Duration: {Duration.TotalSeconds:F2} seconds");
            summary.AppendLine($"Start Time: {StartTime:yyyy-MM-dd HH:mm:ss}");
            summary.AppendLine($"End Time: {EndTime:yyyy-MM-dd HH:mm:ss}");
            
            if (!IsSuccessful)
            {
                summary.AppendLine($"Failed Step: {FailedStep}");
                summary.AppendLine($"Error: {ErrorMessage}");
            }

            summary.AppendLine($"\nData Shape: [{string.Join(", ", InputShape)}] → [{string.Join(", ", OutputShape)}]");

            if (StepResults.Count > 0)
            {
                summary.AppendLine("\nStep Execution Details:");
                summary.AppendLine(new string('-', 60));
                
                foreach (var step in StepResults)
                {
                    var status = step.IsSuccessful ? "✓" : "✗";
                    summary.AppendLine($"{status} {step.StepName,-30} {step.Duration.TotalMilliseconds,10:F2} ms");
                    
                    if (!step.IsSuccessful && !string.IsNullOrEmpty(step.ErrorMessage))
                    {
                        summary.AppendLine($"  Error: {step.ErrorMessage}");
                    }
                }
            }

            if (BranchResults.Count > 0)
            {
                summary.AppendLine("\nBranch Execution:");
                summary.AppendLine(new string('-', 60));
                
                foreach (var branch in BranchResults)
                {
                    var status = branch.WasExecuted ? "Executed" : "Skipped";
                    summary.AppendLine($"{branch.BranchName,-30} {status,10} {branch.Duration.TotalMilliseconds,10:F2} ms");
                }
            }

            summary.AppendLine("\nPerformance Metrics:");
            summary.AppendLine(new string('-', 60));
            summary.AppendLine($"Throughput: {Performance.Throughput:F2} samples/sec");
            summary.AppendLine($"Peak Memory: {Performance.PeakMemoryUsageBytes / (1024.0 * 1024.0):F2} MB");
            
            if (Performance.CacheHits + Performance.CacheMisses > 0)
            {
                summary.AppendLine($"Cache Hit Rate: {Performance.CacheHitRate:P2}");
            }

            return summary.ToString();
        }

        /// <summary>
        /// Gets detailed step timing information
        /// </summary>
        /// <returns>Dictionary of step names to their execution times</returns>
        public Dictionary<string, TimeSpan> GetStepTimings()
        {
            return StepResults.ToDictionary(s => s.StepName, s => s.Duration);
        }

        /// <summary>
        /// Gets the slowest steps in the pipeline
        /// </summary>
        /// <param name="count">Number of steps to return</param>
        /// <returns>List of the slowest steps</returns>
        public List<StepResult> GetSlowestSteps(int count = 5)
        {
            return StepResults
                .OrderByDescending(s => s.Duration)
                .Take(count)
                .ToList();
        }

        /// <summary>
        /// Calculates the percentage of time spent in each step
        /// </summary>
        /// <returns>Dictionary of step names to percentage of total time</returns>
        public Dictionary<string, double> GetStepTimePercentages()
        {
            var totalStepTime = StepResults.Sum(s => s.Duration.TotalMilliseconds);
            
            if (totalStepTime == 0)
            {
                return new Dictionary<string, double>();
            }

            return StepResults.ToDictionary(
                s => s.StepName,
                s => (s.Duration.TotalMilliseconds / totalStepTime) * 100
            );
        }

        /// <summary>
        /// Exports the result to JSON format
        /// </summary>
        /// <returns>JSON representation of the result</returns>
        public string ToJson()
        {
            var json = System.Text.Json.JsonSerializer.Serialize(this, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
            });
            
            return json;
        }

        /// <summary>
        /// Creates a clone of this result without the output data
        /// </summary>
        /// <returns>A new PipelineResult instance without output data</returns>
        public PipelineResult CloneWithoutData()
        {
            var clone = new PipelineResult
            {
                PipelineId = PipelineId,
                PipelineName = PipelineName,
                StartTime = StartTime,
                EndTime = EndTime,
                IsSuccessful = IsSuccessful,
                ErrorMessage = ErrorMessage,
                FailedStep = FailedStep,
                InputShape = (int[])InputShape.Clone(),
                OutputShape = (int[])OutputShape.Clone(),
                Performance = new PerformanceMetrics
                {
                    PeakMemoryUsageBytes = Performance.PeakMemoryUsageBytes,
                    AverageCpuUsage = Performance.AverageCpuUsage,
                    TotalSamplesProcessed = Performance.TotalSamplesProcessed,
                    Throughput = Performance.Throughput,
                    CacheHits = Performance.CacheHits,
                    CacheMisses = Performance.CacheMisses
                }
            };

            // Copy step results
            foreach (var step in StepResults)
            {
                clone.StepResults.Add(new StepResult
                {
                    StepName = step.StepName,
                    Duration = step.Duration,
                    IsSuccessful = step.IsSuccessful,
                    ErrorMessage = step.ErrorMessage,
                    OutputShape = (int[])step.OutputShape.Clone()
                });
            }

            // Copy branch results
            foreach (var branch in BranchResults)
            {
                clone.BranchResults.Add(new BranchResult
                {
                    BranchName = branch.BranchName,
                    WasExecuted = branch.WasExecuted,
                    Duration = branch.Duration
                });
            }

            // Copy metadata
            foreach (var kvp in Metadata)
            {
                clone.Metadata[kvp.Key] = kvp.Value;
            }

            return clone;
        }
    }
}