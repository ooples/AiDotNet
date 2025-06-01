using AiDotNet.Enums;
using AiDotNet.Interfaces;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Orchestrates the execution of complex machine learning pipelines
    /// </summary>
    public class PipelineOrchestrator
    {
        private readonly List<IPipelineStep> _steps;
        private readonly List<PipelineBranch> _branches;
        private readonly Dictionary<string, object> _globalParameters;
        private readonly List<PipelineCheckpoint> _checkpoints;
        private CancellationTokenSource? _cancellationTokenSource;

        /// <summary>
        /// Gets or sets the name of this pipeline
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets the pipeline steps
        /// </summary>
        public IReadOnlyList<IPipelineStep> Steps => _steps.AsReadOnly();

        /// <summary>
        /// Gets the pipeline branches
        /// </summary>
        public IReadOnlyList<PipelineBranch> Branches => _branches.AsReadOnly();

        /// <summary>
        /// Gets or sets whether to enable checkpointing
        /// </summary>
        public bool EnableCheckpointing { get; set; }

        /// <summary>
        /// Gets or sets whether to continue on step failure
        /// </summary>
        public bool ContinueOnFailure { get; set; }

        /// <summary>
        /// Gets or sets the maximum retry attempts for failed steps
        /// </summary>
        public int MaxRetryAttempts { get; set; }

        /// <summary>
        /// Gets the execution history
        /// </summary>
        public PipelineExecutionHistory History { get; }

        /// <summary>
        /// Event raised when a step starts execution
        /// </summary>
        public event EventHandler<StepExecutionEventArgs>? StepStarted;

        /// <summary>
        /// Event raised when a step completes execution
        /// </summary>
        public event EventHandler<StepExecutionEventArgs>? StepCompleted;

        /// <summary>
        /// Event raised when a step fails
        /// </summary>
        public event EventHandler<StepExecutionEventArgs>? StepFailed;

        /// <summary>
        /// Event raised when the pipeline completes
        /// </summary>
        public event EventHandler<PipelineCompletedEventArgs>? PipelineCompleted;

        /// <summary>
        /// Represents a pipeline checkpoint
        /// </summary>
        private class PipelineCheckpoint
        {
            public string StepName { get; set; } = string.Empty;
            public int StepIndex { get; set; }
            public double[][] Data { get; set; } = Array.Empty<double[]>();
            public DateTime Timestamp { get; set; }
        }

        /// <summary>
        /// Pipeline execution history
        /// </summary>
        public class PipelineExecutionHistory
        {
            private readonly List<ExecutionRecord> _records = new List<ExecutionRecord>();

            public IReadOnlyList<ExecutionRecord> Records => _records.AsReadOnly();

            public void AddRecord(ExecutionRecord record)
            {
                _records.Add(record);
            }

            public ExecutionRecord? GetLastSuccessfulExecution()
            {
                return _records.LastOrDefault(r => r.IsSuccessful);
            }

            public class ExecutionRecord
            {
                public string PipelineId { get; set; } = string.Empty;
                public DateTime StartTime { get; set; }
                public DateTime EndTime { get; set; }
                public TimeSpan Duration => EndTime - StartTime;
                public bool IsSuccessful { get; set; }
                public List<StepExecutionInfo> StepExecutions { get; set; } = new List<StepExecutionInfo>();
                public string? ErrorMessage { get; set; }
            }

            public class StepExecutionInfo
            {
                public string StepName { get; set; } = string.Empty;
                public TimeSpan Duration { get; set; }
                public bool IsSuccessful { get; set; }
                public string? ErrorMessage { get; set; }
            }
        }

        /// <summary>
        /// Step execution event arguments
        /// </summary>
        public class StepExecutionEventArgs : EventArgs
        {
            public string StepName { get; set; } = string.Empty;
            public int StepIndex { get; set; }
            public int TotalSteps { get; set; }
            public TimeSpan? Duration { get; set; }
            public Exception? Exception { get; set; }
        }

        /// <summary>
        /// Pipeline completed event arguments
        /// </summary>
        public class PipelineCompletedEventArgs : EventArgs
        {
            public bool IsSuccessful { get; set; }
            public TimeSpan TotalDuration { get; set; }
            public PipelineResult? Result { get; set; }
            public Exception? Exception { get; set; }
        }

        /// <summary>
        /// Initializes a new instance of the PipelineOrchestrator class
        /// </summary>
        /// <param name="name">Name of the pipeline</param>
        public PipelineOrchestrator(string name)
        {
            Name = name;
            _steps = new List<IPipelineStep>();
            _branches = new List<PipelineBranch>();
            _globalParameters = new Dictionary<string, object>();
            _checkpoints = new List<PipelineCheckpoint>();
            History = new PipelineExecutionHistory();
            EnableCheckpointing = false;
            ContinueOnFailure = false;
            MaxRetryAttempts = 3;
        }

        /// <summary>
        /// Adds a step to the pipeline
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
        /// Adds a branch to the pipeline
        /// </summary>
        /// <param name="branch">The pipeline branch to add</param>
        public void AddBranch(PipelineBranch branch)
        {
            if (branch == null)
            {
                throw new ArgumentNullException(nameof(branch));
            }

            _branches.Add(branch);
        }

        /// <summary>
        /// Sets a global parameter that will be available to all steps
        /// </summary>
        /// <param name="name">Parameter name</param>
        /// <param name="value">Parameter value</param>
        public void SetGlobalParameter(string name, object value)
        {
            _globalParameters[name] = value;
        }

        /// <summary>
        /// Executes the pipeline
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Optional target data</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Pipeline execution result</returns>
        public async Task<PipelineResult> ExecuteAsync(double[][] inputs, double[]? targets = null, 
            CancellationToken cancellationToken = default)
        {
            var executionId = Guid.NewGuid().ToString();
            var startTime = DateTime.UtcNow;
            var executionRecord = new PipelineExecutionHistory.ExecutionRecord
            {
                PipelineId = executionId,
                StartTime = startTime,
                IsSuccessful = false
            };

            _cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            var result = new PipelineResult
            {
                PipelineId = executionId,
                PipelineName = Name,
                StartTime = startTime,
                InputShape = new[] { inputs.Length, inputs[0].Length }
            };

            try
            {
                // Apply global parameters to all steps
                ApplyGlobalParameters();

                // Execute main pipeline steps
                var currentData = inputs;
                foreach (var (step, index) in _steps.Select((s, i) => (s, i)))
                {
                    if (_cancellationTokenSource.Token.IsCancellationRequested)
                    {
                        throw new OperationCanceledException("Pipeline execution was cancelled");
                    }

                    currentData = await ExecuteStepAsync(step, currentData, targets, index, _steps.Count, 
                        executionRecord, result);
                }

                // Execute branches if any
                if (_branches.Count > 0)
                {
                    currentData = await ExecuteBranchesAsync(currentData, targets, executionRecord, result);
                }

                // Finalize result
                result.OutputData = currentData;
                result.OutputShape = new[] { currentData.Length, currentData[0].Length };
                result.EndTime = DateTime.UtcNow;
                result.IsSuccessful = true;

                executionRecord.IsSuccessful = true;
                executionRecord.EndTime = result.EndTime;

                PipelineCompleted?.Invoke(this, new PipelineCompletedEventArgs
                {
                    IsSuccessful = true,
                    TotalDuration = result.Duration,
                    Result = result
                });
            }
            catch (Exception ex)
            {
                result.EndTime = DateTime.UtcNow;
                result.IsSuccessful = false;
                result.ErrorMessage = ex.Message;
                result.FailedStep = result.StepResults.LastOrDefault()?.StepName;

                executionRecord.IsSuccessful = false;
                executionRecord.EndTime = result.EndTime;
                executionRecord.ErrorMessage = ex.Message;

                PipelineCompleted?.Invoke(this, new PipelineCompletedEventArgs
                {
                    IsSuccessful = false,
                    TotalDuration = result.Duration,
                    Result = result,
                    Exception = ex
                });

                throw;
            }
            finally
            {
                History.AddRecord(executionRecord);
                _cancellationTokenSource?.Dispose();
                _cancellationTokenSource = null;
            }

            return result;
        }

        /// <summary>
        /// Executes a single step with retry logic
        /// </summary>
        private async Task<double[][]> ExecuteStepAsync(IPipelineStep step, double[][] inputs, 
            double[]? targets, int stepIndex, int totalSteps, 
            PipelineExecutionHistory.ExecutionRecord executionRecord, PipelineResult result)
        {
            var stepStartTime = DateTime.UtcNow;
            var stepInfo = new PipelineExecutionHistory.StepExecutionInfo
            {
                StepName = step.Name
            };

            StepStarted?.Invoke(this, new StepExecutionEventArgs
            {
                StepName = step.Name,
                StepIndex = stepIndex,
                TotalSteps = totalSteps
            });

            Exception? lastException = null;
            for (int attempt = 0; attempt <= MaxRetryAttempts; attempt++)
            {
                try
                {
                    // Fit if necessary
                    if (!step.IsFitted && targets != null)
                    {
                        await step.FitAsync(inputs, targets).ConfigureAwait(false);
                    }

                    // Transform
                    var transformed = await step.TransformAsync(inputs).ConfigureAwait(false);

                    // Create checkpoint if enabled
                    if (EnableCheckpointing)
                    {
                        CreateCheckpoint(step.Name, stepIndex, transformed);
                    }

                    // Record success
                    var duration = DateTime.UtcNow - stepStartTime;
                    stepInfo.Duration = duration;
                    stepInfo.IsSuccessful = true;
                    executionRecord.StepExecutions.Add(stepInfo);

                    result.StepResults.Add(new PipelineResult.StepResult
                    {
                        StepName = step.Name,
                        Duration = duration,
                        IsSuccessful = true,
                        OutputShape = new[] { transformed.Length, transformed[0].Length }
                    });

                    StepCompleted?.Invoke(this, new StepExecutionEventArgs
                    {
                        StepName = step.Name,
                        StepIndex = stepIndex,
                        TotalSteps = totalSteps,
                        Duration = duration
                    });

                    return transformed;
                }
                catch (Exception ex)
                {
                    lastException = ex;
                    
                    if (attempt < MaxRetryAttempts)
                    {
                        await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt))); // Exponential backoff
                    }
                }
            }

            // All retries failed
            var failDuration = DateTime.UtcNow - stepStartTime;
            stepInfo.Duration = failDuration;
            stepInfo.IsSuccessful = false;
            stepInfo.ErrorMessage = lastException?.Message;
            executionRecord.StepExecutions.Add(stepInfo);

            result.StepResults.Add(new PipelineResult.StepResult
            {
                StepName = step.Name,
                Duration = failDuration,
                IsSuccessful = false,
                ErrorMessage = lastException?.Message
            });

            StepFailed?.Invoke(this, new StepExecutionEventArgs
            {
                StepName = step.Name,
                StepIndex = stepIndex,
                TotalSteps = totalSteps,
                Duration = failDuration,
                Exception = lastException
            });

            if (ContinueOnFailure)
            {
                return inputs; // Return original data and continue
            }

            throw lastException!;
        }

        /// <summary>
        /// Executes branches based on their configuration
        /// </summary>
        private async Task<double[][]> ExecuteBranchesAsync(double[][] inputs, double[]? targets,
            PipelineExecutionHistory.ExecutionRecord executionRecord, PipelineResult result)
        {
            // Group branches by parallel execution preference
            var parallelBranches = _branches.Where(b => b.ExecuteInParallel).OrderByDescending(b => b.Priority).ToList();
            var sequentialBranches = _branches.Where(b => !b.ExecuteInParallel).OrderByDescending(b => b.Priority).ToList();

            var branchResults = new List<(PipelineBranch branch, double[][] data)>();

            // Execute sequential branches first
            var currentData = inputs;
            foreach (var branch in sequentialBranches)
            {
                var branchResult = await branch.ExecuteAsync(currentData, targets).ConfigureAwait(false);
                
                if (branch.ShouldExecute(currentData))
                {
                    branchResults.Add((branch, branchResult));
                    currentData = branchResult;
                }

                result.BranchResults.Add(new PipelineResult.BranchResult
                {
                    BranchName = branch.Name,
                    WasExecuted = branch.ShouldExecute(inputs),
                    Duration = branch.Statistics.AverageExecutionTime
                });
            }

            // Execute parallel branches
            if (parallelBranches.Count > 0)
            {
                var parallelTasks = parallelBranches.Select(async branch =>
                {
                    if (branch.ShouldExecute(currentData))
                    {
                        var branchResult = await branch.ExecuteAsync(currentData, targets).ConfigureAwait(false);
                        return (branch, branchResult, executed: true);
                    }
                    return (branch, currentData, executed: false);
                });

                var parallelResults = await Task.WhenAll(parallelTasks).ConfigureAwait(false);

                foreach (var (branch, data, executed) in parallelResults)
                {
                    if (executed)
                    {
                        branchResults.Add((branch, data));
                    }

                    result.BranchResults.Add(new PipelineResult.BranchResult
                    {
                        BranchName = branch.Name,
                        WasExecuted = executed,
                        Duration = branch.Statistics.AverageExecutionTime
                    });
                }
            }

            // Merge branch results if any were executed
            if (branchResults.Count > 0)
            {
                return PipelineBranch.MergeBranchResults(branchResults);
            }

            return currentData;
        }

        /// <summary>
        /// Applies global parameters to all steps
        /// </summary>
        private void ApplyGlobalParameters()
        {
            foreach (var step in _steps)
            {
                foreach (var param in _globalParameters)
                {
                    var stepParams = step.GetParameters();
                    if (!stepParams.ContainsKey(param.Key))
                    {
                        stepParams[param.Key] = param.Value;
                        step.SetParameters(stepParams);
                    }
                }
            }

            // Apply to branch steps as well
            foreach (var branch in _branches)
            {
                foreach (var step in branch.Steps)
                {
                    foreach (var param in _globalParameters)
                    {
                        var stepParams = step.GetParameters();
                        if (!stepParams.ContainsKey(param.Key))
                        {
                            stepParams[param.Key] = param.Value;
                            step.SetParameters(stepParams);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Creates a checkpoint of the current data
        /// </summary>
        private void CreateCheckpoint(string stepName, int stepIndex, double[][] data)
        {
            var checkpoint = new PipelineCheckpoint
            {
                StepName = stepName,
                StepIndex = stepIndex,
                Data = CloneData(data),
                Timestamp = DateTime.UtcNow
            };

            _checkpoints.Add(checkpoint);

            // Keep only last N checkpoints to avoid memory issues
            const int maxCheckpoints = 10;
            if (_checkpoints.Count > maxCheckpoints)
            {
                _checkpoints.RemoveRange(0, _checkpoints.Count - maxCheckpoints);
            }
        }

        /// <summary>
        /// Restores from a checkpoint
        /// </summary>
        /// <param name="stepIndex">The step index to restore from</param>
        /// <returns>The checkpoint data if found, null otherwise</returns>
        public double[][]? RestoreFromCheckpoint(int stepIndex)
        {
            var checkpoint = _checkpoints.LastOrDefault(c => c.StepIndex == stepIndex);
            return checkpoint?.Data != null ? CloneData(checkpoint.Data) : null;
        }

        /// <summary>
        /// Validates the pipeline configuration
        /// </summary>
        /// <returns>List of validation errors</returns>
        public List<string> Validate()
        {
            var errors = new List<string>();

            if (_steps.Count == 0 && _branches.Count == 0)
            {
                errors.Add("Pipeline must contain at least one step or branch");
            }

            // Check for duplicate step names
            var stepNames = _steps.Select(s => s.Name)
                .Concat(_branches.SelectMany(b => b.Steps.Select(s => s.Name)))
                .ToList();

            var duplicates = stepNames.GroupBy(n => n)
                .Where(g => g.Count() > 1)
                .Select(g => g.Key);

            foreach (var duplicate in duplicates)
            {
                errors.Add($"Duplicate step name found: {duplicate}");
            }

            // Validate branch configurations
            foreach (var branch in _branches)
            {
                if (branch.Steps.Count == 0)
                {
                    errors.Add($"Branch '{branch.Name}' contains no steps");
                }
            }

            return errors;
        }

        /// <summary>
        /// Cancels the pipeline execution
        /// </summary>
        public void Cancel()
        {
            _cancellationTokenSource?.Cancel();
        }

        /// <summary>
        /// Clones data to avoid reference issues
        /// </summary>
        private static double[][] CloneData(double[][] data)
        {
            var clone = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                clone[i] = new double[data[i].Length];
                Array.Copy(data[i], clone[i], data[i].Length);
            }
            return clone;
        }

        /// <summary>
        /// Creates a visual representation of the pipeline
        /// </summary>
        /// <returns>String representation of the pipeline structure</returns>
        public string Visualize()
        {
            var visualization = new System.Text.StringBuilder();
            visualization.AppendLine($"Pipeline: {Name}");
            visualization.AppendLine(new string('=', 50));

            if (_steps.Count > 0)
            {
                visualization.AppendLine("\nMain Pipeline:");
                for (int i = 0; i < _steps.Count; i++)
                {
                    visualization.AppendLine($"  [{i + 1}] {_steps[i].Name}");
                    if (i < _steps.Count - 1)
                    {
                        visualization.AppendLine("   ↓");
                    }
                }
            }

            if (_branches.Count > 0)
            {
                visualization.AppendLine("\nBranches:");
                foreach (var branch in _branches)
                {
                    visualization.AppendLine($"\n  Branch: {branch.Name}");
                    visualization.AppendLine($"  Type: {(branch.IsConditional ? "Conditional" : "Unconditional")}");
                    visualization.AppendLine($"  Parallel: {branch.ExecuteInParallel}");
                    visualization.AppendLine($"  Merge Strategy: {branch.MergeStrategy}");
                    visualization.AppendLine("  Steps:");
                    
                    foreach (var step in branch.Steps)
                    {
                        visualization.AppendLine($"    - {step.Name}");
                    }
                }
            }

            visualization.AppendLine(new string('=', 50));
            return visualization.ToString();
        }
    }
}