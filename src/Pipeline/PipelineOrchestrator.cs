using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Orchestrates complex pipeline workflows with branching, parallelism, and conditional execution
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class PipelineOrchestrator<T>
    {
        private readonly List<IPipelineStep<T>> _mainPipeline;
        private readonly Dictionary<string, PipelineBranch<T>> _branches;
        private readonly Dictionary<string, object> _metadata;
        private readonly object _lock = new object();
        private readonly SemaphoreSlim _semaphore;

        /// <summary>
        /// Gets the name of this orchestrator
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets the unique identifier for this orchestrator
        /// </summary>
        public string Id { get; }

        /// <summary>
        /// Gets the main pipeline steps
        /// </summary>
        public IReadOnlyList<IPipelineStep<T>> MainPipeline => _mainPipeline.AsReadOnly();

        /// <summary>
        /// Gets all branches
        /// </summary>
        public IReadOnlyDictionary<string, PipelineBranch<T>> Branches => _branches;

        /// <summary>
        /// Gets or sets the maximum degree of parallelism
        /// </summary>
        public int MaxDegreeOfParallelism { get; set; }

        /// <summary>
        /// Gets or sets whether to continue on branch failures
        /// </summary>
        public bool ContinueOnBranchFailure { get; set; }

        /// <summary>
        /// Gets execution statistics
        /// </summary>
        public OrchestratorStatistics Statistics { get; }

        /// <summary>
        /// Orchestrator execution statistics
        /// </summary>
        public class OrchestratorStatistics
        {
            public int TotalExecutions { get; set; }
            public int SuccessfulExecutions { get; set; }
            public int FailedExecutions { get; set; }
            public TimeSpan TotalExecutionTime { get; set; }
            public TimeSpan AverageExecutionTime => TotalExecutions > 0
                ? TimeSpan.FromMilliseconds(TotalExecutionTime.TotalMilliseconds / TotalExecutions)
                : TimeSpan.Zero;
            public Dictionary<string, int> BranchExecutionCounts { get; } = new Dictionary<string, int>();
            public DateTime? LastExecutedAt { get; set; }
        }

        /// <summary>
        /// Initializes a new instance of the PipelineOrchestrator class
        /// </summary>
        /// <param name="name">Name of the orchestrator</param>
        public PipelineOrchestrator(string name)
        {
            Name = name;
            Id = Guid.NewGuid().ToString();
            _mainPipeline = new List<IPipelineStep<T>>();
            _branches = new Dictionary<string, PipelineBranch<T>>();
            _metadata = new Dictionary<string, object>();
            MaxDegreeOfParallelism = Environment.ProcessorCount;
            ContinueOnBranchFailure = true;
            Statistics = new OrchestratorStatistics();
            _semaphore = new SemaphoreSlim(MaxDegreeOfParallelism, MaxDegreeOfParallelism);
        }

        /// <summary>
        /// Adds a step to the main pipeline
        /// </summary>
        /// <param name="step">The pipeline step to add</param>
        public void AddMainStep(IPipelineStep<T> step)
        {
            if (step == null)
            {
                throw new ArgumentNullException(nameof(step));
            }

            lock (_lock)
            {
                _mainPipeline.Add(step);
            }
        }

        /// <summary>
        /// Adds a branch to the orchestrator
        /// </summary>
        /// <param name="branch">The branch to add</param>
        public void AddBranch(PipelineBranch<T> branch)
        {
            if (branch == null)
            {
                throw new ArgumentNullException(nameof(branch));
            }

            lock (_lock)
            {
                _branches[branch.Id] = branch;
            }
        }

        /// <summary>
        /// Removes a branch from the orchestrator
        /// </summary>
        /// <param name="branchId">ID of the branch to remove</param>
        /// <returns>True if removed, false otherwise</returns>
        public bool RemoveBranch(string branchId)
        {
            lock (_lock)
            {
                return _branches.Remove(branchId);
            }
        }

        /// <summary>
        /// Executes the orchestrated pipeline
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Target data (optional)</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Result of the pipeline execution</returns>
        public async Task<PipelineResult<T>> ExecuteAsync(
            Matrix<T> inputs, 
            Vector<T>? targets = null,
            CancellationToken cancellationToken = default)
        {
            var startTime = DateTime.UtcNow;
            var result = new PipelineResult<T>
            {
                Id = Guid.NewGuid().ToString(),
                OrchestratorId = Id,
                StartTime = startTime
            };

            Statistics.TotalExecutions++;

            try
            {
                // Execute main pipeline
                var mainData = await ExecuteMainPipelineAsync(inputs, targets, cancellationToken);
                result.MainPipelineOutput = mainData;

                // Execute branches
                var branchResults = await ExecuteBranchesAsync(mainData, targets, cancellationToken);
                result.BranchOutputs = branchResults;

                // Merge results if needed
                var finalOutput = await MergeBranchResultsAsync(mainData, branchResults, cancellationToken);
                result.FinalOutput = finalOutput;

                result.EndTime = DateTime.UtcNow;
                result.Success = true;
                Statistics.SuccessfulExecutions++;
            }
            catch (Exception ex)
            {
                result.EndTime = DateTime.UtcNow;
                result.Success = false;
                result.Error = ex.Message;
                Statistics.FailedExecutions++;
                throw;
            }
            finally
            {
                var executionTime = DateTime.UtcNow - startTime;
                Statistics.TotalExecutionTime = Statistics.TotalExecutionTime.Add(executionTime);
                Statistics.LastExecutedAt = DateTime.UtcNow;
            }

            return result;
        }

        /// <summary>
        /// Executes the main pipeline
        /// </summary>
        private async Task<Matrix<T>> ExecuteMainPipelineAsync(
            Matrix<T> inputs,
            Vector<T>? targets,
            CancellationToken cancellationToken)
        {
            var currentData = inputs;

            foreach (var step in _mainPipeline)
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (!step.IsFitted)
                {
                    await step.FitAsync(currentData, targets);
                }
                currentData = await step.TransformAsync(currentData);
            }

            return currentData;
        }

        /// <summary>
        /// Executes all branches
        /// </summary>
        private async Task<Dictionary<string, Matrix<T>>> ExecuteBranchesAsync(
            Matrix<T> inputs,
            Vector<T>? targets,
            CancellationToken cancellationToken)
        {
            var results = new Dictionary<string, Matrix<T>>();
            var orderedBranches = _branches.Values.OrderByDescending(b => b.Priority).ToList();

            // Group branches by parallel execution
            var parallelGroups = orderedBranches.GroupBy(b => b.ExecuteInParallel);

            foreach (var group in parallelGroups)
            {
                if (group.Key) // Execute in parallel
                {
                    var tasks = group.Select(branch => ExecuteBranchWithSemaphoreAsync(
                        branch, inputs, targets, cancellationToken));
                    
                    var branchResults = await Task.WhenAll(tasks);
                    
                    foreach (var (branch, output) in branchResults)
                    {
                        if (output != null)
                        {
                            results[branch.Id] = output;
                        }
                    }
                }
                else // Execute sequentially
                {
                    foreach (var branch in group)
                    {
                        var output = await ExecuteBranchAsync(branch, inputs, targets, cancellationToken);
                        if (output != null)
                        {
                            results[branch.Id] = output;
                        }
                    }
                }
            }

            return results;
        }

        /// <summary>
        /// Executes a branch with semaphore control
        /// </summary>
        private async Task<(PipelineBranch<T> branch, Matrix<T>? output)> ExecuteBranchWithSemaphoreAsync(
            PipelineBranch<T> branch,
            Matrix<T> inputs,
            Vector<T>? targets,
            CancellationToken cancellationToken)
        {
            await _semaphore.WaitAsync(cancellationToken);
            try
            {
                var output = await ExecuteBranchAsync(branch, inputs, targets, cancellationToken);
                return (branch, output);
            }
            finally
            {
                _semaphore.Release();
            }
        }

        /// <summary>
        /// Executes a single branch
        /// </summary>
        private async Task<Matrix<T>?> ExecuteBranchAsync(
            PipelineBranch<T> branch,
            Matrix<T> inputs,
            Vector<T>? targets,
            CancellationToken cancellationToken)
        {
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                
                var output = await branch.ExecuteAsync(inputs, targets);
                
                lock (_lock)
                {
                    if (!Statistics.BranchExecutionCounts.ContainsKey(branch.Id))
                    {
                        Statistics.BranchExecutionCounts[branch.Id] = 0;
                    }
                    Statistics.BranchExecutionCounts[branch.Id]++;
                }
                
                return output;
            }
            catch (Exception ex)
            {
                if (!ContinueOnBranchFailure)
                {
                    throw;
                }
                
                // Log error and continue
                Console.WriteLine($"Branch '{branch.Name}' failed: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Merges branch results based on their merge strategies
        /// </summary>
        private async Task<Matrix<T>> MergeBranchResultsAsync(
            Matrix<T> mainOutput,
            Dictionary<string, Matrix<T>> branchOutputs,
            CancellationToken cancellationToken)
        {
            if (branchOutputs.Count == 0)
            {
                return mainOutput;
            }

            var result = mainOutput;

            foreach (var (branchId, branchOutput) in branchOutputs)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var branch = _branches[branchId];
                result = MergeBranchOutput(result, branchOutput, branch.MergeStrategy, branch.GetMergeParameters());
            }

            return result;
        }

        /// <summary>
        /// Merges two outputs based on the specified strategy
        /// </summary>
        private Matrix<T> MergeBranchOutput(
            Matrix<T> current,
            Matrix<T> branchOutput,
            BranchMergeStrategy strategy,
            Dictionary<string, object> parameters)
        {
            switch (strategy)
            {
                case BranchMergeStrategy.Replace:
                    return branchOutput;
                    
                case BranchMergeStrategy.Concatenate:
                    return ConcatenateMatrices(current, branchOutput, parameters);
                    
                case BranchMergeStrategy.Average:
                    return AverageMatrices(current, branchOutput);
                    
                case BranchMergeStrategy.WeightedAverage:
                    return WeightedAverageMatrices(current, branchOutput, parameters);
                    
                case BranchMergeStrategy.Custom:
                    if (parameters.TryGetValue("MergeFunction", out var mergeFunc) && 
                        mergeFunc is Func<Matrix<T>, Matrix<T>, Matrix<T>> customMerge)
                    {
                        return customMerge(current, branchOutput);
                    }
                    return current;
                    
                default:
                    return current;
            }
        }

        /// <summary>
        /// Concatenates two matrices
        /// </summary>
        private Matrix<T> ConcatenateMatrices(Matrix<T> a, Matrix<T> b, Dictionary<string, object> parameters)
        {
            var axis = parameters.TryGetValue("Axis", out var axisObj) && axisObj is int axisValue ? axisValue : 1;
            
            if (axis == 0) // Concatenate rows
            {
                if (a.Columns != b.Columns)
                {
                    throw new InvalidOperationException("Cannot concatenate matrices with different column counts");
                }
                
                var result = new Matrix<T>(a.Rows + b.Rows, a.Columns);
                for (int i = 0; i < a.Rows; i++)
                {
                    for (int j = 0; j < a.Columns; j++)
                    {
                        result[i, j] = a[i, j];
                    }
                }
                for (int i = 0; i < b.Rows; i++)
                {
                    for (int j = 0; j < b.Columns; j++)
                    {
                        result[a.Rows + i, j] = b[i, j];
                    }
                }
                return result;
            }
            else // Concatenate columns
            {
                if (a.Rows != b.Rows)
                {
                    throw new InvalidOperationException("Cannot concatenate matrices with different row counts");
                }
                
                var result = new Matrix<T>(a.Rows, a.Columns + b.Columns);
                for (int i = 0; i < a.Rows; i++)
                {
                    for (int j = 0; j < a.Columns; j++)
                    {
                        result[i, j] = a[i, j];
                    }
                    for (int j = 0; j < b.Columns; j++)
                    {
                        result[i, a.Columns + j] = b[i, j];
                    }
                }
                return result;
            }
        }

        /// <summary>
        /// Averages two matrices element-wise
        /// </summary>
        private Matrix<T> AverageMatrices(Matrix<T> a, Matrix<T> b)
        {
            if (a.Rows != b.Rows || a.Columns != b.Columns)
            {
                throw new InvalidOperationException("Cannot average matrices with different dimensions");
            }
            
            var result = new Matrix<T>(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    var sum = Convert.ToDouble(a[i, j]) + Convert.ToDouble(b[i, j]);
                    result[i, j] = (T)Convert.ChangeType(sum / 2.0, typeof(T));
                }
            }
            return result;
        }

        /// <summary>
        /// Performs weighted average of two matrices
        /// </summary>
        private Matrix<T> WeightedAverageMatrices(Matrix<T> a, Matrix<T> b, Dictionary<string, object> parameters)
        {
            if (a.Rows != b.Rows || a.Columns != b.Columns)
            {
                throw new InvalidOperationException("Cannot average matrices with different dimensions");
            }
            
            var weightA = parameters.TryGetValue("WeightA", out var wa) && wa is double wad ? wad : 0.5;
            var weightB = parameters.TryGetValue("WeightB", out var wb) && wb is double wbd ? wbd : 0.5;
            
            // Normalize weights
            var totalWeight = weightA + weightB;
            weightA /= totalWeight;
            weightB /= totalWeight;
            
            var result = new Matrix<T>(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    var value = Convert.ToDouble(a[i, j]) * weightA + Convert.ToDouble(b[i, j]) * weightB;
                    result[i, j] = (T)Convert.ChangeType(value, typeof(T));
                }
            }
            return result;
        }

        /// <summary>
        /// Validates the orchestrator configuration
        /// </summary>
        public (bool IsValid, List<string> Errors) Validate()
        {
            var errors = new List<string>();

            if (_mainPipeline.Count == 0 && _branches.Count == 0)
            {
                errors.Add("Orchestrator must have at least one main step or branch");
            }

            foreach (var branch in _branches.Values)
            {
                var (isValid, error) = branch.Validate();
                if (!isValid && error != null)
                {
                    errors.Add($"Branch '{branch.Name}': {error}");
                }
            }

            return (errors.Count == 0, errors);
        }

        /// <summary>
        /// Disposes of resources
        /// </summary>
        public void Dispose()
        {
            _semaphore?.Dispose();
        }

        /// <summary>
        /// Converts a 2D double array to a Tensor
        /// </summary>
        private static Tensor<double> ArrayToTensor(double[][] array)
        {
            if (array == null || array.Length == 0)
            {
                throw new ArgumentException("Array cannot be null or empty", nameof(array));
            }

            var rows = array.Length;
            var cols = array[0].Length;
            var tensor = new Tensor<double>(new[] { rows, cols });

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    tensor[i, j] = array[i][j];
                }
            }

            return tensor;
        }

        /// <summary>
        /// Converts a 1D double array to a Tensor
        /// </summary>
        private static Tensor<double> ArrayTo1DTensor(double[] array)
        {
            if (array == null || array.Length == 0)
            {
                throw new ArgumentException("Array cannot be null or empty", nameof(array));
            }

            var tensor = new Tensor<double>(new[] { array.Length });
            for (int i = 0; i < array.Length; i++)
            {
                tensor[i] = array[i];
            }

            return tensor;
        }

        /// <summary>
        /// Converts a Tensor to a 2D double array
        /// </summary>
        private static double[][] TensorToArray(Tensor<double> tensor)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            var shape = tensor.Shape;
            if (shape.Length == 1)
            {
                // 1D tensor - convert to 2D with single column
                var result = new double[shape[0]][];
                for (int i = 0; i < shape[0]; i++)
                {
                    result[i] = new double[] { tensor[i] };
                }
                return result;
            }
            else if (shape.Length == 2)
            {
                // 2D tensor - direct conversion
                var rows = shape[0];
                var cols = shape[1];
                var result = new double[rows][];
                for (int i = 0; i < rows; i++)
                {
                    result[i] = new double[cols];
                    for (int j = 0; j < cols; j++)
                    {
                        result[i][j] = tensor[i, j];
                    }
                }
                return result;
            }
            else
            {
                throw new ArgumentException($"Unsupported tensor rank: {shape.Length}. Expected 1D or 2D tensor.", nameof(tensor));
            }
        }
    }
}