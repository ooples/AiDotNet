using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Pipeline.Steps
{
    /// <summary>
    /// Pipeline step that enables parallel processing of data batches
    /// </summary>
    public class ParallelProcessingStep : PipelineStepBase
    {
        private readonly List<IPipelineStep<double, Tensor<double>, Tensor<double>>> _wrappedSteps = default!;
        private int _maxDegreeOfParallelism;
        private int _batchSize;
        private bool _preserveOrder;
        private PartitioningStrategy _partitioningStrategy = default!;
        private readonly object _progressLock = new object();

        /// <summary>
        /// Enum for data partitioning strategies
        /// </summary>
        public enum PartitioningStrategy
        {
            FixedSize,      // Fixed batch sizes
            Dynamic,        // Dynamic batch sizes based on load
            RoundRobin,     // Round-robin distribution
            LoadBalanced,   // Load-balanced distribution
            Custom          // Custom partitioning function
        }

        /// <summary>
        /// Progress information for parallel processing
        /// </summary>
        public class ProgressInfo
        {
            public int TotalBatches { get; set; }
            public int CompletedBatches { get; set; }
            public int FailedBatches { get; set; }
            public TimeSpan ElapsedTime { get; set; }
            public double ProgressPercentage => TotalBatches > 0 ? (double)CompletedBatches / TotalBatches * 100 : 0;
        }

        /// <summary>
        /// Gets the current progress information
        /// </summary>
        public ProgressInfo Progress { get; private set; }

        /// <summary>
        /// Event raised when progress is updated
        /// </summary>
        public event EventHandler<ProgressInfo>? ProgressUpdated;

        /// <summary>
        /// Gets the wrapped pipeline steps
        /// </summary>
        public IReadOnlyList<IPipelineStep<double, Tensor<double>, Tensor<double>>> WrappedSteps => _wrappedSteps.AsReadOnly();

        /// <summary>
        /// Initializes a new instance of the ParallelProcessingStep class
        /// </summary>
        /// <param name="stepsToParallelize">Steps to run in parallel</param>
        /// <param name="name">Optional name for this step</param>
        public ParallelProcessingStep(IEnumerable<IPipelineStep<double, Tensor<double>, Tensor<double>>>? stepsToParallelize = null, string? name = null)
            : base(name ?? "ParallelProcessing")
        {
            Position = PipelinePosition.Parallel;
            SupportsParallelExecution = true;
            _wrappedSteps = new List<IPipelineStep<double, Tensor<double>, Tensor<double>>>(stepsToParallelize ?? Enumerable.Empty<IPipelineStep<double, Tensor<double>, Tensor<double>>>());
            _maxDegreeOfParallelism = Environment.ProcessorCount;
            _batchSize = 1000;
            _preserveOrder = true;
            _partitioningStrategy = PartitioningStrategy.FixedSize;
            Progress = new ProgressInfo();

            // Set default parameters
            SetParameter("MaxDegreeOfParallelism", _maxDegreeOfParallelism);
            SetParameter("BatchSize", _batchSize);
            SetParameter("PreserveOrder", _preserveOrder);
            SetParameter("PartitioningStrategy", _partitioningStrategy);
        }

        /// <summary>
        /// Adds a step to be executed in parallel
        /// </summary>
        /// <param name="step">The pipeline step to add</param>
        public void AddStep(IPipelineStep<double, Tensor<double>, Tensor<double>> step)
        {
            if (step == null)
            {
                throw new ArgumentNullException(nameof(step));
            }

            _wrappedSteps.Add(step);
            ResetFittedState();
        }

        /// <summary>
        /// Core fitting logic that fits wrapped steps in parallel
        /// </summary>
        protected override void FitCore(double[][] inputs, double[]? targets)
        {
            if (_wrappedSteps.Count == 0)
            {
                return;
            }

            var startTime = DateTime.UtcNow;
            var batches = CreateBatches(inputs, targets);
            Progress = new ProgressInfo
            {
                TotalBatches = batches.Count * _wrappedSteps.Count,
                CompletedBatches = 0,
                FailedBatches = 0
            };

            var exceptions = new ConcurrentBag<Exception>();

            // Fit each step in parallel using the batches
            var inputTensor = ArrayToTensor(inputs);
            var targetTensor = targets != null ? ArrayTo1DTensor(targets) : null;

            Parallel.ForEach(_wrappedSteps, new ParallelOptions { MaxDegreeOfParallelism = _maxDegreeOfParallelism }, step =>
            {
                try
                {
                    // For fitting, we typically use all data, not batches
                    step.FitAsync(inputTensor, targetTensor).Wait();
                    UpdateProgress(batches.Count);
                }
                catch (Exception ex)
                {
                    exceptions.Add(new Exception($"Failed to fit step '{step.Name}': {ex.Message}", ex));
                    UpdateProgress(batches.Count, failed: true);
                }
            });

            if (exceptions.Any())
            {
                throw new AggregateException("One or more steps failed during parallel fitting", exceptions);
            }

            Progress.ElapsedTime = DateTime.UtcNow - startTime;
            UpdateMetadata("FitDuration", Progress.ElapsedTime.TotalSeconds.ToString("F2") + "s");
            UpdateMetadata("StepsCount", _wrappedSteps.Count.ToString());
        }

        /// <summary>
        /// Core transformation logic that processes data in parallel
        /// </summary>
        protected override double[][] TransformCore(double[][] inputs)
        {
            if (_wrappedSteps.Count == 0)
            {
                return inputs;
            }

            var startTime = DateTime.UtcNow;
            var batches = CreateBatches(inputs, null);
            Progress = new ProgressInfo
            {
                TotalBatches = batches.Count * _wrappedSteps.Count,
                CompletedBatches = 0,
                FailedBatches = 0
            };

            // Process each step sequentially, but process batches in parallel within each step
            var currentData = inputs;
            foreach (var step in _wrappedSteps)
            {
                currentData = ProcessStepInParallel(step, currentData, batches);
            }

            Progress.ElapsedTime = DateTime.UtcNow - startTime;
            UpdateMetadata("TransformDuration", Progress.ElapsedTime.TotalSeconds.ToString("F2") + "s");
            
            return currentData;
        }

        /// <summary>
        /// Processes a single step in parallel across batches
        /// </summary>
        private double[][] ProcessStepInParallel(IPipelineStep<double, Tensor<double>, Tensor<double>> step, double[][] inputs, List<DataBatch> batches)
        {
            var results = new ConcurrentDictionary<int, double[][]>();
            var exceptions = new ConcurrentBag<Exception>();

            // Process batches in parallel
            Parallel.ForEach(batches, new ParallelOptions { MaxDegreeOfParallelism = _maxDegreeOfParallelism }, batch =>
            {
                try
                {
                    var batchData = ExtractBatchData(inputs, batch);
                    var batchTensor = ArrayToTensor(batchData);
                    var transformedTensor = step.TransformAsync(batchTensor).Result;
                    var transformed = TensorToArray(transformedTensor);
                    results[batch.Index] = transformed;
                    UpdateProgress(1);
                }
                catch (Exception ex)
                {
                    exceptions.Add(new Exception($"Batch {batch.Index} failed in step '{step.Name}': {ex.Message}", ex));
                    UpdateProgress(1, failed: true);
                }
            });

            if (exceptions.Any())
            {
                throw new AggregateException($"One or more batches failed during parallel processing of step '{step.Name}'", exceptions);
            }

            // Combine results maintaining order if required
            return CombineResults(results, batches, inputs[0].Length);
        }

        /// <summary>
        /// Creates data batches based on the partitioning strategy
        /// </summary>
        private List<DataBatch> CreateBatches(double[][] inputs, double[]? targets)
        {
            var batches = new List<DataBatch>();
            int totalSamples = inputs.Length;

            switch (_partitioningStrategy)
            {
                case PartitioningStrategy.FixedSize:
                    CreateFixedSizeBatches(batches, totalSamples);
                    break;

                case PartitioningStrategy.Dynamic:
                    CreateDynamicBatches(batches, totalSamples);
                    break;

                case PartitioningStrategy.RoundRobin:
                    CreateRoundRobinBatches(batches, totalSamples);
                    break;

                case PartitioningStrategy.LoadBalanced:
                    CreateLoadBalancedBatches(batches, totalSamples, inputs);
                    break;

                default:
                    CreateFixedSizeBatches(batches, totalSamples);
                    break;
            }

            return batches;
        }

        /// <summary>
        /// Creates fixed-size batches
        /// </summary>
        private void CreateFixedSizeBatches(List<DataBatch> batches, int totalSamples)
        {
            int batchCount = (int)Math.Ceiling((double)totalSamples / _batchSize);
            
            for (int i = 0; i < batchCount; i++)
            {
                int start = i * _batchSize;
                int end = Math.Min(start + _batchSize, totalSamples);
                
                batches.Add(new DataBatch
                {
                    Index = i,
                    StartIndex = start,
                    EndIndex = end,
                    Size = end - start
                });
            }
        }

        /// <summary>
        /// Creates dynamic batches based on system load
        /// </summary>
        private void CreateDynamicBatches(List<DataBatch> batches, int totalSamples)
        {
            // Adjust batch size based on available CPU and memory
            var availableProcessors = Environment.ProcessorCount;
            var dynamicBatchSize = Math.Max(100, totalSamples / (availableProcessors * 2));
            
            int batchCount = (int)Math.Ceiling((double)totalSamples / dynamicBatchSize);
            
            for (int i = 0; i < batchCount; i++)
            {
                int start = i * dynamicBatchSize;
                int end = Math.Min(start + dynamicBatchSize, totalSamples);
                
                batches.Add(new DataBatch
                {
                    Index = i,
                    StartIndex = start,
                    EndIndex = end,
                    Size = end - start
                });
            }
        }

        /// <summary>
        /// Creates round-robin batches
        /// </summary>
        private void CreateRoundRobinBatches(List<DataBatch> batches, int totalSamples)
        {
            int numBatches = Math.Min(_maxDegreeOfParallelism, (int)Math.Ceiling((double)totalSamples / 100));
            
            // Initialize batches
            for (int i = 0; i < numBatches; i++)
            {
                batches.Add(new DataBatch
                {
                    Index = i,
                    Indices = new List<int>()
                });
            }

            // Distribute samples round-robin
            for (int i = 0; i < totalSamples; i++)
            {
                batches[i % numBatches].Indices!.Add(i);
            }

            // Update batch properties
            foreach (var batch in batches)
            {
                batch.Size = batch.Indices!.Count;
            }
        }

        /// <summary>
        /// Creates load-balanced batches based on data complexity
        /// </summary>
        private void CreateLoadBalancedBatches(List<DataBatch> batches, int totalSamples, double[][] inputs)
        {
            // Estimate complexity based on feature variance
            var complexities = new double[totalSamples];
            
            for (int i = 0; i < totalSamples; i++)
            {
                // Simple complexity measure: variance of features
                complexities[i] = StatisticsHelper<double>.Variance(inputs[i]);
            }

            // Sort by complexity and distribute evenly
            var sortedIndices = Enumerable.Range(0, totalSamples)
                .OrderBy(i => complexities[i])
                .ToList();

            int numBatches = Math.Min(_maxDegreeOfParallelism, (int)Math.Ceiling((double)totalSamples / 100));
            
            // Initialize batches
            for (int i = 0; i < numBatches; i++)
            {
                batches.Add(new DataBatch
                {
                    Index = i,
                    Indices = new List<int>(),
                    TotalComplexity = 0
                });
            }

            // Distribute samples to balance complexity
            foreach (var index in sortedIndices)
            {
                // Find batch with lowest total complexity
                var targetBatch = batches.OrderBy(b => b.TotalComplexity).First();
                targetBatch.Indices!.Add(index);
                targetBatch.TotalComplexity += complexities[index];
            }

            // Update batch properties
            foreach (var batch in batches)
            {
                batch.Size = batch.Indices!.Count;
            }
        }

        /// <summary>
        /// Extracts batch data from the full dataset
        /// </summary>
        private double[][] ExtractBatchData(double[][] inputs, DataBatch batch)
        {
            if (batch.Indices != null)
            {
                // Extract specific indices
                var batchData = new double[batch.Indices.Count][];
                for (int i = 0; i < batch.Indices.Count; i++)
                {
                    batchData[i] = inputs[batch.Indices[i]];
                }
                return batchData;
            }
            else
            {
                // Extract range
                var size = batch.EndIndex - batch.StartIndex;
                var batchData = new double[size][];
                Array.Copy(inputs, batch.StartIndex, batchData, 0, size);
                return batchData;
            }
        }

        /// <summary>
        /// Combines results from parallel processing
        /// </summary>
        private double[][] CombineResults(ConcurrentDictionary<int, double[][]> results, 
            List<DataBatch> batches, int featureCount)
        {
            if (!_preserveOrder)
            {
                // Just concatenate all results
                return results.Values.SelectMany(r => r).ToArray();
            }

            // Preserve original order
            var combined = new List<double[]>();
            
            foreach (var batch in batches.OrderBy(b => b.Index))
            {
                if (results.TryGetValue(batch.Index, out var batchResult))
                {
                    combined.AddRange(batchResult);
                }
            }

            return combined.ToArray();
        }

        /// <summary>
        /// Updates progress information
        /// </summary>
        private void UpdateProgress(int increment, bool failed = false)
        {
            lock (_progressLock)
            {
                if (failed)
                {
                    Progress.FailedBatches += increment;
                }
                else
                {
                    Progress.CompletedBatches += increment;
                }

                ProgressUpdated?.Invoke(this, Progress);
            }
        }

        /// <summary>
        /// Sets a single parameter value
        /// </summary>
        protected override void SetParameter(string name, object value)
        {
            base.SetParameter(name, value);

            switch (name)
            {
                case "MaxDegreeOfParallelism":
                    _maxDegreeOfParallelism = Convert.ToInt32(value);
                    break;
                case "BatchSize":
                    _batchSize = Convert.ToInt32(value);
                    break;
                case "PreserveOrder":
                    _preserveOrder = Convert.ToBoolean(value);
                    break;
                case "PartitioningStrategy":
                    _partitioningStrategy = (PartitioningStrategy)value;
                    break;
            }
        }

        /// <summary>
        /// Validates that this step can process the given input
        /// </summary>
        public override bool ValidateInput(Tensor<double> inputs)
        {
            if (!base.ValidateInput(inputs))
            {
                return false;
            }

            // Validate all wrapped steps
            return _wrappedSteps.All(step => step.ValidateInput(inputs));
        }

        /// <summary>
        /// Gets metadata about this pipeline step
        /// </summary>
        public override Dictionary<string, string> GetMetadata()
        {
            var metadata = base.GetMetadata();
            metadata["WrappedSteps"] = string.Join(", ", _wrappedSteps.Select(s => s.Name));
            metadata["MaxDegreeOfParallelism"] = _maxDegreeOfParallelism.ToString();
            metadata["BatchSize"] = _batchSize.ToString();
            metadata["PartitioningStrategy"] = _partitioningStrategy.ToString();
            return metadata;
        }

        /// <summary>
        /// Represents a data batch for parallel processing
        /// </summary>
        private class DataBatch
        {
            public int Index { get; set; }
            public int StartIndex { get; set; }
            public int EndIndex { get; set; }
            public List<int>? Indices { get; set; }
            public int Size { get; set; }
            public double TotalComplexity { get; set; }
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