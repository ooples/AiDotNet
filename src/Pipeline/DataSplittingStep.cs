using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Generic data splitting pipeline step for train/validation/test splits
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class DataSplittingStep<T> : PipelineStepBase<T>
    {
        private readonly DataSplittingConfig config;
        private List<int>? trainIndices;
        private List<int>? valIndices;
        private List<int>? testIndices;
        
        public DataSplittingStep(DataSplittingConfig config) 
            : base("DataSplitting", MathHelper.GetNumericOperations<T>())
        {
            this.config = config ?? throw new ArgumentNullException(nameof(config));
            
            // Data splitting doesn't modify data, just selects indices
            Position = Enums.PipelinePosition.Any;
            IsCacheable = true;
            SupportsParallelExecution = false; // Order matters for splits
        }
        
        protected override bool RequiresFitting() => true;
        
        protected override void FitCore(Matrix<T> inputs, Vector<T>? targets)
        {
            var totalSamples = inputs.Rows;
            var indices = Enumerable.Range(0, totalSamples).ToList();
            
            // Shuffle if not doing time series split
            if (config.Shuffle && config.SplitMethod != SplitMethod.TimeSeries)
            {
                var rng = new Random(config.RandomSeed ?? DateTime.Now.Millisecond);
                indices = indices.OrderBy(x => rng.Next()).ToList();
            }
            
            // Calculate split sizes
            var trainSize = (int)(totalSamples * config.TrainRatio);
            var valSize = (int)(totalSamples * config.ValidationRatio);
            var testSize = totalSamples - trainSize - valSize;
            
            // Perform split based on method
            switch (config.SplitMethod)
            {
                case SplitMethod.Random:
                    PerformRandomSplit(indices, trainSize, valSize, testSize);
                    break;
                    
                case SplitMethod.Stratified:
                    if (targets == null)
                    {
                        throw new ArgumentException("Stratified split requires target labels");
                    }
                    PerformStratifiedSplit(indices, targets, trainSize, valSize, testSize);
                    break;
                    
                case SplitMethod.TimeSeries:
                    PerformTimeSeriesSplit(indices, trainSize, valSize, testSize);
                    break;
                    
                default:
                    throw new NotSupportedException($"Split method {config.SplitMethod} is not supported");
            }
            
            UpdateMetadata("TrainSamples", trainIndices?.Count.ToString() ?? "0");
            UpdateMetadata("ValidationSamples", valIndices?.Count.ToString() ?? "0");
            UpdateMetadata("TestSamples", testIndices?.Count.ToString() ?? "0");
            UpdateMetadata("SplitMethod", config.SplitMethod.ToString());
        }
        
        protected override Matrix<T> TransformCore(Matrix<T> inputs)
        {
            // Data splitting doesn't transform data, it just provides indices
            return inputs;
        }
        
        private void PerformRandomSplit(List<int> indices, int trainSize, int valSize, int testSize)
        {
            trainIndices = indices.Take(trainSize).ToList();
            valIndices = indices.Skip(trainSize).Take(valSize).ToList();
            testIndices = indices.Skip(trainSize + valSize).ToList();
        }
        
        private void PerformStratifiedSplit(List<int> indices, Vector<T> targets, int trainSize, int valSize, int testSize)
        {
            // Group indices by class label
            var classGroups = indices.GroupBy(i => targets[i]).ToList();
            
            trainIndices = new List<int>();
            valIndices = new List<int>();
            testIndices = new List<int>();
            
            // Split each class proportionally
            foreach (var group in classGroups)
            {
                var classIndices = group.ToList();
                var classSize = classIndices.Count;
                
                var classTrain = (int)(classSize * config.TrainRatio);
                var classVal = (int)(classSize * config.ValidationRatio);
                
                trainIndices.AddRange(classIndices.Take(classTrain));
                valIndices.AddRange(classIndices.Skip(classTrain).Take(classVal));
                testIndices.AddRange(classIndices.Skip(classTrain + classVal));
            }
            
            // Shuffle the indices within each set
            if (config.Shuffle)
            {
                var rng = new Random(config.RandomSeed ?? DateTime.Now.Millisecond);
                trainIndices = trainIndices.OrderBy(x => rng.Next()).ToList();
                valIndices = valIndices.OrderBy(x => rng.Next()).ToList();
                testIndices = testIndices.OrderBy(x => rng.Next()).ToList();
            }
        }
        
        private void PerformTimeSeriesSplit(List<int> indices, int trainSize, int valSize, int testSize)
        {
            // For time series, maintain temporal order
            trainIndices = indices.Take(trainSize).ToList();
            valIndices = indices.Skip(trainSize).Take(valSize).ToList();
            testIndices = indices.Skip(trainSize + valSize).ToList();
        }
        
        /// <summary>
        /// Gets the training data based on the split
        /// </summary>
        public (Matrix<T> trainData, Vector<T>? trainLabels) GetTrainData(Matrix<T> allData, Vector<T>? allLabels)
        {
            if (trainIndices == null)
            {
                throw new InvalidOperationException("Data splitting has not been fitted yet");
            }
            
            return ExtractData(allData, allLabels, trainIndices);
        }
        
        /// <summary>
        /// Gets the validation data based on the split
        /// </summary>
        public (Matrix<T> valData, Vector<T>? valLabels) GetValidationData(Matrix<T> allData, Vector<T>? allLabels)
        {
            if (valIndices == null)
            {
                throw new InvalidOperationException("Data splitting has not been fitted yet");
            }
            
            return ExtractData(allData, allLabels, valIndices);
        }
        
        /// <summary>
        /// Gets the test data based on the split
        /// </summary>
        public (Matrix<T> testData, Vector<T>? testLabels) GetTestData(Matrix<T> allData, Vector<T>? allLabels)
        {
            if (testIndices == null)
            {
                throw new InvalidOperationException("Data splitting has not been fitted yet");
            }
            
            return ExtractData(allData, allLabels, testIndices);
        }
        
        private (Matrix<T> data, Vector<T>? labels) ExtractData(Matrix<T> allData, Vector<T>? allLabels, List<int> indices)
        {
            var data = new Matrix<T>(indices.Count, allData.Columns);
            Vector<T>? labels = null;
            
            if (allLabels != null)
            {
                labels = new Vector<T>(indices.Count);
            }
            
            for (int i = 0; i < indices.Count; i++)
            {
                var idx = indices[i];
                for (int j = 0; j < allData.Columns; j++)
                {
                    data[i, j] = allData[idx, j];
                }
                
                if (labels != null && allLabels != null)
                {
                    labels[i] = allLabels[idx];
                }
            }
            
            return (data, labels);
        }
        
        /// <summary>
        /// Gets the indices for each split
        /// </summary>
        public (int[]? train, int[]? validation, int[]? test) GetIndices()
        {
            return (trainIndices?.ToArray(), valIndices?.ToArray(), testIndices?.ToArray());
        }
    }
    
    /// <summary>
    /// Configuration for data splitting
    /// </summary>
    public class DataSplittingConfig
    {
        /// <summary>
        /// Ratio of data for training (0-1)
        /// </summary>
        public double TrainRatio { get; set; } = 0.7;
        
        /// <summary>
        /// Ratio of data for validation (0-1)
        /// </summary>
        public double ValidationRatio { get; set; } = 0.15;
        
        /// <summary>
        /// Test ratio is automatically calculated as 1 - TrainRatio - ValidationRatio
        /// </summary>
        public double TestRatio => 1.0 - TrainRatio - ValidationRatio;
        
        /// <summary>
        /// Whether to shuffle data before splitting
        /// </summary>
        public bool Shuffle { get; set; } = true;
        
        /// <summary>
        /// Random seed for reproducibility
        /// </summary>
        public int? RandomSeed { get; set; }
        
        /// <summary>
        /// Method for splitting data
        /// </summary>
        public SplitMethod SplitMethod { get; set; } = SplitMethod.Random;
    }
    
    /// <summary>
    /// Methods for splitting data
    /// </summary>
    public enum SplitMethod
    {
        /// <summary>
        /// Random split
        /// </summary>
        Random,
        
        /// <summary>
        /// Stratified split (maintains class distribution)
        /// </summary>
        Stratified,
        
        /// <summary>
        /// Time series split (maintains temporal order)
        /// </summary>
        TimeSeries
    }
}