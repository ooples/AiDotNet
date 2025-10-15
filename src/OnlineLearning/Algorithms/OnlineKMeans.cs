using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Helpers;
using AiDotNet.Statistics;

namespace AiDotNet.OnlineLearning.Algorithms;

/// <summary>
/// Online K-Means clustering algorithm for streaming data.
/// Uses mini-batch updates and can adapt to changing cluster centers.
/// </summary>
public class OnlineKMeans<T> : OnlineModelBase<T, Vector<T>, T>
{
    private readonly OnlineModelOptions<T> _options = default!;
    private readonly int _k; // Number of clusters
    private readonly int _numFeatures;
    private readonly Vector<T>[] _centroids;
    private readonly int[] _clusterCounts;
    private T _forgettingFactor; // For exponential decay of old data influence
    private readonly Random _random = default!;
    private readonly bool _useMiniKMeans; // true for mini-batch K-means
    private int _miniBatchSize;
    private readonly List<Vector<T>> _miniBatch = default!;
    
    /// <summary>
    /// Initializes a new instance of the OnlineKMeans class.
    /// </summary>
    public OnlineKMeans(int numFeatures, int k, bool useMiniKMeans = true,
                       OnlineModelOptions<T>? options = null, ILogging? logger = null)
        : base(options != null ? options.InitialLearningRate : MathHelper.GetNumericOperations<T>().FromDouble(0.01), logger)
    {
        if (k <= 0)
            throw new ArgumentException("Number of clusters must be positive", nameof(k));
            
        _options = options ?? new OnlineModelOptions<T>
        {
            InitialLearningRate = NumOps.FromDouble(0.01),
            UseAdaptiveLearningRate = true,
            LearningRateDecay = NumOps.FromDouble(0.999), // Used as forgetting factor
            RegularizationParameter = NumOps.Zero,
            MiniBatchSize = 100
        };
        
        _k = k;
        _numFeatures = numFeatures;
        _forgettingFactor = _options.LearningRateDecay;
        _useMiniKMeans = useMiniKMeans;
        _miniBatchSize = _options.MiniBatchSize;
        _random = new Random();
        
        // Initialize centroids randomly
        _centroids = new Vector<T>[k];
        _clusterCounts = new int[k];
        _miniBatch = new List<Vector<T>>(_miniBatchSize);
        
        InitializeCentroids();
        
        _learningRate = _options.InitialLearningRate;
        AdaptiveLearningRate = _options.UseAdaptiveLearningRate;
    }
    
    /// <summary>
    /// Initializes centroids with small random values.
    /// </summary>
    private void InitializeCentroids()
    {
        for (int i = 0; i < _k; i++)
        {
            var centroid = new T[_numFeatures];
            for (int j = 0; j < _numFeatures; j++)
            {
                // Random initialization between -1 and 1
                centroid[j] = NumOps.FromDouble(_random.NextDouble() * 2 - 1);
            }
            _centroids[i] = new Vector<T>(centroid);
            _clusterCounts[i] = 1; // Avoid division by zero
        }
    }
    
    /// <summary>
    /// Initializes centroids using K-means++ algorithm on first batch.
    /// </summary>
    private void InitializeCentroidsKMeansPlusPlus(List<Vector<T>> initialData)
    {
        if (initialData.Count == 0)
            return;
            
        // Choose first centroid randomly
        _centroids[0] = initialData[_random.Next(initialData.Count)].Clone();
        
        for (int i = 1; i < _k; i++)
        {
            // Calculate distances to nearest centroid for each point
            var distances = new T[initialData.Count];
            var totalDistance = NumOps.Zero;
            
            for (int j = 0; j < initialData.Count; j++)
            {
                var minDist = NumOps.FromDouble(double.MaxValue);
                for (int c = 0; c < i; c++)
                {
                    var dist = ComputeDistance(initialData[j], _centroids[c]);
                    if (NumOps.LessThan(dist, minDist))
                    {
                        minDist = dist;
                    }
                }
                distances[j] = minDist;
                totalDistance = NumOps.Add(totalDistance, minDist);
            }
            
            // Choose next centroid with probability proportional to squared distance
            var threshold = NumOps.Multiply(totalDistance, NumOps.FromDouble(_random.NextDouble()));
            var cumSum = NumOps.Zero;
            
            for (int j = 0; j < initialData.Count; j++)
            {
                cumSum = NumOps.Add(cumSum, distances[j]);
                if (NumOps.GreaterThanOrEquals(cumSum, threshold))
                {
                    _centroids[i] = initialData[j].Clone();
                    break;
                }
            }
        }
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        if (_useMiniKMeans)
        {
            // Add to mini-batch
            _miniBatch.Add(input.Clone());
            
            // Process when batch is full
            if (_miniBatch.Count >= _miniBatchSize)
            {
                ProcessMiniBatch(learningRate);
                _miniBatch.Clear();
            }
        }
        else
        {
            // Sequential K-means update
            UpdateSinglePoint(input, learningRate);
        }
    }
    
    /// <summary>
    /// Updates centroids based on a single data point.
    /// </summary>
    private void UpdateSinglePoint(Vector<T> input, T learningRate)
    {
        // Find nearest centroid
        int nearestCluster = FindNearestCluster(input);
        
        // Update centroid using exponential moving average
        var eta = ComputeAdaptiveLearningRate(nearestCluster, learningRate);
        
        for (int j = 0; j < _numFeatures; j++)
        {
            // centroid = (1 - eta) * centroid + eta * input
            _centroids[nearestCluster][j] = NumOps.Add(
                NumOps.Multiply(NumOps.Subtract(NumOps.One, eta), _centroids[nearestCluster][j]),
                NumOps.Multiply(eta, input[j])
            );
        }
        
        _clusterCounts[nearestCluster]++;
    }
    
    /// <summary>
    /// Processes a mini-batch of data points.
    /// </summary>
    private void ProcessMiniBatch(T learningRate)
    {
        if (_miniBatch.Count == 0)
            return;
            
        // If this is the first batch and we haven't initialized properly, use K-means++
        if (SamplesSeen <= _miniBatchSize && _miniBatch.Count >= Math.Min(_k, _miniBatchSize))
        {
            InitializeCentroidsKMeansPlusPlus(_miniBatch);
        }
        
        // Assign points to clusters
        var assignments = new int[_miniBatch.Count];
        var clusterSums = new Vector<T>[_k];
        var clusterBatchCounts = new int[_k];
        
        // Initialize cluster sums
        for (int i = 0; i < _k; i++)
        {
            clusterSums[i] = new Vector<T>(new T[_numFeatures]);
            for (int j = 0; j < _numFeatures; j++)
            {
                clusterSums[i][j] = NumOps.Zero;
            }
        }
        
        // Assign each point to nearest cluster and accumulate
        for (int i = 0; i < _miniBatch.Count; i++)
        {
            assignments[i] = FindNearestCluster(_miniBatch[i]);
            clusterBatchCounts[assignments[i]]++;
            
            for (int j = 0; j < _numFeatures; j++)
            {
                clusterSums[assignments[i]][j] = NumOps.Add(
                    clusterSums[assignments[i]][j],
                    _miniBatch[i][j]
                );
            }
        }
        
        // Update centroids
        for (int i = 0; i < _k; i++)
        {
            if (clusterBatchCounts[i] > 0)
            {
                var eta = ComputeAdaptiveLearningRate(i, learningRate);
                var batchCentroid = new Vector<T>(new T[_numFeatures]);
                
                // Compute batch centroid
                for (int j = 0; j < _numFeatures; j++)
                {
                    batchCentroid[j] = NumOps.Divide(
                        clusterSums[i][j],
                        NumOps.FromDouble(clusterBatchCounts[i])
                    );
                }
                
                // Update using exponential moving average
                for (int j = 0; j < _numFeatures; j++)
                {
                    _centroids[i][j] = NumOps.Add(
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, eta), _centroids[i][j]),
                        NumOps.Multiply(eta, batchCentroid[j])
                    );
                }
                
                _clusterCounts[i] += clusterBatchCounts[i];
            }
        }
    }
    
    /// <summary>
    /// Computes adaptive learning rate based on cluster visit count.
    /// </summary>
    private T ComputeAdaptiveLearningRate(int clusterIndex, T baseLearningRate)
    {
        if (!_adaptiveLearningRate)
            return baseLearningRate;
            
        // eta = base_rate / (1 + cluster_count * decay)
        var count = NumOps.FromDouble(_clusterCounts[clusterIndex]);
        var denominator = NumOps.Add(NumOps.One, 
            NumOps.Multiply(count, NumOps.Subtract(NumOps.One, _forgettingFactor)));
        
        return NumOps.Divide(baseLearningRate, denominator);
    }
    
    /// <summary>
    /// Finds the nearest cluster for a data point.
    /// </summary>
    private int FindNearestCluster(Vector<T> input)
    {
        int nearestCluster = 0;
        T minDistance = ComputeDistance(input, _centroids[0]);
        
        for (int i = 1; i < _k; i++)
        {
            var distance = ComputeDistance(input, _centroids[i]);
            if (NumOps.LessThan(distance, minDistance))
            {
                minDistance = distance;
                nearestCluster = i;
            }
        }
        
        return nearestCluster;
    }
    
    /// <summary>
    /// Computes Euclidean distance between two vectors.
    /// </summary>
    private T ComputeDistance(Vector<T> a, Vector<T> b)
    {
        var sum = NumOps.Zero;
        
        for (int i = 0; i < _numFeatures; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        
        return NumOps.Sqrt(sum);
    }
    
    /// <inheritdoc/>
    public override T Predict(Vector<T> input)
    {
        // Return cluster index as prediction
        return NumOps.FromDouble(FindNearestCluster(input));
    }
    
    /// <summary>
    /// Gets distances to all centroids.
    /// </summary>
    public Vector<T> GetDistancesToCentroids(Vector<T> input)
    {
        var distances = new T[_k];
        
        for (int i = 0; i < _k; i++)
        {
            distances[i] = ComputeDistance(input, _centroids[i]);
        }
        
        return new Vector<T>(distances);
    }
    
    /// <summary>
    /// Gets the current cluster centroids.
    /// </summary>
    public Vector<T>[] GetCentroids()
    {
        var centroids = new Vector<T>[_k];
        for (int i = 0; i < _k; i++)
        {
            centroids[i] = _centroids[i].Clone();
        }
        return centroids;
    }
    
    /// <inheritdoc/>
    public override void Train(Vector<T> input, T expectedOutput)
    {
        // For clustering, we ignore expectedOutput
        PartialFit(input, NumOps.Zero);
    }
    
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.SelfOrganizingMap, // Closest match for online clustering
            FeatureCount = _numFeatures,
            Complexity = _k * _numFeatures,
            Description = $"Online K-Means with {_k} clusters{(_useMiniKMeans ? " (mini-batch)" : " (sequential)")}",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = SamplesSeen,
                ["NumClusters"] = _k,
                ["UseMiniKMeans"] = _useMiniKMeans,
                ["MiniBatchSize"] = _miniBatchSize,
                ["ClusterCounts"] = _clusterCounts.ToArray(),
                ["ForgettingFactor"] = Convert.ToDouble(_forgettingFactor),
                ["Centroids"] = _centroids.Select(c => c.Clone()).ToArray()
            }
        };
    }
    
    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        
        // Version number
        writer.Write(1);
        
        // Basic info
        writer.Write(_numFeatures);
        writer.Write(_k);
        writer.Write(_useMiniKMeans);
        writer.Write(_miniBatchSize);
        writer.Write(Convert.ToDouble(_forgettingFactor));
        writer.Write(SamplesSeen);
        
        // Centroids
        for (int i = 0; i < _k; i++)
        {
            for (int j = 0; j < _numFeatures; j++)
            {
                writer.Write(Convert.ToDouble(_centroids[i][j]));
            }
        }
        
        // Cluster counts
        for (int i = 0; i < _k; i++)
        {
            writer.Write(_clusterCounts[i]);
        }
        
        return ms.ToArray();
    }
    
    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));
        if (data.Length == 0)
            throw new ArgumentException("Serialized data cannot be empty.", nameof(data));
            
        try
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);
            
            // Version number
            int version = reader.ReadInt32();
            
            // Basic info
            int numFeatures = reader.ReadInt32();
            int k = reader.ReadInt32();
            bool useMiniKMeans = reader.ReadBoolean();
            _miniBatchSize = reader.ReadInt32();
            _forgettingFactor = NumOps.FromDouble(reader.ReadDouble());
            _samplesSeen = reader.ReadInt64();
            
            // Centroids
            for (int i = 0; i < k; i++)
            {
                var centroid = new T[numFeatures];
                for (int j = 0; j < numFeatures; j++)
                {
                    centroid[j] = NumOps.FromDouble(reader.ReadDouble());
                }
                _centroids[i] = new Vector<T>(centroid);
            }
            
            // Cluster counts
            for (int i = 0; i < k; i++)
            {
                _clusterCounts[i] = reader.ReadInt32();
            }
        }
        catch (Exception ex) when (!(ex is ArgumentNullException || ex is ArgumentException))
        {
            throw new ArgumentException("Failed to deserialize the model. The data may be corrupted or in an invalid format.", nameof(data), ex);
        }
    }
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> WithParameters(IDictionary<string, object> parameters)
    {
        var newOptions = new OnlineModelOptions<T>
        {
            InitialLearningRate = parameters.ContainsKey("LearningRate") 
                ? (T)parameters["LearningRate"] 
                : _options.InitialLearningRate,
            UseAdaptiveLearningRate = parameters.ContainsKey("AdaptiveLearningRate") 
                ? (bool)parameters["AdaptiveLearningRate"] 
                : _options.UseAdaptiveLearningRate,
            LearningRateDecay = parameters.ContainsKey("ForgettingFactor") 
                ? (T)parameters["ForgettingFactor"] 
                : _forgettingFactor,
            MiniBatchSize = parameters.ContainsKey("MiniBatchSize") 
                ? (int)parameters["MiniBatchSize"] 
                : _miniBatchSize
        };
        
        int k = parameters.ContainsKey("NumClusters") ? (int)parameters["NumClusters"] : _k;
        bool useMiniKMeans = parameters.ContainsKey("UseMiniKMeans") 
            ? (bool)parameters["UseMiniKMeans"] 
            : _useMiniKMeans;
        
        return new OnlineKMeans<T>(_numFeatures, k, useMiniKMeans, newOptions, _logger);
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _numFeatures;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => _k;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var clone = new OnlineKMeans<T>(_numFeatures, _k, _useMiniKMeans, _options, _logger);
        var serialized = Serialize();
        clone.Deserialize(serialized);
        return clone;
    }
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> DeepCopy()
    {
        return Clone();
    }
    
    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Return learning rate and forgetting factor
        return new Vector<T>(new[] { _learningRate, _forgettingFactor });
    }
    
    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length >= 1)
        {
            _learningRate = parameters[0];
        }
        if (parameters.Length >= 2)
        {
            _forgettingFactor = parameters[1];
        }
    }
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> WithParameters(Vector<T> parameters)
    {
        var clone = Clone();
        clone.SetParameters(parameters);
        return clone;
    }
    
    /// <inheritdoc/>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        // All features are active in K-means
        return Enumerable.Range(0, _numFeatures);
    }
    
    /// <inheritdoc/>
    public override bool IsFeatureUsed(int featureIndex)
    {
        return featureIndex >= 0 && featureIndex < _numFeatures;
    }
    
    /// <inheritdoc/>
    public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // K-means uses all features, so this is a no-op
        _logger.Warning("SetActiveFeatureIndices is not supported for K-means");
    }
    
    /// <summary>
    /// Computes the within-cluster sum of squares (WCSS).
    /// </summary>
    public T ComputeWCSS(IEnumerable<Vector<T>> data)
    {
        var wcss = NumOps.Zero;
        
        foreach (var point in data)
        {
            var nearestCluster = FindNearestCluster(point);
            var distance = ComputeDistance(point, _centroids[nearestCluster]);
            wcss = NumOps.Add(wcss, NumOps.Multiply(distance, distance));
        }
        
        return wcss;
    }
    
    /// <summary>
    /// Gets cluster assignment probabilities based on distances (soft assignment).
    /// </summary>
    public Vector<T> GetClusterProbabilities(Vector<T> input)
    {
        var distances = GetDistancesToCentroids(input);
        var probabilities = new T[_k];
        
        // Convert distances to similarities using exponential
        var sumExp = NumOps.Zero;
        for (int i = 0; i < _k; i++)
        {
            // Use negative distance for similarity
            probabilities[i] = NumOps.Exp(NumOps.Negate(distances[i]));
            sumExp = NumOps.Add(sumExp, probabilities[i]);
        }
        
        // Normalize
        for (int i = 0; i < _k; i++)
        {
            probabilities[i] = NumOps.Divide(probabilities[i], sumExp);
        }
        
        return new Vector<T>(probabilities);
    }

    
    /// <inheritdoc/>
    public override int InputDimensions => _numFeatures;
    
    /// <inheritdoc/>
    public override int OutputDimensions => _k;
    
    /// <inheritdoc/>
    public override bool IsTrained => _samplesSeen > 0;
    
    /// <inheritdoc/>
    public override T[] PredictBatch(Vector<T>[] inputBatch)
    {
        var predictions = new T[inputBatch.Length];
        for (int i = 0; i < inputBatch.Length; i++)
        {
            predictions[i] = Predict(inputBatch[i]);
        }
        return predictions;
    }
    
    /// <inheritdoc/>
    public override Dictionary<string, double> Evaluate(Vector<T> testData, T testLabels)
    {
        // This method should accept arrays, but for now return basic metrics
        var prediction = Predict(testData);
        var error = NumOps.Zero;
        
        return new Dictionary<string, double>
        {
            ["ClusterAssignment"] = Convert.ToDouble(prediction)
        };
    }
    
    /// <inheritdoc/>
    public override void SaveModel(string filePath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filePath, data);
    }
    
    /// <inheritdoc/>
    public override double GetTrainingLoss()
    {
        // Default implementation for models that don't track loss
        return 0.0;
    }
    
    /// <inheritdoc/>
    public override double GetValidationLoss()
    {
        // In online learning, we don't have separate validation loss
        return GetTrainingLoss();
    }
    
    /// <inheritdoc/>
    public override Vector<T> GetModelParameters()
    {
        return GetParameters();
    }
    
    /// <inheritdoc/>
    public override ModelStats<T> GetStats()
    {
        return new ModelStats<T>
        {
            SampleCount = SamplesSeen,
            LearningRate = _learningRate,
            TrainingLoss = NumOps.FromDouble(GetTrainingLoss()),
            ValidationLoss = NumOps.FromDouble(GetValidationLoss()),
            AdditionalMetrics = new Dictionary<string, T>
            {
                ["NumClusters"] = NumOps.FromDouble(_k),
                ["ForgettingFactor"] = _forgettingFactor
            }
        };
    }
    
    /// <inheritdoc/>
    public override void Save()
    {
        // Default implementation saves to a standard location
        SaveModel($"online_kmeans_model_{DateTime.Now:yyyyMMddHHmmss}.bin");
    }
    
    /// <inheritdoc/>
    public override void Load()
    {
        // Default implementation would load from a standard location
        // For now, this is a no-op as we need a file path
        throw new NotImplementedException("Load requires a file path. Use Deserialize instead.");
    }
    
    /// <inheritdoc/>
    public override void Dispose()
    {
        // Clean up any resources if needed
    }
}