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
/// Online Naive Bayes classifier for streaming data.
/// Supports both Gaussian (continuous) and Multinomial (discrete) features.
/// </summary>
public class OnlineNaiveBayes<T> : OnlineModelBase<T, Vector<T>, T>
{
    private readonly OnlineModelOptions<T> _options = default!;
    private readonly int _numClasses;
    private readonly int _numFeatures;
    private readonly bool _isGaussian; // true for Gaussian, false for Multinomial
    
    // Class statistics
    private readonly int[] _classCounts;
    private int _totalSamples;
    
    // Gaussian parameters: mean and variance for each feature per class
    private T[,] _featureMeans = null!;
    private T[,] _featureVariances = null!;
    private T[,] _featureSumSquares = null!; // For incremental variance calculation
    
    // Multinomial parameters: feature counts per class
    private Dictionary<int, T>[,] _featureCounts = null!; // [class, feature] -> value -> count
    private T[] _classTotalFeatureCounts = null!;
    
    // Laplace smoothing parameter
    private T _alpha = default!;
    
    /// <summary>
    /// Initializes a new instance of the OnlineNaiveBayes class.
    /// </summary>
    public OnlineNaiveBayes(int numFeatures, int numClasses, bool isGaussian = true,
                           OnlineModelOptions<T>? options = null, ILogging? logger = null)
        : base(options != null ? options.InitialLearningRate : MathHelper.GetNumericOperations<T>().FromDouble(1.0), logger)
    {
        _options = options ?? new OnlineModelOptions<T>
        {
            InitialLearningRate = NumOps.FromDouble(1.0),
            UseAdaptiveLearningRate = false,
            LearningRateDecay = NumOps.Zero,
            RegularizationParameter = NumOps.FromDouble(1.0) // Used as Laplace smoothing parameter
        };
        
        _numFeatures = numFeatures;
        _numClasses = numClasses;
        _isGaussian = isGaussian;
        _alpha = _options.RegularizationParameter; // Laplace smoothing
        
        _classCounts = new int[numClasses];
        _totalSamples = 0;
        
        if (_isGaussian)
        {
            _featureMeans = new T[numClasses, numFeatures];
            _featureVariances = new T[numClasses, numFeatures];
            _featureSumSquares = new T[numClasses, numFeatures];
            
            // Initialize with small values
            for (int c = 0; c < numClasses; c++)
            {
                for (int f = 0; f < numFeatures; f++)
                {
                    _featureMeans[c, f] = NumOps.Zero;
                    _featureVariances[c, f] = NumOps.One; // Initial variance of 1
                    _featureSumSquares[c, f] = NumOps.Zero;
                }
            }
        }
        else
        {
            _featureCounts = new Dictionary<int, T>[numClasses, numFeatures];
            _classTotalFeatureCounts = new T[numClasses];
            
            for (int c = 0; c < numClasses; c++)
            {
                _classTotalFeatureCounts[c] = NumOps.Zero;
                for (int f = 0; f < numFeatures; f++)
                {
                    _featureCounts[c, f] = new Dictionary<int, T>();
                }
            }
        }
        
        _learningRate = _options.InitialLearningRate;
        AdaptiveLearningRate = _options.UseAdaptiveLearningRate;
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        int classIndex = Convert.ToInt32(expectedOutput);
        if (classIndex < 0 || classIndex >= _numClasses)
        {
            throw new ArgumentException($"Class index {classIndex} is out of range [0, {_numClasses})");
        }
        
        _classCounts[classIndex]++;
        _totalSamples++;
        
        if (_isGaussian)
        {
            UpdateGaussianParameters(input, classIndex);
        }
        else
        {
            UpdateMultinomialParameters(input, classIndex);
        }
    }
    
    /// <summary>
    /// Updates Gaussian parameters using Welford's online algorithm.
    /// </summary>
    private void UpdateGaussianParameters(Vector<T> input, int classIndex)
    {
        var n = _classCounts[classIndex];
        
        for (int f = 0; f < _numFeatures; f++)
        {
            var value = input[f];
            var oldMean = _featureMeans[classIndex, f];
            
            // Update mean: new_mean = old_mean + (value - old_mean) / n
            var delta = NumOps.Subtract(value, oldMean);
            var newMean = NumOps.Add(oldMean, NumOps.Divide(delta, NumOps.FromDouble(n)));
            _featureMeans[classIndex, f] = newMean;
            
            // Update sum of squares for variance calculation
            var delta2 = NumOps.Subtract(value, newMean);
            _featureSumSquares[classIndex, f] = NumOps.Add(
                _featureSumSquares[classIndex, f],
                NumOps.Multiply(delta, delta2)
            );
            
            // Update variance
            if (n > 1)
            {
                _featureVariances[classIndex, f] = NumOps.Divide(
                    _featureSumSquares[classIndex, f],
                    NumOps.FromDouble(n - 1)
                );
            }
        }
    }
    
    /// <summary>
    /// Updates multinomial parameters (feature counts).
    /// </summary>
    private void UpdateMultinomialParameters(Vector<T> input, int classIndex)
    {
        for (int f = 0; f < _numFeatures; f++)
        {
            var value = Convert.ToInt32(input[f]);
            
            if (!_featureCounts[classIndex, f].ContainsKey(value))
            {
                _featureCounts[classIndex, f][value] = NumOps.Zero;
            }
            
            _featureCounts[classIndex, f][value] = NumOps.Add(
                _featureCounts[classIndex, f][value],
                NumOps.One
            );
            
            _classTotalFeatureCounts[classIndex] = NumOps.Add(
                _classTotalFeatureCounts[classIndex],
                NumOps.One
            );
        }
    }
    
    /// <inheritdoc/>
    public override T Predict(Vector<T> input)
    {
        var logProbabilities = ComputeLogProbabilities(input);
        
        // Find class with maximum log probability
        int maxClass = 0;
        var maxLogProb = logProbabilities[0];
        
        for (int c = 1; c < _numClasses; c++)
        {
            if (NumOps.GreaterThan(logProbabilities[c], maxLogProb))
            {
                maxLogProb = logProbabilities[c];
                maxClass = c;
            }
        }
        
        return NumOps.FromDouble(maxClass);
    }
    
    /// <summary>
    /// Gets prediction probabilities for all classes.
    /// </summary>
    public Vector<T> PredictProbabilities(Vector<T> input)
    {
        var logProbabilities = ComputeLogProbabilities(input);
        
        // Convert log probabilities to probabilities using exp and normalization
        var probabilities = new T[_numClasses];
        var maxLogProb = logProbabilities[0];
        
        // Find max for numerical stability
        for (int c = 1; c < _numClasses; c++)
        {
            if (NumOps.GreaterThan(logProbabilities[c], maxLogProb))
            {
                maxLogProb = logProbabilities[c];
            }
        }
        
        // Compute exp(logProb - maxLogProb) for numerical stability
        var sumExp = NumOps.Zero;
        for (int c = 0; c < _numClasses; c++)
        {
            probabilities[c] = NumOps.Exp(NumOps.Subtract(logProbabilities[c], maxLogProb));
            sumExp = NumOps.Add(sumExp, probabilities[c]);
        }
        
        // Normalize
        for (int c = 0; c < _numClasses; c++)
        {
            probabilities[c] = NumOps.Divide(probabilities[c], sumExp);
        }
        
        return new Vector<T>(probabilities);
    }
    
    /// <summary>
    /// Computes log probabilities for all classes.
    /// </summary>
    private T[] ComputeLogProbabilities(Vector<T> input)
    {
        var logProbabilities = new T[_numClasses];
        
        for (int c = 0; c < _numClasses; c++)
        {
            // Log prior probability: log(P(class))
            var logPrior = _totalSamples > 0
                ? NumOps.Log(NumOps.Divide(
                    NumOps.Add(NumOps.FromDouble(_classCounts[c]), _alpha),
                    NumOps.Add(NumOps.FromDouble(_totalSamples), NumOps.Multiply(_alpha, NumOps.FromDouble(_numClasses)))
                  ))
                : NumOps.Log(NumOps.Divide(NumOps.One, NumOps.FromDouble(_numClasses)));
            
            // Log likelihood: sum of log(P(feature|class))
            var logLikelihood = NumOps.Zero;
            
            if (_isGaussian)
            {
                for (int f = 0; f < _numFeatures; f++)
                {
                    var mean = _featureMeans[c, f];
                    var variance = _featureVariances[c, f];
                    
                    // Gaussian log probability: -0.5 * log(2π*σ²) - 0.5 * (x-μ)²/σ²
                    var diff = NumOps.Subtract(input[f], mean);
                    var normalizedSquare = NumOps.Divide(
                        NumOps.Multiply(diff, diff),
                        variance
                    );
                    
                    var logProb = NumOps.Subtract(
                        NumOps.Multiply(NumOps.FromDouble(-0.5), NumOps.Log(
                            NumOps.Multiply(NumOps.FromDouble(2 * Math.PI), variance)
                        )),
                        NumOps.Multiply(NumOps.FromDouble(0.5), normalizedSquare)
                    );
                    
                    logLikelihood = NumOps.Add(logLikelihood, logProb);
                }
            }
            else
            {
                for (int f = 0; f < _numFeatures; f++)
                {
                    var value = Convert.ToInt32(input[f]);
                    
                    // Multinomial log probability with Laplace smoothing
                    T count = _featureCounts[c, f].ContainsKey(value)
                        ? _featureCounts[c, f][value]
                        : NumOps.Zero;
                    
                    var numerator = NumOps.Add(count, _alpha);
                    var denominator = NumOps.Add(
                        _classTotalFeatureCounts[c],
                        NumOps.Multiply(_alpha, NumOps.FromDouble(GetVocabularySize(f)))
                    );
                    
                    var logProb = NumOps.Log(NumOps.Divide(numerator, denominator));
                    logLikelihood = NumOps.Add(logLikelihood, logProb);
                }
            }
            
            logProbabilities[c] = NumOps.Add(logPrior, logLikelihood);
        }
        
        return logProbabilities;
    }
    
    /// <summary>
    /// Gets the vocabulary size for a feature (for multinomial).
    /// </summary>
    private int GetVocabularySize(int featureIndex)
    {
        var uniqueValues = new HashSet<int>();
        for (int c = 0; c < _numClasses; c++)
        {
            foreach (var value in _featureCounts[c, featureIndex].Keys)
            {
                uniqueValues.Add(value);
            }
        }
        return Math.Max(uniqueValues.Count, 2); // At least binary
    }
    
    /// <inheritdoc/>
    public override void Train(Vector<T> input, T expectedOutput)
    {
        PartialFit(input, expectedOutput);
    }
    
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.OnlineNaiveBayes,
            FeatureCount = _numFeatures,
            Complexity = _isGaussian ? _numClasses * _numFeatures * 2 : _numClasses * _numFeatures,
            Description = $"{(_isGaussian ? "Gaussian" : "Multinomial")} Naive Bayes with {_numClasses} classes",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = SamplesSeen,
                ["TotalSamples"] = _totalSamples,
                ["IsGaussian"] = _isGaussian,
                ["NumClasses"] = _numClasses,
                ["LaplaceSmoothing"] = Convert.ToDouble(_alpha)
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
        writer.Write(_numClasses);
        writer.Write(_isGaussian);
        writer.Write(Convert.ToDouble(_alpha));
        writer.Write(SamplesSeen);
        writer.Write(_totalSamples);
        
        // Class counts
        for (int c = 0; c < _numClasses; c++)
        {
            writer.Write(_classCounts[c]);
        }
        
        if (_isGaussian)
        {
            // Gaussian parameters
            for (int c = 0; c < _numClasses; c++)
            {
                for (int f = 0; f < _numFeatures; f++)
                {
                    writer.Write(Convert.ToDouble(_featureMeans[c, f]));
                    writer.Write(Convert.ToDouble(_featureVariances[c, f]));
                    writer.Write(Convert.ToDouble(_featureSumSquares[c, f]));
                }
            }
        }
        else
        {
            // Multinomial parameters
            for (int c = 0; c < _numClasses; c++)
            {
                writer.Write(Convert.ToDouble(_classTotalFeatureCounts[c]));
                
                for (int f = 0; f < _numFeatures; f++)
                {
                    writer.Write(_featureCounts[c, f].Count);
                    foreach (var kvp in _featureCounts[c, f])
                    {
                        writer.Write(kvp.Key);
                        writer.Write(Convert.ToDouble(kvp.Value));
                    }
                }
            }
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
            int numClasses = reader.ReadInt32();
            bool isGaussian = reader.ReadBoolean();
            _alpha = NumOps.FromDouble(reader.ReadDouble());
            _samplesSeen = reader.ReadInt64();
            _totalSamples = reader.ReadInt32();
            
            // Class counts
            for (int c = 0; c < numClasses; c++)
            {
                _classCounts[c] = reader.ReadInt32();
            }
            
            if (isGaussian)
            {
                // Gaussian parameters
                for (int c = 0; c < numClasses; c++)
                {
                    for (int f = 0; f < numFeatures; f++)
                    {
                        _featureMeans[c, f] = NumOps.FromDouble(reader.ReadDouble());
                        _featureVariances[c, f] = NumOps.FromDouble(reader.ReadDouble());
                        _featureSumSquares[c, f] = NumOps.FromDouble(reader.ReadDouble());
                    }
                }
            }
            else
            {
                // Multinomial parameters
                for (int c = 0; c < numClasses; c++)
                {
                    _classTotalFeatureCounts[c] = NumOps.FromDouble(reader.ReadDouble());
                    
                    for (int f = 0; f < numFeatures; f++)
                    {
                        _featureCounts[c, f].Clear();
                        int count = reader.ReadInt32();
                        
                        for (int i = 0; i < count; i++)
                        {
                            int value = reader.ReadInt32();
                            T countValue = NumOps.FromDouble(reader.ReadDouble());
                            _featureCounts[c, f][value] = countValue;
                        }
                    }
                }
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
            RegularizationParameter = parameters.ContainsKey("LaplaceSmoothing") 
                ? (T)parameters["LaplaceSmoothing"] 
                : _alpha
        };
        
        bool isGaussian = parameters.ContainsKey("IsGaussian") ? (bool)parameters["IsGaussian"] : _isGaussian;
        
        return new OnlineNaiveBayes<T>(_numFeatures, _numClasses, isGaussian, newOptions, _logger);
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _numFeatures;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => _numClasses;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var clone = new OnlineNaiveBayes<T>(_numFeatures, _numClasses, _isGaussian, _options, _logger);
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
        // Return Laplace smoothing parameter
        return new Vector<T>(new[] { _alpha });
    }
    
    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length >= 1)
        {
            _alpha = parameters[0];
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
        // All features are active in Naive Bayes
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
        // Naive Bayes uses all features, so this is a no-op
        _logger.Warning("SetActiveFeatureIndices is not supported for Naive Bayes");
    }
    
    /// <summary>
    /// Gets the feature importance based on variance (Gaussian) or entropy (Multinomial).
    /// </summary>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();

        if (_isGaussian)
        {
            // For Gaussian, use variance ratio between classes
            for (int f = 0; f < _numFeatures; f++)
            {
                var maxVar = NumOps.Zero;
                var minVar = NumOps.FromDouble(double.MaxValue);

                for (int c = 0; c < _numClasses; c++)
                {
                    var var = _featureVariances[c, f];
                    if (NumOps.GreaterThan(var, maxVar)) maxVar = var;
                    if (NumOps.LessThan(var, minVar)) minVar = var;
                }

                // Higher ratio means more discriminative
                result[$"Feature_{f}"] = NumOps.Equals(minVar, NumOps.Zero)
                    ? NumOps.One
                    : NumOps.Divide(maxVar, minVar);
            }
        }
        else
        {
            // For Multinomial, use entropy
            for (int f = 0; f < _numFeatures; f++)
            {
                var entropy = NumOps.Zero;
                var totalCount = NumOps.Zero;

                // Aggregate counts across all classes
                var valueCounts = new Dictionary<int, T>();
                for (int c = 0; c < _numClasses; c++)
                {
                    foreach (var kvp in _featureCounts[c, f])
                    {
                        if (!valueCounts.ContainsKey(kvp.Key))
                            valueCounts[kvp.Key] = NumOps.Zero;
                        valueCounts[kvp.Key] = NumOps.Add(valueCounts[kvp.Key], kvp.Value);
                        totalCount = NumOps.Add(totalCount, kvp.Value);
                    }
                }

                // Calculate entropy
                if (NumOps.GreaterThan(totalCount, NumOps.Zero))
                {
                    foreach (var count in valueCounts.Values)
                    {
                        var p = NumOps.Divide(count, totalCount);
                        if (NumOps.GreaterThan(p, NumOps.Zero))
                        {
                            entropy = NumOps.Subtract(entropy,
                                NumOps.Multiply(p, NumOps.Log(p)));
                        }
                    }
                }

                result[$"Feature_{f}"] = entropy;
            }
        }

        return result;
    }

    
    /// <inheritdoc/>
    public override int InputDimensions => _numFeatures;
    
    /// <inheritdoc/>
    public override int OutputDimensions => 1;
    
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
        var error = CalculateError(prediction, testLabels);
        
        return new Dictionary<string, double>
        {
            ["Accuracy"] = NumOps.Equals(prediction, testLabels) ? 1.0 : 0.0,
            ["Error"] = Convert.ToDouble(error)
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
                ["NumClasses"] = NumOps.FromDouble(_numClasses),
                ["IsGaussian"] = NumOps.FromDouble(_isGaussian ? 1.0 : 0.0),
                ["Alpha"] = _alpha
            }
        };
    }
    
    /// <inheritdoc/>
    public override void Save()
    {
        // Default implementation saves to a standard location
        SaveModel($"online_naive_bayes_model_{DateTime.Now:yyyyMMddHHmmss}.bin");
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