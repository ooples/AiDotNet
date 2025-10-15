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
/// Adaptive Regularization of Weights (AROW) algorithm.
/// A confidence-weighted online learning algorithm that maintains a diagonal covariance matrix
/// for the weight vector and adapts to different feature scales.
/// </summary>
public class AROW<T> : AdaptiveOnlineModelBase<T, Vector<T>, T>
{
    private Vector<T> _mean; // Mean weight vector
    private Vector<T> _variance; // Diagonal of covariance matrix
    private T _r; // Regularization parameter
    private readonly OnlineModelOptions<T> _options = default!;
    
    /// <summary>
    /// Initializes a new instance of the AROW class.
    /// </summary>
    public AROW(int inputDimension, OnlineModelOptions<T>? options = null, ILogging? logger = null)
        : base(options != null ? options.InitialLearningRate : MathHelper.GetNumericOperations<T>().FromDouble(1.0), 
               DriftDetectionMethod.ADWIN, 
               MathHelper.GetNumericOperations<T>().FromDouble(0.5), 
               logger)
    {
        _options = options ?? new OnlineModelOptions<T>
        {
            InitialLearningRate = NumOps.FromDouble(1.0),
            UseAdaptiveLearningRate = true,
            LearningRateDecay = NumOps.FromDouble(0.0),
            RegularizationParameter = NumOps.FromDouble(1.0)
        };
        
        // Initialize mean to zero
        _mean = new Vector<T>(Enumerable.Repeat(NumOps.Zero, inputDimension).ToArray());
        
        // Initialize variance to 1 (identity covariance)
        _variance = new Vector<T>(Enumerable.Repeat(NumOps.One, inputDimension).ToArray());
        
        _r = _options.RegularizationParameter;
        _learningRate = _options.InitialLearningRate;
        AdaptiveLearningRate = _options.UseAdaptiveLearningRate;
        
        // No need to initialize confidence bounds here as they're calculated from variance
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        // Convert expected output to -1 or 1 for binary classification
        var y = NumOps.GreaterThan(expectedOutput, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One);
        
        // Compute prediction: μ^T * x
        var prediction = input.DotProduct(_mean);
        var margin = NumOps.Multiply(y, prediction);
        
        // Compute confidence: x^T * Σ * x (where Σ is diagonal)
        var confidence = ComputeConfidence(input);
        
        // Compute loss: ℓ = max(0, 1 - margin)
        var diff = NumOps.Subtract(NumOps.One, margin);
        var loss = NumOps.GreaterThan(diff, NumOps.Zero) ? diff : NumOps.Zero;
        
        if (NumOps.GreaterThan(loss, NumOps.Zero))
        {
            // Update only if there's a loss
            
            // Compute beta: 1 / (confidence + r)
            var beta = NumOps.Divide(NumOps.One, NumOps.Add(confidence, _r));
            
            // Compute alpha: loss * beta
            var alpha = NumOps.Multiply(loss, beta);
            
            // Update mean: μ = μ + α * y * Σ * x
            for (int i = 0; i < _mean.Length; i++)
            {
                var update = NumOps.Multiply(NumOps.Multiply(alpha, y), 
                            NumOps.Multiply(_variance[i], input[i]));
                _mean[i] = NumOps.Add(_mean[i], update);
            }
            
            // Update variance: Σ = Σ - β * Σ * x * x^T * Σ
            var newVariance = new T[_variance.Length];
            for (int i = 0; i < _variance.Length; i++)
            {
                var xiSquared = NumOps.Multiply(input[i], input[i]);
                var varianceUpdate = NumOps.Multiply(NumOps.Multiply(beta, _variance[i]), 
                                    NumOps.Multiply(xiSquared, _variance[i]));
                newVariance[i] = NumOps.Subtract(_variance[i], varianceUpdate);
                
                // Ensure variance doesn't become too small
                if (NumOps.LessThan(newVariance[i], NumOps.FromDouble(0.0001)))
                {
                    newVariance[i] = NumOps.FromDouble(0.0001);
                }
            }
            _variance = new Vector<T>(newVariance);
            
            // Confidence bounds are calculated dynamically from variance
        }
    }
    
    /// <summary>
    /// Computes the confidence score for a given input.
    /// </summary>
    private T ComputeConfidence(Vector<T> input)
    {
        var confidence = NumOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            var term = NumOps.Multiply(NumOps.Multiply(input[i], _variance[i]), input[i]);
            confidence = NumOps.Add(confidence, term);
        }
        return confidence;
    }
    
    /// <inheritdoc/>
    public override T Predict(Vector<T> input)
    {
        var rawPrediction = PredictRaw(input);
        // Return 1 if positive, 0 if negative (for binary classification)
        return NumOps.GreaterThan(rawPrediction, NumOps.Zero) ? NumOps.One : NumOps.Zero;
    }
    
    /// <summary>
    /// Gets the raw decision function value.
    /// </summary>
    protected T PredictRaw(Vector<T> input)
    {
        return input.DotProduct(_mean);
    }
    
    /// <summary>
    /// Gets the prediction with confidence interval.
    /// </summary>
    public (T prediction, T lowerBound, T upperBound) PredictWithConfidence(Vector<T> input)
    {
        var prediction = PredictRaw(input);
        var confidence = ComputeConfidence(input);
        var stdDev = NumOps.Sqrt(confidence);
        
        // 95% confidence interval (approximately 2 standard deviations)
        var margin = NumOps.Multiply(NumOps.FromDouble(2.0), stdDev);
        var lowerBound = NumOps.Subtract(prediction, margin);
        var upperBound = NumOps.Add(prediction, margin);
        
        return (prediction, lowerBound, upperBound);
    }
    
    /// <inheritdoc/>
    protected override T CalculateError(T prediction, T expectedOutput)
    {
        // For binary classification, calculate 0-1 loss
        var y = NumOps.GreaterThan(expectedOutput, NumOps.Zero) ? NumOps.One : NumOps.Zero;
        var predY = NumOps.GreaterThan(prediction, NumOps.Zero) ? NumOps.One : NumOps.Zero;
        return NumOps.Equals(y, predY) ? NumOps.Zero : NumOps.One;
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
            ModelType = ModelType.AROW,
            FeatureCount = _mean.Length,
            Complexity = _mean.Length * 2, // Mean and variance vectors
            Description = $"AROW (Adaptive Regularization of Weights) with {_mean.Length} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = SamplesSeen,
                ["LearningRate"] = Convert.ToDouble(_learningRate),
                ["RegularizationParameter"] = Convert.ToDouble(_r),
                ["DriftDetected"] = DriftDetected,
                ["DriftLevel"] = Convert.ToDouble(DriftLevel)
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
        writer.Write(_mean.Length);
        writer.Write(Convert.ToDouble(_r));
        
        // Mean vector
        for (int i = 0; i < _mean.Length; i++)
        {
            writer.Write(Convert.ToDouble(_mean[i]));
        }
        
        // Variance vector
        for (int i = 0; i < _variance.Length; i++)
        {
            writer.Write(Convert.ToDouble(_variance[i]));
        }
        
        // Samples seen
        writer.Write(SamplesSeen);
        
        // Learning rate
        writer.Write(Convert.ToDouble(_learningRate));
        
        // Drift detection
        writer.Write((int)_driftMethod);
        writer.Write(Convert.ToDouble(_driftSensitivity));
        writer.Write(_driftDetected);
        writer.Write(Convert.ToDouble(_driftLevel));
        
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
            int featureCount = reader.ReadInt32();
            _r = NumOps.FromDouble(reader.ReadDouble());
            
            // Mean vector
            var mean = new T[featureCount];
            for (int i = 0; i < featureCount; i++)
            {
                mean[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _mean = new Vector<T>(mean);
            
            // Variance vector
            var variance = new T[featureCount];
            for (int i = 0; i < featureCount; i++)
            {
                variance[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _variance = new Vector<T>(variance);
            
            // Samples seen
            _samplesSeen = reader.ReadInt64();
            
            // Learning rate
            _learningRate = NumOps.FromDouble(reader.ReadDouble());
            
            // Drift detection
            _driftMethod = (DriftDetectionMethod)reader.ReadInt32();
            _driftSensitivity = NumOps.FromDouble(reader.ReadDouble());
            _driftDetected = reader.ReadBoolean();
            _driftLevel = NumOps.FromDouble(reader.ReadDouble());
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
            RegularizationParameter = parameters.ContainsKey("RegularizationParameter") 
                ? (T)parameters["RegularizationParameter"] 
                : _r
        };
        
        return new AROW<T>(_mean.Length, newOptions, _logger);
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _mean.Length;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => 1;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var clone = new AROW<T>(_mean.Length, _options, _logger)
        {
            _mean = new Vector<T>(_mean.ToArray()),
            _variance = new Vector<T>(_variance.ToArray()),
            _r = _r,
            _samplesSeen = SamplesSeen,
            _learningRate = _learningRate,
            _adaptiveLearningRate = _adaptiveLearningRate,
            _driftMethod = _driftMethod,
            _driftSensitivity = _driftSensitivity,
            _driftDetected = _driftDetected,
            _driftLevel = _driftLevel
        };
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
        // Return mean and variance as a single vector
        var parameters = new T[_mean.Length * 2];
        _mean.ToArray().CopyTo(parameters, 0);
        _variance.ToArray().CopyTo(parameters, _mean.Length);
        return new Vector<T>(parameters);
    }
    
    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _mean.Length * 2)
        {
            throw new ArgumentException($"Parameter vector must have length {_mean.Length * 2}");
        }
        
        _mean = new Vector<T>(parameters.Take(_mean.Length).ToArray());
        _variance = new Vector<T>(parameters.Skip(_mean.Length).Take(_mean.Length).ToArray());
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
        // All features are active in AROW
        return Enumerable.Range(0, _mean.Length);
    }
    
    /// <inheritdoc/>
    public override bool IsFeatureUsed(int featureIndex)
    {
        return featureIndex >= 0 && featureIndex < _mean.Length;
    }
    
    /// <inheritdoc/>
    public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // AROW uses all features, so this is a no-op
        // Could implement feature masking if needed
    }
    
    /// <summary>
    /// Gets the current mean weight vector.
    /// </summary>
    public Vector<T> Mean => _mean.Clone();
    
    /// <summary>
    /// Gets the current variance (diagonal of covariance matrix).
    /// </summary>
    public Vector<T> Variance => _variance.Clone();
    
    /// <summary>
    /// Gets the feature importance based on the inverse of variance.
    /// Lower variance indicates higher confidence in the feature weight.
    /// </summary>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();
        for (int i = 0; i < _variance.Length; i++)
        {
            // Importance is inversely proportional to variance
            result[$"Feature_{i}"] = NumOps.Divide(NumOps.One, _variance[i]);
        }
        return result;
    }
    
    /// <inheritdoc/>
    public override int InputDimensions => _mean.Length;
    
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
        if (_recentErrors.Count == 0)
            return 0.0;
            
        var avgError = CalculateMean(_recentErrors.ToArray());
        return Convert.ToDouble(avgError);
    }
    
    /// <inheritdoc/>
    public override double GetValidationLoss()
    {
        // In online learning, we don't have separate validation loss
        // Return the same as training loss
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
                ["DriftLevel"] = _driftLevel,
                ["RegularizationParameter"] = _r
            }
        };
    }
    
    /// <inheritdoc/>
    public override void Save()
    {
        // Default implementation saves to a standard location
        SaveModel($"arow_model_{DateTime.Now:yyyyMMddHHmmss}.bin");
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
        _recentErrors?.Clear();
    }
}