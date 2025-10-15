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
/// Confidence-Weighted (CW) learning algorithm.
/// Maintains a Gaussian distribution over weight vectors and updates both mean and covariance.
/// </summary>
public class ConfidenceWeighted<T> : AdaptiveOnlineModelBase<T, Vector<T>, T>
{
    private Vector<T> _mean; // Mean weight vector
    private Vector<T> _variance; // Diagonal of covariance matrix
    private T _eta; // Confidence parameter (typically 0.95)
    private T _phi; // Φ^(-1)(η) - inverse cumulative distribution function value
    private readonly OnlineModelOptions<T> _options = default!;
    
    /// <summary>
    /// Initializes a new instance of the ConfidenceWeighted class.
    /// </summary>
    public ConfidenceWeighted(int inputDimension, OnlineModelOptions<T>? options = null, ILogging? logger = null)
        : base(options != null ? options.InitialLearningRate : MathHelper.GetNumericOperations<T>().FromDouble(0.95), 
               DriftDetectionMethod.ADWIN, 
               MathHelper.GetNumericOperations<T>().FromDouble(0.5), 
               logger)
    {
        _options = options ?? new OnlineModelOptions<T>
        {
            InitialLearningRate = NumOps.FromDouble(0.95), // Used as confidence parameter
            UseAdaptiveLearningRate = true,
            LearningRateDecay = NumOps.FromDouble(0.0),
            RegularizationParameter = NumOps.FromDouble(0.1)
        };
        
        // Initialize mean to zero
        _mean = new Vector<T>(Enumerable.Repeat(NumOps.Zero, inputDimension).ToArray());
        
        // Initialize variance to a (where a is regularization parameter)
        var initialVariance = _options.RegularizationParameter;
        _variance = new Vector<T>(Enumerable.Repeat(initialVariance, inputDimension).ToArray());
        
        _eta = _options.InitialLearningRate; // Confidence parameter
        _phi = ComputePhiInverse(_eta); // Φ^(-1)(η)
        
        _learningRate = _options.InitialLearningRate;
        AdaptiveLearningRate = _options.UseAdaptiveLearningRate;
        
        // No need to initialize confidence bounds here as they're calculated from variance
    }
    
    /// <summary>
    /// Computes the inverse cumulative distribution function of standard normal at eta.
    /// Approximation of Φ^(-1)(η) for η = 0.95 ≈ 1.645
    /// </summary>
    private T ComputePhiInverse(T eta)
    {
        // For simplicity, using fixed values for common confidence levels
        var etaDouble = Convert.ToDouble(eta);
        double phiInverse;
        
        if (Math.Abs(etaDouble - 0.95) < 0.01)
            phiInverse = 1.645;
        else if (Math.Abs(etaDouble - 0.99) < 0.01)
            phiInverse = 2.326;
        else if (Math.Abs(etaDouble - 0.90) < 0.01)
            phiInverse = 1.282;
        else
            phiInverse = 1.645; // Default to 95% confidence
            
        return NumOps.FromDouble(phiInverse);
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        // Convert expected output to -1 or 1 for binary classification
        var y = NumOps.GreaterThan(expectedOutput, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One);
        
        // Compute prediction: μ^T * x
        var prediction = input.DotProduct(_mean);
        var margin = NumOps.Multiply(y, prediction);
        
        // Compute variance of prediction: v = x^T * Σ * x (where Σ is diagonal)
        var v = ComputeVariance(input);
        
        // Compute margin threshold: M = φ * sqrt(v)
        var M = NumOps.Multiply(_phi, NumOps.Sqrt(v));
        
        // Update only if margin < M (not confident enough)
        if (NumOps.LessThan(margin, M))
        {
            // Compute alpha using closed-form solution
            var alpha = ComputeAlpha(margin, v);
            
            // Compute u = 0.5 * (-alpha * v * φ + sqrt((alpha * v * φ)^2 + 4 * v))
            var alphavphi = NumOps.Multiply(NumOps.Multiply(alpha, v), _phi);
            var discriminant = NumOps.Add(NumOps.Multiply(alphavphi, alphavphi), 
                                         NumOps.Multiply(NumOps.FromDouble(4.0), v));
            var u = NumOps.Multiply(NumOps.FromDouble(0.5), 
                    NumOps.Add(NumOps.Negate(alphavphi), NumOps.Sqrt(discriminant)));
            
            // Compute beta = alpha * φ / (sqrt(u) + v * alpha * φ)
            var denominator = NumOps.Add(NumOps.Sqrt(u), NumOps.Multiply(v, NumOps.Multiply(alpha, _phi)));
            var beta = NumOps.Divide(NumOps.Multiply(alpha, _phi), denominator);
            
            // Update mean: μ = μ + α * y * Σ * x
            for (int i = 0; i < _mean.Length; i++)
            {
                var update = NumOps.Multiply(NumOps.Multiply(alpha, y), 
                            NumOps.Multiply(_variance[i], input[i]));
                _mean[i] = NumOps.Add(_mean[i], update);
            }
            
            // Update variance: Σ_ii = 1 / (1/Σ_ii + 2 * α * φ * x_i^2)
            var newVariance = new T[_variance.Length];
            for (int i = 0; i < _variance.Length; i++)
            {
                var xiSquared = NumOps.Multiply(input[i], input[i]);
                var term = NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(2.0), alpha), 
                          NumOps.Multiply(_phi, xiSquared));
                var invVariance = NumOps.Add(NumOps.Divide(NumOps.One, _variance[i]), term);
                newVariance[i] = NumOps.Divide(NumOps.One, invVariance);
                
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
    /// Computes the optimal alpha for the update.
    /// </summary>
    private T ComputeAlpha(T margin, T v)
    {
        // α = max(0, (φ * sqrt(φ^2 * v + 4 * margin * v) - 2 * margin) / (2 * (φ^2 * v + v)))
        var phiSquared = NumOps.Multiply(_phi, _phi);
        var phiSquaredV = NumOps.Multiply(phiSquared, v);
        var fourMarginV = NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(4.0), margin), v);
        
        var discriminant = NumOps.Add(phiSquaredV, fourMarginV);
        var sqrtDiscriminant = NumOps.Sqrt(discriminant);
        
        var numerator = NumOps.Subtract(NumOps.Multiply(_phi, sqrtDiscriminant), 
                                       NumOps.Multiply(NumOps.FromDouble(2.0), margin));
        var denominator = NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Add(phiSquaredV, v));
        
        var alpha = NumOps.Divide(numerator, denominator);
        return NumOps.GreaterThan(alpha, NumOps.Zero) ? alpha : NumOps.Zero;
    }
    
    /// <summary>
    /// Computes the variance of the prediction.
    /// </summary>
    private T ComputeVariance(Vector<T> input)
    {
        var variance = NumOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            var term = NumOps.Multiply(NumOps.Multiply(input[i], _variance[i]), input[i]);
            variance = NumOps.Add(variance, term);
        }
        return variance;
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
        var variance = ComputeVariance(input);
        var stdDev = NumOps.Sqrt(variance);
        
        // Confidence interval based on eta parameter
        var margin = NumOps.Multiply(_phi, stdDev);
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
            ModelType = ModelType.ConfidenceWeighted,
            FeatureCount = _mean.Length,
            Complexity = _mean.Length * 2, // Mean and variance vectors
            Description = $"Confidence-Weighted learning with {_mean.Length} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = SamplesSeen,
                ["LearningRate"] = Convert.ToDouble(_learningRate),
                ["ConfidenceParameter"] = Convert.ToDouble(_eta),
                ["Phi"] = Convert.ToDouble(_phi),
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
        writer.Write(Convert.ToDouble(_eta));
        writer.Write(Convert.ToDouble(_phi));
        
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
            _eta = NumOps.FromDouble(reader.ReadDouble());
            _phi = NumOps.FromDouble(reader.ReadDouble());
            
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
            InitialLearningRate = parameters.ContainsKey("ConfidenceParameter") 
                ? (T)parameters["ConfidenceParameter"] 
                : _eta,
            UseAdaptiveLearningRate = parameters.ContainsKey("AdaptiveLearningRate") 
                ? (bool)parameters["AdaptiveLearningRate"] 
                : _options.UseAdaptiveLearningRate,
            RegularizationParameter = parameters.ContainsKey("RegularizationParameter") 
                ? (T)parameters["RegularizationParameter"] 
                : _options.RegularizationParameter
        };
        
        return new ConfidenceWeighted<T>(_mean.Length, newOptions, _logger);
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _mean.Length;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => 1;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var clone = new ConfidenceWeighted<T>(_mean.Length, _options, _logger)
        {
            _mean = new Vector<T>(_mean.ToArray()),
            _variance = new Vector<T>(_variance.ToArray()),
            _eta = _eta,
            _phi = _phi,
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
        // All features are active in Confidence-Weighted
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
        // Confidence-Weighted uses all features, so this is a no-op
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
    /// Gets the confidence parameter (eta).
    /// </summary>
    public T ConfidenceParameter => _eta;
    
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
                ["ConfidenceParameter"] = _eta,
                ["Phi"] = _phi
            }
        };
    }
    
    /// <inheritdoc/>
    public override void Save()
    {
        // Default implementation saves to a standard location
        SaveModel($"confidence_weighted_model_{DateTime.Now:yyyyMMddHHmmss}.bin");
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