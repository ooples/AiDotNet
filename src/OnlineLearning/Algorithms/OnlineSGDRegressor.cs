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
/// Online Stochastic Gradient Descent regressor with various loss functions.
/// </summary>
public class OnlineSGDRegressor<T> : AdaptiveOnlineModelBase<T, Vector<T>, T>
{
    private Vector<T> _weights = default!;
    private T _bias = default!;
    private Vector<T>? _momentum; // For momentum-based updates
    private readonly AdaptiveOnlineModelOptions<T> _options = default!;
    private FitnessCalculatorType _lossType = default!;
    
    /// <summary>
    /// Initializes a new instance of the OnlineSGDRegressor class.
    /// </summary>
    public OnlineSGDRegressor(
        int inputDimension,
        FitnessCalculatorType lossType = FitnessCalculatorType.MeanSquaredError,
        AdaptiveOnlineModelOptions<T>? options = null,
        DriftDetectionMethod driftMethod = DriftDetectionMethod.ADWIN,
        ILogging? logger = null)
        : base(
            options != null ? options.InitialLearningRate : MathHelper.GetNumericOperations<T>().FromDouble(0.01),
            driftMethod,
            options != null ? options.DriftSensitivity : MathHelper.GetNumericOperations<T>().FromDouble(0.5),
            logger)
    {
        _options = options ?? new AdaptiveOnlineModelOptions<T>
        {
            InitialLearningRate = NumOps.FromDouble(0.01),
            UseAdaptiveLearningRate = true,
            LearningRateDecay = NumOps.FromDouble(0.0001),
            RegularizationParameter = NumOps.FromDouble(0.0001),
            UseMomentum = true,
            MomentumFactor = NumOps.FromDouble(0.9)
        };
        
        _lossType = lossType;
        _weights = new Vector<T>(Enumerable.Repeat(NumOps.Zero, inputDimension).ToArray());
        _bias = NumOps.Zero;
        
        if (_options.UseMomentum)
        {
            _momentum = new Vector<T>(Enumerable.Repeat(NumOps.Zero, inputDimension).ToArray());
        }
        
        _learningRate = _options.InitialLearningRate;
        AdaptiveLearningRate = _options.UseAdaptiveLearningRate;
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        // Calculate prediction
        var prediction = PredictRaw(input);
        
        // Calculate gradient based on loss function
        var gradient = CalculateGradient(prediction, expectedOutput, input);
        
        // Apply momentum if enabled
        if (_options.UseMomentum && _momentum != null)
        {
            // v = momentum * v + (1 - momentum) * gradient
            var scaledGradient = gradient.Select(g => 
                NumOps.Multiply(NumOps.Subtract(NumOps.One, _options.MomentumFactor), g)).ToArray();
            var scaledMomentum = _momentum.Select(m => 
                NumOps.Multiply(_options.MomentumFactor, m)).ToArray();
            
            _momentum = new Vector<T>(
                scaledMomentum.Zip(scaledGradient, (m, g) => NumOps.Add(m, g)).ToArray()
            );
            
            gradient = _momentum;
        }
        
        // Update weights with gradient and regularization
        for (int i = 0; i < _weights.Length; i++)
        {
            // Apply L2 regularization
            var regularization = NumOps.Multiply(_options.RegularizationParameter, _weights[i]);
            var update = NumOps.Add(gradient[i], regularization);
            
            // Update weight
            _weights[i] = NumOps.Subtract(_weights[i], NumOps.Multiply(learningRate, update));
        }
        
        // Update bias (no regularization on bias)
        var biasGradient = CalculateBiasGradient(prediction, expectedOutput);
        _bias = NumOps.Subtract(_bias, NumOps.Multiply(learningRate, biasGradient));
    }
    
    /// <summary>
    /// Calculates the gradient based on the loss function.
    /// </summary>
    private Vector<T> CalculateGradient(T prediction, T expectedOutput, Vector<T> input)
    {
        T errorDerivative;
        
        switch (_lossType)
        {
            case FitnessCalculatorType.MeanSquaredError:
                // Derivative: 2 * (prediction - expected)
                errorDerivative = NumOps.Multiply(
                    NumOps.FromDouble(2.0),
                    NumOps.Subtract(prediction, expectedOutput)
                );
                break;
                
            case FitnessCalculatorType.MeanAbsoluteError:
                // Derivative: sign(prediction - expected)
                var diff = NumOps.Subtract(prediction, expectedOutput);
                errorDerivative = NumOps.GreaterThan(diff, NumOps.Zero) 
                    ? NumOps.One 
                    : NumOps.Negate(NumOps.One);
                break;
                
            case FitnessCalculatorType.HuberLoss:
                // Huber loss with delta = 1.0
                var delta = NumOps.One;
                var absDiff = NumOps.Abs(NumOps.Subtract(prediction, expectedOutput));
                if (NumOps.LessThanOrEquals(absDiff, delta))
                {
                    errorDerivative = NumOps.Subtract(prediction, expectedOutput);
                }
                else
                {
                    errorDerivative = NumOps.Multiply(delta, 
                        NumOps.GreaterThan(NumOps.Subtract(prediction, expectedOutput), NumOps.Zero) 
                            ? NumOps.One 
                            : NumOps.Negate(NumOps.One));
                }
                break;
                
            default:
                // Default to MSE
                errorDerivative = NumOps.Multiply(
                    NumOps.FromDouble(2.0),
                    NumOps.Subtract(prediction, expectedOutput)
                );
                break;
        }
        
        // Gradient = error_derivative * input
        return new Vector<T>(input.Select(xi => NumOps.Multiply(errorDerivative, xi)).ToArray());
    }
    
    /// <summary>
    /// Calculates the bias gradient.
    /// </summary>
    private T CalculateBiasGradient(T prediction, T expectedOutput)
    {
        switch (_lossType)
        {
            case FitnessCalculatorType.MeanSquaredError:
                return NumOps.Multiply(
                    NumOps.FromDouble(2.0),
                    NumOps.Subtract(prediction, expectedOutput)
                );
                
            case FitnessCalculatorType.MeanAbsoluteError:
                var diff = NumOps.Subtract(prediction, expectedOutput);
                return NumOps.GreaterThan(diff, NumOps.Zero) 
                    ? NumOps.One 
                    : NumOps.Negate(NumOps.One);
                
            default:
                return NumOps.Multiply(
                    NumOps.FromDouble(2.0),
                    NumOps.Subtract(prediction, expectedOutput)
                );
        }
    }
    
    /// <summary>
    /// Gets the raw prediction value.
    /// </summary>
    private T PredictRaw(Vector<T> input)
    {
        var dotProduct = _weights.Zip(input, (w, x) => NumOps.Multiply(w, x))
            .Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
        return NumOps.Add(dotProduct, _bias);
    }
    
    /// <inheritdoc/>
    protected override T CalculateError(T prediction, T expectedOutput)
    {
        return NumOps.Abs(NumOps.Subtract(prediction, expectedOutput));
    }
    
    /// <inheritdoc/>
    public override void Train(Vector<T> input, T expectedOutput)
    {
        PartialFit(input, expectedOutput);
    }
    
    /// <inheritdoc/>
    public override T Predict(Vector<T> input)
    {
        return PredictRaw(input);
    }
    
    /// <inheritdoc/>
    protected override T GetAdaptiveLearningRate()
    {
        if (!_adaptiveLearningRate)
        {
            return _learningRate;
        }
        
        // Use decay schedule from options
        var t = NumOps.FromDouble((double)_samplesSeen);
        var denominator = NumOps.Add(NumOps.One, NumOps.Multiply(_options.LearningRateDecay, t));
        return NumOps.Divide(_options.InitialLearningRate, denominator);
    }
    
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.OnlineSGD,
            FeatureCount = _weights.Length,
            Complexity = _weights.Length + 1,
            Description = $"Online SGD Regressor with {_lossType} loss",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = _samplesSeen,
                ["LearningRate"] = Convert.ToDouble(_learningRate),
                ["LossType"] = _lossType.ToString(),
                ["DriftDetected"] = DriftDetected,
                ["DriftLevel"] = Convert.ToDouble(DriftLevel),
                ["WeightNorm"] = Convert.ToDouble(_weights.Select(w => NumOps.Multiply(w, w))
                    .Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val)))
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
        
        // Weights
        writer.Write(_weights.Length);
        for (int i = 0; i < _weights.Length; i++)
        {
            writer.Write(Convert.ToDouble(_weights[i]));
        }
        
        // Bias
        writer.Write(Convert.ToDouble(_bias));
        
        // Momentum
        writer.Write(_momentum != null);
        if (_momentum != null)
        {
            for (int i = 0; i < _momentum.Length; i++)
            {
                writer.Write(Convert.ToDouble(_momentum[i]));
            }
        }
        
        // Samples seen
        writer.Write(_samplesSeen);
        
        // Learning rate
        writer.Write(Convert.ToDouble(_learningRate));
        
        // Loss type
        writer.Write((int)_lossType);
        
        // Options
        writer.Write(_options.UseAdaptiveLearningRate);
        writer.Write(Convert.ToDouble(_options.LearningRateDecay));
        writer.Write(Convert.ToDouble(_options.RegularizationParameter));
        writer.Write(_options.UseMomentum);
        writer.Write(Convert.ToDouble(_options.MomentumFactor));
        
        // Drift parameters
        writer.Write((int)_driftMethod);
        writer.Write(Convert.ToDouble(_driftSensitivity));
        
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
            
            // Weights
            int weightCount = reader.ReadInt32();
            var weights = new T[weightCount];
            for (int i = 0; i < weightCount; i++)
            {
                weights[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _weights = new Vector<T>(weights);
            
            // Bias
            _bias = NumOps.FromDouble(reader.ReadDouble());
            
            // Momentum
            bool hasMomentum = reader.ReadBoolean();
            if (hasMomentum)
            {
                var momentum = new T[weightCount];
                for (int i = 0; i < weightCount; i++)
                {
                    momentum[i] = NumOps.FromDouble(reader.ReadDouble());
                }
                _momentum = new Vector<T>(momentum);
            }
            
            // Samples seen
            _samplesSeen = reader.ReadInt64();
            
            // Learning rate
            _learningRate = NumOps.FromDouble(reader.ReadDouble());
            
            // Loss type
            _lossType = (FitnessCalculatorType)reader.ReadInt32();
            
            // Options
            _options.UseAdaptiveLearningRate = reader.ReadBoolean();
            _options.LearningRateDecay = NumOps.FromDouble(reader.ReadDouble());
            _options.RegularizationParameter = NumOps.FromDouble(reader.ReadDouble());
            _options.UseMomentum = reader.ReadBoolean();
            _options.MomentumFactor = NumOps.FromDouble(reader.ReadDouble());
            
            // Drift parameters
            _driftMethod = (DriftDetectionMethod)reader.ReadInt32();
            _driftSensitivity = NumOps.FromDouble(reader.ReadDouble());
        }
        catch (Exception ex) when (!(ex is ArgumentNullException || ex is ArgumentException))
        {
            throw new ArgumentException("Failed to deserialize the model. The data may be corrupted or in an invalid format.", nameof(data), ex);
        }
    }
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> WithParameters(IDictionary<string, object> parameters)
    {
        var newOptions = new AdaptiveOnlineModelOptions<T>
        {
            InitialLearningRate = parameters.ContainsKey("LearningRate") 
                ? (T)parameters["LearningRate"] 
                : _options.InitialLearningRate,
            UseAdaptiveLearningRate = parameters.ContainsKey("AdaptiveLearningRate") 
                ? (bool)parameters["AdaptiveLearningRate"] 
                : _options.UseAdaptiveLearningRate,
            RegularizationParameter = parameters.ContainsKey("Regularization") 
                ? (T)parameters["Regularization"] 
                : _options.RegularizationParameter,
            UseMomentum = parameters.ContainsKey("UseMomentum") 
                ? (bool)parameters["UseMomentum"] 
                : _options.UseMomentum,
            MomentumFactor = parameters.ContainsKey("MomentumFactor") 
                ? (T)parameters["MomentumFactor"] 
                : _options.MomentumFactor,
            LearningRateDecay = _options.LearningRateDecay,
            DriftSensitivity = _options.DriftSensitivity
        };
        
        var lossType = parameters.ContainsKey("LossType") 
            ? (FitnessCalculatorType)parameters["LossType"] 
            : _lossType;
            
        return new OnlineSGDRegressor<T>(_weights.Length, lossType, newOptions, _driftMethod, _logger);
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _weights.Length;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => 1;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var clone = new OnlineSGDRegressor<T>(_weights.Length, _lossType, _options, _driftMethod, _logger)
        {
            _weights = new Vector<T>(_weights.ToArray()),
            _bias = _bias,
            _samplesSeen = _samplesSeen,
            _learningRate = _learningRate,
            _adaptiveLearningRate = _adaptiveLearningRate,
            _driftSensitivity = _driftSensitivity
        };
        
        if (_momentum != null)
        {
            clone._momentum = new Vector<T>(_momentum.ToArray());
        }
        
        return clone;
    }
    
    /// <inheritdoc/>
    protected override void OnDriftAdaptation()
    {
        // Reset momentum when drift is detected
        if (_momentum != null)
        {
            _momentum = new Vector<T>(Enumerable.Repeat(NumOps.Zero, _weights.Length).ToArray());
        }
    }
    
    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Return weights concatenated with bias
        var parameters = new T[_weights.Length + 1];
        Array.Copy(_weights.ToArray(), parameters, _weights.Length);
        parameters[_weights.Length] = _bias;
        return new Vector<T>(parameters);
    }
    
    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _weights.Length + 1)
        {
            throw new ArgumentException($"Expected {_weights.Length + 1} parameters but got {parameters.Length}");
        }
        
        var paramArray = parameters.ToArray();
        _weights = new Vector<T>(paramArray.Take(_weights.Length).ToArray());
        _bias = paramArray[_weights.Length];
    }
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> WithParameters(Vector<T> parameters)
    {
        var clone = (OnlineSGDRegressor<T>)Clone();
        clone.SetParameters(parameters);
        return clone;
    }
    
    /// <inheritdoc/>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        // Return indices where weights are non-zero
        for (int i = 0; i < _weights.Length; i++)
        {
            if (!NumOps.Equals(_weights[i], NumOps.Zero))
            {
                yield return i;
            }
        }
    }
    
    /// <inheritdoc/>
    public override bool IsFeatureUsed(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= _weights.Length)
        {
            return false;
        }
        return !NumOps.Equals(_weights[featureIndex], NumOps.Zero);
    }
    
    /// <inheritdoc/>
    public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Reset all weights to zero
        var newWeights = Enumerable.Repeat(NumOps.Zero, _weights.Length).ToArray();
        
        // Keep only the specified features
        foreach (var index in featureIndices)
        {
            if (index >= 0 && index < _weights.Length)
            {
                newWeights[index] = _weights[index];
            }
        }
        
        _weights = new Vector<T>(newWeights);
        
        // Reset momentum for inactive features if using momentum
        if (_momentum != null)
        {
            var newMomentum = Enumerable.Repeat(NumOps.Zero, _weights.Length).ToArray();
            foreach (var index in featureIndices)
            {
                if (index >= 0 && index < _weights.Length)
                {
                    newMomentum[index] = _momentum[index];
                }
            }
            _momentum = new Vector<T>(newMomentum);
        }
    }
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> DeepCopy()
    {
        // Clone already creates a deep copy
        return Clone();
    }

    
    /// <inheritdoc/>
    public override int InputDimensions => _weights.Length;
    
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
                ["LossType"] = NumOps.FromDouble((int)_lossType)
            }
        };
    }
    
    /// <inheritdoc/>
    public override void Save()
    {
        // Default implementation saves to a standard location
        SaveModel($"online_sgd_regressor_model_{DateTime.Now:yyyyMMddHHmmss}.bin");
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
        // Clean up any resources if needed _recentErrors?.Clear();
    }
}