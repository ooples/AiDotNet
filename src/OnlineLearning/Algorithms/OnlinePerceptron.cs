using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Helpers;

namespace AiDotNet.OnlineLearning.Algorithms;

/// <summary>
/// Online Perceptron for binary classification.
/// </summary>
public class OnlinePerceptron<T> : OnlineModelBase<T, Vector<T>, T>
{
    private Vector<T> _weights;
    private T _bias;
    private readonly OnlineModelOptions<T> _options;
    
    /// <summary>
    /// Initializes a new instance of the OnlinePerceptron class.
    /// </summary>
    public OnlinePerceptron(int inputDimension, OnlineModelOptions<T>? options = null, ILogging? logger = null)
        : base(options != null ? options.InitialLearningRate : MathHelper.GetNumericOperations<T>().FromDouble(0.1), logger)
    {
        _options = options ?? new OnlineModelOptions<T>
        {
            InitialLearningRate = NumOps.FromDouble(0.1),
            UseAdaptiveLearningRate = true,
            LearningRateDecay = NumOps.FromDouble(0.001),
            RegularizationParameter = NumOps.Zero
        };
        
        _weights = new Vector<T>(Enumerable.Repeat(NumOps.Zero, inputDimension).ToArray());
        _bias = NumOps.Zero;
        _learningRate = _options.InitialLearningRate;
        AdaptiveLearningRate = _options.UseAdaptiveLearningRate;
        
        // Initialize ModelParameters
        UpdateModelParameters();
    }
    
    private void UpdateModelParameters()
    {
        // Update ModelParameters to reflect weights and bias
        var parameters = new T[_weights.Length + 1];
        _weights.ToArray().CopyTo(parameters, 0);
        parameters[_weights.Length] = _bias;
        ModelParameters = new Vector<T>(parameters);
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        // Predict current output
        var prediction = PredictRaw(input);
        
        // Convert expected output to -1 or 1
        var y = NumOps.GreaterThan(expectedOutput, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One);
        
        // Check if update is needed (perceptron update rule)
        var yPred = NumOps.Multiply(y, prediction);
        if (NumOps.LessThanOrEquals(yPred, NumOps.Zero))
        {
            // Update weights: w = w + learning_rate * y * x
            var update = input.Select(xi => NumOps.Multiply(NumOps.Multiply(learningRate, y), xi)).ToArray();
            _weights = _weights.Add(new Vector<T>(update));
            
            // Update bias: b = b + learning_rate * y
            _bias = NumOps.Add(_bias, NumOps.Multiply(learningRate, y));
            
            // Apply L2 regularization if configured
            if (!NumOps.Equals(_options.RegularizationParameter, NumOps.Zero))
            {
                var regularization = NumOps.Multiply(_options.RegularizationParameter, learningRate);
                _weights = new Vector<T>(_weights.Select(w => NumOps.Multiply(w, NumOps.Subtract(NumOps.One, regularization))).ToArray());
            }
            
            // Update ModelParameters to reflect the new weights and bias
            UpdateModelParameters();
        }
    }
    
    /// <summary>
    /// Gets the raw prediction value (before thresholding).
    /// </summary>
    private T PredictRaw(Vector<T> input)
    {
        var dotProduct = _weights.Zip(input, (w, x) => NumOps.Multiply(w, x))
            .Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
        return NumOps.Add(dotProduct, _bias);
    }
    
    /// <inheritdoc/>
    public override void Train(Vector<T> input, T expectedOutput)
    {
        PartialFit(input, expectedOutput);
    }
    
    /// <inheritdoc/>
    public override T Predict(Vector<T> input)
    {
        var raw = PredictRaw(input);
        return NumOps.GreaterThan(raw, NumOps.Zero) ? NumOps.One : NumOps.Zero;
    }
    
    /// <inheritdoc/>
    public override ModelMetaData<T> GetModelMetaData()
    {
        return new ModelMetaData<T>
        {
            ModelType = ModelType.OnlinePerceptron,
            FeatureCount = _weights.Length,
            Complexity = _weights.Length + 1,
            Description = $"Online Perceptron with {_weights.Length} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = _samplesSeen,
                ["LearningRate"] = Convert.ToDouble(_learningRate),
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
        
        // Samples seen
        writer.Write(_samplesSeen);
        
        // Learning rate
        writer.Write(Convert.ToDouble(_learningRate));
        
        // Options
        writer.Write(_options.UseAdaptiveLearningRate);
        writer.Write(Convert.ToDouble(_options.RegularizationParameter));
        
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
            
            // Samples seen
            _samplesSeen = reader.ReadInt64();
            
            // Learning rate
            _learningRate = NumOps.FromDouble(reader.ReadDouble());
            
            // Options
            _options.UseAdaptiveLearningRate = reader.ReadBoolean();
            _options.RegularizationParameter = NumOps.FromDouble(reader.ReadDouble());
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
            RegularizationParameter = parameters.ContainsKey("Regularization") 
                ? (T)parameters["Regularization"] 
                : _options.RegularizationParameter
        };
        
        return new OnlinePerceptron<T>(_weights.Length, newOptions, _logger);
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _weights.Length;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => 1;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var clone = new OnlinePerceptron<T>(_weights.Length, _options, _logger)
        {
            _weights = new Vector<T>(_weights.ToArray()),
            _bias = _bias,
            _samplesSeen = _samplesSeen,
            _learningRate = _learningRate,
            _adaptiveLearningRate = _adaptiveLearningRate
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
        // Return weights and bias as a single vector
        var parameters = new T[_weights.Length + 1];
        _weights.ToArray().CopyTo(parameters, 0);
        parameters[_weights.Length] = _bias;
        return new Vector<T>(parameters);
    }
    
    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _weights.Length + 1)
        {
            throw new ArgumentException($"Parameter vector must have length {_weights.Length + 1}");
        }
        
        _weights = new Vector<T>(parameters.Take(_weights.Length).ToArray());
        _bias = parameters[_weights.Length];
        
        // Update ModelParameters to reflect the new weights and bias
        UpdateModelParameters();
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
        // All features are active in perceptron
        return Enumerable.Range(0, _weights.Length);
    }
    
    /// <inheritdoc/>
    public override bool IsFeatureUsed(int featureIndex)
    {
        return featureIndex >= 0 && featureIndex < _weights.Length;
    }
    
    /// <inheritdoc/>
    public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Perceptron uses all features, so this is a no-op
        // Could implement feature masking if needed
    }
}