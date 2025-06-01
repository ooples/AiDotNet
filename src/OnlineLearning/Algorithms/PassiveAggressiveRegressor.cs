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
/// Passive-Aggressive online regression algorithm.
/// </summary>
public class PassiveAggressiveRegressor<T> : OnlineModelBase<T, Vector<T>, T>
{
    private Vector<T> _weights;
    private T _bias;
    private readonly OnlineModelOptions<T> _options;
    private T _epsilon;
    private T _C; // Aggressiveness parameter
    
    /// <summary>
    /// Initializes a new instance of the PassiveAggressiveRegressor class.
    /// </summary>
    public PassiveAggressiveRegressor(
        int inputDimension, 
        OnlineModelOptions<T>? options = null, 
        ILogging? logger = null)
        : base(MathHelper.GetNumericOperations<T>().One, logger) // PA doesn't use traditional learning rate
    {
        _options = options ?? new OnlineModelOptions<T>
        {
            RegularizationParameter = NumOps.Zero
        };
        
        _weights = new Vector<T>(Enumerable.Repeat(NumOps.Zero, inputDimension).ToArray());
        _bias = NumOps.Zero;
        _C = options != null && options.AggressivenessParameter != null 
            ? options.AggressivenessParameter 
            : NumOps.FromDouble(1.0);
        _epsilon = options != null && options.Epsilon != null 
            ? options.Epsilon 
            : NumOps.FromDouble(0.1);
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        // Calculate prediction
        var prediction = PredictRaw(input);
        
        // Calculate loss
        var loss = NumOps.Abs(NumOps.Subtract(prediction, expectedOutput));
        
        // Update only if loss exceeds epsilon (passive when loss is small)
        if (NumOps.GreaterThan(loss, _epsilon))
        {
            // Calculate squared norm of input
            var normSquared = input.Select(x => NumOps.Multiply(x, x))
                .Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
            
            // Add small value to avoid division by zero
            normSquared = NumOps.Add(normSquared, NumOps.FromDouble(1e-10));
            
            // Calculate step size (tau)
            var numerator = NumOps.Subtract(loss, _epsilon);
            var tau = NumOps.Divide(numerator, normSquared);
            
            // Apply aggressiveness constraint
            if (NumOps.GreaterThan(tau, _C))
            {
                tau = _C;
            }
            
            // Calculate signed tau based on prediction error
            var error = NumOps.Subtract(expectedOutput, prediction);
            var signedTau = NumOps.GreaterThan(error, NumOps.Zero) ? tau : NumOps.Negate(tau);
            
            // Update weights: w = w + tau * sign(error) * x
            var update = input.Select(xi => NumOps.Multiply(signedTau, xi)).ToArray();
            _weights = _weights.Add(new Vector<T>(update));
            
            // Update bias
            _bias = NumOps.Add(_bias, signedTau);
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
    public override ModelMetaData<T> GetModelMetaData()
    {
        return new ModelMetaData<T>
        {
            ModelType = ModelType.PassiveAggressive,
            FeatureCount = _weights.Length,
            Complexity = _weights.Length + 1,
            Description = $"Passive-Aggressive Regressor (C={_C}, ε={_epsilon})",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = _samplesSeen,
                ["Aggressiveness"] = Convert.ToDouble(_C),
                ["Epsilon"] = Convert.ToDouble(_epsilon),
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
        
        // Aggressiveness and epsilon
        writer.Write(Convert.ToDouble(_C));
        writer.Write(Convert.ToDouble(_epsilon));
        
        // Options
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
            
            // Aggressiveness and epsilon
            _C = NumOps.FromDouble(reader.ReadDouble());
            _epsilon = NumOps.FromDouble(reader.ReadDouble());
            
            // Options
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
            RegularizationParameter = _options.RegularizationParameter,
            AggressivenessParameter = parameters.ContainsKey("Aggressiveness") 
                ? (T)parameters["Aggressiveness"] 
                : _C,
            Epsilon = parameters.ContainsKey("Epsilon") 
                ? (T)parameters["Epsilon"] 
                : _epsilon
        };
            
        return new PassiveAggressiveRegressor<T>(_weights.Length, newOptions, _logger);
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _weights.Length;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => 1;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var cloneOptions = new OnlineModelOptions<T>
        {
            RegularizationParameter = _options.RegularizationParameter,
            AggressivenessParameter = _C,
            Epsilon = _epsilon
        };
        var clone = new PassiveAggressiveRegressor<T>(_weights.Length, cloneOptions, _logger)
        {
            _weights = new Vector<T>(_weights.ToArray()),
            _bias = _bias,
            _samplesSeen = _samplesSeen
        };
        return clone;
    }
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> DeepCopy()
    {
        // For this model, Clone and DeepCopy are equivalent since we're copying all data
        return Clone();
    }
    
    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Return weights concatenated with bias
        var parameters = new T[_weights.Length + 1];
        for (int i = 0; i < _weights.Length; i++)
        {
            parameters[i] = _weights[i];
        }
        parameters[_weights.Length] = _bias;
        return new Vector<T>(parameters);
    }
    
    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _weights.Length + 1)
        {
            throw new ArgumentException($"Expected {_weights.Length + 1} parameters, but got {parameters.Length}");
        }
        
        // Extract weights
        var weightArray = new T[_weights.Length];
        for (int i = 0; i < _weights.Length; i++)
        {
            weightArray[i] = parameters[i];
        }
        _weights = new Vector<T>(weightArray);
        
        // Extract bias
        _bias = parameters[_weights.Length];
    }
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> WithParameters(Vector<T> parameters)
    {
        var modelOptions = new OnlineModelOptions<T>
        {
            RegularizationParameter = _options.RegularizationParameter,
            AggressivenessParameter = _C,
            Epsilon = _epsilon
        };
        var model = new PassiveAggressiveRegressor<T>(_weights.Length, modelOptions, _logger);
        model.SetParameters(parameters);
        model._samplesSeen = _samplesSeen;
        return model;
    }
    
    /// <inheritdoc/>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        // Return indices of non-zero weights
        var indices = new List<int>();
        for (int i = 0; i < _weights.Length; i++)
        {
            if (!NumOps.Equals(_weights[i], NumOps.Zero))
            {
                indices.Add(i);
            }
        }
        return indices;
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
        var newWeights = new T[_weights.Length];
        for (int i = 0; i < newWeights.Length; i++)
        {
            newWeights[i] = NumOps.Zero;
        }
        
        // Keep only the weights for active features
        foreach (var index in featureIndices)
        {
            if (index >= 0 && index < _weights.Length)
            {
                newWeights[index] = _weights[index];
            }
        }
        
        _weights = new Vector<T>(newWeights);
    }
}