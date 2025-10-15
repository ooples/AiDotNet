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
/// Follow-The-Regularized-Leader (FTRL) algorithm for online learning.
/// Particularly effective for sparse, high-dimensional problems like click-through rate prediction.
/// </summary>
public class FTRL<T> : OnlineModelBase<T, Vector<T>, T>
{
    private readonly Dictionary<int, T> _z; // Per-coordinate learning rate schedules
    private readonly Dictionary<int, T> _n; // Sum of squared gradients
    private readonly Dictionary<int, T> _weights; // Model weights (sparse)
    private readonly OnlineModelOptions<T> _options = default!;
    private T _alpha; // Learning rate parameter
    private T _beta; // Learning rate parameter
    private T _lambda1; // L1 regularization
    private T _lambda2; // L2 regularization
    private readonly int _numFeatures;
    
    /// <summary>
    /// Initializes a new instance of the FTRL class.
    /// </summary>
    public FTRL(int numFeatures, OnlineModelOptions<T>? options = null, ILogging? logger = null)
        : base(options != null ? options.InitialLearningRate : MathHelper.GetNumericOperations<T>().FromDouble(0.1), logger)
    {
        _options = options ?? new OnlineModelOptions<T>
        {
            InitialLearningRate = NumOps.FromDouble(0.1),
            UseAdaptiveLearningRate = true,
            LearningRateDecay = NumOps.FromDouble(1.0), // Used as beta
            RegularizationParameter = NumOps.FromDouble(1.0) // Used as lambda1
        };
        
        _numFeatures = numFeatures;
        _z = new Dictionary<int, T>();
        _n = new Dictionary<int, T>();
        _weights = new Dictionary<int, T>();
        
        _alpha = _options.InitialLearningRate;
        _beta = _options.LearningRateDecay;
        _lambda1 = _options.RegularizationParameter;
        _lambda2 = NumOps.FromDouble(1.0); // Default L2 regularization
        
        _learningRate = _alpha;
        AdaptiveLearningRate = _options.UseAdaptiveLearningRate;
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        // First, compute current weights based on z and n
        UpdateWeights(input);
        
        // Make prediction with current weights
        var prediction = PredictRaw(input);
        
        // Compute gradient (for logistic regression)
        var p = Sigmoid(prediction);
        var gradient = NumOps.Subtract(p, expectedOutput);
        
        // Update z and n for non-zero features
        for (int i = 0; i < input.Length; i++)
        {
            if (!NumOps.Equals(input[i], NumOps.Zero))
            {
                // Get current values or initialize
                if (!_n.ContainsKey(i))
                {
                    _n[i] = NumOps.Zero;
                    _z[i] = NumOps.Zero;
                }
                
                var g_i = NumOps.Multiply(gradient, input[i]);
                var sigma_i = NumOps.Subtract(
                    NumOps.Sqrt(NumOps.Add(_n[i], NumOps.Multiply(g_i, g_i))),
                    NumOps.Sqrt(_n[i])
                );
                
                // Update z[i]
                var z_update = NumOps.Subtract(g_i, NumOps.Multiply(sigma_i, GetWeight(i)));
                _z[i] = NumOps.Add(_z[i], z_update);
                
                // Update n[i]
                _n[i] = NumOps.Add(_n[i], NumOps.Multiply(g_i, g_i));
            }
        }
    }
    
    /// <summary>
    /// Updates weights for active features based on FTRL closed-form solution.
    /// </summary>
    private void UpdateWeights(Vector<T> input)
    {
        for (int i = 0; i < input.Length; i++)
        {
            if (!NumOps.Equals(input[i], NumOps.Zero) && _z.ContainsKey(i))
            {
                var z_i = _z[i];
                var n_i = _n[i];
                
                // Check if |z_i| > lambda1
                var abs_z = NumOps.Abs(z_i);
                if (NumOps.GreaterThan(abs_z, _lambda1))
                {
                    // Compute weight using closed-form solution
                    var sign_z = NumOps.GreaterThan(z_i, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One);
                    var numerator = NumOps.Multiply(sign_z, NumOps.Subtract(abs_z, _lambda1));
                    var denominator = NumOps.Add(
                        NumOps.Divide(NumOps.Add(_beta, NumOps.Sqrt(n_i)), _alpha),
                        _lambda2
                    );
                    
                    _weights[i] = NumOps.Negate(NumOps.Divide(numerator, denominator));
                }
                else
                {
                    // Weight is zero due to L1 regularization
                    if (_weights.ContainsKey(i))
                    {
                        _weights.Remove(i);
                    }
                }
            }
        }
    }
    
    /// <summary>
    /// Gets the weight for a feature index.
    /// </summary>
    private T GetWeight(int index)
    {
        return _weights.ContainsKey(index) ? _weights[index] : NumOps.Zero;
    }
    
    /// <summary>
    /// Computes the sigmoid function.
    /// </summary>
    private T Sigmoid(T x)
    {
        // 1 / (1 + exp(-x))
        var exp_neg_x = NumOps.Exp(NumOps.Negate(x));
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, exp_neg_x));
    }
    
    /// <summary>
    /// Gets the raw prediction value (before sigmoid).
    /// </summary>
    private T PredictRaw(Vector<T> input)
    {
        var sum = NumOps.Zero;
        
        // Only compute for non-zero features that have weights
        for (int i = 0; i < input.Length; i++)
        {
            if (!NumOps.Equals(input[i], NumOps.Zero) && _weights.ContainsKey(i))
            {
                sum = NumOps.Add(sum, NumOps.Multiply(input[i], _weights[i]));
            }
        }
        
        return sum;
    }
    
    /// <inheritdoc/>
    public override T Predict(Vector<T> input)
    {
        // Update weights for active features
        UpdateWeights(input);
        
        var raw = PredictRaw(input);
        var probability = Sigmoid(raw);
        
        // Return 1 if probability > 0.5, else 0
        return NumOps.GreaterThan(probability, NumOps.FromDouble(0.5)) ? NumOps.One : NumOps.Zero;
    }
    
    /// <summary>
    /// Gets the probability of the positive class.
    /// </summary>
    public T PredictProbability(Vector<T> input)
    {
        UpdateWeights(input);
        var raw = PredictRaw(input);
        return Sigmoid(raw);
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
            ModelType = ModelType.FTRL,
            FeatureCount = _numFeatures,
            Complexity = _weights.Count, // Number of non-zero weights
            Description = $"FTRL with {_weights.Count} non-zero weights out of {_numFeatures} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = SamplesSeen,
                ["Alpha"] = Convert.ToDouble(_alpha),
                ["Beta"] = Convert.ToDouble(_beta),
                ["Lambda1"] = Convert.ToDouble(_lambda1),
                ["Lambda2"] = Convert.ToDouble(_lambda2),
                ["NonZeroWeights"] = _weights.Count,
                ["Sparsity"] = 1.0 - (double)_weights.Count / _numFeatures
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
        
        // Parameters
        writer.Write(_numFeatures);
        writer.Write(Convert.ToDouble(_alpha));
        writer.Write(Convert.ToDouble(_beta));
        writer.Write(Convert.ToDouble(_lambda1));
        writer.Write(Convert.ToDouble(_lambda2));
        writer.Write(SamplesSeen);
        
        // Sparse weights
        writer.Write(_weights.Count);
        foreach (var kvp in _weights)
        {
            writer.Write(kvp.Key);
            writer.Write(Convert.ToDouble(kvp.Value));
        }
        
        // z values
        writer.Write(_z.Count);
        foreach (var kvp in _z)
        {
            writer.Write(kvp.Key);
            writer.Write(Convert.ToDouble(kvp.Value));
        }
        
        // n values
        writer.Write(_n.Count);
        foreach (var kvp in _n)
        {
            writer.Write(kvp.Key);
            writer.Write(Convert.ToDouble(kvp.Value));
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
            
            // Parameters
            int numFeatures = reader.ReadInt32();
            _alpha = NumOps.FromDouble(reader.ReadDouble());
            _beta = NumOps.FromDouble(reader.ReadDouble());
            _lambda1 = NumOps.FromDouble(reader.ReadDouble());
            _lambda2 = NumOps.FromDouble(reader.ReadDouble());
            _samplesSeen = reader.ReadInt64();
            
            // Sparse weights
            _weights.Clear();
            int weightCount = reader.ReadInt32();
            for (int i = 0; i < weightCount; i++)
            {
                int index = reader.ReadInt32();
                T value = NumOps.FromDouble(reader.ReadDouble());
                _weights[index] = value;
            }
            
            // z values
            _z.Clear();
            int zCount = reader.ReadInt32();
            for (int i = 0; i < zCount; i++)
            {
                int index = reader.ReadInt32();
                T value = NumOps.FromDouble(reader.ReadDouble());
                _z[index] = value;
            }
            
            // n values
            _n.Clear();
            int nCount = reader.ReadInt32();
            for (int i = 0; i < nCount; i++)
            {
                int index = reader.ReadInt32();
                T value = NumOps.FromDouble(reader.ReadDouble());
                _n[index] = value;
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
            InitialLearningRate = parameters.ContainsKey("Alpha") 
                ? (T)parameters["Alpha"] 
                : _alpha,
            LearningRateDecay = parameters.ContainsKey("Beta") 
                ? (T)parameters["Beta"] 
                : _beta,
            RegularizationParameter = parameters.ContainsKey("Lambda1") 
                ? (T)parameters["Lambda1"] 
                : _lambda1,
            UseAdaptiveLearningRate = _options.UseAdaptiveLearningRate
        };
        
        var ftrl = new FTRL<T>(_numFeatures, newOptions, _logger);
        
        if (parameters.ContainsKey("Lambda2"))
        {
            ftrl._lambda2 = (T)parameters["Lambda2"];
        }
        
        return ftrl;
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _numFeatures;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => 1;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var clone = new FTRL<T>(_numFeatures, _options, _logger)
        {
            _alpha = _alpha,
            _beta = _beta,
            _lambda1 = _lambda1,
            _lambda2 = _lambda2,
            _samplesSeen = SamplesSeen,
            _learningRate = _learningRate,
            _adaptiveLearningRate = _adaptiveLearningRate
        };
        
        // Deep copy sparse data structures
        foreach (var kvp in _weights)
            clone._weights[kvp.Key] = kvp.Value;
        foreach (var kvp in _z)
            clone._z[kvp.Key] = kvp.Value;
        foreach (var kvp in _n)
            clone._n[kvp.Key] = kvp.Value;
        
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
        // Return hyperparameters
        return new Vector<T>(new[] { _alpha, _beta, _lambda1, _lambda2 });
    }
    
    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length >= 4)
        {
            _alpha = parameters[0];
            _beta = parameters[1];
            _lambda1 = parameters[2];
            _lambda2 = parameters[3];
            _learningRate = _alpha;
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
        // Return indices of non-zero weights
        return _weights.Keys.OrderBy(k => k);
    }
    
    /// <inheritdoc/>
    public override bool IsFeatureUsed(int featureIndex)
    {
        return _weights.ContainsKey(featureIndex);
    }
    
    /// <inheritdoc/>
    public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // FTRL automatically manages sparsity, so this is informational only
        _logger.Information("FTRL manages feature sparsity automatically. Current non-zero features: {Count}", 
            _weights.Count);
    }
    
    /// <summary>
    /// Gets the sparse weight vector.
    /// </summary>
    public IDictionary<int, T> GetSparseWeights()
    {
        UpdateWeights(new Vector<T>(new T[_numFeatures])); // Update all weights
        return new Dictionary<int, T>(_weights);
    }
    
    /// <summary>
    /// Gets feature importance based on weight magnitudes.
    /// </summary>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();

        for (int i = 0; i < _numFeatures; i++)
        {
            result[$"Feature_{i}"] = _weights.ContainsKey(i) ? NumOps.Abs(_weights[i]) : NumOps.Zero;
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
        var error = NumOps.Subtract(prediction, testLabels);
        var squaredError = NumOps.Multiply(error, error);
        
        return new Dictionary<string, double>
        {
            ["MSE"] = Convert.ToDouble(squaredError),
            ["RMSE"] = Math.Sqrt(Convert.ToDouble(squaredError))
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
        // FTRL doesn't track recent errors by default, return 0
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
            LearningRate = _alpha,
            TrainingLoss = NumOps.FromDouble(GetTrainingLoss()),
            ValidationLoss = NumOps.FromDouble(GetValidationLoss()),
            AdditionalMetrics = new Dictionary<string, T>
            {
                ["Alpha"] = _alpha,
                ["Beta"] = _beta,
                ["Lambda1"] = _lambda1,
                ["Lambda2"] = _lambda2,
                ["NonZeroWeights"] = NumOps.FromDouble(_weights.Count)
            }
        };
    }
    
    /// <inheritdoc/>
    public override void Save()
    {
        // Default implementation saves to a standard location
        SaveModel($"ftrl_model_{DateTime.Now:yyyyMMddHHmmss}.bin");
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
        _z?.Clear();
        _n?.Clear();
        _weights?.Clear();
    }
}