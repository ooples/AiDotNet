using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Helpers;
using AiDotNet.Statistics;

namespace AiDotNet.OnlineLearning.Algorithms;

/// <summary>
/// Online Support Vector<double> Machine using stochastic gradient descent with hinge loss.
/// Supports both linear and kernelized versions.
/// </summary>
public class OnlineSVM<T> : OnlineModelBase<T, Vector<T>, T>
{
    private Vector<T> _weights = default!;
    private T _bias = default!;
    private readonly OnlineModelOptions<T> _options = default!;
    private readonly IKernelFunction<T>? _kernel;
    private readonly List<Vector<T>> _supportVectors = default!;
    private readonly List<T> _alphas = default!;
    private T _C; // Regularization parameter
    
    /// <summary>
    /// Initializes a new instance of the OnlineSVM class.
    /// </summary>
    public OnlineSVM(int inputDimension, OnlineModelOptions<T>? options = null, 
                     IKernelFunction<T>? kernel = null, ILogging? logger = null)
        : base(options != null ? options.InitialLearningRate : MathHelper.GetNumericOperations<T>().FromDouble(0.01), logger)
    {
        _options = options ?? new OnlineModelOptions<T>
        {
            InitialLearningRate = NumOps.FromDouble(0.01),
            UseAdaptiveLearningRate = true,
            LearningRateDecay = NumOps.FromDouble(0.001),
            RegularizationParameter = NumOps.FromDouble(1.0)
        };
        
        _weights = new Vector<T>(Enumerable.Repeat(NumOps.Zero, inputDimension).ToArray());
        _bias = NumOps.Zero;
        _learningRate = _options.InitialLearningRate;
        AdaptiveLearningRate = _options.UseAdaptiveLearningRate;
        _C = _options.RegularizationParameter;
        _kernel = kernel;
        
        _supportVectors = new List<Vector<T>>();
        _alphas = new List<T>();
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        // Convert expected output to -1 or 1
        var y = NumOps.GreaterThan(expectedOutput, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One);
        
        if (_kernel != null)
        {
            // Kernelized SVM update
            UpdateKernelized(input, y, learningRate);
        }
        else
        {
            // Linear SVM update using subgradient of hinge loss
            UpdateLinear(input, y, learningRate);
        }
    }
    
    private void UpdateLinear(Vector<T> input, T y, T learningRate)
    {
        // Compute margin: y * (w^T * x + b)
        var score = NumOps.Add(input.DotProduct(_weights), _bias);
        var margin = NumOps.Multiply(y, score);
        
        // Hinge loss: max(0, 1 - margin)
        if (NumOps.LessThan(margin, NumOps.One))
        {
            // Misclassified or within margin
            // w = w - η * (λ * w - C * y * x)
            // b = b - η * (-C * y)
            
            // Regularization term: λ * w
            var regTerm = _weights.Multiply(NumOps.Divide(_options.RegularizationParameter, _C));
            
            // Gradient term: -C * y * x
            var gradTerm = input.Multiply(NumOps.Multiply(NumOps.Multiply(NumOps.Negate(_C), y), learningRate));
            
            // Update weights
            _weights = _weights.Subtract(regTerm.Multiply(learningRate)).Add(gradTerm);
            
            // Update bias
            _bias = NumOps.Add(_bias, NumOps.Multiply(NumOps.Multiply(_C, y), learningRate));
        }
        else
        {
            // Correctly classified outside margin - only apply regularization
            _weights = _weights.Multiply(NumOps.Subtract(NumOps.One, 
                NumOps.Multiply(learningRate, NumOps.Divide(_options.RegularizationParameter, _C))));
        }
    }
    
    private void UpdateKernelized(Vector<T> input, T y, T learningRate)
    {
        if (_kernel == null)
            throw new InvalidOperationException("Kernel function is null but kernelized update was called.");
            
        // Compute kernel-based prediction
        var score = _bias;
        for (int i = 0; i < _supportVectors.Count; i++)
        {
            var kernelValue = _kernel.Calculate(input, _supportVectors[i]);
            score = NumOps.Add(score, NumOps.Multiply(_alphas[i], kernelValue));
        }
        
        var margin = NumOps.Multiply(y, score);
        
        if (NumOps.LessThan(margin, NumOps.One))
        {
            // Add as support vector or update existing
            _supportVectors.Add((Vector<T>)input.Clone());
            _alphas.Add(NumOps.Multiply(NumOps.Multiply(y, _C), learningRate));
            
            // Update bias
            _bias = NumOps.Add(_bias, NumOps.Multiply(NumOps.Multiply(y, _C), learningRate));
            
            // Apply budget constraint if needed
            if (_supportVectors.Count > 1000) // Budget size
            {
                // Remove oldest support vector
                _supportVectors.RemoveAt(0);
                _alphas.RemoveAt(0);
            }
        }
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
        if (_kernel != null)
        {
            // Kernel prediction
            var score = _bias;
            for (int i = 0; i < _supportVectors.Count; i++)
            {
                var kernelValue = _kernel.Calculate(input, _supportVectors[i]);
                score = NumOps.Add(score, NumOps.Multiply(_alphas[i], kernelValue));
            }
            return score;
        }
        else
        {
            // Linear prediction: w^T * x + b
            return NumOps.Add(input.DotProduct(_weights), _bias);
        }
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
            ModelType = ModelType.OnlineSVM,
            FeatureCount = _weights.Length,
            Complexity = _kernel != null ? _supportVectors.Count : _weights.Length + 1,
            Description = _kernel != null ? 
                $"Kernelized Online SVM with {_supportVectors.Count} support vectors" : 
                $"Linear Online SVM with {_weights.Length} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = SamplesSeen,
                ["LearningRate"] = Convert.ToDouble(_learningRate),
                ["IsKernelized"] = _kernel != null,
                ["SupportVectorCount"] = _supportVectors.Count,
                ["C"] = Convert.ToDouble(_C)
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
        writer.Write(_weights.Length);
        writer.Write(_kernel != null);
        
        // Weights
        for (int i = 0; i < _weights.Length; i++)
        {
            writer.Write(Convert.ToDouble(_weights[i]));
        }
        
        // Bias and C
        writer.Write(Convert.ToDouble(_bias));
        writer.Write(Convert.ToDouble(_C));
        
        // Samples seen
        writer.Write(SamplesSeen);
        
        // Learning rate
        writer.Write(Convert.ToDouble(_learningRate));
        
        // Kernel-specific data
        if (_kernel != null)
        {
            writer.Write(_supportVectors.Count);
            foreach (var sv in _supportVectors)
            {
                for (int i = 0; i < sv.Length; i++)
                {
                    writer.Write(Convert.ToDouble(sv[i]));
                }
            }
            
            foreach (var alpha in _alphas)
            {
                writer.Write(Convert.ToDouble(alpha));
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
            int weightCount = reader.ReadInt32();
            bool isKernelized = reader.ReadBoolean();
            
            // Weights
            var weights = new T[weightCount];
            for (int i = 0; i < weightCount; i++)
            {
                weights[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _weights = new Vector<T>(weights);
            
            // Bias and C
            _bias = NumOps.FromDouble(reader.ReadDouble());
            _C = NumOps.FromDouble(reader.ReadDouble());
            
            // Samples seen
            _samplesSeen = reader.ReadInt64();
            
            // Learning rate
            _learningRate = NumOps.FromDouble(reader.ReadDouble());
            
            // Kernel-specific data
            if (isKernelized && _kernel != null)
            {
                _supportVectors.Clear();
                _alphas.Clear();
                
                int svCount = reader.ReadInt32();
                for (int i = 0; i < svCount; i++)
                {
                    var sv = new T[weightCount];
                    for (int j = 0; j < weightCount; j++)
                    {
                        sv[j] = NumOps.FromDouble(reader.ReadDouble());
                    }
                    _supportVectors.Add(new Vector<T>(sv));
                }
                
                for (int i = 0; i < svCount; i++)
                {
                    _alphas.Add(NumOps.FromDouble(reader.ReadDouble()));
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
            RegularizationParameter = parameters.ContainsKey("C") 
                ? (T)parameters["C"] 
                : _C
        };
        
        return new OnlineSVM<T>(_weights.Length, newOptions, _kernel, _logger);
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _weights.Length;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => 1;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var clone = new OnlineSVM<T>(_weights.Length, _options, _kernel, _logger)
        {
            _weights = new Vector<T>(_weights.ToArray()),
            _bias = _bias,
            _C = _C,
            _samplesSeen = SamplesSeen,
            _learningRate = _learningRate,
            _adaptiveLearningRate = _adaptiveLearningRate
        };
        
        // Clone support vectors if kernelized
        if (_kernel != null)
        {
            clone._supportVectors.Clear();
            clone._supportVectors.AddRange(_supportVectors.Select(sv => new Vector<T>(sv.ToArray())));
            clone._alphas.Clear();
            clone._alphas.AddRange(_alphas);
        }
        
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
        if (_kernel != null)
        {
            // For kernelized SVM, return alphas and bias
            var parameters = new T[_alphas.Count + 1];
            _alphas.CopyTo(parameters, 0);
            parameters[_alphas.Count] = _bias;
            return new Vector<T>(parameters);
        }
        else
        {
            // For linear SVM, return weights and bias
            var parameters = new T[_weights.Length + 1];
            _weights.ToArray().CopyTo(parameters, 0);
            parameters[_weights.Length] = _bias;
            return new Vector<T>(parameters);
        }
    }
    
    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (_kernel != null)
        {
            if (parameters.Length != _alphas.Count + 1)
            {
                throw new ArgumentException($"Parameter vector must have length {_alphas.Count + 1} for kernelized SVM");
            }
            
            _alphas.Clear();
            _alphas.AddRange(parameters.Take(parameters.Length - 1));
            _bias = parameters[parameters.Length - 1];
        }
        else
        {
            if (parameters.Length != _weights.Length + 1)
            {
                throw new ArgumentException($"Parameter vector must have length {_weights.Length + 1} for linear SVM");
            }
            
            _weights = new Vector<T>(parameters.Take(_weights.Length).ToArray());
            _bias = parameters[_weights.Length];
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
        // All features are active in SVM
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
        // SVM uses all features, so this is a no-op
        // Could implement feature masking if needed
    }
    
    /// <summary>
    /// Gets the number of support vectors (for kernelized SVM).
    /// </summary>
    public int SupportVectorCount => _supportVectors.Count;
    
    /// <summary>
    /// Gets whether this is a kernelized SVM.
    /// </summary>
    public bool IsKernelized => _kernel != null;

    
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
                ["C"] = _C,
                ["NumSupportVectors"] = NumOps.FromDouble(_supportVectors.Count)
            }
        };
    }
    
    /// <inheritdoc/>
    public override void Save()
    {
        // Default implementation saves to a standard location
        SaveModel($"online_svm_model_{DateTime.Now:yyyyMMddHHmmss}.bin");
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