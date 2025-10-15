using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Helpers;
using AiDotNet.Statistics;

namespace AiDotNet.OnlineLearning.Algorithms;

/// <summary>
/// Online Bagging (Bootstrap Aggregating) for streaming data.
/// Can use any base online learner and combines predictions through voting or averaging.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OnlineBagging<T> : OnlineModelBase<T, Vector<T>, T>
{
    private readonly List<IOnlineModel<T, Vector<T>, T>> _baseLearners = default!;
    private readonly OnlineModelOptions<T> _options = default!;
    private readonly int _ensembleSize;
    private readonly OnlineLearningAlgorithm _baseAlgorithm = default!;
    private readonly int _numFeatures;
    private readonly int _numClasses;
    private readonly Random _random = default!;
    private readonly bool _useParallel;
    private readonly bool _isClassification;
    
    /// <summary>
    /// Initializes a new instance of the OnlineBagging class.
    /// </summary>
    public OnlineBagging(int numFeatures, int numClasses, OnlineLearningAlgorithm baseAlgorithm,
                        int ensembleSize = 10, bool isClassification = true,
                        OnlineModelOptions<T>? options = null, ILogging? logger = null)
        : base(options != null ? options.InitialLearningRate : MathHelper.GetNumericOperations<T>().FromDouble(1.0), logger)
    {
        _options = options ?? new OnlineModelOptions<T>
        {
            InitialLearningRate = NumOps.FromDouble(1.0),
            UseAdaptiveLearningRate = false,
            LearningRateDecay = NumOps.Zero,
            RegularizationParameter = NumOps.FromDouble(0.01)
        };
        
        _ensembleSize = ensembleSize;
        _baseAlgorithm = baseAlgorithm;
        _numFeatures = numFeatures;
        _numClasses = numClasses;
        _isClassification = isClassification;
        _baseLearners = new List<IOnlineModel<T, Vector<T>, T>>(ensembleSize);
        _random = new Random();
        _useParallel = ensembleSize >= 4; // Use parallel processing for 4+ learners
        
        // Initialize base learners
        for (int i = 0; i < ensembleSize; i++)
        {
            var learner = CreateBaseLearner();
            _baseLearners.Add(learner);
        }
        
        _learningRate = _options.InitialLearningRate;
        AdaptiveLearningRate = _options.UseAdaptiveLearningRate;
    }
    
    /// <summary>
    /// Creates a base learner based on the specified algorithm.
    /// </summary>
    private IOnlineModel<T, Vector<T>, T> CreateBaseLearner()
    {
        switch (_baseAlgorithm)
        {
            case OnlineLearningAlgorithm.Perceptron:
                return new OnlinePerceptron<T>(_numFeatures, _options, _logger);
                
            case OnlineLearningAlgorithm.PassiveAggressive:
                return new PassiveAggressiveRegressor<T>(_numFeatures, _options, _logger);
                
            case OnlineLearningAlgorithm.StochasticGradientDescent:
                return new OnlineSGDRegressor<T>(_numFeatures, FitnessCalculatorType.MeanSquaredError, 
                    new AdaptiveOnlineModelOptions<T>
                    {
                        InitialLearningRate = _options.InitialLearningRate,
                        UseAdaptiveLearningRate = _options.UseAdaptiveLearningRate,
                        LearningRateDecay = _options.LearningRateDecay,
                        RegularizationParameter = _options.RegularizationParameter
                    }, 
                    DriftDetectionMethod.ADWIN, _logger);
                
            case OnlineLearningAlgorithm.OnlineSVM:
                return new OnlineSVM<T>(_numFeatures, _options, null, _logger);
                
            case OnlineLearningAlgorithm.AROW:
                return new AROW<T>(_numFeatures, _options, _logger);
                
            case OnlineLearningAlgorithm.ConfidenceWeighted:
                return new ConfidenceWeighted<T>(_numFeatures, _options, _logger);
                
            case OnlineLearningAlgorithm.HoeffdingTree:
                return new HoeffdingTree<T>(_numFeatures, _numClasses, _options, _logger);
                
            default:
                throw new NotSupportedException($"Base algorithm {_baseAlgorithm} is not supported for online bagging");
        }
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        // Use Poisson(1) for online bagging
        if (_useParallel)
        {
            Parallel.For(0, _ensembleSize, i =>
            {
                UpdateLearner(i, input, expectedOutput, learningRate);
            });
        }
        else
        {
            for (int i = 0; i < _ensembleSize; i++)
            {
                UpdateLearner(i, input, expectedOutput, learningRate);
            }
        }
    }
    
    private void UpdateLearner(int learnerIndex, Vector<T> input, T expectedOutput, T learningRate)
    {
        // Sample weight from Poisson(1) distribution for online bagging
        int k = SamplePoisson(1.0);
        
        // Update the learner k times
        for (int j = 0; j < k; j++)
        {
            _baseLearners[learnerIndex].PartialFit(input, expectedOutput, learningRate);
        }
    }
    
    /// <summary>
    /// Samples from Poisson distribution using Knuth's algorithm.
    /// </summary>
    private int SamplePoisson(double lambda)
    {
        double L = Math.Exp(-lambda);
        double p = 1.0;
        int k = 0;
        
        do
        {
            k++;
            p *= _random.NextDouble();
        } while (p > L);
        
        return k - 1;
    }
    
    /// <inheritdoc/>
    public override T Predict(Vector<T> input)
    {
        if (_isClassification)
        {
            return PredictClassification(input);
        }
        else
        {
            return PredictRegression(input);
        }
    }
    
    /// <summary>
    /// Predicts using majority voting for classification.
    /// </summary>
    private T PredictClassification(Vector<T> input)
    {
        var votes = new int[_numClasses];
        
        if (_useParallel)
        {
            var localVotes = new int[_ensembleSize][];
            Parallel.For(0, _ensembleSize, i =>
            {
                localVotes[i] = new int[_numClasses];
                var prediction = Convert.ToInt32(_baseLearners[i].Predict(input));
                if (prediction >= 0 && prediction < _numClasses)
                {
                    localVotes[i][prediction] = 1;
                }
            });
            
            // Aggregate votes
            for (int i = 0; i < _ensembleSize; i++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    votes[c] += localVotes[i][c];
                }
            }
        }
        else
        {
            foreach (var learner in _baseLearners)
            {
                var prediction = Convert.ToInt32(learner.Predict(input));
                if (prediction >= 0 && prediction < _numClasses)
                {
                    votes[prediction]++;
                }
            }
        }
        
        // Find class with most votes
        int maxVotes = 0;
        int predictedClass = 0;
        for (int i = 0; i < _numClasses; i++)
        {
            if (votes[i] > maxVotes)
            {
                maxVotes = votes[i];
                predictedClass = i;
            }
        }
        
        return NumOps.FromDouble(predictedClass);
    }
    
    /// <summary>
    /// Predicts using averaging for regression.
    /// </summary>
    private T PredictRegression(Vector<T> input)
    {
        T sum = NumOps.Zero;
        
        if (_useParallel)
        {
            var predictions = new T[_ensembleSize];
            Parallel.For(0, _ensembleSize, i =>
            {
                predictions[i] = _baseLearners[i].Predict(input);
            });
            
            foreach (var pred in predictions)
            {
                sum = NumOps.Add(sum, pred);
            }
        }
        else
        {
            foreach (var learner in _baseLearners)
            {
                sum = NumOps.Add(sum, learner.Predict(input));
            }
        }
        
        return NumOps.Divide(sum, NumOps.FromDouble(_ensembleSize));
    }
    
    /// <summary>
    /// Gets prediction probabilities by averaging learner probabilities (for classification).
    /// </summary>
    public Vector<T> PredictProbabilities(Vector<T> input)
    {
        if (!_isClassification)
        {
            throw new InvalidOperationException("PredictProbabilities is only available for classification");
        }
        
        var avgProbs = new T[_numClasses];
        for (int i = 0; i < _numClasses; i++)
        {
            avgProbs[i] = NumOps.Zero;
        }
        
        // For each learner, get probabilities if supported
        foreach (var learner in _baseLearners)
        {
            Vector<T> probs;
            
            // Check if learner supports probability prediction
            if (learner is HoeffdingTree<T> tree)
            {
                probs = tree.PredictProbabilities(input);
            }
            else if (learner is AROW<T> arow)
            {
                // AROW returns confidence, convert to probability
                var (prediction, lower, upper) = arow.PredictWithConfidence(input);
                probs = ConvertConfidenceToProbability(prediction, lower, upper);
            }
            else if (learner is ConfidenceWeighted<T> cw)
            {
                // CW returns confidence, convert to probability
                var (prediction, lower, upper) = cw.PredictWithConfidence(input);
                probs = ConvertConfidenceToProbability(prediction, lower, upper);
            }
            else
            {
                // Fallback to one-hot encoding
                var probArray = new T[_numClasses];
                var prediction = Convert.ToInt32(learner.Predict(input));
                for (int j = 0; j < _numClasses; j++)
                {
                    probArray[j] = j == prediction ? NumOps.One : NumOps.Zero;
                }
                probs = new Vector<T>(probArray);
            }
            
            for (int c = 0; c < _numClasses; c++)
            {
                avgProbs[c] = NumOps.Add(avgProbs[c], probs[c]);
            }
        }
        
        // Normalize
        var divisor = NumOps.FromDouble(_ensembleSize);
        for (int i = 0; i < _numClasses; i++)
        {
            avgProbs[i] = NumOps.Divide(avgProbs[i], divisor);
        }
        
        return new Vector<T>(avgProbs);
    }
    
    /// <summary>
    /// Converts confidence bounds to probability distribution for binary classification.
    /// </summary>
    private Vector<T> ConvertConfidenceToProbability(T prediction, T lower, T upper)
    {
        // For binary classification, use sigmoid-like transformation
        var probs = new T[_numClasses];
        
        if (_numClasses == 2)
        {
            // Map confidence to [0, 1] probability
            var confidence = NumOps.Subtract(upper, lower);
            var normalizedPred = NumOps.Divide(
                NumOps.Add(prediction, NumOps.One),
                NumOps.FromDouble(2.0)
            );
            
            probs[1] = normalizedPred;
            probs[0] = NumOps.Subtract(NumOps.One, normalizedPred);
        }
        else
        {
            // For multi-class, use softmax-like approach
            var pred = Convert.ToInt32(NumOps.GreaterThan(prediction, NumOps.Zero) ? NumOps.One : NumOps.Zero);
            for (int i = 0; i < _numClasses; i++)
            {
                probs[i] = i == pred ? NumOps.One : NumOps.Zero;
            }
        }
        
        return new Vector<T>(probs);
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
            ModelType = ModelType.OnlineBagging,
            FeatureCount = _numFeatures,
            Complexity = _ensembleSize,
            Description = $"Online Bagging with {_ensembleSize} {_baseAlgorithm} learners",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = SamplesSeen,
                ["EnsembleSize"] = _ensembleSize,
                ["BaseAlgorithm"] = _baseAlgorithm.ToString(),
                ["IsClassification"] = _isClassification,
                ["UseParallel"] = _useParallel
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
        
        // Ensemble parameters
        writer.Write(_ensembleSize);
        writer.Write((int)_baseAlgorithm);
        writer.Write(_numFeatures);
        writer.Write(_numClasses);
        writer.Write(_isClassification);
        writer.Write(SamplesSeen);
        
        // Base learners
        for (int i = 0; i < _ensembleSize; i++)
        {
            var learnerData = _baseLearners[i].Serialize();
            writer.Write(learnerData.Length);
            writer.Write(learnerData);
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
            
            // Ensemble parameters
            int ensembleSize = reader.ReadInt32();
            var baseAlgorithm = (OnlineLearningAlgorithm)reader.ReadInt32();
            int numFeatures = reader.ReadInt32();
            int numClasses = reader.ReadInt32();
            bool isClassification = reader.ReadBoolean();
            _samplesSeen = reader.ReadInt64();
            
            // Base learners
            _baseLearners.Clear();
            for (int i = 0; i < ensembleSize; i++)
            {
                int learnerDataLength = reader.ReadInt32();
                var learnerData = reader.ReadBytes(learnerDataLength);
                
                var learner = CreateBaseLearner();
                learner.Deserialize(learnerData);
                _baseLearners.Add(learner);
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
            RegularizationParameter = parameters.ContainsKey("RegularizationParameter") 
                ? (T)parameters["RegularizationParameter"] 
                : _options.RegularizationParameter
        };
        
        int ensembleSize = parameters.ContainsKey("EnsembleSize") ? (int)parameters["EnsembleSize"] : _ensembleSize;
        
        return new OnlineBagging<T>(_numFeatures, _numClasses, _baseAlgorithm, 
                                    ensembleSize, _isClassification, newOptions, _logger);
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _numFeatures;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => _isClassification ? _numClasses : 1;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var clone = new OnlineBagging<T>(_numFeatures, _numClasses, _baseAlgorithm,
                                         _ensembleSize, _isClassification, _options, _logger);
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
        // Return ensemble hyperparameters
        return new Vector<T>(new[] { 
            NumOps.FromDouble(_ensembleSize),
            _options.InitialLearningRate,
            _options.RegularizationParameter 
        });
    }
    
    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Can only change learning rate and regularization for existing learners
        if (parameters.Length >= 2)
        {
            _learningRate = parameters[1];
            
            if (parameters.Length >= 3)
            {
                _options.RegularizationParameter = parameters[2];
            }
            
            // Update all learners
            // Since we need to maintain the learning rate across all base learners,
            // we update them using SetParameters if they support it
            var newParams = new Vector<T>(new[] { _learningRate, _options.RegularizationParameter });
            
            foreach (var learner in _baseLearners)
            {
                learner.SetParameters(newParams);
            }
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
        // Return union of all active features from base learners
        var allFeatures = new HashSet<int>();
        foreach (var learner in _baseLearners)
        {
            foreach (var feature in learner.GetActiveFeatureIndices())
            {
                allFeatures.Add(feature);
            }
        }
        return allFeatures.OrderBy(f => f);
    }
    
    /// <inheritdoc/>
    public override bool IsFeatureUsed(int featureIndex)
    {
        return _baseLearners.Any(learner => learner.IsFeatureUsed(featureIndex));
    }
    
    /// <inheritdoc/>
    public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Propagate to all base learners
        foreach (var learner in _baseLearners)
        {
            learner.SetActiveFeatureIndices(featureIndices);
        }
    }
    
    /// <summary>
    /// Gets the diversity of the ensemble (percentage of learners that disagree on average).
    /// </summary>
    public T GetEnsembleDiversity(IEnumerable<Vector<T>> testInputs)
    {
        int totalDisagreements = 0;
        int totalComparisons = 0;
        
        foreach (var input in testInputs)
        {
            var predictions = _baseLearners.Select(l => l.Predict(input)).ToArray();
            
            // Count pairwise disagreements
            for (int i = 0; i < _ensembleSize - 1; i++)
            {
                for (int j = i + 1; j < _ensembleSize; j++)
                {
                    if (!NumOps.Equals(predictions[i], predictions[j]))
                    {
                        totalDisagreements++;
                    }
                    totalComparisons++;
                }
            }
        }
        
        return totalComparisons > 0 
            ? NumOps.Divide(NumOps.FromDouble(totalDisagreements), NumOps.FromDouble(totalComparisons))
            : NumOps.Zero;
    }
    
    /// <inheritdoc/>
    public override int InputDimensions => _numFeatures;
    
    /// <inheritdoc/>
    public override int OutputDimensions => _isClassification ? _numClasses : 1;
    
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
        
        if (_isClassification)
        {
            var classIndex = Convert.ToInt32(testLabels);
            var predIndex = Convert.ToInt32(prediction);
            return new Dictionary<string, double>
            {
                ["Accuracy"] = classIndex == predIndex ? 1.0 : 0.0,
                ["Error"] = classIndex == predIndex ? 0.0 : 1.0
            };
        }
        else
        {
            var error = NumOps.Subtract(prediction, testLabels);
            var squaredError = NumOps.Multiply(error, error);
            return new Dictionary<string, double>
            {
                ["MSE"] = Convert.ToDouble(squaredError),
                ["RMSE"] = Math.Sqrt(Convert.ToDouble(squaredError))
            };
        }
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
        // Average loss from all base learners
        if (_baseLearners.Count == 0)
            return 0.0;
            
        double totalLoss = 0.0;
        foreach (var learner in _baseLearners)
        {
            totalLoss += learner.GetTrainingLoss();
        }
        return totalLoss / _baseLearners.Count;
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
                ["EnsembleSize"] = NumOps.FromDouble(_ensembleSize),
                ["BaseAlgorithm"] = NumOps.FromDouble((int)_baseAlgorithm),
                ["IsClassification"] = NumOps.FromDouble(_isClassification ? 1.0 : 0.0)
            }
        };
    }
    
    /// <inheritdoc/>
    public override void Save()
    {
        // Default implementation saves to a standard location
        SaveModel($"online_bagging_model_{DateTime.Now:yyyyMMddHHmmss}.bin");
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
        foreach (var learner in _baseLearners)
        {
            learner.Dispose();
        }
        _baseLearners.Clear();
    }
}