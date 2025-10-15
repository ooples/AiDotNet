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
/// Online Random Forest for streaming data using an ensemble of Hoeffding Trees.
/// Combines bagging with random feature subsets for each tree.
/// </summary>
public class OnlineRandomForest<T> : OnlineModelBase<T, Vector<T>, T>
{
    private readonly List<HoeffdingTree<T>> _trees = default!;
    private readonly OnlineModelOptions<T> _options = default!;
    private readonly int _numTrees;
    private readonly int _numClasses;
    private readonly int _numFeatures;
    private readonly int _subspaceSize;
    private readonly List<int[]> _featureSubsets;
    private readonly Random _random = default!;
    private readonly bool _useParallel;
    
    /// <summary>
    /// Initializes a new instance of the OnlineRandomForest class.
    /// </summary>
    public OnlineRandomForest(int numFeatures, int numClasses, int numTrees = 10, 
                             OnlineModelOptions<T>? options = null, ILogging? logger = null)
        : base(options != null ? options.InitialLearningRate : MathHelper.GetNumericOperations<T>().FromDouble(1.0), logger)
    {
        _options = options ?? new OnlineModelOptions<T>
        {
            InitialLearningRate = NumOps.FromDouble(1.0),
            UseAdaptiveLearningRate = false,
            LearningRateDecay = NumOps.Zero,
            RegularizationParameter = NumOps.FromDouble(0.0001) // Used as split confidence for trees
        };
        
        _numTrees = numTrees;
        _numClasses = numClasses;
        _numFeatures = numFeatures;
        _subspaceSize = (int)Math.Sqrt(numFeatures); // Common choice for classification
        _trees = new List<HoeffdingTree<T>>(numTrees);
        _featureSubsets = new List<int[]>(numTrees);
        _random = new Random();
        _useParallel = numTrees >= 4; // Use parallel processing for 4+ trees
        
        // Initialize trees with random feature subsets
        for (int i = 0; i < numTrees; i++)
        {
            var featureSubset = SelectRandomFeatures();
            _featureSubsets.Add(featureSubset);
            
            var tree = new HoeffdingTree<T>(featureSubset.Length, numClasses, _options, logger);
            _trees.Add(tree);
        }
        
        _learningRate = _options.InitialLearningRate;
        AdaptiveLearningRate = _options.UseAdaptiveLearningRate;
    }
    
    /// <summary>
    /// Selects a random subset of features for a tree.
    /// </summary>
    private int[] SelectRandomFeatures()
    {
        var allFeatures = Enumerable.Range(0, _numFeatures).ToList();
        var selectedFeatures = new List<int>(_subspaceSize);
        
        for (int i = 0; i < _subspaceSize && allFeatures.Count > 0; i++)
        {
            int index = _random.Next(allFeatures.Count);
            selectedFeatures.Add(allFeatures[index]);
            allFeatures.RemoveAt(index);
        }
        
        return selectedFeatures.OrderBy(f => f).ToArray();
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        // Use Poisson(1) for online bagging
        if (_useParallel)
        {
            Parallel.For(0, _numTrees, i =>
            {
                UpdateTree(i, input, expectedOutput, learningRate);
            });
        }
        else
        {
            for (int i = 0; i < _numTrees; i++)
            {
                UpdateTree(i, input, expectedOutput, learningRate);
            }
        }
    }
    
    private void UpdateTree(int treeIndex, Vector<T> input, T expectedOutput, T learningRate)
    {
        // Sample weight from Poisson(1) distribution for online bagging
        int k = SamplePoisson(1.0);
        
        if (k > 0)
        {
            // Extract features for this tree
            var treeInput = ExtractFeatures(input, _featureSubsets[treeIndex]);
            
            // Update the tree k times
            for (int j = 0; j < k; j++)
            {
                _trees[treeIndex].PartialFit(treeInput, expectedOutput, learningRate);
            }
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
    
    /// <summary>
    /// Extracts the features specified by the indices.
    /// </summary>
    private Vector<T> ExtractFeatures(Vector<T> fullInput, int[] featureIndices)
    {
        var extractedFeatures = new T[featureIndices.Length];
        for (int i = 0; i < featureIndices.Length; i++)
        {
            extractedFeatures[i] = fullInput[featureIndices[i]];
        }
        return new Vector<T>(extractedFeatures);
    }
    
    /// <inheritdoc/>
    public override T Predict(Vector<T> input)
    {
        // Majority voting
        var votes = new int[_numClasses];
        
        if (_useParallel)
        {
            var localVotes = new int[_numTrees][];
            Parallel.For(0, _numTrees, i =>
            {
                localVotes[i] = new int[_numClasses];
                var treeInput = ExtractFeatures(input, _featureSubsets[i]);
                var prediction = Convert.ToInt32(_trees[i].Predict(treeInput));
                localVotes[i][prediction] = 1;
            });
            
            // Aggregate votes
            for (int i = 0; i < _numTrees; i++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    votes[c] += localVotes[i][c];
                }
            }
        }
        else
        {
            for (int i = 0; i < _numTrees; i++)
            {
                var treeInput = ExtractFeatures(input, _featureSubsets[i]);
                var prediction = Convert.ToInt32(_trees[i].Predict(treeInput));
                votes[prediction]++;
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
    /// Gets prediction probabilities by averaging tree probabilities.
    /// </summary>
    public Vector<T> PredictProbabilities(Vector<T> input)
    {
        var avgProbs = new T[_numClasses];
        for (int i = 0; i < _numClasses; i++)
        {
            avgProbs[i] = NumOps.Zero;
        }
        
        if (_useParallel)
        {
            var treeProbs = new T[_numTrees][];
            Parallel.For(0, _numTrees, i =>
            {
                var treeInput = ExtractFeatures(input, _featureSubsets[i]);
                var tree = _trees[i] as HoeffdingTree<T>;
                if (tree != null)
                {
                    treeProbs[i] = tree.PredictProbabilities(treeInput).ToArray();
                }
                else
                {
                    // Fallback to one-hot encoding
                    treeProbs[i] = new T[_numClasses];
                    var prediction = Convert.ToInt32(_trees[i].Predict(treeInput));
                    for (int j = 0; j < _numClasses; j++)
                    {
                        treeProbs[i][j] = j == prediction ? NumOps.One : NumOps.Zero;
                    }
                }
            });
            
            // Average probabilities
            for (int i = 0; i < _numTrees; i++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    avgProbs[c] = NumOps.Add(avgProbs[c], treeProbs[i][c]);
                }
            }
        }
        else
        {
            for (int i = 0; i < _numTrees; i++)
            {
                var treeInput = ExtractFeatures(input, _featureSubsets[i]);
                var tree = _trees[i] as HoeffdingTree<T>;
                Vector<T> probs;
                
                if (tree != null)
                {
                    probs = tree.PredictProbabilities(treeInput);
                }
                else
                {
                    // Fallback to one-hot encoding
                    var probArray = new T[_numClasses];
                    var prediction = Convert.ToInt32(_trees[i].Predict(treeInput));
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
        }
        
        // Normalize
        var divisor = NumOps.FromDouble(_numTrees);
        for (int i = 0; i < _numClasses; i++)
        {
            avgProbs[i] = NumOps.Divide(avgProbs[i], divisor);
        }
        
        return new Vector<T>(avgProbs);
    }
    
    /// <inheritdoc/>
    public override void Train(Vector<T> input, T expectedOutput)
    {
        PartialFit(input, expectedOutput);
    }
    
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var totalNodes = _trees.Sum(t => 
        {
            var meta = t.GetModelMetadata();
            return (int)meta.AdditionalInfo["DecisionNodes"] + (int)meta.AdditionalInfo["ActiveLeaves"];
        });
        
        return new ModelMetadata<T>
        {
            ModelType = ModelType.OnlineRandomForest,
            FeatureCount = _numFeatures,
            Complexity = totalNodes,
            Description = $"Online Random Forest with {_numTrees} trees",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = SamplesSeen,
                ["NumTrees"] = _numTrees,
                ["SubspaceSize"] = _subspaceSize,
                ["TotalNodes"] = totalNodes,
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
        
        // Forest parameters
        writer.Write(_numTrees);
        writer.Write(_numClasses);
        writer.Write(_numFeatures);
        writer.Write(_subspaceSize);
        writer.Write(SamplesSeen);
        
        // Feature subsets
        for (int i = 0; i < _numTrees; i++)
        {
            writer.Write(_featureSubsets[i].Length);
            foreach (var feature in _featureSubsets[i])
            {
                writer.Write(feature);
            }
        }
        
        // Trees
        for (int i = 0; i < _numTrees; i++)
        {
            var treeData = _trees[i].Serialize();
            writer.Write(treeData.Length);
            writer.Write(treeData);
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
            
            // Forest parameters
            int numTrees = reader.ReadInt32();
            int numClasses = reader.ReadInt32();
            int numFeatures = reader.ReadInt32();
            int subspaceSize = reader.ReadInt32();
            _samplesSeen = reader.ReadInt64();
            
            // Feature subsets
            _featureSubsets.Clear();
            for (int i = 0; i < numTrees; i++)
            {
                int subsetLength = reader.ReadInt32();
                var subset = new int[subsetLength];
                for (int j = 0; j < subsetLength; j++)
                {
                    subset[j] = reader.ReadInt32();
                }
                _featureSubsets.Add(subset);
            }
            
            // Trees
            _trees.Clear();
            for (int i = 0; i < numTrees; i++)
            {
                int treeDataLength = reader.ReadInt32();
                var treeData = reader.ReadBytes(treeDataLength);
                
                var tree = new HoeffdingTree<T>(_featureSubsets[i].Length, numClasses, _options, _logger);
                tree.Deserialize(treeData);
                _trees.Add(tree);
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
            RegularizationParameter = parameters.ContainsKey("SplitConfidence") 
                ? (T)parameters["SplitConfidence"] 
                : _options.RegularizationParameter
        };
        
        int numTrees = parameters.ContainsKey("NumTrees") ? (int)parameters["NumTrees"] : _numTrees;
        
        return new OnlineRandomForest<T>(_numFeatures, _numClasses, numTrees, newOptions, _logger);
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _numFeatures;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => _numClasses;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var clone = new OnlineRandomForest<T>(_numFeatures, _numClasses, _numTrees, _options, _logger);
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
        // Return forest hyperparameters
        return new Vector<T>(new[] { 
            NumOps.FromDouble(_numTrees), 
            NumOps.FromDouble(_subspaceSize),
            _options.RegularizationParameter 
        });
    }
    
    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Can only change split confidence for existing trees
        if (parameters.Length >= 3)
        {
            _options.RegularizationParameter = parameters[2];
            
            // Update all trees
            foreach (var tree in _trees)
            {
                tree.SetParameters(new Vector<T>(new[] { parameters[2] }));
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
        // Return all features that have been selected in any tree
        var allFeatures = new HashSet<int>();
        foreach (var subset in _featureSubsets)
        {
            foreach (var feature in subset)
            {
                allFeatures.Add(feature);
            }
        }
        return allFeatures.OrderBy(f => f);
    }
    
    /// <inheritdoc/>
    public override bool IsFeatureUsed(int featureIndex)
    {
        return _featureSubsets.Any(subset => subset.Contains(featureIndex));
    }
    
    /// <inheritdoc/>
    public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Online Random Forest uses features adaptively, so this is a no-op
        _logger.Warning("SetActiveFeatureIndices is not supported for Online Random Forest");
    }
    
    /// <summary>
    /// Gets the feature importance scores based on how often features are used in splits.
    /// </summary>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();

        // Count how many trees use each feature
        for (int f = 0; f < _numFeatures; f++)
        {
            int count = 0;
            for (int t = 0; t < _numTrees; t++)
            {
                if (_featureSubsets[t].Contains(f))
                {
                    // Check if the feature is actually used in splits
                    var treeFeatureIndex = Array.IndexOf(_featureSubsets[t], f);
                    if (_trees[t].IsFeatureUsed(treeFeatureIndex))
                    {
                        count++;
                    }
                }
            }
            result[$"Feature_{f}"] = NumOps.Divide(NumOps.FromDouble(count), NumOps.FromDouble(_numTrees));
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
                ["NumTrees"] = NumOps.FromDouble(_numTrees),
                ["SubspaceSize"] = NumOps.FromDouble(_subspaceSize)
            }
        };
    }
    
    /// <inheritdoc/>
    public override void Save()
    {
        // Default implementation saves to a standard location
        SaveModel($"online_random_forest_model_{DateTime.Now:yyyyMMddHHmmss}.bin");
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