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
/// Hoeffding Tree (Very Fast Decision Tree) for streaming data classification.
/// Uses the Hoeffding bound to decide when a node has enough statistical evidence to split.
/// </summary>
public class HoeffdingTree<T> : OnlineModelBase<T, Vector<T>, T>
{
    private HoeffdingNode _root = default!;
    private readonly OnlineModelOptions<T> _options = default!;
    private readonly int _numClasses;
    private T _splitConfidence = default!;
    private int _gracePeriod;
    private T _tieThreshold = default!;
    private int _maxDepth;
    private int _activeLeafCount;
    private int _inactiveLeafCount;
    private int _decisionNodeCount;
    
    /// <summary>
    /// Initializes a new instance of the HoeffdingTree class.
    /// </summary>
    public HoeffdingTree(int numFeatures, int numClasses, OnlineModelOptions<T>? options = null, ILogging? logger = null)
        : base(options != null ? options.InitialLearningRate : MathHelper.GetNumericOperations<T>().FromDouble(1.0), logger)
    {
        _options = options ?? new OnlineModelOptions<T>
        {
            InitialLearningRate = NumOps.FromDouble(1.0),
            UseAdaptiveLearningRate = false,
            LearningRateDecay = NumOps.Zero,
            RegularizationParameter = NumOps.FromDouble(0.0001) // Used as split confidence
        };
        
        _numClasses = numClasses;
        _splitConfidence = _options.RegularizationParameter;
        _gracePeriod = 200; // Minimum samples between split attempts
        _tieThreshold = NumOps.FromDouble(0.05);
        _maxDepth = 20;
        
        _root = new HoeffdingNode(numFeatures, numClasses, NumOps, 0);
        _activeLeafCount = 1;
        _inactiveLeafCount = 0;
        _decisionNodeCount = 0;
        
        _learningRate = _options.InitialLearningRate;
        AdaptiveLearningRate = _options.UseAdaptiveLearningRate;
    }
    
    /// <inheritdoc/>
    protected override void PerformUpdate(Vector<T> input, T expectedOutput, T learningRate)
    {
        // Convert output to class index
        int classIndex = Convert.ToInt32(expectedOutput);
        if (classIndex < 0 || classIndex >= _numClasses)
        {
            throw new ArgumentException($"Class index {classIndex} is out of range [0, {_numClasses})");
        }
        
        // Update the tree
        var leafNode = _root.FilterToLeaf(input, NumOps);
        leafNode.UpdateStatistics(input, classIndex, NumOps);
        
        // Check if we should attempt to split
        if (leafNode.GetSeenInstances() % _gracePeriod == 0)
        {
            AttemptToSplit(leafNode, input);
        }
    }
    
    /// <summary>
    /// Attempts to split a leaf node if there's enough statistical evidence.
    /// </summary>
    private void AttemptToSplit(HoeffdingNode node, Vector<T> lastInstance)
    {
        if (!node.IsLeaf || node.GetSeenInstances() < _gracePeriod)
            return;
            
        // Calculate information gain for each attribute
        var gains = new T[lastInstance.Length];
        var bestSplitValues = new T[lastInstance.Length];
        
        for (int i = 0; i < lastInstance.Length; i++)
        {
            (gains[i], bestSplitValues[i]) = node.CalculateBestSplitForAttribute(i, NumOps);
        }
        
        // Find best and second best attributes
        int bestAttribute = -1;
        int secondBestAttribute = -1;
        T bestGain = NumOps.Zero;
        T secondBestGain = NumOps.Zero;
        
        for (int i = 0; i < gains.Length; i++)
        {
            if (NumOps.GreaterThan(gains[i], bestGain))
            {
                secondBestGain = bestGain;
                secondBestAttribute = bestAttribute;
                bestGain = gains[i];
                bestAttribute = i;
            }
            else if (NumOps.GreaterThan(gains[i], secondBestGain))
            {
                secondBestGain = gains[i];
                secondBestAttribute = i;
            }
        }
        
        // Calculate Hoeffding bound
        T hoeffdingBound = CalculateHoeffdingBound(node.GetSeenInstances());
        
        // Check if we have enough confidence to split
        T gainDifference = NumOps.Subtract(bestGain, secondBestGain);
        if (NumOps.GreaterThan(gainDifference, hoeffdingBound) || 
            NumOps.LessThan(hoeffdingBound, _tieThreshold))
        {
            // Create split
            node.Split(bestAttribute, bestSplitValues[bestAttribute], NumOps);
            _activeLeafCount += 2; // Two new leaves
            _activeLeafCount -= 1; // Remove the split leaf
            _decisionNodeCount += 1;
            
            _logger.Information("Split node at depth {Depth} on attribute {Attribute} with gain {Gain}", 
                node.Depth, bestAttribute, Convert.ToDateTime(bestGain));
        }
    }
    
    /// <summary>
    /// Calculates the Hoeffding bound for the given number of instances.
    /// </summary>
    private T CalculateHoeffdingBound(int numInstances)
    {
        // ε = sqrt(R²ln(1/δ) / 2n)
        // where R is the range of the random variable (1 for information gain)
        // δ is the split confidence
        // n is the number of instances
        
        var logTerm = Math.Log(1.0 / Convert.ToDouble(_splitConfidence));
        var bound = Math.Sqrt(logTerm / (2.0 * numInstances));
        return NumOps.FromDouble(bound);
    }
    
    /// <inheritdoc/>
    public override T Predict(Vector<T> input)
    {
        var leaf = _root.FilterToLeaf(input, NumOps);
        return NumOps.FromDouble(leaf.GetMajorityClass());
    }
    
    /// <summary>
    /// Gets prediction probabilities for each class.
    /// </summary>
    public Vector<T> PredictProbabilities(Vector<T> input)
    {
        var leaf = _root.FilterToLeaf(input, NumOps);
        var probs = leaf.GetClassProbabilities(NumOps);
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
            ModelType = ModelType.HoeffdingTree,
            FeatureCount = _root.NumFeatures,
            Complexity = _decisionNodeCount + _activeLeafCount,
            Description = $"Hoeffding Tree with {_decisionNodeCount} decision nodes and {_activeLeafCount} active leaves",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SamplesSeen"] = SamplesSeen,
                ["ActiveLeaves"] = _activeLeafCount,
                ["InactiveLeaves"] = _inactiveLeafCount,
                ["DecisionNodes"] = _decisionNodeCount,
                ["MaxDepth"] = GetTreeDepth(),
                ["SplitConfidence"] = Convert.ToDouble(_splitConfidence)
            }
        };
    }
    
    /// <summary>
    /// Gets the maximum depth of the tree.
    /// </summary>
    private int GetTreeDepth()
    {
        return _root.GetMaxDepth();
    }
    
    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        
        // Version number
        writer.Write(1);
        
        // Tree parameters
        writer.Write(_numClasses);
        writer.Write(Convert.ToDouble(_splitConfidence));
        writer.Write(_gracePeriod);
        writer.Write(Convert.ToDouble(_tieThreshold));
        writer.Write(_maxDepth);
        
        // Tree statistics
        writer.Write(SamplesSeen);
        writer.Write(_activeLeafCount);
        writer.Write(_inactiveLeafCount);
        writer.Write(_decisionNodeCount);
        
        // Serialize tree structure
        SerializeNode(writer, _root);
        
        return ms.ToArray();
    }
    
    private void SerializeNode(BinaryWriter writer, HoeffdingNode node)
    {
        writer.Write(node.IsLeaf);
        writer.Write(node.Depth);
        writer.Write(node.GetSeenInstances());
        
        if (node.IsLeaf)
        {
            // Serialize leaf statistics
            var classCounts = node.GetClassCounts();
            for (int i = 0; i < _numClasses; i++)
            {
                writer.Write(classCounts[i]);
            }
        }
        else
        {
            // Serialize split info
            writer.Write(node.SplitAttribute);
            writer.Write(Convert.ToDouble(node.SplitValue));
            
            // Serialize children
            SerializeNode(writer, node.LeftChild!);
            SerializeNode(writer, node.RightChild!);
        }
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
            
            // Tree parameters
            int numClasses = reader.ReadInt32();
            _splitConfidence = NumOps.FromDouble(reader.ReadDouble());
            _gracePeriod = reader.ReadInt32();
            _tieThreshold = NumOps.FromDouble(reader.ReadDouble());
            _maxDepth = reader.ReadInt32();
            
            // Tree statistics
            _samplesSeen = reader.ReadInt64();
            _activeLeafCount = reader.ReadInt32();
            _inactiveLeafCount = reader.ReadInt32();
            _decisionNodeCount = reader.ReadInt32();
            
            // Deserialize tree structure
            _root = DeserializeNode(reader, 0);
        }
        catch (Exception ex) when (!(ex is ArgumentNullException || ex is ArgumentException))
        {
            throw new ArgumentException("Failed to deserialize the model. The data may be corrupted or in an invalid format.", nameof(data), ex);
        }
    }
    
    private HoeffdingNode DeserializeNode(BinaryReader reader, int depth)
    {
        bool isLeaf = reader.ReadBoolean();
        int nodeDepth = reader.ReadInt32();
        int seenInstances = reader.ReadInt32();
        
        var node = new HoeffdingNode(_root.NumFeatures, _numClasses, NumOps, nodeDepth);
        
        if (isLeaf)
        {
            // Deserialize leaf statistics
            var classCounts = new int[_numClasses];
            for (int i = 0; i < _numClasses; i++)
            {
                classCounts[i] = reader.ReadInt32();
            }
            node.SetClassCounts(classCounts, seenInstances);
        }
        else
        {
            // Deserialize split info
            int splitAttribute = reader.ReadInt32();
            T splitValue = NumOps.FromDouble(reader.ReadDouble());
            
            // Deserialize children
            var leftChild = DeserializeNode(reader, depth + 1);
            var rightChild = DeserializeNode(reader, depth + 1);
            
            node.SetSplit(splitAttribute, splitValue, leftChild, rightChild);
        }
        
        return node;
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
                : _splitConfidence
        };
        
        return new HoeffdingTree<T>(_root.NumFeatures, _numClasses, newOptions, _logger);
    }
    
    /// <inheritdoc/>
    public override int GetInputFeatureCount() => _root.NumFeatures;
    
    /// <inheritdoc/>
    public override int GetOutputFeatureCount() => _numClasses;
    
    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, T> Clone()
    {
        var clone = new HoeffdingTree<T>(_root.NumFeatures, _numClasses, _options, _logger);
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
        // Return split confidence and other hyperparameters
        return new Vector<T>(new[] { _splitConfidence, _tieThreshold, NumOps.FromDouble(_gracePeriod) });
    }
    
    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length < 3)
        {
            throw new ArgumentException("Parameter vector must have at least 3 elements");
        }
        
        _splitConfidence = parameters[0];
        _tieThreshold = parameters[1];
        _gracePeriod = Convert.ToInt32(parameters[2]);
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
        // Return all features that have been used in splits
        var usedFeatures = new HashSet<int>();
        _root.CollectUsedFeatures(usedFeatures);
        return usedFeatures.OrderBy(f => f);
    }
    
    /// <inheritdoc/>
    public override bool IsFeatureUsed(int featureIndex)
    {
        return GetActiveFeatureIndices().Contains(featureIndex);
    }
    
    /// <inheritdoc/>
    public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Hoeffding Tree uses features adaptively, so this is a no-op
        _logger.Warning("SetActiveFeatureIndices is not supported for Hoeffding Tree");
    }
    
    /// <inheritdoc/>
    public override int InputDimensions => _root.NumFeatures;
    
    /// <inheritdoc/>
    public override int OutputDimensions => _numClasses;
    
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
        var classIndex = Convert.ToInt32(testLabels);
        var predIndex = Convert.ToInt32(prediction);
        
        return new Dictionary<string, double>
        {
            ["Accuracy"] = classIndex == predIndex ? 1.0 : 0.0,
            ["Error"] = classIndex == predIndex ? 0.0 : 1.0
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
        // Hoeffding tree doesn't track loss directly
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
                ["ActiveLeaves"] = NumOps.FromDouble(_activeLeafCount),
                ["InactiveLeaves"] = NumOps.FromDouble(_inactiveLeafCount),
                ["DecisionNodes"] = NumOps.FromDouble(_decisionNodeCount),
                ["MaxDepth"] = NumOps.FromDouble(_maxDepth),
                ["SplitConfidence"] = _splitConfidence,
                ["GracePeriod"] = NumOps.FromDouble(_gracePeriod)
            }
        };
    }
    
    /// <inheritdoc/>
    public override void Save()
    {
        // Default implementation saves to a standard location
        SaveModel($"hoeffding_tree_model_{DateTime.Now:yyyyMMddHHmmss}.bin");
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
        _root = null!;
    }
    
    /// <summary>
    /// Inner class representing a node in the Hoeffding tree.
    /// </summary>
    private class HoeffdingNode
    {
        public bool IsLeaf { get; private set; }
        public int Depth { get; }
        public int NumFeatures { get; }
        public int SplitAttribute { get; private set; }
        public T SplitValue { get; private set; }
        public HoeffdingNode? LeftChild { get; private set; }
        public HoeffdingNode? RightChild { get; private set; }
        
        private readonly int _numClasses;
        private readonly int[] _classCounts;
        private readonly Dictionary<int, GaussianEstimator>[] _attributeObservers = default!;
        private int _seenInstances;
        
        public HoeffdingNode(int numFeatures, int numClasses, INumericOperations<T> numOps, int depth)
        {
            IsLeaf = true;
            Depth = depth;
            NumFeatures = numFeatures;
            _numClasses = numClasses;
            _classCounts = new int[numClasses];
            _attributeObservers = new Dictionary<int, GaussianEstimator>[numFeatures];
            
            for (int i = 0; i < numFeatures; i++)
            {
                _attributeObservers[i] = new Dictionary<int, GaussianEstimator>();
            }
            
            _seenInstances = 0;
            SplitAttribute = -1;
            SplitValue = numOps.Zero;
        }
        
        public void UpdateStatistics(Vector<T> instance, int classIndex, INumericOperations<T> numOps)
        {
            if (!IsLeaf)
                return;
                
            _classCounts[classIndex]++;
            _seenInstances++;
            
            // Update attribute statistics
            for (int i = 0; i < instance.Length; i++)
            {
                if (!_attributeObservers[i].ContainsKey(classIndex))
                {
                    _attributeObservers[i][classIndex] = new GaussianEstimator();
                }
                
                _attributeObservers[i][classIndex].Update(Convert.ToDouble(instance[i]));
            }
        }
        
        public HoeffdingNode FilterToLeaf(Vector<T> instance, INumericOperations<T> numOps)
        {
            if (IsLeaf)
                return this;
                
            if (numOps.LessThanOrEquals(instance[SplitAttribute], SplitValue))
                return LeftChild!.FilterToLeaf(instance, numOps);
            else
                return RightChild!.FilterToLeaf(instance, numOps);
        }
        
        public (T gain, T splitValue) CalculateBestSplitForAttribute(int attributeIndex, INumericOperations<T> numOps)
        {
            if (_seenInstances == 0)
                return (numOps.Zero, numOps.Zero);
                
            // Calculate entropy before split
            double entropyBefore = CalculateEntropy(_classCounts, _seenInstances);
            
            // Find best split value using Gaussian estimators
            var splitCandidates = new List<double>();
            foreach (var estimator in _attributeObservers[attributeIndex].Values)
            {
                if (estimator.Count > 0)
                {
                    splitCandidates.Add(estimator.Mean);
                    splitCandidates.Add(estimator.Mean - estimator.StdDev);
                    splitCandidates.Add(estimator.Mean + estimator.StdDev);
                }
            }
            
            if (splitCandidates.Count == 0)
                return (numOps.Zero, numOps.Zero);
                
            splitCandidates = splitCandidates.Distinct().OrderBy(x => x).ToList();
            
            double bestGain = 0;
            double bestSplitValue = 0;
            
            // Try each split candidate
            foreach (var splitCandidate in splitCandidates)
            {
                var leftCounts = new int[_numClasses];
                var rightCounts = new int[_numClasses];
                int leftTotal = 0, rightTotal = 0;
                
                // Estimate split counts using Gaussian distributions
                foreach (var kvp in _attributeObservers[attributeIndex])
                {
                    int classIdx = kvp.Key;
                    var estimator = kvp.Value;
                    
                    if (estimator.Count > 0)
                    {
                        double probLeft = estimator.ProbabilityLessThan(splitCandidate);
                        int leftCount = (int)(probLeft * _classCounts[classIdx]);
                        int rightCount = _classCounts[classIdx] - leftCount;
                        
                        leftCounts[classIdx] = leftCount;
                        rightCounts[classIdx] = rightCount;
                        leftTotal += leftCount;
                        rightTotal += rightCount;
                    }
                }
                
                if (leftTotal > 0 && rightTotal > 0)
                {
                    double leftEntropy = CalculateEntropy(leftCounts, leftTotal);
                    double rightEntropy = CalculateEntropy(rightCounts, rightTotal);
                    
                    double entropyAfter = (leftTotal * leftEntropy + rightTotal * rightEntropy) / _seenInstances;
                    double gain = entropyBefore - entropyAfter;
                    
                    if (gain > bestGain)
                    {
                        bestGain = gain;
                        bestSplitValue = splitCandidate;
                    }
                }
            }
            
            return (numOps.FromDouble(bestGain), numOps.FromDouble(bestSplitValue));
        }
        
        private double CalculateEntropy(int[] counts, int total)
        {
            if (total == 0)
                return 0;
                
            double entropy = 0;
            for (int i = 0; i < counts.Length; i++)
            {
                if (counts[i] > 0)
                {
                    double p = (double)counts[i] / total;
                    entropy -= p * Math.Log(p, 2);
                }
            }
            return entropy;
        }
        
        public void Split(int attribute, T splitValue, INumericOperations<T> numOps)
        {
            IsLeaf = false;
            SplitAttribute = attribute;
            SplitValue = splitValue;
            LeftChild = new HoeffdingNode(NumFeatures, _numClasses, numOps, Depth + 1);
            RightChild = new HoeffdingNode(NumFeatures, _numClasses, numOps, Depth + 1);
        }
        
        public int GetMajorityClass()
        {
            int maxCount = 0;
            int majorityClass = 0;
            
            for (int i = 0; i < _classCounts.Length; i++)
            {
                if (_classCounts[i] > maxCount)
                {
                    maxCount = _classCounts[i];
                    majorityClass = i;
                }
            }
            
            return majorityClass;
        }
        
        public T[] GetClassProbabilities(INumericOperations<T> numOps)
        {
            var probs = new T[_numClasses];
            
            if (_seenInstances == 0)
            {
                // Uniform distribution
                var uniformProb = numOps.Divide(numOps.One, numOps.FromDouble(_numClasses));
                for (int i = 0; i < _numClasses; i++)
                {
                    probs[i] = uniformProb;
                }
            }
            else
            {
                for (int i = 0; i < _numClasses; i++)
                {
                    probs[i] = numOps.Divide(numOps.FromDouble(_classCounts[i]), numOps.FromDouble(_seenInstances));
                }
            }
            
            return probs;
        }
        
        public int GetSeenInstances() => _seenInstances;
        
        public int[] GetClassCounts() => _classCounts.ToArray();
        
        public void SetClassCounts(int[] counts, int seenInstances)
        {
            Array.Copy(counts, _classCounts, counts.Length);
            _seenInstances = seenInstances;
        }
        
        public void SetSplit(int attribute, T value, HoeffdingNode left, HoeffdingNode right)
        {
            IsLeaf = false;
            SplitAttribute = attribute;
            SplitValue = value;
            LeftChild = left;
            RightChild = right;
        }
        
        public int GetMaxDepth()
        {
            if (IsLeaf)
                return Depth;
                
            return Math.Max(LeftChild!.GetMaxDepth(), RightChild!.GetMaxDepth());
        }
        
        public void CollectUsedFeatures(HashSet<int> usedFeatures)
        {
            if (!IsLeaf)
            {
                usedFeatures.Add(SplitAttribute);
                LeftChild!.CollectUsedFeatures(usedFeatures);
                RightChild!.CollectUsedFeatures(usedFeatures);
            }
        }
        
        /// <summary>
        /// Helper class for estimating Gaussian distributions of numeric attributes.
        /// </summary>
        private class GaussianEstimator
        {
            public int Count { get; private set; }
            public double Mean { get; private set; }
            public double M2 { get; private set; } // Sum of squares of differences from mean
            
            public double Variance => Count > 1 ? M2 / (Count - 1) : 0;
            public double StdDev => Math.Sqrt(Variance);
            
            public void Update(double value)
            {
                Count++;
                double delta = value - Mean;
                Mean += delta / Count;
                double delta2 = value - Mean;
                M2 += delta * delta2;
            }
            
            public double ProbabilityLessThan(double threshold)
            {
                if (Count == 0)
                    return 0.5;
                    
                if (StdDev == 0)
                    return Mean < threshold ? 1.0 : 0.0;
                    
                // Use normal CDF approximation
                double z = (threshold - Mean) / StdDev;
                return 0.5 * (1 + Erf(z / Math.Sqrt(2)));
            }
            
            private static double Erf(double x)
            {
                // Approximation of error function
                double a1 = 0.254829592;
                double a2 = -0.284496736;
                double a3 = 1.421413741;
                double a4 = -1.453152027;
                double a5 = 1.061405429;
                double p = 0.3275911;
                
                int sign = x < 0 ? -1 : 1;
                x = Math.Abs(x);
                
                double t = 1.0 / (1.0 + p * x);
                double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
                
                return sign * y;
            }
        }
    }
}