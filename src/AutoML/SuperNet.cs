using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NumericOperations;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// SuperNet implementation for gradient-based neural architecture search (DARTS).
    /// Implements a differentiable architecture search by maintaining architecture parameters (alpha)
    /// and network weights simultaneously.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class SuperNet<T> : IFullModel<T, Tensor<T>, Tensor<T>>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpace<T> _searchSpace;
        private readonly int _numNodes;
        private readonly int _numOperations;
        private readonly Random _random; // Shared Random instance to avoid time-based seeding issues

        // Architecture parameters (alpha) - learnable parameters that determine operation weights
        private readonly List<Matrix<T>> _architectureParams;

        // Network weights - parameters for each operation
        private readonly Dictionary<string, Vector<T>> _weights;

        // Gradients
        private readonly List<Matrix<T>> _architectureGradients;
        private readonly Dictionary<string, Vector<T>> _weightGradients;

        // Model metadata
        private int _inputSize;
        private int _outputSize;

        // IInterpretableModel fields
        private readonly HashSet<InterpretationMethod> _enabledMethods = new();
        private Vector<int>? _sensitiveFeatures;
        private readonly List<FairnessMetric> _fairnessMetrics = new();
        private IModel<Tensor<T>, Tensor<T>, ModelMetaData<T>>? _baseModel;

        public ModelType Type => ModelType.NeuralNetwork;
        public string[] FeatureNames { get; set; } = Array.Empty<string>();
        public int ParameterCount => _weights.Values.Sum(w => w.Length) +
                                      _architectureParams.Sum(a => a.Rows * a.Columns);

        /// <summary>
        /// Initializes a new SuperNet for differentiable architecture search.
        /// </summary>
        /// <param name="searchSpace">The search space defining available operations</param>
        /// <param name="numNodes">Number of nodes in the architecture</param>
        public SuperNet(SearchSpace<T> searchSpace, int numNodes = 4)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _searchSpace = searchSpace;
            _numNodes = numNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5; // Default operations: identity, conv3x3, conv5x5, maxpool, avgpool
            _random = new Random(42); // Initialize with seed for reproducibility

            // Initialize architecture parameters (alpha) with small random values
            _architectureParams = new List<Matrix<T>>();
            _architectureGradients = new List<Matrix<T>>();

            for (int i = 0; i < _numNodes; i++)
            {
                // Each node can receive input from all previous nodes
                // Alpha is initialized near zero so all operations have equal weight after softmax
                var alpha = new Matrix<T>(i + 1, _numOperations);
                for (int j = 0; j < alpha.Rows; j++)
                {
                    for (int k = 0; k < alpha.Columns; k++)
                    {
                        // Small random initialization: range [-0.1, 0.1]
                        alpha[j, k] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.2);
                    }
                }
                _architectureParams.Add(alpha);
                _architectureGradients.Add(new Matrix<T>(i + 1, _numOperations));
            }

            // Initialize network weights
            _weights = new Dictionary<string, Vector<T>>();
            _weightGradients = new Dictionary<string, Vector<T>>();
        }

        /// <summary>
        /// Forward pass through the SuperNet with mixed operations
        /// </summary>
        public Tensor<T> Predict(Tensor<T> input)
        {
            _inputSize = input.Shape[input.Shape.Length - 1];
            _outputSize = _inputSize; // For simplicity, maintain same dimensions

            // Store intermediate node outputs
            var nodeOutputs = new List<Tensor<T>> { input };

            // Process each node
            for (int nodeIdx = 0; nodeIdx < _numNodes; nodeIdx++)
            {
                var nodeOutput = new Tensor<T>(input.Shape);
                var alpha = _architectureParams[nodeIdx];

                // Apply softmax to architecture parameters for this node
                var softmaxWeights = ApplySoftmax(alpha);

                // Mix operations from all previous nodes
                for (int prevNodeIdx = 0; prevNodeIdx <= nodeIdx; prevNodeIdx++)
                {
                    var prevOutput = nodeOutputs[prevNodeIdx];

                    // Apply each operation and mix with softmax weights
                    for (int opIdx = 0; opIdx < _numOperations; opIdx++)
                    {
                        var opOutput = ApplyOperation(prevOutput, opIdx, $"node{nodeIdx}_from{prevNodeIdx}_op{opIdx}");
                        var weight = softmaxWeights[prevNodeIdx, opIdx];

                        // Accumulate weighted operation outputs
                        for (int batchIdx = 0; batchIdx < nodeOutput.Shape[0]; batchIdx++)
                        {
                            for (int featureIdx = 0; featureIdx < nodeOutput.Shape[1]; featureIdx++)
                            {
                                nodeOutput[batchIdx, featureIdx] = _ops.Add(
                                    nodeOutput[batchIdx, featureIdx],
                                    _ops.Multiply(weight, opOutput[batchIdx, featureIdx]));
                            }
                        }
                    }
                }

                nodeOutputs.Add(nodeOutput);
            }

            // Return the output of the final node
            return nodeOutputs[nodeOutputs.Count - 1];
        }

        /// <summary>
        /// Training is handled externally by alternating architecture and weight updates
        /// </summary>
        public void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            throw new NotSupportedException(
                "SuperNet training is handled through alternating optimization. " +
                "Use UpdateArchitectureParameters() and UpdateWeights() instead.");
        }

        /// <summary>
        /// Computes validation loss for architecture parameter updates
        /// </summary>
        public T ComputeValidationLoss(Tensor<T> valData, Tensor<T> valLabels)
        {
            var predictions = Predict(valData);
            return ComputeLoss(predictions, valLabels);
        }

        /// <summary>
        /// Computes training loss for weight updates
        /// </summary>
        public T ComputeTrainingLoss(Tensor<T> trainData, Tensor<T> trainLabels)
        {
            var predictions = Predict(trainData);
            return ComputeLoss(predictions, trainLabels);
        }

        /// <summary>
        /// Computes mean squared error loss
        /// </summary>
        private T ComputeLoss(Tensor<T> predictions, Tensor<T> targets)
        {
            T sumSquaredError = _ops.Zero;
            int count = 0;

            // Access tensors using proper 2D indexing
            for (int batchIdx = 0; batchIdx < predictions.Shape[0]; batchIdx++)
            {
                for (int featureIdx = 0; featureIdx < predictions.Shape[1]; featureIdx++)
                {
                    var diff = _ops.Subtract(predictions[batchIdx, featureIdx], targets[batchIdx, featureIdx]);
                    sumSquaredError = _ops.Add(sumSquaredError, _ops.Multiply(diff, diff));
                    count++;
                }
            }

            return _ops.Divide(sumSquaredError, _ops.FromDouble(count));
        }

        /// <summary>
        /// Backward pass to compute gradients for architecture parameters
        /// </summary>
        public void BackwardArchitecture(Tensor<T> input, Tensor<T> target)
        {
            // Simplified gradient computation
            // In a full implementation, this would use automatic differentiation
            var output = Predict(input);
            var loss = ComputeLoss(output, target);

            // Compute gradients using finite differences (simplified)
            T epsilon = _ops.FromDouble(1e-5);

            for (int nodeIdx = 0; nodeIdx < _architectureParams.Count; nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];
                var grad = _architectureGradients[nodeIdx];

                for (int i = 0; i < alpha.Rows; i++)
                {
                    for (int j = 0; j < alpha.Columns; j++)
                    {
                        // Finite difference approximation
                        T originalValue = alpha[i, j];

                        alpha[i, j] = _ops.Add(originalValue, epsilon);
                        var lossPlus = ComputeValidationLoss(input, target);

                        alpha[i, j] = _ops.Subtract(originalValue, epsilon);
                        var lossMinus = ComputeValidationLoss(input, target);

                        alpha[i, j] = originalValue;

                        // Gradient = (f(x+ε) - f(x-ε)) / (2ε)
                        grad[i, j] = _ops.Divide(
                            _ops.Subtract(lossPlus, lossMinus),
                            _ops.Multiply(_ops.FromDouble(2), epsilon)
                        );
                    }
                }
            }
        }

        /// <summary>
        /// Backward pass to compute gradients for network weights
        /// </summary>
        public void BackwardWeights(Tensor<T> input, Tensor<T> target)
        {
            // Simplified gradient computation for weights
            var output = Predict(input);
            T epsilon = _ops.FromDouble(1e-5);

            foreach (var kvp in _weights)
            {
                var key = kvp.Key;
                var weight = kvp.Value;
                var grad = _weightGradients[key];

                for (int i = 0; i < weight.Length; i++)
                {
                    T originalValue = weight[i];

                    weight[i] = _ops.Add(originalValue, epsilon);
                    var lossPlus = ComputeTrainingLoss(input, target);

                    weight[i] = _ops.Subtract(originalValue, epsilon);
                    var lossMinus = ComputeTrainingLoss(input, target);

                    weight[i] = originalValue;

                    grad[i] = _ops.Divide(
                        _ops.Subtract(lossPlus, lossMinus),
                        _ops.Multiply(_ops.FromDouble(2), epsilon)
                    );
                }
            }
        }

        /// <summary>
        /// Gets architecture parameters for optimization
        /// </summary>
        public List<Matrix<T>> GetArchitectureParameters()
        {
            return _architectureParams;
        }

        /// <summary>
        /// Gets architecture gradients
        /// </summary>
        public List<Matrix<T>> GetArchitectureGradients()
        {
            return _architectureGradients;
        }

        /// <summary>
        /// Gets weight parameters for optimization
        /// </summary>
        public Dictionary<string, Vector<T>> GetWeightParameters()
        {
            return _weights;
        }

        /// <summary>
        /// Gets weight gradients
        /// </summary>
        public Dictionary<string, Vector<T>> GetWeightGradients()
        {
            return _weightGradients;
        }

        /// <summary>
        /// Derives discrete architecture from continuous parameters (argmax selection)
        /// </summary>
        public Architecture<T> DeriveArchitecture()
        {
            var architecture = new Architecture<T>();

            for (int nodeIdx = 0; nodeIdx < _numNodes; nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];
                var softmaxWeights = ApplySoftmax(alpha);

                // For each previous node connection, select operation with highest weight
                for (int prevNodeIdx = 0; prevNodeIdx <= nodeIdx; prevNodeIdx++)
                {
                    int bestOpIdx = 0;
                    T bestWeight = softmaxWeights[prevNodeIdx, 0];

                    for (int opIdx = 1; opIdx < _numOperations; opIdx++)
                    {
                        if (_ops.GreaterThan(softmaxWeights[prevNodeIdx, opIdx], bestWeight))
                        {
                            bestWeight = softmaxWeights[prevNodeIdx, opIdx];
                            bestOpIdx = opIdx;
                        }
                    }

                    // Add selected operation to architecture
                    var operation = GetOperationName(bestOpIdx);
                    architecture.AddOperation(nodeIdx, prevNodeIdx, operation);
                }
            }

            return architecture;
        }

        /// <summary>
        /// Apply softmax to architecture parameters
        /// </summary>
        private Matrix<T> ApplySoftmax(Matrix<T> alpha)
        {
            var result = new Matrix<T>(alpha.Rows, alpha.Columns);

            for (int row = 0; row < alpha.Rows; row++)
            {
                // Compute softmax for this row
                T maxVal = alpha[row, 0];
                for (int col = 1; col < alpha.Columns; col++)
                {
                    if (_ops.GreaterThan(alpha[row, col], maxVal))
                        maxVal = alpha[row, col];
                }

                // Compute exp(x - max) for numerical stability
                T sumExp = _ops.Zero;
                var expValues = new T[alpha.Columns];
                for (int col = 0; col < alpha.Columns; col++)
                {
                    expValues[col] = _ops.Exp(_ops.Subtract(alpha[row, col], maxVal));
                    sumExp = _ops.Add(sumExp, expValues[col]);
                }

                // Normalize
                for (int col = 0; col < alpha.Columns; col++)
                {
                    result[row, col] = _ops.Divide(expValues[col], sumExp);
                }
            }

            return result;
        }

        /// <summary>
        /// Apply a specific operation to input
        /// </summary>
        private Tensor<T> ApplyOperation(Tensor<T> input, int opIdx, string weightKey)
        {
            // Initialize weights if needed
            if (!_weights.ContainsKey(weightKey))
            {
                _weights[weightKey] = new Vector<T>(input.Length);
                _weightGradients[weightKey] = new Vector<T>(input.Length);

                // Initialize with small random values
                for (int i = 0; i < input.Length; i++)
                {
                    _weights[weightKey][i] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.1);
                }
            }

            var output = new Tensor<T>(input.Shape);
            var weight = _weights[weightKey];

            // Apply operation (simplified) using proper 2D tensor indexing
            switch (opIdx)
            {
                case 0: // Identity
                    for (int batchIdx = 0; batchIdx < input.Shape[0]; batchIdx++)
                    {
                        for (int featureIdx = 0; featureIdx < input.Shape[1]; featureIdx++)
                        {
                            output[batchIdx, featureIdx] = input[batchIdx, featureIdx];
                        }
                    }
                    break;

                case 1: // 3x3 Conv (simplified as weighted pass)
                    {
                        for (int batchIdx = 0; batchIdx < input.Shape[0]; batchIdx++)
                        {
                            for (int featureIdx = 0; featureIdx < input.Shape[1]; featureIdx++)
                            {
                                if (featureIdx < weight.Length)
                                {
                                    output[batchIdx, featureIdx] = _ops.Multiply(
                                        input[batchIdx, featureIdx],
                                        _ops.Add(_ops.One, weight[featureIdx]));
                                }
                            }
                        }
                    }
                    break;

                case 2: // 5x5 Conv (simplified)
                    {
                        for (int batchIdx = 0; batchIdx < input.Shape[0]; batchIdx++)
                        {
                            for (int featureIdx = 0; featureIdx < input.Shape[1]; featureIdx++)
                            {
                                if (featureIdx < weight.Length)
                                {
                                    output[batchIdx, featureIdx] = _ops.Multiply(
                                        input[batchIdx, featureIdx],
                                        _ops.Add(_ops.One, _ops.Multiply(_ops.FromDouble(1.5), weight[featureIdx])));
                                }
                            }
                        }
                    }
                    break;

                case 3: // MaxPool (simplified)
                    for (int batchIdx = 0; batchIdx < input.Shape[0]; batchIdx++)
                    {
                        for (int featureIdx = 0; featureIdx < input.Shape[1]; featureIdx++)
                        {
                            output[batchIdx, featureIdx] = _ops.Multiply(input[batchIdx, featureIdx], _ops.FromDouble(0.9));
                        }
                    }
                    break;

                case 4: // AvgPool (simplified)
                    for (int batchIdx = 0; batchIdx < input.Shape[0]; batchIdx++)
                    {
                        for (int featureIdx = 0; featureIdx < input.Shape[1]; featureIdx++)
                        {
                            output[batchIdx, featureIdx] = _ops.Multiply(input[batchIdx, featureIdx], _ops.FromDouble(0.8));
                        }
                    }
                    break;

                default:
                    for (int batchIdx = 0; batchIdx < input.Shape[0]; batchIdx++)
                    {
                        for (int featureIdx = 0; featureIdx < input.Shape[1]; featureIdx++)
                        {
                            output[batchIdx, featureIdx] = input[batchIdx, featureIdx];
                        }
                    }
                    break;
            }

            return output;
        }

        /// <summary>
        /// Gets the human-readable name for a given operation index.
        /// Maps operation indices to their corresponding operation types in the NAS search space.
        /// </summary>
        /// <param name="opIdx">The operation index (0-4)</param>
        /// <returns>The operation name (identity, conv3x3, conv5x5, maxpool, avgpool)</returns>
        private string GetOperationName(int opIdx)
        {
            return opIdx switch
            {
                0 => "identity",
                1 => "conv3x3",
                2 => "conv5x5",
                3 => "maxpool",
                4 => "avgpool",
                _ => "identity"
            };
        }

        // IFullModel implementation
        public Vector<T> GetParameters()
        {
            var allParams = new List<T>();

            // Add architecture parameters
            foreach (var alpha in _architectureParams)
            {
                for (int i = 0; i < alpha.Rows; i++)
                    for (int j = 0; j < alpha.Columns; j++)
                        allParams.Add(alpha[i, j]);
            }

            // Add weights
            foreach (var weight in _weights.Values)
            {
                for (int i = 0; i < weight.Length; i++)
                    allParams.Add(weight[i]);
            }

            return new Vector<T>(allParams.ToArray());
        }

        public void SetParameters(Vector<T> parameters)
        {
            int idx = 0;

            // Set architecture parameters
            foreach (var alpha in _architectureParams)
            {
                for (int i = 0; i < alpha.Rows; i++)
                    for (int j = 0; j < alpha.Columns; j++)
                        alpha[i, j] = parameters[idx++];
            }

            // Set weights
            foreach (var weight in _weights.Values)
            {
                for (int i = 0; i < weight.Length; i++)
                    weight[i] = parameters[idx++];
            }
        }

        public IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
        {
            var clone = (SuperNet<T>)Clone();
            clone.SetParameters(parameters);
            return clone;
        }

        public ModelMetaData<T> GetModelMetaData()
        {
            return new ModelMetaData<T>
            {
                ModelType = ModelType.NeuralNetwork,
                Description = "Differentiable Architecture Search SuperNet",
                FeatureCount = _inputSize,
                Complexity = _numNodes,
                AdditionalInfo = new Dictionary<string, object>
                {
                    ["NumNodes"] = _numNodes,
                    ["NumOperations"] = _numOperations,
                    ["ParameterCount"] = ParameterCount
                }
            };
        }

        public void SaveModel(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

            // Validate path security: prevent directory traversal attacks
            // Use canonicalized path and ensure it is within the current working directory
            var fullPath = System.IO.Path.GetFullPath(filePath);

            // Additional validation: ensure the resolved path doesn't escape the working directory
            var currentDirectory = System.IO.Path.GetFullPath(Environment.CurrentDirectory);
            // Ensure trailing separator for strict directory containment (prevents /app vs /app-data bypass)
            var currentDirWithSep = currentDirectory.EndsWith(System.IO.Path.DirectorySeparatorChar.ToString())
                ? currentDirectory
                : currentDirectory + System.IO.Path.DirectorySeparatorChar;
            if (!fullPath.StartsWith(currentDirWithSep, StringComparison.OrdinalIgnoreCase))
                throw new UnauthorizedAccessException($"Attempted to save model outside of the current directory. Path: {fullPath}");

            using var fs = new System.IO.FileStream(fullPath, System.IO.FileMode.Create);
            using var writer = new System.IO.BinaryWriter(fs);

            writer.Write(_numNodes);
            writer.Write(_numOperations);
            writer.Write(_inputSize);
            writer.Write(_outputSize);

            // Serialize architecture parameters
            writer.Write(_architectureParams.Count);
            foreach (var alpha in _architectureParams)
            {
                writer.Write(alpha.Rows);
                writer.Write(alpha.Columns);
                for (int i = 0; i < alpha.Rows; i++)
                {
                    for (int j = 0; j < alpha.Columns; j++)
                    {
                        writer.Write(Convert.ToDouble(alpha[i, j]));
                    }
                }
            }

            // Serialize weights
            writer.Write(_weights.Count);
            foreach (var kvp in _weights)
            {
                writer.Write(kvp.Key);
                writer.Write(kvp.Value.Length);
                for (int i = 0; i < kvp.Value.Length; i++)
                {
                    writer.Write(Convert.ToDouble(kvp.Value[i]));
                }
            }
        }
        public void LoadModel(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

            // Validate path security: prevent directory traversal attacks
            // Use canonicalized path and ensure it is within the current working directory
            var fullPath = System.IO.Path.GetFullPath(filePath);

            // Additional validation: ensure the resolved path doesn't escape the working directory
            var currentDirectory = System.IO.Path.GetFullPath(Environment.CurrentDirectory);
            // Ensure trailing separator for strict directory containment (prevents /app vs /app-data bypass)
            var currentDirWithSep = currentDirectory.EndsWith(System.IO.Path.DirectorySeparatorChar.ToString())
                ? currentDirectory
                : currentDirectory + System.IO.Path.DirectorySeparatorChar;
            if (!fullPath.StartsWith(currentDirWithSep, StringComparison.OrdinalIgnoreCase))
                throw new UnauthorizedAccessException($"Attempted to load model from outside the current directory. Path: {fullPath}");

            if (!System.IO.File.Exists(fullPath))
                throw new System.IO.FileNotFoundException($"Model file not found: {filePath}");

            using var fs = new System.IO.FileStream(fullPath, System.IO.FileMode.Open);
            using var reader = new System.IO.BinaryReader(fs);

            // Deserialize _numNodes and _numOperations (read-only fields need reflection or constructor)
            var numNodes = reader.ReadInt32();
            var numOperations = reader.ReadInt32();

            // Validate that deserialized structure matches this instance
            if (numNodes != _numNodes || numOperations != _numOperations)
            {
                throw new InvalidOperationException(
                    $"Model file structure mismatch: file has numNodes={numNodes}, numOperations={numOperations}, " +
                    $"but this instance has numNodes={_numNodes}, numOperations={_numOperations}.");
            }

            _inputSize = reader.ReadInt32();
            _outputSize = reader.ReadInt32();

            // Deserialize architecture parameters
            int alphaCount = reader.ReadInt32();
            _architectureParams.Clear();
            _architectureGradients.Clear();
            for (int idx = 0; idx < alphaCount; idx++)
            {
                int rows = reader.ReadInt32();
                int cols = reader.ReadInt32();
                var alpha = new Matrix<T>(rows, cols);
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        alpha[i, j] = _ops.FromDouble(reader.ReadDouble());
                    }
                }
                _architectureParams.Add(alpha);
                _architectureGradients.Add(new Matrix<T>(rows, cols));
            }

            // Deserialize weights
            int weightCount = reader.ReadInt32();
            _weights.Clear();
            _weightGradients.Clear();
            for (int idx = 0; idx < weightCount; idx++)
            {
                string key = reader.ReadString();
                int length = reader.ReadInt32();
                var weight = new Vector<T>(length);
                for (int i = 0; i < length; i++)
                {
                    weight[i] = _ops.FromDouble(reader.ReadDouble());
                }
                _weights[key] = weight;
                _weightGradients[key] = new Vector<T>(length);
            }
        }
        public byte[] Serialize()
        {
            using var ms = new System.IO.MemoryStream();
            using var writer = new System.IO.BinaryWriter(ms);

            writer.Write(_numNodes);
            writer.Write(_numOperations);
            writer.Write(_inputSize);
            writer.Write(_outputSize);

            // Serialize architecture parameters
            writer.Write(_architectureParams.Count);
            foreach (var alpha in _architectureParams)
            {
                writer.Write(alpha.Rows);
                writer.Write(alpha.Columns);
                for (int i = 0; i < alpha.Rows; i++)
                {
                    for (int j = 0; j < alpha.Columns; j++)
                    {
                        writer.Write(Convert.ToDouble(alpha[i, j]));
                    }
                }
            }

            // Serialize weights
            writer.Write(_weights.Count);
            foreach (var kvp in _weights)
            {
                writer.Write(kvp.Key);
                writer.Write(kvp.Value.Length);
                for (int i = 0; i < kvp.Value.Length; i++)
                {
                    writer.Write(Convert.ToDouble(kvp.Value[i]));
                }
            }

            return ms.ToArray();
        }
        public void Deserialize(byte[] data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data), "The data parameter passed to Deserialize cannot be null.");

            using var ms = new System.IO.MemoryStream(data);
            using var reader = new System.IO.BinaryReader(ms);

            // Deserialize _numNodes and _numOperations (read-only fields need reflection or constructor)
            var numNodes = reader.ReadInt32();
            var numOperations = reader.ReadInt32();

            // Validate that deserialized structure matches this instance
            if (numNodes != _numNodes || numOperations != _numOperations)
            {
                throw new InvalidOperationException(
                    $"Deserialized model structure does not match this instance. " +
                    $"Expected numNodes={_numNodes}, numOperations={_numOperations}, " +
                    $"but got numNodes={numNodes}, numOperations={numOperations}.");
            }

            _inputSize = reader.ReadInt32();
            _outputSize = reader.ReadInt32();

            // Deserialize architecture parameters
            int alphaCount = reader.ReadInt32();
            _architectureParams.Clear();
            _architectureGradients.Clear();
            for (int idx = 0; idx < alphaCount; idx++)
            {
                int rows = reader.ReadInt32();
                int cols = reader.ReadInt32();
                var alpha = new Matrix<T>(rows, cols);
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        alpha[i, j] = _ops.FromDouble(reader.ReadDouble());
                    }
                }
                _architectureParams.Add(alpha);
                _architectureGradients.Add(new Matrix<T>(rows, cols));
            }

            // Deserialize weights
            int weightCount = reader.ReadInt32();
            _weights.Clear();
            _weightGradients.Clear();
            for (int idx = 0; idx < weightCount; idx++)
            {
                string key = reader.ReadString();
                int length = reader.ReadInt32();
                var weight = new Vector<T>(length);
                for (int i = 0; i < length; i++)
                {
                    weight[i] = _ops.FromDouble(reader.ReadDouble());
                }
                _weights[key] = weight;
                _weightGradients[key] = new Vector<T>(length);
            }
        }

        public Dictionary<string, T> GetFeatureImportance() => new Dictionary<string, T>();
        public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputSize);
        public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _inputSize;
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }

        public IFullModel<T, Tensor<T>, Tensor<T>> Clone()
        {
            return new SuperNet<T>(_searchSpace, _numNodes);
        }

        public IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

        #region IInterpretableModel Implementation

        /// <summary>
        /// Gets the operation importance for SuperNet architecture search.
        /// Returns importance scores for architectural operations rather than input features.
        /// </summary>
        /// <param name="inputs">Input tensor (required for interface compliance; not used in this implementation)</param>
        /// <returns>Dictionary mapping operation indices to their importance scores</returns>
        /// <remarks>
        /// <para>
        /// <b>Note:</b> SuperNet reinterprets "feature importance" as "operation importance" in the context of Neural Architecture Search (NAS).
        /// The returned dictionary maps operation indices (0=identity, 1=conv3x3, 2=conv5x5, etc.) to their importance scores,
        /// calculated by aggregating the absolute values of architecture parameters across all nodes.
        /// </para>
        /// <para>
        /// The 'inputs' parameter is required for IInterpretableModel interface compliance but is not used.
        /// SuperNet analyzes operation importance based on learned architecture parameters rather than input data.
        /// </para>
        /// </remarks>
        public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync(Tensor<T> inputs)
        {
            var importance = new Dictionary<int, T>();

            // For SuperNet, we analyze operation importance rather than input feature importance
            // Each operation index represents a different architectural operation (identity, conv3x3, etc.)
            for (int opIdx = 0; opIdx < _numOperations; opIdx++)
            {
                T sum = _ops.Zero;

                // Aggregate importance across all nodes and connections
                foreach (var alpha in _architectureParams)
                {
                    // Sum absolute values of architecture parameters for this operation
                    for (int i = 0; i < alpha.Rows; i++)
                    {
                        if (opIdx < alpha.Columns)
                        {
                            sum = _ops.Add(sum, _ops.Abs(alpha[i, opIdx]));
                        }
                    }
                }

                importance[opIdx] = sum;
            }

            return await Task.FromResult(importance);
        }

        /// <summary>
        /// Gets the local feature importance for a specific input.
        /// Provides importance based on softmax weights, analyzing which operations are most active.
        /// </summary>
        public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Tensor<T> input)
        {
            var importance = new Dictionary<int, T>();

            // For local importance, we use softmax-transformed architecture parameters
            // to determine which operations are most active for this specific input
            for (int opIdx = 0; opIdx < _numOperations; opIdx++)
            {
                T sum = _ops.Zero;

                // Apply softmax and aggregate weights for each operation
                foreach (var alpha in _architectureParams)
                {
                    var softmaxWeights = ApplySoftmax(alpha);

                    for (int i = 0; i < softmaxWeights.Rows; i++)
                    {
                        if (opIdx < softmaxWeights.Columns)
                        {
                            sum = _ops.Add(sum, softmaxWeights[i, opIdx]);
                        }
                    }
                }

                importance[opIdx] = sum;
            }

            return await Task.FromResult(importance);
        }

        /// <summary>
        /// Gets SHAP values for the given inputs.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<Matrix<T>> GetShapValuesAsync(Tensor<T> inputs)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "SHAP values are not supported for SuperNet architecture search models. " +
                "SuperNet uses differentiable architecture search and does not have traditional feature attribution.");
        }

        /// <summary>
        /// Gets LIME explanation for a specific input.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Tensor<T> input, int numFeatures = 10)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "LIME explanations are not supported for SuperNet architecture search models. " +
                "Use GetGlobalFeatureImportanceAsync or GetLocalFeatureImportanceAsync instead.");
        }

        /// <summary>
        /// Gets partial dependence data for specified features.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "Partial dependence plots are not supported for SuperNet architecture search models. " +
                "SuperNet focuses on architecture optimization rather than feature-level analysis.");
        }

        /// <summary>
        /// Gets counterfactual explanation for a given input and desired output.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Tensor<T> input, Tensor<T> desiredOutput, int maxChanges = 5)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "Counterfactual explanations are not supported for SuperNet architecture search models. " +
                "SuperNet is designed for architecture search, not instance-level counterfactuals.");
        }

        /// <summary>
        /// Gets model-specific interpretability information for SuperNet.
        /// Returns architecture parameters and their importance.
        /// </summary>
        public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
        {
            var info = new Dictionary<string, object>
            {
                ["ModelType"] = "SuperNet (Differentiable Architecture Search)",
                ["NumNodes"] = _numNodes,
                ["NumOperations"] = _numOperations,
                ["ParameterCount"] = ParameterCount,
                ["ArchitectureParameterCount"] = _architectureParams.Sum(a => a.Rows * a.Columns),
                ["WeightParameterCount"] = _weights.Values.Sum(w => w.Length),
                ["InputSize"] = _inputSize,
                ["OutputSize"] = _outputSize
            };

            // Add architecture parameter statistics
            var archStats = new List<Dictionary<string, object>>();
            for (int i = 0; i < _architectureParams.Count; i++)
            {
                var alpha = _architectureParams[i];
                var softmax = ApplySoftmax(alpha);

                var nodeStats = new Dictionary<string, object>
                {
                    ["NodeIndex"] = i,
                    ["Rows"] = alpha.Rows,
                    ["Columns"] = alpha.Columns,
                    ["ParameterCount"] = alpha.Rows * alpha.Columns
                };

                archStats.Add(nodeStats);
            }

            info["ArchitectureNodes"] = archStats;

            return await Task.FromResult(info);
        }

        /// <summary>
        /// Generates a text explanation for a prediction.
        /// Provides a description of which operations are most important in the SuperNet.
        /// </summary>
        public virtual async Task<string> GenerateTextExplanationAsync(Tensor<T> input, Tensor<T> prediction)
        {
            var explanation = $"SuperNet Architecture Search Model:\n";
            explanation += $"- Network contains {_numNodes} nodes with {_numOperations} operations each\n";
            explanation += $"- Total parameters: {ParameterCount}\n";
            explanation += $"- Architecture is determined by learned softmax weights over operations\n\n";

            explanation += "Most important architectural decisions:\n";

            // Identify most important nodes based on architecture parameters
            for (int nodeIdx = 0; nodeIdx < Math.Min(3, _numNodes); nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];
                var softmax = ApplySoftmax(alpha);

                // Find the operation with highest weight
                if (softmax.Rows > 0 && softmax.Columns > 0)
                {
                    int bestOp = 0;
                    T bestWeight = softmax[0, 0];

                    for (int i = 0; i < softmax.Rows; i++)
                    {
                        for (int j = 0; j < softmax.Columns; j++)
                        {
                            if (_ops.GreaterThan(softmax[i, j], bestWeight))
                            {
                                bestWeight = softmax[i, j];
                                bestOp = j;
                            }
                        }
                    }

                    explanation += $"- Node {nodeIdx}: {GetOperationName(bestOp)} operation is dominant\n";
                }
                else
                {
                    explanation += $"- Node {nodeIdx}: No operations available (empty softmax matrix)\n";
                }
            }

            return await Task.FromResult(explanation);
        }

        /// <summary>
        /// Gets feature interaction effects between two features.
        /// Analyzes interactions between operations based on architecture parameter correlations.
        /// </summary>
        public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
        {
            // In SuperNet context, feature indices represent operation indices
            if (feature1Index < 0 || feature1Index >= _numOperations ||
                feature2Index < 0 || feature2Index >= _numOperations)
            {
                throw new ArgumentOutOfRangeException(
                    $"Feature indices must be in the range [0, {_numOperations - 1}]. " +
                    $"Received feature1Index={feature1Index}, feature2Index={feature2Index}.");
            }

            // Calculate correlation between two operations across all architecture parameters
            T sum1 = _ops.Zero;
            T sum2 = _ops.Zero;
            T sumProduct = _ops.Zero;
            T sumSquares1 = _ops.Zero;
            T sumSquares2 = _ops.Zero;
            int count = 0;

            foreach (var alpha in _architectureParams)
            {
                for (int i = 0; i < alpha.Rows; i++)
                {
                    if (feature1Index < alpha.Columns && feature2Index < alpha.Columns)
                    {
                        T val1 = alpha[i, feature1Index];
                        T val2 = alpha[i, feature2Index];

                        sum1 = _ops.Add(sum1, val1);
                        sum2 = _ops.Add(sum2, val2);
                        sumProduct = _ops.Add(sumProduct, _ops.Multiply(val1, val2));
                        sumSquares1 = _ops.Add(sumSquares1, _ops.Multiply(val1, val1));
                        sumSquares2 = _ops.Add(sumSquares2, _ops.Multiply(val2, val2));
                        count++;
                    }
                }
            }

            if (count == 0)
            {
                return _ops.Zero;
            }

            // Calculate correlation coefficient
            T n = _ops.FromDouble(count);
            T numerator = _ops.Subtract(
                _ops.Multiply(n, sumProduct),
                _ops.Multiply(sum1, sum2)
            );

            T denom1 = _ops.Subtract(
                _ops.Multiply(n, sumSquares1),
                _ops.Multiply(sum1, sum1)
            );

            T denom2 = _ops.Subtract(
                _ops.Multiply(n, sumSquares2),
                _ops.Multiply(sum2, sum2)
            );

            T denominator = _ops.Multiply(denom1, denom2);

            // Avoid division by zero
            if (_ops.Equals(denominator, _ops.Zero))
            {
                return _ops.Zero;
            }

            T correlation = _ops.Divide(numerator, _ops.Sqrt(denominator));

            return await Task.FromResult(correlation);
        }

        /// <summary>
        /// Validates fairness metrics for the given inputs.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Tensor<T> inputs, int sensitiveFeatureIndex)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "Fairness validation is not supported for SuperNet architecture search models. " +
                "SuperNet focuses on architecture optimization rather than fairness evaluation.");
        }

        /// <summary>
        /// Gets anchor explanation for a given input.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Tensor<T> input, T threshold)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "Anchor explanations are not supported for SuperNet architecture search models. " +
                "SuperNet focuses on architecture optimization rather than instance-level explanations.");
        }

        /// <summary>
        /// Sets the base model for interpretability analysis.
        /// </summary>
        public virtual void SetBaseModel(IModel<Tensor<T>, Tensor<T>, ModelMetaData<T>> model)
        {
            _baseModel = model ?? throw new ArgumentNullException(nameof(model));
        }

        /// <summary>
        /// Enables specific interpretation methods.
        /// </summary>
        public virtual void EnableMethod(params InterpretationMethod[] methods)
        {
            if (methods == null)
                return;

            foreach (var method in methods)
            {
                _enabledMethods.Add(method);
            }
        }

        /// <summary>
        /// Configures fairness evaluation settings.
        /// </summary>
        public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
        {
            _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
            _fairnessMetrics.Clear();
            if (fairnessMetrics != null)
            {
                _fairnessMetrics.AddRange(fairnessMetrics);
            }
        }

        #endregion
    }
}

