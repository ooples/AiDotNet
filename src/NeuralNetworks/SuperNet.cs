using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.AutoML;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.LossFunctions;
using AiDotNet.Validation;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// SuperNet implementation for gradient-based neural architecture search (DARTS).
    /// Implements a differentiable architecture search by maintaining architecture parameters (alpha)
    /// and network weights simultaneously.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class SuperNet<T> : IFullModel<T, Tensor<T>, Tensor<T>>
    {
        /// <summary>
        /// Provides numeric operations for type T.
        /// </summary>
        protected readonly INumericOperations<T> NumOps;
        private readonly SearchSpaceBase<T> _searchSpace;
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
        private IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>>? _baseModel;

        /// <summary>
        /// The default loss function used by this model for gradient computation.
        /// </summary>
        private readonly ILossFunction<T> _defaultLossFunction;

        public ModelType Type => ModelType.NeuralNetwork;
        public string[] FeatureNames { get; set; } = Array.Empty<string>();
        public int ParameterCount => _weights.Values.Sum(w => w.Length) +
                                      _architectureParams.Sum(a => a.Rows * a.Columns);

        /// <summary>
        /// Gets the default loss function used by this model for gradient computation.
        /// </summary>
        /// <remarks>
        /// <para>
        /// For SuperNet (Neural Architecture Search), the default loss function is Mean Squared Error (MSE),
        /// which is used for computing both architecture and weight gradients.
        /// </para>
        /// </remarks>
        public ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

        /// <summary>
        /// Initializes a new SuperNet for differentiable architecture search.
        /// </summary>
        /// <param name="searchSpace">The search space defining available operations</param>
        /// <param name="numNodes">Number of nodes in the architecture</param>
        /// <param name="lossFunction">Optional loss function to use for training. If null, uses Mean Squared Error (MSE) for neural architecture search.</param>
        public SuperNet(SearchSpaceBase<T> searchSpace, int numNodes = 4, ILossFunction<T>? lossFunction = null)
        {
            NumOps = MathHelper.GetNumericOperations<T>();
            _searchSpace = searchSpace;
            _numNodes = numNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5; // Default operations: identity, conv3x3, conv5x5, maxpool, avgpool
            _random = RandomHelper.CreateSeededRandom(42); // Initialize with seed for reproducibility

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
                        alpha[j, k] = NumOps.FromDouble((_random.NextDouble() - 0.5) * 0.2);
                    }
                }
                _architectureParams.Add(alpha);
                _architectureGradients.Add(new Matrix<T>(i + 1, _numOperations));
            }

            // Initialize network weights
            _weights = new Dictionary<string, Vector<T>>();
            _weightGradients = new Dictionary<string, Vector<T>>();

            // Initialize default loss function (MSE for SuperNet)
            _defaultLossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        }

        /// <summary>
        /// Forward pass through the SuperNet with mixed operations
        /// </summary>
        public Tensor<T> Predict(Tensor<T> input)
        {
            // Handle 1D input by reshaping to 2D [1, features]
            bool was1D = input.Shape.Length == 1;
            int[] originalShape = input.Shape;
            if (was1D)
            {
                input = input.Reshape([1, input.Shape[0]]);
            }
            else if (input.Shape.Length > 2)
            {
                // For higher-rank tensors, flatten to 2D [batch, features]
                int batchSize = 1;
                for (int i = 0; i < input.Shape.Length - 1; i++)
                    batchSize *= input.Shape[i];
                int features = input.Shape[input.Shape.Length - 1];
                input = input.Reshape([batchSize, features]);
            }

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
                                nodeOutput[batchIdx, featureIdx] = NumOps.Add(
                                    nodeOutput[batchIdx, featureIdx],
                                    NumOps.Multiply(weight, opOutput[batchIdx, featureIdx]));
                            }
                        }
                    }
                }

                nodeOutputs.Add(nodeOutput);
            }

            // Get final output and restore original shape if needed
            var result = nodeOutputs[nodeOutputs.Count - 1];
            if (was1D)
            {
                result = result.Reshape(originalShape);
            }
            else if (originalShape.Length > 2)
            {
                result = result.Reshape(originalShape);
            }

            return result;
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
            T sumSquaredError = NumOps.Zero;
            int count = 0;

            // Access tensors using proper 2D indexing
            for (int batchIdx = 0; batchIdx < predictions.Shape[0]; batchIdx++)
            {
                for (int featureIdx = 0; featureIdx < predictions.Shape[1]; featureIdx++)
                {
                    var diff = NumOps.Subtract(predictions[batchIdx, featureIdx], targets[batchIdx, featureIdx]);
                    sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Multiply(diff, diff));
                    count++;
                }
            }

            return NumOps.Divide(sumSquaredError, NumOps.FromDouble(count));
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
            T epsilon = NumOps.FromDouble(1e-5);

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

                        alpha[i, j] = NumOps.Add(originalValue, epsilon);
                        var lossPlus = ComputeValidationLoss(input, target);

                        alpha[i, j] = NumOps.Subtract(originalValue, epsilon);
                        var lossMinus = ComputeValidationLoss(input, target);

                        alpha[i, j] = originalValue;

                        // Gradient = (f(x+ε) - f(x-ε)) / (2ε)
                        grad[i, j] = NumOps.Divide(
                            NumOps.Subtract(lossPlus, lossMinus),
                            NumOps.Multiply(NumOps.FromDouble(2), epsilon)
                        );
                    }
                }
            }
        }

        /// <summary>
        /// Backward pass to compute gradients for network weights using the specified loss function.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="target">The target tensor.</param>
        /// <param name="lossFunction">The loss function to use for gradient computation.</param>
        public void BackwardWeights(Tensor<T> input, Tensor<T> target, ILossFunction<T> lossFunction)
        {
            // Simplified gradient computation for weights
            var output = Predict(input);
            T epsilon = NumOps.FromDouble(1e-5);

            foreach (var kvp in _weights)
            {
                var key = kvp.Key;
                var weight = kvp.Value;
                var grad = _weightGradients[key];

                for (int i = 0; i < weight.Length; i++)
                {
                    T originalValue = weight[i];

                    weight[i] = NumOps.Add(originalValue, epsilon);
                    var lossPlus = ComputeLossWithFunction(input, target, lossFunction);

                    weight[i] = NumOps.Subtract(originalValue, epsilon);
                    var lossMinus = ComputeLossWithFunction(input, target, lossFunction);

                    weight[i] = originalValue;

                    grad[i] = NumOps.Divide(
                        NumOps.Subtract(lossPlus, lossMinus),
                        NumOps.Multiply(NumOps.FromDouble(2), epsilon)
                    );
                }
            }
        }

        /// <summary>
        /// Computes loss using the specified loss function.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="target">The target tensor.</param>
        /// <param name="lossFunction">The loss function to use.</param>
        /// <returns>The computed loss value.</returns>
        private T ComputeLossWithFunction(Tensor<T> input, Tensor<T> target, ILossFunction<T> lossFunction)
        {
            var predictions = Predict(input);

            // Flatten tensors to vectors for ILossFunction
            var predVector = FlattenTensor(predictions);
            var targetVector = FlattenTensor(target);

            return lossFunction.CalculateLoss(predVector, targetVector);
        }

        /// <summary>
        /// Flattens a 2D tensor to a vector.
        /// </summary>
        private Vector<T> FlattenTensor(Tensor<T> tensor)
        {
            var flattenedData = new List<T>();
            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    flattenedData.Add(tensor[i, j]);
                }
            }
            return new Vector<T>(flattenedData.ToArray());
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
        /// Computes gradients of the loss function with respect to model parameters WITHOUT updating parameters.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="target">The target/expected output tensor.</param>
        /// <param name="lossFunction">The loss function to use. If null, uses the model's default loss function.</param>
        /// <returns>A vector containing gradients with respect to all model parameters (both architecture and weights).</returns>
        /// <exception cref="ArgumentNullException">If input or target is null.</exception>
        /// <remarks>
        /// <para>
        /// For SuperNet, this computes gradients for weight parameters only (not architecture parameters).
        /// Architecture parameters are updated separately in DARTS using validation data.
        /// The method uses the existing BackwardWeights method and collects gradients from all layers.
        /// </para>
        /// <para><b>For Beginners:</b>
        /// SuperNet has two types of parameters:
        /// - Architecture parameters (α): which operations to use
        /// - Weight parameters (w): the actual neural network weights
        ///
        /// This method computes gradients for the weight parameters based on training data.
        /// In DARTS, architecture parameters are optimized separately on validation data.
        /// </para>
        /// </remarks>
        public Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (target == null)
                throw new ArgumentNullException(nameof(target));

            // Use the effective loss function (supplied or default)
            var effectiveLoss = lossFunction ?? _defaultLossFunction;

            // Use BackwardWeights to compute gradients for weight parameters
            BackwardWeights(input, target, effectiveLoss);

            // Collect all gradients into a single vector
            var gradients = new List<T>();

            // Add architecture parameter gradients as ZEROS (not computed in this method)
            // Architecture parameters are optimized separately in DARTS on validation data
            // We include zeros here to maintain consistent vector length with GetParameters()
            var zero = NumOps.FromDouble(0.0);
            foreach (var alpha in _architectureParams)
            {
                for (int i = 0; i < alpha.Rows; i++)
                    for (int j = 0; j < alpha.Columns; j++)
                        gradients.Add(zero);  // Zero gradient since not computed here
            }

            // Add weight gradients (freshly computed by BackwardWeights above)
            foreach (var weightGrad in _weightGradients.Values)
            {
                for (int i = 0; i < weightGrad.Length; i++)
                    gradients.Add(weightGrad[i]);
            }

            return new Vector<T>(gradients.ToArray());
        }

        /// <summary>
        /// Applies pre-computed gradients to update the model parameters.
        /// </summary>
        /// <param name="gradients">The gradient vector to apply.</param>
        /// <param name="learningRate">The learning rate for the update.</param>
        /// <exception cref="ArgumentNullException">If gradients is null.</exception>
        /// <exception cref="ArgumentException">If gradient vector length doesn't match parameter count.</exception>
        /// <remarks>
        /// <para>
        /// Updates both architecture and weight parameters using: θ = θ - learningRate * gradients
        /// </para>
        /// <para><b>For Beginners:</b>
        /// This method applies the gradient updates to both:
        /// - Architecture parameters (which operations are selected)
        /// - Weight parameters (the neural network weights)
        ///
        /// In DARTS, you typically call this with different learning rates for
        /// architecture and weight parameters.
        /// </para>
        /// </remarks>
        public void ApplyGradients(Vector<T> gradients, T learningRate)
        {
            if (gradients == null)
                throw new ArgumentNullException(nameof(gradients));

            var currentParams = GetParameters();

            if (gradients.Length != currentParams.Length)
            {
                throw new ArgumentException(
                    $"Gradient vector length ({gradients.Length}) must match parameter count ({currentParams.Length})",
                    nameof(gradients));
            }

            int idx = 0;

            // Update architecture parameters
            foreach (var alpha in _architectureParams)
            {
                for (int i = 0; i < alpha.Rows; i++)
                {
                    for (int j = 0; j < alpha.Columns; j++)
                    {
                        T update = NumOps.Multiply(learningRate, gradients[idx++]);
                        alpha[i, j] = NumOps.Subtract(alpha[i, j], update);
                    }
                }
            }

            // Update weights
            foreach (var key in _weights.Keys.ToList())
            {
                var weight = _weights[key];
                for (int i = 0; i < weight.Length; i++)
                {
                    T update = NumOps.Multiply(learningRate, gradients[idx++]);
                    weight[i] = NumOps.Subtract(weight[i], update);
                }
            }
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
                        if (NumOps.GreaterThan(softmaxWeights[prevNodeIdx, opIdx], bestWeight))
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
                    if (NumOps.GreaterThan(alpha[row, col], maxVal))
                        maxVal = alpha[row, col];
                }

                // Compute exp(x - max) for numerical stability
                T sumExp = NumOps.Zero;
                var expValues = new T[alpha.Columns];
                for (int col = 0; col < alpha.Columns; col++)
                {
                    expValues[col] = NumOps.Exp(NumOps.Subtract(alpha[row, col], maxVal));
                    sumExp = NumOps.Add(sumExp, expValues[col]);
                }

                // Normalize
                for (int col = 0; col < alpha.Columns; col++)
                {
                    result[row, col] = NumOps.Divide(expValues[col], sumExp);
                }
            }

            return result;
        }

        /// <summary>
        /// Apply a specific operation to input
        /// </summary>
        private Tensor<T> ApplyOperation(Tensor<T> input, int opIdx, string weightKey)
        {
            // Handle 1D input by reshaping to 2D [1, features]
            bool was1D = input.Shape.Length == 1;
            int[] originalShape = input.Shape;
            if (was1D)
            {
                input = input.Reshape([1, input.Shape[0]]);
            }
            else if (input.Shape.Length > 2)
            {
                // For higher-rank tensors, flatten to 2D [batch, features]
                int batchSize = 1;
                for (int i = 0; i < input.Shape.Length - 1; i++)
                    batchSize *= input.Shape[i];
                int features = input.Shape[input.Shape.Length - 1];
                input = input.Reshape([batchSize, features]);
            }

            // Initialize weights if needed
            if (!_weights.ContainsKey(weightKey))
            {
                _weights[weightKey] = new Vector<T>(input.Length);
                _weightGradients[weightKey] = new Vector<T>(input.Length);

                // Initialize with small random values
                for (int i = 0; i < input.Length; i++)
                {
                    _weights[weightKey][i] = NumOps.FromDouble((_random.NextDouble() - 0.5) * 0.1);
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
                                output[batchIdx, featureIdx] = NumOps.Multiply(
                                    input[batchIdx, featureIdx],
                                    NumOps.Add(NumOps.One, weight[featureIdx]));
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
                                output[batchIdx, featureIdx] = NumOps.Multiply(
                                    input[batchIdx, featureIdx],
                                    NumOps.Add(NumOps.One, NumOps.Multiply(NumOps.FromDouble(1.5), weight[featureIdx])));
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
                            output[batchIdx, featureIdx] = NumOps.Multiply(input[batchIdx, featureIdx], NumOps.FromDouble(0.9));
                        }
                    }
                    break;

                case 4: // AvgPool (simplified)
                    for (int batchIdx = 0; batchIdx < input.Shape[0]; batchIdx++)
                    {
                        for (int featureIdx = 0; featureIdx < input.Shape[1]; featureIdx++)
                        {
                            output[batchIdx, featureIdx] = NumOps.Multiply(input[batchIdx, featureIdx], NumOps.FromDouble(0.8));
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

            // Restore original shape if input was 1D or higher-rank
            if (was1D)
            {
                output = output.Reshape(originalShape);
            }
            else if (originalShape.Length > 2)
            {
                output = output.Reshape(originalShape);
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

        public ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
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
                        alpha[i, j] = NumOps.FromDouble(reader.ReadDouble());
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
                    weight[i] = NumOps.FromDouble(reader.ReadDouble());
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
                        alpha[i, j] = NumOps.FromDouble(reader.ReadDouble());
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
                    weight[i] = NumOps.FromDouble(reader.ReadDouble());
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
                T sum = NumOps.Zero;

                // Aggregate importance across all nodes and connections
                foreach (var alpha in _architectureParams)
                {
                    // Sum absolute values of architecture parameters for this operation
                    for (int i = 0; i < alpha.Rows; i++)
                    {
                        if (opIdx < alpha.Columns)
                        {
                            sum = NumOps.Add(sum, NumOps.Abs(alpha[i, opIdx]));
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
                T sum = NumOps.Zero;

                // Apply softmax and aggregate weights for each operation
                foreach (var alpha in _architectureParams)
                {
                    var softmaxWeights = ApplySoftmax(alpha);

                    for (int i = 0; i < softmaxWeights.Rows; i++)
                    {
                        if (opIdx < softmaxWeights.Columns)
                        {
                            sum = NumOps.Add(sum, softmaxWeights[i, opIdx]);
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
                            if (NumOps.GreaterThan(softmax[i, j], bestWeight))
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
            T sum1 = NumOps.Zero;
            T sum2 = NumOps.Zero;
            T sumProduct = NumOps.Zero;
            T sumSquares1 = NumOps.Zero;
            T sumSquares2 = NumOps.Zero;
            int count = 0;

            foreach (var alpha in _architectureParams)
            {
                for (int i = 0; i < alpha.Rows; i++)
                {
                    if (feature1Index < alpha.Columns && feature2Index < alpha.Columns)
                    {
                        T val1 = alpha[i, feature1Index];
                        T val2 = alpha[i, feature2Index];

                        sum1 = NumOps.Add(sum1, val1);
                        sum2 = NumOps.Add(sum2, val2);
                        sumProduct = NumOps.Add(sumProduct, NumOps.Multiply(val1, val2));
                        sumSquares1 = NumOps.Add(sumSquares1, NumOps.Multiply(val1, val1));
                        sumSquares2 = NumOps.Add(sumSquares2, NumOps.Multiply(val2, val2));
                        count++;
                    }
                }
            }

            if (count == 0)
            {
                return NumOps.Zero;
            }

            // Calculate correlation coefficient
            T n = NumOps.FromDouble(count);
            T numerator = NumOps.Subtract(
                NumOps.Multiply(n, sumProduct),
                NumOps.Multiply(sum1, sum2)
            );

            T denom1 = NumOps.Subtract(
                NumOps.Multiply(n, sumSquares1),
                NumOps.Multiply(sum1, sum1)
            );

            T denom2 = NumOps.Subtract(
                NumOps.Multiply(n, sumSquares2),
                NumOps.Multiply(sum2, sum2)
            );

            T denominator = NumOps.Multiply(denom1, denom2);

            // Avoid division by zero
            if (NumOps.Equals(denominator, NumOps.Zero))
            {
                return NumOps.Zero;
            }

            T correlation = NumOps.Divide(numerator, NumOps.Sqrt(denominator));

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
        public virtual void SetBaseModel(IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> model)
        {
            Guard.NotNull(model);
            _baseModel = model;
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
            Guard.NotNull(sensitiveFeatures);
            _sensitiveFeatures = sensitiveFeatures;
            _fairnessMetrics.Clear();
            if (fairnessMetrics != null)
            {
                _fairnessMetrics.AddRange(fairnessMetrics);
            }
        }

        #endregion

        /// <summary>
        /// Saves the SuperNet's current state (architecture parameters and weights) to a stream.
        /// </summary>
        /// <param name="stream">The stream to write the model state to.</param>
        /// <remarks>
        /// <para>
        /// This method serializes all the information needed to recreate the SuperNet's current state,
        /// including architecture parameters, operation weights, and model configuration.
        /// It uses the existing Serialize method and writes the data to the provided stream.
        /// </para>
        /// <para><b>For Beginners:</b> This is like creating a snapshot of your neural architecture search model.
        ///
        /// When you call SaveState:
        /// - All architecture parameters (alpha values) are written to the stream
        /// - All operation weights are saved
        /// - The model's configuration and structure are preserved
        ///
        /// This is particularly useful for:
        /// - Checkpointing during neural architecture search
        /// - Saving the best architecture found during search
        /// - Knowledge distillation from SuperNet to final architecture
        /// - Resuming interrupted architecture search
        ///
        /// You can later use LoadState to restore the model to this exact state.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
        /// <exception cref="IOException">Thrown when there's an error writing to the stream.</exception>
        public virtual void SaveState(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            if (!stream.CanWrite)
                throw new ArgumentException("Stream must be writable.", nameof(stream));

            try
            {
                var data = this.Serialize();
                stream.Write(data, 0, data.Length);
                stream.Flush();
            }
            catch (IOException ex)
            {
                throw new IOException($"Failed to save SuperNet state to stream: {ex.Message}", ex);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Unexpected error while saving SuperNet state: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Loads the SuperNet's state (architecture parameters and weights) from a stream.
        /// </summary>
        /// <param name="stream">The stream to read the model state from.</param>
        /// <remarks>
        /// <para>
        /// This method deserializes SuperNet state that was previously saved with SaveState,
        /// restoring all architecture parameters, operation weights, and configuration.
        /// It uses the existing Deserialize method after reading data from the stream.
        /// </para>
        /// <para><b>For Beginners:</b> This is like loading a saved snapshot of your neural architecture search model.
        ///
        /// When you call LoadState:
        /// - All architecture parameters (alpha values) are read from the stream
        /// - All operation weights are restored
        /// - The model is configured to match the saved state
        ///
        /// After loading, the model can:
        /// - Continue architecture search from where it left off
        /// - Make predictions using the restored architecture
        /// - Be used for further optimization or deployment
        ///
        /// This is essential for:
        /// - Resuming interrupted architecture search
        /// - Loading the best architecture found during search
        /// - Deploying searched architectures to production
        /// - Knowledge distillation workflows
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
        /// <exception cref="IOException">Thrown when there's an error reading from the stream.</exception>
        /// <exception cref="InvalidOperationException">Thrown when the stream contains invalid or incompatible data.</exception>
        public virtual void LoadState(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            if (!stream.CanRead)
                throw new ArgumentException("Stream must be readable.", nameof(stream));

            try
            {
                using var ms = new MemoryStream();
                stream.CopyTo(ms);
                var data = ms.ToArray();

                if (data.Length == 0)
                    throw new InvalidOperationException("Stream contains no data.");

                this.Deserialize(data);
            }
            catch (IOException ex)
            {
                throw new IOException($"Failed to read SuperNet state from stream: {ex.Message}", ex);
            }
            catch (InvalidOperationException)
            {
                // Re-throw InvalidOperationException from Deserialize
                throw;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Failed to deserialize SuperNet state. The stream may contain corrupted or incompatible data: {ex.Message}", ex);
            }
        }

        #region IJitCompilable Implementation

        /// <summary>
        /// Gets whether this SuperNet supports JIT compilation.
        /// </summary>
        /// <value>
        /// <c>true</c> after at least one forward pass has been performed to initialize weights.
        /// </value>
        /// <remarks>
        /// <para>
        /// SuperNet implements Differentiable Architecture Search (DARTS), which is specifically
        /// designed to be differentiable. The softmax-weighted operation mixing that defines DARTS
        /// is a fully differentiable computation that can be exported as a computation graph.
        /// </para>
        /// <para><b>Key Insight:</b> While the architecture parameters (alpha) are learned during
        /// training, at inference time they are fixed values. The computation graph includes:
        /// </para>
        /// <list type="bullet">
        /// <item><description>Softmax over architecture parameters for each node</description></item>
        /// <item><description>All operation outputs computed in parallel</description></item>
        /// <item><description>Weighted sum of operation outputs using softmax weights</description></item>
        /// </list>
        /// <para>
        /// This is exactly what makes DARTS "differentiable" - the entire forward pass can be
        /// expressed as continuous, differentiable operations that are JIT-compilable.
        /// </para>
        /// <para><b>For Beginners:</b> DARTS uses a clever trick called "continuous relaxation":
        ///
        /// Instead of choosing ONE operation at each step (which would be discrete and non-differentiable),
        /// DARTS computes ALL operations and combines them with softmax weights. This weighted
        /// combination IS differentiable and CAN be JIT compiled.
        ///
        /// The JIT-compiled SuperNet will:
        /// - Use the current architecture parameters (alpha values)
        /// - Compute softmax weights over operations
        /// - Evaluate all operations
        /// - Combine outputs using the computed weights
        ///
        /// After architecture search is complete, you can also call DeriveArchitecture() to create
        /// a simpler, discrete architecture that uses only the best operations.
        /// </para>
        /// </remarks>
        public bool SupportsJitCompilation => _weights.Count > 0;

        /// <summary>
        /// Exports the model's computation graph for JIT compilation.
        /// </summary>
        /// <param name="inputNodes">List to populate with input computation nodes (parameters).</param>
        /// <returns>The output computation node representing the SuperNet forward pass.</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown if called before any forward pass has initialized the weights.
        /// </exception>
        /// <remarks>
        /// <para>
        /// Exports the DARTS continuous relaxation as a computation graph. The graph includes:
        /// </para>
        /// <list type="bullet">
        /// <item><description>Input tensor variable</description></item>
        /// <item><description>Architecture parameters embedded as constants</description></item>
        /// <item><description>Softmax computation over architecture parameters</description></item>
        /// <item><description>All operation outputs</description></item>
        /// <item><description>Weighted sum using softmax weights</description></item>
        /// </list>
        /// <para><b>For Beginners:</b> This exports the current state of the SuperNet as a
        /// JIT-compilable graph. The architecture parameters (alpha values) are baked into
        /// the graph as constants, so the exported graph represents the current "snapshot"
        /// of the architecture search.
        ///
        /// You can export at different points during training to capture the evolving architecture,
        /// or export after search completes to get the final continuous relaxation.
        /// </para>
        /// </remarks>
        public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
        {
            if (_weights.Count == 0)
            {
                throw new InvalidOperationException(
                    "SuperNet must be initialized with at least one forward pass before exporting computation graph. " +
                    "Call Predict() first to initialize the network weights.");
            }

            // Create input node for the data tensor
            // We assume 2D input: [batch, features]
            var inputShape = new[] { 1, _inputSize > 0 ? _inputSize : 1 };
            var inputTensor = new Tensor<T>(inputShape);
            var input = TensorOperations<T>.Variable(inputTensor, "input");
            inputNodes.Add(input);

            // Build the computation graph for DARTS forward pass
            // Store intermediate node outputs as computation nodes
            var nodeOutputs = new List<ComputationNode<T>> { input };

            // Process each node in the architecture
            for (int nodeIdx = 0; nodeIdx < _numNodes; nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];

                // Compute softmax weights for this node's architecture parameters
                // Create a constant tensor from the softmax weights
                var softmaxWeights = ApplySoftmax(alpha);

                // Initialize accumulated output for this node
                ComputationNode<T>? nodeOutput = null;

                // Mix operations from all previous nodes
                for (int prevNodeIdx = 0; prevNodeIdx <= nodeIdx; prevNodeIdx++)
                {
                    var prevOutput = nodeOutputs[prevNodeIdx];

                    // Apply each operation and mix with softmax weights
                    for (int opIdx = 0; opIdx < _numOperations; opIdx++)
                    {
                        var weightKey = $"node{nodeIdx}_from{prevNodeIdx}_op{opIdx}";
                        var weight = softmaxWeights[prevNodeIdx, opIdx];

                        // Create computation for this operation's output
                        var opOutput = ExportOperationGraph(prevOutput, opIdx, weightKey);

                        // Scale by softmax weight (create constant for the weight)
                        var weightTensor = new Tensor<T>(new[] { 1 });
                        weightTensor[0] = weight;
                        var weightNode = TensorOperations<T>.Constant(weightTensor, $"weight_{nodeIdx}_{prevNodeIdx}_{opIdx}");

                        var scaledOutput = TensorOperations<T>.ElementwiseMultiply(opOutput, weightNode);

                        // Accumulate
                        if (nodeOutput == null)
                        {
                            nodeOutput = scaledOutput;
                        }
                        else
                        {
                            nodeOutput = TensorOperations<T>.Add(nodeOutput, scaledOutput);
                        }
                    }
                }

                nodeOutputs.Add(nodeOutput ?? input);
            }

            // Return the output of the final node
            return nodeOutputs[nodeOutputs.Count - 1];
        }

        /// <summary>
        /// Exports a single operation as a computation graph.
        /// </summary>
        private ComputationNode<T> ExportOperationGraph(ComputationNode<T> input, int opIdx, string weightKey)
        {
            // Get or create weight constants
            Vector<T>? weight = null;
            if (_weights.TryGetValue(weightKey, out var w))
            {
                weight = w;
            }

            switch (opIdx)
            {
                case 0: // Identity
                    return input;

                case 1: // 3x3 Conv (simplified as weighted pass)
                    if (weight != null)
                    {
                        var weightTensor = new Tensor<T>(new[] { weight.Length });
                        for (int i = 0; i < weight.Length; i++)
                        {
                            weightTensor[i] = NumOps.Add(NumOps.One, weight[i]);
                        }
                        var weightNode = TensorOperations<T>.Constant(weightTensor, $"weights_{weightKey}");
                        return TensorOperations<T>.ElementwiseMultiply(input, weightNode);
                    }
                    return input;

                case 2: // 5x5 Conv (simplified)
                    if (weight != null)
                    {
                        var weightTensor = new Tensor<T>(new[] { weight.Length });
                        for (int i = 0; i < weight.Length; i++)
                        {
                            weightTensor[i] = NumOps.Add(NumOps.One, NumOps.Multiply(NumOps.FromDouble(1.5), weight[i]));
                        }
                        var weightNode = TensorOperations<T>.Constant(weightTensor, $"weights_{weightKey}");
                        return TensorOperations<T>.ElementwiseMultiply(input, weightNode);
                    }
                    return input;

                case 3: // MaxPool (simplified as scaling)
                {
                    var scaleTensor = new Tensor<T>(new[] { 1 });
                    scaleTensor[0] = NumOps.FromDouble(0.9);
                    var scaleNode = TensorOperations<T>.Constant(scaleTensor, $"maxpool_scale_{weightKey}");
                    return TensorOperations<T>.ElementwiseMultiply(input, scaleNode);
                }

                case 4: // AvgPool (simplified as scaling)
                {
                    var scaleTensor = new Tensor<T>(new[] { 1 });
                    scaleTensor[0] = NumOps.FromDouble(0.8);
                    var scaleNode = TensorOperations<T>.Constant(scaleTensor, $"avgpool_scale_{weightKey}");
                    return TensorOperations<T>.ElementwiseMultiply(input, scaleNode);
                }

                default:
                    return input;
            }
        }

        /// <summary>
        /// Performs forward pass through the model (required by IJitCompilable).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public Tensor<T> Forward(Tensor<T> input)
        {
            return Predict(input);
        }

        #endregion
    }
}
