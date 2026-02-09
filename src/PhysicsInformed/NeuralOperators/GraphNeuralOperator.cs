using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PhysicsInformed.NeuralOperators
{
    /// <summary>
    /// Implements Graph Neural Operators for learning operators on graph-structured data.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Graph Neural Operators extend neural operators to irregular, graph-structured domains.
    ///
    /// Why Graphs?
    /// Many physical systems are naturally represented as graphs:
    /// - Molecular structures (atoms = nodes, bonds = edges)
    /// - Mesh-based simulations (mesh points = nodes, connectivity = edges)
    /// - Traffic networks (intersections = nodes, roads = edges)
    /// - Social networks, power grids, etc.
    ///
    /// Regular operators (FNO, DeepONet) work on:
    /// - Structured grids (images, regular spatial domains)
    /// - Euclidean spaces
    ///
    /// Graph operators work on:
    /// - Irregular geometries
    /// - Non-Euclidean spaces
    /// - Variable-size domains
    ///
    /// Key Idea - Message Passing:
    /// Information propagates through the graph via message passing:
    /// 1. Each node has features (e.g., temperature, velocity)
    /// 2. Nodes send messages to neighbors
    /// 3. Nodes aggregate messages and update their features
    /// 4. Repeat for multiple layers
    ///
    /// Applications:
    /// - Molecular dynamics (predict molecular properties)
    /// - Computational fluid dynamics (irregular meshes)
    /// - Material science (crystal structures)
    /// - Climate modeling (irregular Earth grids)
    /// - Particle systems
    /// </remarks>
    public class GraphNeuralOperator<T> : NeuralNetworkBase<T>
    {
        private readonly GraphNeuralOperatorOptions _options;

        /// <inheritdoc/>
        public override ModelOptions GetOptions() => _options;

        private readonly int _numMessagePassingLayers;
        private readonly int _hiddenDim;
        private readonly int _inputDim;
        private readonly bool _normalizeAdjacency;
        private readonly List<GraphConvolutionalLayer<T>> _graphLayers;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;

        public GraphNeuralOperator(
            NeuralNetworkArchitecture<T> architecture,
            int numLayers = 4,
            int hiddenDim = 64,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int inputDim = 0,
            bool normalizeAdjacency = true,
            GraphNeuralOperatorOptions? options = null)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            _options = options ?? new GraphNeuralOperatorOptions();
            Options = _options;

            if (architecture == null)
            {
                throw new ArgumentNullException(nameof(architecture));
            }

            if (numLayers <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be positive.");
            }

            if (hiddenDim <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(hiddenDim), "Hidden dimension must be positive.");
            }

            _numMessagePassingLayers = numLayers;
            _hiddenDim = hiddenDim;
            _inputDim = inputDim > 0 ? inputDim : architecture.InputSize;
            if (_inputDim <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(inputDim), "Input dimension must be positive.");
            }

            _normalizeAdjacency = normalizeAdjacency;
            _graphLayers = new List<GraphConvolutionalLayer<T>>();
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = optimizer == null;

            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            for (int i = 0; i < _numMessagePassingLayers; i++)
            {
                int layerInputDim = i == 0 ? _inputDim : _hiddenDim;
                IActivationFunction<T>? activation = i < _numMessagePassingLayers - 1
                    ? new ReLUActivation<T>()
                    : null;
                var layer = new GraphConvolutionalLayer<T>(layerInputDim, _hiddenDim, activation);
                layer.UseAutodiff = true;
                _graphLayers.Add(layer);
            }
        }

        /// <summary>
        /// Forward pass through the graph neural operator.
        /// </summary>
        /// <param name="nodeFeatures">Features for each node.</param>
        /// <param name="adjacencyMatrix">Graph adjacency matrix.</param>
        /// <returns>Updated node features.</returns>
        public T[,] Forward(T[,] nodeFeatures, T[,] adjacencyMatrix)
        {
            ValidateGraphInput(nodeFeatures, adjacencyMatrix);
            var featureTensor = ToTensor3D(nodeFeatures);
            var adjacencyTensor = ToTensor2D(adjacencyMatrix);
            var outputTensor = Forward(featureTensor, adjacencyTensor);
            return ToArray2D(outputTensor.GetSlice(0));
        }

        /// <summary>
        /// Forward pass using an identity adjacency matrix.
        /// </summary>
        /// <param name="input">Node feature tensor.</param>
        /// <returns>Updated node features.</returns>
        public Tensor<T> Forward(Tensor<T> input)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (input.Rank != 2 && input.Rank != 3)
            {
                throw new ArgumentException("GraphNeuralOperator expects a 2D [nodes, features] or 3D [batch, nodes, features] tensor.", nameof(input));
            }

            int numNodes = input.Rank == 2 ? input.Shape[0] : input.Shape[1];
            var adjacency = CreateIdentityAdjacency(numNodes);
            return Forward(input, adjacency);
        }

        /// <summary>
        /// Makes a prediction using the graph neural operator.
        /// </summary>
        /// <param name="input">Input tensor containing node features (and optional adjacency).</param>
        /// <returns>Predicted node features.</returns>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            bool wasTraining = IsTrainingMode;
            SetTrainingMode(false);

            try
            {
                if (input.Rank != 2 && input.Rank != 3)
                {
                    throw new ArgumentException("GraphNeuralOperator expects a 2D [nodes, features] or 3D [batch, nodes, features] tensor.");
                }

                int numNodes = input.Rank == 2 ? input.Shape[0] : input.Shape[1];
                var adjacency = CreateIdentityAdjacency(numNodes);
                var predicted = Forward(input, adjacency);
                return input.Rank == 2 ? predicted.GetSlice(0) : predicted;
            }
            finally
            {
                SetTrainingMode(wasTraining);
            }
        }


        /// <summary>
        /// Trains the graph neural operator on a single graph.
        /// </summary>
        /// <param name="nodeFeatures">Node feature matrix.</param>
        /// <param name="adjacencyMatrix">Adjacency matrix.</param>
        /// <param name="targetValues">Target node features.</param>
        /// <param name="epochs">Number of training epochs.</param>
        /// <param name="learningRate">Learning rate.</param>
        /// <param name="verbose">Whether to print progress.</param>
        /// <returns>Training history.</returns>
        public TrainingHistory<T> TrainOnGraph(
            T[,] nodeFeatures,
            T[,] adjacencyMatrix,
            T[,] targetValues,
            int epochs = 200,
            double learningRate = 0.001,
            bool verbose = true)
        {
            ValidateGraphInput(nodeFeatures, adjacencyMatrix);
            if (targetValues == null)
            {
                throw new ArgumentNullException(nameof(targetValues));
            }
            if (targetValues.GetLength(0) != nodeFeatures.GetLength(0) ||
                targetValues.GetLength(1) != _hiddenDim)
            {
                throw new ArgumentException("Target values must match node count and hidden dimension.");
            }

            var history = new TrainingHistory<T>();
            var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();

            if (_usesDefaultOptimizer)
            {
                var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = learningRate
                };
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
            }

            SetTrainingMode(true);
            foreach (var layer in _graphLayers)
            {
                layer.SetTrainingMode(true);
            }

            try
            {
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    var featureTensor = ToTensor3D(nodeFeatures);
                    var adjacencyTensor = ToTensor2D(adjacencyMatrix);
                    var targetTensor = ToTensor3D(targetValues);

                    var prediction = Forward(featureTensor, adjacencyTensor);
                    var loss = lossFunction.CalculateLoss(prediction.ToVector(), targetTensor.ToVector());
                    history.AddEpoch(loss);

                    var outputGradientVector = lossFunction.CalculateDerivative(prediction.ToVector(), targetTensor.ToVector());
                    var outputGradient = new Tensor<T>(prediction.Shape, outputGradientVector);
                    Backpropagate(outputGradient);

                    var gradients = GetGradients();
                    var parameters = GetParameters();
                    if (parameters.Length > 0)
                    {
                        var updatedParameters = _optimizer.UpdateParameters(parameters, gradients);
                        UpdateParameters(updatedParameters);
                    }

                    ClearGradients();

                    if (verbose && epoch % 10 == 0)
                    {
                        Console.WriteLine($"Epoch {epoch}/{epochs}, Loss: {loss}");
                    }
                }
            }
            finally
            {
                foreach (var layer in _graphLayers)
                {
                    layer.SetTrainingMode(false);
                }
                SetTrainingMode(false);
            }

            return history;
        }

        /// <summary>
        /// Updates the operator parameters from a flattened vector.
        /// </summary>
        /// <param name="parameters">Parameter vector.</param>
        public override void UpdateParameters(Vector<T> parameters)
        {
            int index = 0;
            foreach (var layer in _graphLayers)
            {
                int layerParameterCount = layer.ParameterCount;
                if (layerParameterCount > 0)
                {
                    Vector<T> layerParameters = parameters.GetSubVector(index, layerParameterCount);
                    layer.SetParameters(layerParameters);
                    index += layerParameterCount;
                }
            }
        }

        public override Tensor<T> Backpropagate(Tensor<T> outputGradients)
        {
            if (!IsTrainingMode)
            {
                throw new InvalidOperationException("Cannot backpropagate when network is not in training mode");
            }

            if (!SupportsTraining)
            {
                throw new InvalidOperationException("This network does not support backpropagation");
            }

            var gradientTensor = outputGradients;
            for (int i = _graphLayers.Count - 1; i >= 0; i--)
            {
                gradientTensor = _graphLayers[i].Backward(gradientTensor);
            }

            return gradientTensor;
        }

        public override Vector<T> GetGradients()
        {
            var gradients = new List<T>();
            foreach (var layer in _graphLayers)
            {
                var layerGradients = layer.GetParameterGradients();
                if (layerGradients.Length > 0)
                {
                    gradients.AddRange(layerGradients.ToArray());
                }
            }

            return new Vector<T>(gradients.ToArray());
        }

        private void ClearGradients()
        {
            foreach (var layer in _graphLayers)
            {
                layer.ClearGradients();
            }
        }

        /// <summary>
        /// Performs a basic supervised training step using MSE loss.
        /// </summary>
        /// <param name="input">Training input tensor.</param>
        /// <param name="expectedOutput">Expected output tensor.</param>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (expectedOutput == null)
            {
                throw new ArgumentNullException(nameof(expectedOutput));
            }

            if (input.Rank != 2 && input.Rank != 3)
            {
                throw new ArgumentException("GraphNeuralOperator expects a 2D [nodes, features] or 3D [batch, nodes, features] tensor.", nameof(input));
            }

            if (expectedOutput.Rank != input.Rank)
            {
                throw new ArgumentException("Expected output rank must match input rank.", nameof(expectedOutput));
            }

            int numNodes = input.Rank == 2 ? input.Shape[0] : input.Shape[1];
            int outputDim = _hiddenDim;

            if (expectedOutput.Rank == 2)
            {
                if (expectedOutput.Shape[0] != numNodes || expectedOutput.Shape[1] != outputDim)
                {
                    throw new ArgumentException($"Expected output shape [nodes, {outputDim}] to match the input node count.", nameof(expectedOutput));
                }
            }
            else
            {
                int batchSize = input.Shape[0];
                if (expectedOutput.Shape[0] != batchSize || expectedOutput.Shape[1] != numNodes || expectedOutput.Shape[2] != outputDim)
                {
                    throw new ArgumentException($"Expected output shape [batch, nodes, {outputDim}] to match the input batch and node counts.", nameof(expectedOutput));
                }
            }

            SetTrainingMode(true);
            foreach (var layer in _graphLayers)
            {
                layer.SetTrainingMode(true);
            }

            try
            {
                var adjacency = CreateIdentityAdjacency(numNodes);
                var prediction = Forward(input, adjacency);
                var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();
                LastLoss = lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

                var outputGradientVector = lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
                var outputGradient = new Tensor<T>(prediction.Shape, outputGradientVector);

                Backpropagate(outputGradient);

                var gradients = GetGradients();
                var parameters = GetParameters();
                if (parameters.Length > 0)
                {
                    var updatedParameters = _optimizer.UpdateParameters(parameters, gradients);
                    UpdateParameters(updatedParameters);
                }

                ClearGradients();
            }
            finally
            {
                foreach (var layer in _graphLayers)
                {
                    layer.SetTrainingMode(false);
                }

                SetTrainingMode(false);
            }
        }

        /// <summary>
        /// Gets metadata about the graph neural operator.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "MessagePassingLayers", _numMessagePassingLayers },
                    { "HiddenDimension", _hiddenDim },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelData = Serialize()
            };
        }

        /// <summary>
        /// Serializes graph operator-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_numMessagePassingLayers);
            writer.Write(_hiddenDim);
            writer.Write(_inputDim);
            writer.Write(_normalizeAdjacency);
            writer.Write(_graphLayers.Count);

            foreach (var layer in _graphLayers)
            {
                SerializationHelper<T>.SerializeVector(writer, layer.GetParameters());
            }
        }

        /// <summary>
        /// Deserializes graph operator-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            int storedLayers = reader.ReadInt32();
            int storedHidden = reader.ReadInt32();
            int storedInputDim = reader.ReadInt32();
            bool storedNormalize = reader.ReadBoolean();
            int storedLayerCount = reader.ReadInt32();

            if (storedLayers != _numMessagePassingLayers ||
                storedHidden != _hiddenDim ||
                storedInputDim != _inputDim ||
                storedNormalize != _normalizeAdjacency ||
                storedLayerCount != _graphLayers.Count)
            {
                throw new InvalidOperationException("Serialized graph operator configuration does not match the current instance.");
            }

            foreach (var layer in _graphLayers)
            {
                layer.SetParameters(SerializationHelper<T>.DeserializeVector(reader));
            }
        }

        /// <summary>
        /// Creates a new instance with the same configuration.
        /// </summary>
        /// <returns>New graph operator instance.</returns>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new GraphNeuralOperator<T>(Architecture, _numMessagePassingLayers, _hiddenDim, null, _inputDim, _normalizeAdjacency);
        }

        /// <summary>
        /// Gets the total number of parameters across graph layers.
        /// </summary>
        public override int ParameterCount => _graphLayers.Sum(layer => layer.ParameterCount);

        /// <summary>
        /// Gets the operator parameters as a flattened vector.
        /// </summary>
        public override Vector<T> GetParameters()
        {
            var parameters = new Vector<T>(ParameterCount);
            int index = 0;

            foreach (var layer in _graphLayers)
            {
                var layerParameters = layer.GetParameters();
                for (int i = 0; i < layerParameters.Length; i++)
                {
                    parameters[index + i] = layerParameters[i];
                }

                index += layerParameters.Length;
            }

            return parameters;
        }

        public Tensor<T> Forward(Tensor<T> nodeFeatures, Tensor<T> adjacencyMatrix)
        {
            ValidateGraphTensorInputs(nodeFeatures, adjacencyMatrix);

            var features = nodeFeatures.Rank == 2 ? AddBatchDimension(nodeFeatures) : nodeFeatures;
            var preparedAdjacency = PrepareAdjacency(adjacencyMatrix, features.Shape[0]);

            foreach (var layer in _graphLayers)
            {
                layer.SetAdjacencyMatrix(preparedAdjacency);
                features = layer.Forward(features);
            }

            return features;
        }

        private Tensor<T> CreateIdentityAdjacency(int numNodes)
        {
            var adjacency = new Tensor<T>(new int[] { numNodes, numNodes });
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    adjacency[i, j] = i == j ? NumOps.One : NumOps.Zero;
                }
            }

            return adjacency;
        }

        private static T[,] ToArray2D(Tensor<T> tensor)
        {
            if (tensor.Rank != 2)
            {
                throw new ArgumentException("Expected a 2D tensor.");
            }

            var result = new T[tensor.Shape[0], tensor.Shape[1]];
            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    result[i, j] = tensor[i, j];
                }
            }

            return result;
        }

        private static Tensor<T> ToTensor2D(T[,] data)
        {
            var tensor = new Tensor<T>(new int[] { data.GetLength(0), data.GetLength(1) });
            for (int i = 0; i < data.GetLength(0); i++)
            {
                for (int j = 0; j < data.GetLength(1); j++)
                {
                    tensor[i, j] = data[i, j];
                }
            }

            return tensor;
        }

        private static Tensor<T> ToTensor3D(T[,] data)
        {
            var tensor = new Tensor<T>(new int[] { 1, data.GetLength(0), data.GetLength(1) });
            for (int i = 0; i < data.GetLength(0); i++)
            {
                for (int j = 0; j < data.GetLength(1); j++)
                {
                    tensor[0, i, j] = data[i, j];
                }
            }

            return tensor;
        }

        private Tensor<T> AddBatchDimension(Tensor<T> tensor)
        {
            if (tensor.Rank != 2)
            {
                throw new ArgumentException("Expected a 2D tensor.");
            }

            var output = new Tensor<T>(new int[] { 1, tensor.Shape[0], tensor.Shape[1] });
            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    output[0, i, j] = tensor[i, j];
                }
            }

            return output;
        }

        private T ComputeMSE(T[,] predictions, T[,] targets)
        {
            int rows = predictions.GetLength(0);
            int cols = predictions.GetLength(1);
            if (rows != targets.GetLength(0) || cols != targets.GetLength(1))
            {
                throw new ArgumentException("Prediction and target shapes must match.");
            }

            T sumSquaredError = NumOps.Zero;
            int count = 0;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    T error = NumOps.Subtract(predictions[i, j], targets[i, j]);
                    sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Multiply(error, error));
                    count++;
                }
            }

            return count > 0
                ? NumOps.Divide(sumSquaredError, NumOps.FromDouble(count))
                : NumOps.Zero;
        }

        public override bool SupportsTraining => true;
        public override bool SupportsJitCompilation => false;

        private void ValidateGraphInput(T[,] nodeFeatures, T[,] adjacencyMatrix)
        {
            if (nodeFeatures == null)
            {
                throw new ArgumentNullException(nameof(nodeFeatures));
            }

            if (adjacencyMatrix == null)
            {
                throw new ArgumentNullException(nameof(adjacencyMatrix));
            }

            int numNodes = nodeFeatures.GetLength(0);
            int featureDim = nodeFeatures.GetLength(1);
            if (featureDim != _inputDim)
            {
                throw new ArgumentException($"Input feature dimension ({featureDim}) must match configured input dimension ({_inputDim}).", nameof(nodeFeatures));
            }

            int adjRows = adjacencyMatrix.GetLength(0);
            int adjCols = adjacencyMatrix.GetLength(1);
            if (adjRows != adjCols)
            {
                throw new ArgumentException("Adjacency matrix must be square.", nameof(adjacencyMatrix));
            }

            if (adjRows != numNodes)
            {
                throw new ArgumentException($"Adjacency matrix size ({adjRows}) must match number of nodes ({numNodes}).", nameof(adjacencyMatrix));
            }
        }

        private void ValidateGraphTensorInputs(Tensor<T> nodeFeatures, Tensor<T> adjacencyMatrix)
        {
            if (nodeFeatures == null)
            {
                throw new ArgumentNullException(nameof(nodeFeatures));
            }

            if (adjacencyMatrix == null)
            {
                throw new ArgumentNullException(nameof(adjacencyMatrix));
            }

            if (nodeFeatures.Rank != 2 && nodeFeatures.Rank != 3)
            {
                throw new ArgumentException("Node feature tensor must have rank 2 or 3.", nameof(nodeFeatures));
            }

            int numNodes = nodeFeatures.Rank == 2 ? nodeFeatures.Shape[0] : nodeFeatures.Shape[1];
            int featureDim = nodeFeatures.Rank == 2 ? nodeFeatures.Shape[1] : nodeFeatures.Shape[2];
            if (featureDim != _inputDim)
            {
                throw new ArgumentException($"Input feature dimension ({featureDim}) must match configured input dimension ({_inputDim}).", nameof(nodeFeatures));
            }

            if (adjacencyMatrix.Rank != 2 && adjacencyMatrix.Rank != 3)
            {
                throw new ArgumentException("Adjacency tensor must have rank 2 or 3.", nameof(adjacencyMatrix));
            }

            int adjRows = adjacencyMatrix.Rank == 2 ? adjacencyMatrix.Shape[0] : adjacencyMatrix.Shape[1];
            int adjCols = adjacencyMatrix.Rank == 2 ? adjacencyMatrix.Shape[1] : adjacencyMatrix.Shape[2];
            if (adjRows != adjCols)
            {
                throw new ArgumentException("Adjacency matrix must be square.", nameof(adjacencyMatrix));
            }

            if (adjRows != numNodes)
            {
                throw new ArgumentException($"Adjacency matrix size ({adjRows}) must match number of nodes ({numNodes}).", nameof(adjacencyMatrix));
            }

            if (adjacencyMatrix.Rank == 3)
            {
                int featureBatch = nodeFeatures.Rank == 3 ? nodeFeatures.Shape[0] : 1;
                int adjacencyBatch = adjacencyMatrix.Shape[0];
                if (adjacencyBatch != 1 && adjacencyBatch != featureBatch)
                {
                    throw new ArgumentException("Adjacency batch size must be 1 or match the node feature batch size.", nameof(adjacencyMatrix));
                }
            }
        }

        private Tensor<T> PrepareAdjacency(Tensor<T> adjacencyMatrix, int batchSize)
        {
            int numNodes = adjacencyMatrix.Rank == 2 ? adjacencyMatrix.Shape[0] : adjacencyMatrix.Shape[1];

            if (adjacencyMatrix.Rank == 2)
            {
                var baseAdjacency = _normalizeAdjacency ? NormalizeAdjacency(adjacencyMatrix) : adjacencyMatrix;
                var batchedAdjacency = baseAdjacency.Reshape([1, numNodes, numNodes]);
                if (batchSize <= 1)
                {
                    return batchedAdjacency;
                }

                return Engine.TensorRepeatElements(batchedAdjacency, batchSize, axis: 0);
            }

            int adjBatch = adjacencyMatrix.Shape[0];
            if (adjBatch == 1 && batchSize > 1)
            {
                var baseAdjacency = _normalizeAdjacency ? NormalizeAdjacency(adjacencyMatrix) : adjacencyMatrix;
                return Engine.TensorRepeatElements(baseAdjacency, batchSize, axis: 0);
            }

            if (adjBatch != batchSize)
            {
                throw new ArgumentException("Adjacency batch size must match node feature batch size.");
            }

            return _normalizeAdjacency ? NormalizeAdjacency(adjacencyMatrix) : adjacencyMatrix;
        }

        private Tensor<T> NormalizeAdjacency(Tensor<T> adjacency)
        {
            return adjacency.Rank == 2
                ? NormalizeAdjacencySingle(adjacency)
                : NormalizeAdjacencyBatch(adjacency);
        }

        private Tensor<T> NormalizeAdjacencySingle(Tensor<T> adjacency)
        {
            int numNodes = adjacency.Shape[0];
            var normalized = new Tensor<T>(adjacency.Shape);
            var degrees = new T[numNodes];
            var invSqrt = new T[numNodes];
            T epsilon = NumOps.FromDouble(1e-10);

            for (int i = 0; i < numNodes; i++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < numNodes; j++)
                {
                    sum = NumOps.Add(sum, adjacency[i, j]);
                }
                degrees[i] = NumOps.Add(sum, epsilon);
                invSqrt[i] = NumOps.Divide(NumOps.One, NumOps.Sqrt(degrees[i]));
            }

            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    normalized[i, j] = NumOps.Multiply(adjacency[i, j], NumOps.Multiply(invSqrt[i], invSqrt[j]));
                }
            }

            return normalized;
        }

        private Tensor<T> NormalizeAdjacencyBatch(Tensor<T> adjacency)
        {
            int batch = adjacency.Shape[0];
            int numNodes = adjacency.Shape[1];
            var normalized = new Tensor<T>(adjacency.Shape);
            var degrees = new T[numNodes];
            var invSqrt = new T[numNodes];
            T epsilon = NumOps.FromDouble(1e-10);

            for (int b = 0; b < batch; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    T sum = NumOps.Zero;
                    for (int j = 0; j < numNodes; j++)
                    {
                        sum = NumOps.Add(sum, adjacency[b, i, j]);
                    }
                    degrees[i] = NumOps.Add(sum, epsilon);
                    invSqrt[i] = NumOps.Divide(NumOps.One, NumOps.Sqrt(degrees[i]));
                }

                for (int i = 0; i < numNodes; i++)
                {
                    for (int j = 0; j < numNodes; j++)
                    {
                        normalized[b, i, j] = NumOps.Multiply(adjacency[b, i, j], NumOps.Multiply(invSqrt[i], invSqrt[j]));
                    }
                }
            }

            return normalized;
        }

    }


}
