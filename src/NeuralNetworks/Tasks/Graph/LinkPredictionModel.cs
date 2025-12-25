using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tasks.Graph;

/// <summary>
/// Implements a complete neural network model for link prediction tasks on graphs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Link prediction predicts whether edges should exist between node pairs using:
/// - Node features
/// - Graph structure
/// - Learned node embeddings
/// </para>
/// <para><b>For Beginners:</b> This model predicts connections between nodes.
///
/// **How it works:**
///
/// 1. **Encode**: Learn embeddings for all nodes using GNN layers
///    ```
///    Input: Node features + Graph structure
///    Process: Stack of graph conv layers
///    Output: Node embeddings [num_nodes, embedding_dim]
///    ```
///
/// 2. **Decode**: Score node pairs to predict edges
///    ```
///    Input: Node pair (i, j)
///    Compute: score = f(embedding[i], embedding[j])
///    Common functions:
///    - Dot product: z_i * z_j
///    - Concatenation + MLP: MLP([z_i || z_j])
///    - Distance-based: -||z_i - z_j||^2
///    ```
///
/// 3. **Train**: Learn to score existing edges high, non-existing edges low
///
/// **Example:**
/// ```
/// Friend recommendation:
/// - Encode users as embeddings using friend network
/// - For user pair (Alice, Bob):
///   * Compute score from their embeddings
///   * High score -> Likely to be friends
///   * Low score -> Unlikely to be friends
/// ```
/// </para>
/// </remarks>
public class LinkPredictionModel<T> : NeuralNetworkBase<T>
{
    private static new readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ILossFunction<T> _lossFunction;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly LinkPredictionDecoder _decoderType;
    private Tensor<T>? _cachedAdjacencyMatrix;
    private Tensor<T>? _nodeEmbeddings;

    /// <summary>
    /// Decoder types for combining node embeddings into edge scores.
    /// </summary>
    public enum LinkPredictionDecoder
    {
        /// <summary>Dot product: score = z_i * z_j</summary>
        DotProduct,

        /// <summary>Cosine similarity: score = (z_i * z_j) / (||z_i|| ||z_j||)</summary>
        CosineSimilarity,

        /// <summary>Element-wise product: score = sum(z_i * z_j)</summary>
        Hadamard,

        /// <summary>L2 distance: score = -||z_i - z_j||^2</summary>
        Distance
    }

    /// <summary>
    /// Gets the number of input features per node.
    /// </summary>
    public int InputFeatures { get; }

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDim { get; }

    /// <summary>
    /// Gets the hidden dimension size.
    /// </summary>
    public int HiddenDim { get; }

    /// <summary>
    /// Gets the number of GNN layers.
    /// </summary>
    public int NumLayers { get; }

    /// <summary>
    /// Gets the dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="LinkPredictionModel{T}"/> class.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output sizes and layers.</param>
    /// <param name="hiddenDim">Hidden dimension for intermediate layers (default: 64).</param>
    /// <param name="embeddingDim">Dimension of node embeddings (default: 32).</param>
    /// <param name="numLayers">Number of graph convolutional layers (default: 2).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.5).</param>
    /// <param name="decoderType">Method for combining node embeddings into edge scores.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creating a link prediction model:
    ///
    /// ```csharp
    /// // Create architecture for friend prediction
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     InputType.OneDimensional,
    ///     NeuralNetworkTaskType.BinaryClassification,
    ///     NetworkComplexity.Simple,
    ///     inputSize: 128,    // User features
    ///     outputSize: 1);    // Edge score
    ///
    /// // Create model with default layers
    /// var model = new LinkPredictionModel&lt;double&gt;(architecture);
    ///
    /// // Train on link prediction task
    /// var history = model.TrainOnTask(task, epochs: 100, learningRate: 0.01);
    /// ```
    /// </para>
    /// </remarks>
    public LinkPredictionModel(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenDim = 64,
        int embeddingDim = 32,
        int numLayers = 2,
        double dropoutRate = 0.5,
        LinkPredictionDecoder decoderType = LinkPredictionDecoder.DotProduct,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>(), maxGradNorm)
    {
        InputFeatures = architecture.InputSize;
        EmbeddingDim = embeddingDim;
        HiddenDim = hiddenDim;
        NumLayers = numLayers;
        DropoutRate = dropoutRate;
        _decoderType = decoderType;

        _lossFunction = lossFunction ?? new BinaryCrossEntropyLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Create default link prediction encoder layers using LayerHelper
            Layers.AddRange(LayerHelper<T>.CreateDefaultLinkPredictionLayers(
                Architecture, HiddenDim, EmbeddingDim, NumLayers, DropoutRate));
        }
    }

    /// <summary>
    /// Sets the adjacency matrix for all graph layers in the model.
    /// </summary>
    /// <param name="adjacencyMatrix">The graph adjacency matrix.</param>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _cachedAdjacencyMatrix = adjacencyMatrix;

        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
            {
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
            }
        }
    }

    /// <summary>
    /// Performs a forward pass through the encoder network.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor [num_nodes, input_features].</param>
    /// <returns>Node embeddings [num_nodes, embedding_dim].</returns>
    public Tensor<T> Forward(Tensor<T> nodeFeatures)
    {
        if (_cachedAdjacencyMatrix is null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set before forward pass. Call SetAdjacencyMatrix() first.");
        }

        // Encode: Pass through GNN layers to get node embeddings
        var current = nodeFeatures;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        _nodeEmbeddings = current;
        return current;
    }

    /// <summary>
    /// Computes edge scores for given node pairs.
    /// </summary>
    /// <param name="edges">Edge tensor of shape [num_edges, 2] where each row is [source, target].</param>
    /// <returns>Edge scores of shape [num_edges].</returns>
    public Tensor<T> PredictEdges(Tensor<T> edges)
    {
        if (_nodeEmbeddings is null)
        {
            throw new InvalidOperationException(
                "Must call Forward() to compute node embeddings before predicting edges.");
        }

        int numEdges = edges.Shape[0];
        var scores = new Tensor<T>([numEdges]);

        for (int e = 0; e < numEdges; e++)
        {
            int sourceIdx = Convert.ToInt32(NumOps.ToDouble(edges[e, 0]));
            int targetIdx = Convert.ToInt32(NumOps.ToDouble(edges[e, 1]));
            scores[e] = ComputeEdgeScore(sourceIdx, targetIdx);
        }

        return scores;
    }

    private T ComputeEdgeScore(int sourceIdx, int targetIdx)
    {
        if (_nodeEmbeddings is null)
        {
            throw new InvalidOperationException("Node embeddings not computed.");
        }

        var sourceEmb = GetNodeEmbedding(sourceIdx);
        var targetEmb = GetNodeEmbedding(targetIdx);

        return _decoderType switch
        {
            LinkPredictionDecoder.DotProduct => DotProduct(sourceEmb, targetEmb),
            LinkPredictionDecoder.CosineSimilarity => CosineSimilarity(sourceEmb, targetEmb),
            LinkPredictionDecoder.Hadamard => Hadamard(sourceEmb, targetEmb),
            LinkPredictionDecoder.Distance => NegativeDistance(sourceEmb, targetEmb),
            _ => DotProduct(sourceEmb, targetEmb)
        };
    }

    private Vector<T> GetNodeEmbedding(int nodeIdx)
    {
        if (_nodeEmbeddings is null) throw new InvalidOperationException("Embeddings not computed.");

        var embedding = new Vector<T>(EmbeddingDim);
        for (int i = 0; i < EmbeddingDim; i++)
        {
            embedding[i] = _nodeEmbeddings[nodeIdx, i];
        }
        return embedding;
    }

    private T DotProduct(Vector<T> a, Vector<T> b)
    {
        // Vectorized dot product using Engine
        var tensorA = new Tensor<T>(a.ToArray(), [a.Length]);
        var tensorB = new Tensor<T>(b.ToArray(), [b.Length]);
        var product = Engine.TensorMultiply(tensorA, tensorB);
        return Engine.TensorSum(product);
    }

    private T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        T dot = DotProduct(a, b);
        T normA = Norm(a);
        T normB = Norm(b);
        T denom = NumOps.Multiply(normA, normB);

        return NumOps.Equals(denom, NumOps.Zero)
            ? NumOps.Zero
            : NumOps.Divide(dot, denom);
    }

    private T Hadamard(Vector<T> a, Vector<T> b)
    {
        // Vectorized element-wise product sum using Engine
        var tensorA = new Tensor<T>(a.ToArray(), [a.Length]);
        var tensorB = new Tensor<T>(b.ToArray(), [b.Length]);
        var product = Engine.TensorMultiply(tensorA, tensorB);
        return Engine.TensorSum(product);
    }

    private T NegativeDistance(Vector<T> a, Vector<T> b)
    {
        // Vectorized L2 distance calculation using Engine
        var tensorA = new Tensor<T>(a.ToArray(), [a.Length]);
        var tensorB = new Tensor<T>(b.ToArray(), [b.Length]);
        var diff = Engine.TensorSubtract(tensorA, tensorB);
        var squaredDiff = Engine.TensorMultiply(diff, diff);
        T sumSquaredDiff = Engine.TensorSum(squaredDiff);
        return NumOps.Multiply(NumOps.FromDouble(-1.0), sumSquaredDiff);
    }

    private T Norm(Vector<T> vec)
    {
        // Vectorized L2 norm using Engine
        var tensor = new Tensor<T>(vec.ToArray(), [vec.Length]);
        var squared = Engine.TensorMultiply(tensor, tensor);
        T sumSquares = Engine.TensorSum(squared);
        return NumOps.Sqrt(sumSquares);
    }

    /// <summary>
    /// Performs a backward pass through the network.
    /// </summary>
    /// <param name="outputGradient">Gradient of loss with respect to output.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var currentGradient = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
        }
        return currentGradient;
    }

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParamCount = layer.ParameterCount;
            if (layerParamCount > 0)
            {
                var layerParams = parameters.SubVector(index, layerParamCount);
                layer.SetParameters(layerParams);
                index += layerParamCount;
            }
        }
    }

    /// <summary>
    /// Trains the model on a link prediction task.
    /// </summary>
    /// <param name="task">The link prediction task with graph data and edge splits.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate for optimization.</param>
    /// <returns>Training history with loss and metrics per epoch.</returns>
    public Dictionary<string, List<double>> TrainOnTask(
        LinkPredictionTask<T> task,
        int epochs,
        double learningRate = 0.01)
    {
        if (task.Graph.AdjacencyMatrix is null)
        {
            throw new ArgumentException("Task graph must have an adjacency matrix.");
        }

        SetAdjacencyMatrix(task.Graph.AdjacencyMatrix);

        var history = new Dictionary<string, List<double>>
        {
            ["train_loss"] = new List<double>(),
            ["val_auc"] = new List<double>()
        };

        var lr = NumOps.FromDouble(learningRate);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Set all layers to training mode
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(true);
            }

            // Encode nodes
            Forward(task.Graph.NodeFeatures);

            // Score training edges
            var posScores = PredictEdges(task.TrainPosEdges);
            var negScores = PredictEdges(task.TrainNegEdges);

            // Compute binary cross-entropy loss
            double loss = ComputeBCELoss(posScores, negScores);
            history["train_loss"].Add(loss);

            // Validation AUC
            if (task.ValPosEdges.Shape[0] > 0)
            {
                var valPosScores = PredictEdges(task.ValPosEdges);
                var valNegScores = PredictEdges(task.ValNegEdges);
                double auc = ComputeAUC(valPosScores, valNegScores);
                history["val_auc"].Add(auc);
            }

            // Compute gradients and backpropagate
            var gradient = ComputeBCEGradients(posScores, negScores, task.TrainPosEdges, task.TrainNegEdges);
            Backward(gradient);

            // Update parameters
            foreach (var layer in Layers)
            {
                layer.UpdateParameters(lr);
            }
        }

        // Set layers back to inference mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        return history;
    }

    /// <summary>
    /// Evaluates the model on test edges.
    /// </summary>
    public double EvaluateOnTask(LinkPredictionTask<T> task)
    {
        if (task.Graph.AdjacencyMatrix is not null)
        {
            SetAdjacencyMatrix(task.Graph.AdjacencyMatrix);
        }

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        Forward(task.Graph.NodeFeatures);

        var testPosScores = PredictEdges(task.TestPosEdges);
        var testNegScores = PredictEdges(task.TestNegEdges);

        return ComputeAUC(testPosScores, testNegScores);
    }

    private Tensor<T> ComputeBCEGradients(
        Tensor<T> posScores,
        Tensor<T> negScores,
        Tensor<T> posEdges,
        Tensor<T> negEdges)
    {
        if (_nodeEmbeddings is null)
        {
            throw new InvalidOperationException("Node embeddings not computed.");
        }

        var gradient = new Tensor<T>(_nodeEmbeddings.Shape);
        int numPos = posEdges.Shape[0];
        int numNeg = negEdges.Shape[0];

        // Process positive edges (target = 1, grad = sigmoid(score) - 1)
        for (int e = 0; e < numPos; e++)
        {
            int srcIdx = Convert.ToInt32(NumOps.ToDouble(posEdges[e, 0]));
            int tgtIdx = Convert.ToInt32(NumOps.ToDouble(posEdges[e, 1]));
            double score = NumOps.ToDouble(posScores[e]);
            double sigmoidGrad = 1.0 / (1.0 + Math.Exp(-score)) - 1.0;

            AccumulateGradients(gradient, srcIdx, tgtIdx, sigmoidGrad);
        }

        // Process negative edges (target = 0, grad = sigmoid(score))
        for (int e = 0; e < numNeg; e++)
        {
            int srcIdx = Convert.ToInt32(NumOps.ToDouble(negEdges[e, 0]));
            int tgtIdx = Convert.ToInt32(NumOps.ToDouble(negEdges[e, 1]));
            double score = NumOps.ToDouble(negScores[e]);
            double sigmoidGrad = 1.0 / (1.0 + Math.Exp(-score));

            AccumulateGradients(gradient, srcIdx, tgtIdx, sigmoidGrad);
        }

        return gradient;
    }

    private void AccumulateGradients(Tensor<T> gradient, int srcIdx, int tgtIdx, double lossGrad)
    {
        var srcEmb = GetNodeEmbedding(srcIdx);
        var tgtEmb = GetNodeEmbedding(tgtIdx);

        for (int d = 0; d < EmbeddingDim; d++)
        {
            T srcGrad = NumOps.FromDouble(lossGrad * NumOps.ToDouble(tgtEmb[d]));
            T tgtGrad = NumOps.FromDouble(lossGrad * NumOps.ToDouble(srcEmb[d]));

            gradient[srcIdx, d] = NumOps.Add(gradient[srcIdx, d], srcGrad);
            gradient[tgtIdx, d] = NumOps.Add(gradient[tgtIdx, d], tgtGrad);
        }
    }

    private double ComputeBCELoss(Tensor<T> posScores, Tensor<T> negScores)
    {
        int numPos = posScores.Shape[0];
        int numNeg = negScores.Shape[0];
        double loss = 0.0;
        T one = NumOps.One;
        T epsilon = NumOps.FromDouble(1e-10);

        // Vectorized computation for positive scores: loss = -log(sigmoid(score))
        if (numPos > 0)
        {
            // sigmoid(x) = 1 / (1 + exp(-x))
            // -log(sigmoid(x)) = log(1 + exp(-x))
            var negPosScores = Engine.TensorMultiplyScalar(posScores, NumOps.FromDouble(-1.0));
            var expNegPos = Engine.TensorExp(negPosScores);
            var onePlusExp = Engine.TensorAddScalar(expNegPos, one);
            var clamped = Engine.TensorClamp(onePlusExp, epsilon, NumOps.FromDouble(1e10));
            var logValues = Engine.TensorLog(clamped);
            T posLoss = Engine.TensorSum(logValues);
            loss += NumOps.ToDouble(posLoss);
        }

        // Vectorized computation for negative scores: loss = -log(1 - sigmoid(score))
        if (numNeg > 0)
        {
            // 1 - sigmoid(x) = exp(-x) / (1 + exp(-x)) = 1 / (1 + exp(x))
            // -log(1 - sigmoid(x)) = log(1 + exp(x))
            var expNeg = Engine.TensorExp(negScores);
            var onePlusExp = Engine.TensorAddScalar(expNeg, one);
            var clamped = Engine.TensorClamp(onePlusExp, epsilon, NumOps.FromDouble(1e10));
            var logValues = Engine.TensorLog(clamped);
            T negLoss = Engine.TensorSum(logValues);
            loss += NumOps.ToDouble(negLoss);
        }

        return (numPos + numNeg) > 0 ? loss / (numPos + numNeg) : 0.0;
    }

    private double ComputeAUC(Tensor<T> posScores, Tensor<T> negScores)
    {
        int correctRankings = 0;
        int totalPairs = 0;
        int numPos = posScores.Shape[0];
        int numNeg = negScores.Shape[0];

        for (int i = 0; i < numPos; i++)
        {
            for (int j = 0; j < numNeg; j++)
            {
                if (NumOps.GreaterThan(posScores[i], negScores[j]))
                {
                    correctRankings++;
                }
                totalPairs++;
            }
        }

        return totalPairs > 0 ? (double)correctRankings / totalPairs : 0.5;
    }

    /// <summary>
    /// Gets all parameters as a vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                allParams.Add(layerParams[i]);
            }
        }
        return new Vector<T>([.. allParams]);
    }

    #region Abstract Method Implementations

    /// <summary>
    /// Makes a prediction using the trained network.
    /// </summary>
    /// <param name="input">The input tensor containing node features.</param>
    /// <returns>The node embeddings tensor.</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (_cachedAdjacencyMatrix is null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set using SetAdjacencyMatrix before calling Predict.");
        }

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        return Forward(input);
    }

    /// <summary>
    /// Trains the network on a single batch of data.
    /// </summary>
    /// <param name="input">The input node features.</param>
    /// <param name="expectedOutput">The expected output (edge scores).</param>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_cachedAdjacencyMatrix is null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set using SetAdjacencyMatrix before calling Train.");
        }

        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(true);
        }

        var predictions = Forward(input);

        var flattenedPredictions = predictions.ToVector();
        var flattenedExpected = expectedOutput.ToVector();

        LastLoss = _lossFunction.CalculateLoss(flattenedPredictions, flattenedExpected);

        var outputGradients = _lossFunction.CalculateDerivative(flattenedPredictions, flattenedExpected);
        var gradOutput = Tensor<T>.FromVector(outputGradients);

        if (gradOutput.Shape.Length == 1 && predictions.Shape.Length > 1)
        {
            gradOutput = gradOutput.Reshape(predictions.Shape);
        }

        Backward(gradOutput);

        Vector<T> parameterGradients = GetParameterGradients();
        Vector<T> currentParameters = GetParameters();
        Vector<T> updatedParameters = _optimizer.UpdateParameters(currentParameters, parameterGradients);

        UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Gets metadata about this model for serialization and identification.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GraphNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["NetworkType"] = "LinkPredictionModel",
                ["InputFeatures"] = InputFeatures,
                ["EmbeddingDim"] = EmbeddingDim,
                ["HiddenDim"] = HiddenDim,
                ["NumLayers"] = NumLayers,
                ["DropoutRate"] = DropoutRate,
                ["DecoderType"] = _decoderType.ToString()
            }
        };
    }

    /// <summary>
    /// Serializes network-specific data to a binary writer.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(InputFeatures);
        writer.Write(EmbeddingDim);
        writer.Write(HiddenDim);
        writer.Write(NumLayers);
        writer.Write(DropoutRate);
        writer.Write((int)_decoderType);

        SerializationHelper<T>.SerializeInterface(writer, _lossFunction);
        SerializationHelper<T>.SerializeInterface(writer, _optimizer);
    }

    /// <summary>
    /// Deserializes network-specific data from a binary reader.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // InputFeatures
        _ = reader.ReadInt32(); // EmbeddingDim
        _ = reader.ReadInt32(); // HiddenDim
        _ = reader.ReadInt32(); // NumLayers
        _ = reader.ReadDouble(); // DropoutRate
        _ = reader.ReadInt32(); // DecoderType

        _ = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader);
        _ = DeserializationHelper.DeserializeInterface<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>(reader);
    }

    /// <summary>
    /// Creates a new instance of this network type for cloning or deserialization.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new LinkPredictionModel<T>(
            architecture: Architecture,
            hiddenDim: HiddenDim,
            embeddingDim: EmbeddingDim,
            numLayers: NumLayers,
            dropoutRate: DropoutRate,
            decoderType: _decoderType);
    }

    #endregion
}
