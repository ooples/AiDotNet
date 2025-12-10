using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tasks.Graph;

/// <summary>
/// Implements a complete model for link prediction tasks on graphs.
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
///    - Dot product: z_i · z_j
///    - Concatenation + MLP: MLP([z_i || z_j])
///    - Distance-based: -||z_i - z_j||²
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
///   * High score → Likely to be friends
///   * Low score → Unlikely to be friends
/// ```
/// </para>
/// </remarks>
public class LinkPredictionModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly List<ILayer<T>> _encoderLayers;
    private readonly LinkPredictionDecoder _decoder;
    private Tensor<T>? _adjacencyMatrix;
    private Tensor<T>? _nodeEmbeddings; // Cached after forward pass
    private bool _isTrainingMode;

    /// <summary>
    /// Decoder types for combining node embeddings into edge scores.
    /// </summary>
    public enum LinkPredictionDecoder
    {
        /// <summary>Dot product: score = z_i · z_j</summary>
        DotProduct,

        /// <summary>Cosine similarity: score = (z_i · z_j) / (||z_i|| ||z_j||)</summary>
        CosineSimilarity,

        /// <summary>Element-wise product: score = sum(z_i ⊙ z_j)</summary>
        Hadamard,

        /// <summary>L2 distance: score = -||z_i - z_j||²</summary>
        Distance
    }

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDim { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="LinkPredictionModel{T}"/> class.
    /// </summary>
    /// <param name="encoderLayers">GNN layers that encode nodes into embeddings.</param>
    /// <param name="embeddingDim">Dimension of node embeddings.</param>
    /// <param name="decoder">Method for combining node embeddings into edge scores.</param>
    /// <remarks>
    /// <para>
    /// Typical encoder configuration:
    /// 1. Graph convolutional layer (GCN, GAT, GraphSAGE)
    /// 2. Activation (ReLU)
    /// 3. Dropout
    /// 4. Additional graph conv layers
    /// 5. Final layer outputs embeddings of dimension embeddingDim
    /// </para>
    /// <para><b>For Beginners:</b> Choosing a decoder:
    ///
    /// - **Dot Product**: Simple, fast, assumes similarity in embedding space
    ///   * Good for: Large graphs, initial experiments
    ///   * Limitation: Can't capture complex relationships
    ///
    /// - **Cosine Similarity**: Normalized dot product
    ///   * Good for: When embedding magnitudes vary
    ///   * Handles: Different node degrees better
    ///
    /// - **Hadamard**: Element-wise multiplication
    ///   * Good for: Capturing feature interactions
    ///   * More expressive than dot product
    ///
    /// - **Distance**: Negative squared L2 distance
    ///   * Good for: Embedding space as metric space
    ///   * Similar nodes close, dissimilar far apart
    /// </para>
    /// </remarks>
    public LinkPredictionModel(
        List<ILayer<T>> encoderLayers,
        int embeddingDim,
        LinkPredictionDecoder decoder = LinkPredictionDecoder.DotProduct)
    {
        _encoderLayers = encoderLayers ?? throw new ArgumentNullException(nameof(encoderLayers));
        EmbeddingDim = embeddingDim;
        _decoder = decoder;
    }

    /// <summary>
    /// Sets the adjacency matrix for all graph layers in the encoder.
    /// </summary>
    /// <param name="adjacencyMatrix">The graph adjacency matrix.</param>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _adjacencyMatrix = adjacencyMatrix;

        foreach (var layer in _encoderLayers.OfType<IGraphConvolutionLayer<T>>())
        {
            layer.SetAdjacencyMatrix(adjacencyMatrix);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> Forward(Tensor<T> input)
    {
        if (_adjacencyMatrix == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set before forward pass. Call SetAdjacencyMatrix() first.");
        }

        // Encode: Pass through GNN layers to get node embeddings
        var current = input;
        foreach (var layer in _encoderLayers)
        {
            current = layer.Forward(current);
        }

        _nodeEmbeddings = current;
        return current;
    }

    /// <summary>
    /// Computes edge scores for given node pairs.
    /// </summary>
    /// <param name="edges">Edge tensor of shape [num_edges, 2] where each row is [source, target],
    /// or [batch_size, num_edges, 2] for batched operation.</param>
    /// <returns>Edge scores of shape [num_edges] or [batch_size, num_edges].</returns>
    /// <remarks>
    /// <para>
    /// After encoding nodes into embeddings with Forward(), this method scores specific edges.
    /// Higher scores indicate higher likelihood of edge existence.
    /// </para>
    /// <para><b>For Beginners:</b> Edge scoring process:
    ///
    /// ```
    /// For edge (node_i, node_j):
    /// 1. Get embeddings: z_i = nodeEmbeddings[i], z_j = nodeEmbeddings[j]
    /// 2. Compute score using decoder:
    ///    - Dot product: z_i · z_j
    ///    - Cosine: (z_i · z_j) / (||z_i|| ||z_j||)
    ///    - Hadamard: sum(z_i ⊙ z_j)
    ///    - Distance: -||z_i - z_j||²
    /// 3. Return score
    /// ```
    ///
    /// During training:
    /// - Positive edges (exist): Want high scores
    /// - Negative edges (don't exist): Want low scores
    /// - Use binary cross-entropy loss
    /// </para>
    /// </remarks>
    public Tensor<T> PredictEdges(Tensor<T> edges)
    {
        if (_nodeEmbeddings == null)
        {
            throw new InvalidOperationException(
                "Must call Forward() to compute node embeddings before predicting edges.");
        }

        // Handle both [num_edges, 2] and [batch_size, num_edges, 2] formats
        if (edges.Shape.Length == 2)
        {
            // Shape [num_edges, 2] - non-batched
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
        else
        {
            // Shape [batch_size, num_edges, 2] - batched
            int batchSize = edges.Shape[0];
            int numEdges = edges.Shape[1];
            var scores = new Tensor<T>([batchSize, numEdges]);

            for (int b = 0; b < batchSize; b++)
            {
                for (int e = 0; e < numEdges; e++)
                {
                    int sourceIdx = Convert.ToInt32(NumOps.ToDouble(edges[b, e, 0]));
                    int targetIdx = Convert.ToInt32(NumOps.ToDouble(edges[b, e, 1]));
                    scores[b, e] = ComputeEdgeScore(sourceIdx, targetIdx);
                }
            }

            return scores;
        }
    }

    /// <summary>
    /// Computes the score for a single edge between two nodes.
    /// </summary>
    private T ComputeEdgeScore(int sourceIdx, int targetIdx)
    {
        if (_nodeEmbeddings == null)
        {
            throw new InvalidOperationException("Node embeddings not computed.");
        }

        // Get embeddings for source and target nodes
        var sourceEmb = GetNodeEmbedding(sourceIdx);
        var targetEmb = GetNodeEmbedding(targetIdx);

        return _decoder switch
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
        if (_nodeEmbeddings == null) throw new InvalidOperationException("Embeddings not computed.");

        var embedding = new Vector<T>(EmbeddingDim);
        for (int i = 0; i < EmbeddingDim; i++)
        {
            // Handle both 2D and 3D embedding tensors
            embedding[i] = _nodeEmbeddings.Shape.Length == 3
                ? _nodeEmbeddings[0, nodeIdx, i]
                : _nodeEmbeddings[nodeIdx, i];
        }
        return embedding;
    }

    private T DotProduct(Vector<T> a, Vector<T> b)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(a[i], b[i]));
        }
        return sum;
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
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(a[i], b[i]));
        }
        return sum;
    }

    private T NegativeDistance(Vector<T> a, Vector<T> b)
    {
        T sumSquaredDiff = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
        }
        return NumOps.Multiply(NumOps.FromDouble(-1.0), sumSquaredDiff);
    }

    private T Norm(Vector<T> vec)
    {
        T sumSquares = NumOps.Zero;
        for (int i = 0; i < vec.Length; i++)
        {
            sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(vec[i], vec[i]));
        }
        return NumOps.Sqrt(sumSquares);
    }

    /// <inheritdoc/>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var currentGradient = outputGradient;
        for (int i = _encoderLayers.Count - 1; i >= 0; i--)
        {
            currentGradient = _encoderLayers[i].Backward(currentGradient);
        }
        return currentGradient;
    }

    /// <inheritdoc/>
    public void UpdateParameters(T learningRate)
    {
        foreach (var layer in _encoderLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <inheritdoc/>
    public void SetTrainingMode(bool isTraining)
    {
        _isTrainingMode = isTraining;
        foreach (var layer in _encoderLayers)
        {
            layer.SetTrainingMode(isTraining);
        }
    }

    /// <summary>
    /// Trains the model on a link prediction task.
    /// </summary>
    /// <param name="task">The link prediction task with graph data and edge splits.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate for optimization.</param>
    /// <returns>Training history with loss and metrics per epoch.</returns>
    /// <remarks>
    /// <para>
    /// Training uses binary cross-entropy loss:
    /// - Positive edges (exist): Target = 1
    /// - Negative edges (don't exist): Target = 0
    ///
    /// The model learns to assign high scores to positive edges and low scores to negative edges.
    /// </para>
    /// <para><b>For Beginners:</b> Link prediction training:
    ///
    /// **Each training step:**
    /// 1. Encode all nodes using current graph structure
    /// 2. Score positive and negative edge examples
    /// 3. Compute loss: BCE(positive_scores, 1) + BCE(negative_scores, 0)
    /// 4. Backpropagate gradients
    /// 5. Update encoder parameters
    ///
    /// **Evaluation metrics:**
    /// - **AUC** (Area Under ROC Curve): Ranking quality
    ///   * 1.0 = Perfect ranking (all positives scored higher than negatives)
    ///   * 0.5 = Random guessing
    ///
    /// - **Accuracy**: Classification with threshold 0.5
    ///   * score > 0.5 → Predict edge exists
    ///   * score ≤ 0.5 → Predict edge doesn't exist
    /// </para>
    /// </remarks>
    public Dictionary<string, List<double>> Train(
        LinkPredictionTask<T> task,
        int epochs,
        T learningRate)
    {
        if (task.Graph.AdjacencyMatrix == null)
        {
            throw new ArgumentException("Task graph must have an adjacency matrix.");
        }

        SetAdjacencyMatrix(task.Graph.AdjacencyMatrix);
        SetTrainingMode(true);

        var history = new Dictionary<string, List<double>>
        {
            ["train_loss"] = new List<double>(),
            ["val_auc"] = new List<double>()
        };

        for (int epoch = 0; epoch < epochs; epoch++)
        {
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

            // Compute actual BCE gradients and backprop through encoder
            var gradient = ComputeBCEGradients(posScores, negScores, task.TrainPosEdges, task.TrainNegEdges);
            Backward(gradient);
            UpdateParameters(learningRate);
        }

        SetTrainingMode(false);
        return history;
    }

    /// <summary>
    /// Computes BCE loss gradients and accumulates them to node embeddings.
    /// For dot product decoder: d(score)/d(z_i) = z_j and d(score)/d(z_j) = z_i
    /// BCE gradient w.r.t. score: sigmoid(score) - target
    /// </summary>
    private Tensor<T> ComputeBCEGradients(
        Tensor<T> posScores,
        Tensor<T> negScores,
        Tensor<T> posEdges,
        Tensor<T> negEdges)
    {
        if (_nodeEmbeddings == null)
        {
            throw new InvalidOperationException("Node embeddings not computed.");
        }

        var gradient = new Tensor<T>(_nodeEmbeddings.Shape);
        bool is2D = posEdges.Shape.Length == 2;

        // Process positive edges (target = 1, grad = sigmoid(score) - 1)
        if (is2D)
        {
            int numPos = posEdges.Shape[0];
            for (int e = 0; e < numPos; e++)
            {
                int srcIdx = Convert.ToInt32(NumOps.ToDouble(posEdges[e, 0]));
                int tgtIdx = Convert.ToInt32(NumOps.ToDouble(posEdges[e, 1]));
                double score = NumOps.ToDouble(posScores[e]);
                double sigmoidGrad = 1.0 / (1.0 + Math.Exp(-score)) - 1.0; // sigmoid(s) - 1

                AccumulateGradients(gradient, srcIdx, tgtIdx, sigmoidGrad);
            }

            // Process negative edges (target = 0, grad = sigmoid(score))
            int numNeg = negEdges.Shape[0];
            for (int e = 0; e < numNeg; e++)
            {
                int srcIdx = Convert.ToInt32(NumOps.ToDouble(negEdges[e, 0]));
                int tgtIdx = Convert.ToInt32(NumOps.ToDouble(negEdges[e, 1]));
                double score = NumOps.ToDouble(negScores[e]);
                double sigmoidGrad = 1.0 / (1.0 + Math.Exp(-score)); // sigmoid(s) - 0

                AccumulateGradients(gradient, srcIdx, tgtIdx, sigmoidGrad);
            }
        }
        else
        {
            // Batched case
            int batchSize = posEdges.Shape[0];
            int numPos = posEdges.Shape[1];
            int numNeg = negEdges.Shape[1];

            for (int b = 0; b < batchSize; b++)
            {
                for (int e = 0; e < numPos; e++)
                {
                    int srcIdx = Convert.ToInt32(NumOps.ToDouble(posEdges[b, e, 0]));
                    int tgtIdx = Convert.ToInt32(NumOps.ToDouble(posEdges[b, e, 1]));
                    double score = NumOps.ToDouble(posScores[b, e]);
                    double sigmoidGrad = 1.0 / (1.0 + Math.Exp(-score)) - 1.0;

                    AccumulateGradientsBatched(gradient, b, srcIdx, tgtIdx, sigmoidGrad);
                }

                for (int e = 0; e < numNeg; e++)
                {
                    int srcIdx = Convert.ToInt32(NumOps.ToDouble(negEdges[b, e, 0]));
                    int tgtIdx = Convert.ToInt32(NumOps.ToDouble(negEdges[b, e, 1]));
                    double score = NumOps.ToDouble(negScores[b, e]);
                    double sigmoidGrad = 1.0 / (1.0 + Math.Exp(-score));

                    AccumulateGradientsBatched(gradient, b, srcIdx, tgtIdx, sigmoidGrad);
                }
            }
        }

        return gradient;
    }

    private void AccumulateGradients(Tensor<T> gradient, int srcIdx, int tgtIdx, double lossGrad)
    {
        // For dot product: d(z_i·z_j)/d(z_i) = z_j, d(z_i·z_j)/d(z_j) = z_i
        // Full grad = lossGrad * decoder_grad
        var srcEmb = GetNodeEmbedding(srcIdx);
        var tgtEmb = GetNodeEmbedding(tgtIdx);

        bool is3D = gradient.Shape.Length == 3;

        for (int d = 0; d < EmbeddingDim; d++)
        {
            T srcGrad = NumOps.FromDouble(lossGrad * NumOps.ToDouble(tgtEmb[d]));
            T tgtGrad = NumOps.FromDouble(lossGrad * NumOps.ToDouble(srcEmb[d]));

            if (is3D)
            {
                gradient[0, srcIdx, d] = NumOps.Add(gradient[0, srcIdx, d], srcGrad);
                gradient[0, tgtIdx, d] = NumOps.Add(gradient[0, tgtIdx, d], tgtGrad);
            }
            else
            {
                gradient[srcIdx, d] = NumOps.Add(gradient[srcIdx, d], srcGrad);
                gradient[tgtIdx, d] = NumOps.Add(gradient[tgtIdx, d], tgtGrad);
            }
        }
    }

    private void AccumulateGradientsBatched(Tensor<T> gradient, int batch, int srcIdx, int tgtIdx, double lossGrad)
    {
        var srcEmb = GetNodeEmbedding(srcIdx);
        var tgtEmb = GetNodeEmbedding(tgtIdx);

        for (int d = 0; d < EmbeddingDim; d++)
        {
            T srcGrad = NumOps.FromDouble(lossGrad * NumOps.ToDouble(tgtEmb[d]));
            T tgtGrad = NumOps.FromDouble(lossGrad * NumOps.ToDouble(srcEmb[d]));

            gradient[batch, srcIdx, d] = NumOps.Add(gradient[batch, srcIdx, d], srcGrad);
            gradient[batch, tgtIdx, d] = NumOps.Add(gradient[batch, tgtIdx, d], tgtGrad);
        }
    }

    private double ComputeBCELoss(Tensor<T> posScores, Tensor<T> negScores)
    {
        double loss = 0.0;

        // Handle both 1D [num_edges] and 2D [batch_size, num_edges] scores
        if (posScores.Shape.Length == 1)
        {
            // 1D case: non-batched
            int numPos = posScores.Shape[0];
            int numNeg = negScores.Shape[0];

            for (int i = 0; i < numPos; i++)
            {
                double score = NumOps.ToDouble(posScores[i]);
                double sigmoid = 1.0 / (1.0 + Math.Exp(-score));
                loss -= Math.Log(Math.Max(sigmoid, 1e-10));
            }

            for (int i = 0; i < numNeg; i++)
            {
                double score = NumOps.ToDouble(negScores[i]);
                double sigmoid = 1.0 / (1.0 + Math.Exp(-score));
                loss -= Math.Log(Math.Max(1.0 - sigmoid, 1e-10));
            }

            return (numPos + numNeg) > 0 ? loss / (numPos + numNeg) : 0.0;
        }
        else
        {
            // 2D case: batched
            int batchSize = posScores.Shape[0];
            int numPos = posScores.Shape[1];
            int numNeg = negScores.Shape[1];

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numPos; i++)
                {
                    double score = NumOps.ToDouble(posScores[b, i]);
                    double sigmoid = 1.0 / (1.0 + Math.Exp(-score));
                    loss -= Math.Log(Math.Max(sigmoid, 1e-10));
                }

                for (int i = 0; i < numNeg; i++)
                {
                    double score = NumOps.ToDouble(negScores[b, i]);
                    double sigmoid = 1.0 / (1.0 + Math.Exp(-score));
                    loss -= Math.Log(Math.Max(1.0 - sigmoid, 1e-10));
                }
            }

            int totalSamples = batchSize * (numPos + numNeg);
            return totalSamples > 0 ? loss / totalSamples : 0.0;
        }
    }

    private double ComputeAUC(Tensor<T> posScores, Tensor<T> negScores)
    {
        // Simplified AUC: fraction of (pos, neg) pairs correctly ranked
        int correctRankings = 0;
        int totalPairs = 0;

        // Handle both 1D [num_edges] and 2D [batch_size, num_edges] scores
        if (posScores.Shape.Length == 1)
        {
            // 1D case: non-batched
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
        }
        else
        {
            // 2D case: batched
            int batchSize = posScores.Shape[0];
            int numPos = posScores.Shape[1];
            int numNeg = negScores.Shape[1];

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numPos; i++)
                {
                    for (int j = 0; j < numNeg; j++)
                    {
                        if (NumOps.GreaterThan(posScores[b, i], negScores[b, j]))
                        {
                            correctRankings++;
                        }
                        totalPairs++;
                    }
                }
            }
        }

        return totalPairs > 0 ? (double)correctRankings / totalPairs : 0.5;
    }
}
