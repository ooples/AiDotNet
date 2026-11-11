using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Attention Visualization explainer for Transformer models.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Transformers use "attention" to decide which parts of the input
/// to focus on when making predictions. This explainer visualizes these attention patterns.
///
/// What is attention?
/// - Each position in the input "attends" to all other positions
/// - Attention weights show how much each position influences another
/// - Higher weight = more influence/importance
///
/// Types of attention in Transformers:
/// 1. <b>Self-attention</b>: Input attends to itself (e.g., words attending to other words)
/// 2. <b>Cross-attention</b>: One sequence attends to another (e.g., decoder attending to encoder)
///
/// Visualization methods:
/// - <b>Raw attention</b>: Direct attention weights from the model
/// - <b>Attention rollout</b>: Combines attention across all layers
/// - <b>Attention flow</b>: Tracks information flow through the network
///
/// Example use cases:
/// - NLP: See which words the model focuses on for classification
/// - Vision Transformers: See which image patches are important
/// - Time Series: See which past timesteps influence predictions
///
/// Multi-head attention:
/// - Transformers have multiple "heads" that attend to different aspects
/// - This explainer can show individual heads or their combination
/// </para>
/// </remarks>
public class AttentionVisualizationExplainer<T> : ILocalExplainer<T, AttentionExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Tensor<T>, Tensor<T>>? _predictFunction;
    private readonly Func<Tensor<T>, int, Tensor<T>>? _getAttentionWeights;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _sequenceLength;
    private readonly string[]? _tokenLabels;

    /// <inheritdoc/>
    public string MethodName => "AttentionVisualization";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <summary>
    /// Initializes a new Attention Visualization explainer.
    /// </summary>
    /// <param name="predictFunction">Function that makes predictions.</param>
    /// <param name="getAttentionWeights">Function that returns attention weights for a layer.
    /// Takes (input, layerIndex) and returns [batch, heads, seq_len, seq_len] or [heads, seq_len, seq_len].</param>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numHeads">Number of attention heads per layer.</param>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="tokenLabels">Optional labels for tokens/positions.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>numLayers</b>: Transformers stack multiple layers (BERT-base has 12, GPT-3 has 96)
    /// - <b>numHeads</b>: Multiple heads = multiple perspectives (BERT-base has 12)
    /// - <b>tokenLabels</b>: For text, these are the actual words/tokens being attended to
    /// </para>
    /// </remarks>
    public AttentionVisualizationExplainer(
        Func<Tensor<T>, Tensor<T>>? predictFunction,
        Func<Tensor<T>, int, Tensor<T>>? getAttentionWeights,
        int numLayers,
        int numHeads,
        int sequenceLength,
        string[]? tokenLabels = null)
    {
        _predictFunction = predictFunction;
        _getAttentionWeights = getAttentionWeights;

        if (numLayers < 1)
            throw new ArgumentException("Number of layers must be at least 1.", nameof(numLayers));
        if (numHeads < 1)
            throw new ArgumentException("Number of heads must be at least 1.", nameof(numHeads));
        if (sequenceLength < 1)
            throw new ArgumentException("Sequence length must be at least 1.", nameof(sequenceLength));
        if (tokenLabels != null && tokenLabels.Length != sequenceLength)
            throw new ArgumentException($"tokenLabels length ({tokenLabels.Length}) must match sequenceLength ({sequenceLength}).", nameof(tokenLabels));

        _numLayers = numLayers;
        _numHeads = numHeads;
        _sequenceLength = sequenceLength;
        _tokenLabels = tokenLabels;
    }

    /// <summary>
    /// Visualizes attention patterns for an input.
    /// </summary>
    /// <param name="instance">The input as a flattened vector.</param>
    /// <returns>Attention explanation with attention weights.</returns>
    public AttentionExplanation<T> Explain(Vector<T> instance)
    {
        // The first sequenceLength tokens, zero-padded to the sequence length
        int count = Math.Min(instance.Length, _sequenceLength);
        var tokens = new Tensor<T>(new[] { _sequenceLength });
        if (count > 0)
        {
            tokens = count < _sequenceLength
                ? AiDotNetEngine.Current.TensorConcatenate(
                    new[] { instance.GetSubVector(0, count).AsTensor(), new Tensor<T>(new[] { _sequenceLength - count }) }, 0)
                : instance.GetSubVector(0, count).AsTensor();
        }

        return ExplainTensor(tokens.Reshape(new[] { 1, _sequenceLength }));
    }

    /// <summary>
    /// Visualizes attention patterns for an input tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>Attention explanation with attention weights.</returns>
    public AttentionExplanation<T> ExplainTensor(Tensor<T> input)
    {
        // Collect attention weights from all layers
        var allLayerAttention = new List<Tensor<T>>();

        if (_getAttentionWeights != null)
        {
            for (int layer = 0; layer < _numLayers; layer++)
            {
                var attnTensor = _getAttentionWeights(input, layer);
                var layerAttn = ExtractAttentionMatrix(attnTensor);
                allLayerAttention.Add(layerAttn);
            }
        }
        else
        {
            // Create placeholder attention (uniform distribution)
            for (int layer = 0; layer < _numLayers; layer++)
            {
                T uniformWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(_sequenceLength));
                var layerAttn = AiDotNetEngine.Current.TensorFullLike(
                    new Tensor<T>(new[] { _numHeads, _sequenceLength, _sequenceLength }), uniformWeight);
                allLayerAttention.Add(layerAttn);
            }
        }

        // Compute average attention across heads
        var avgAttentionPerLayer = allLayerAttention.Select(ComputeAverageAcrossHeads).ToList();

        // Compute attention rollout
        var rollout = ComputeAttentionRollout(avgAttentionPerLayer);

        // Compute token importance (sum of attention received)
        var tokenImportance = ComputeTokenImportance(rollout).ToArray();

        return new AttentionExplanation<T>
        {
            LayerAttention = allLayerAttention,
            AverageAttentionPerLayer = avgAttentionPerLayer,
            AttentionRollout = rollout.ToMatrix(),
            TokenImportance = tokenImportance,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            SequenceLength = _sequenceLength,
            TokenLabels = _tokenLabels ?? Enumerable.Range(0, _sequenceLength).Select(i => $"Position {i}").ToArray()
        };
    }

    /// <inheritdoc/>
    public AttentionExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new AttentionExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Gets attention for a specific token/position.
    /// </summary>
    /// <param name="explanation">The attention explanation.</param>
    /// <param name="queryPosition">Position of the token doing the attending.</param>
    /// <param name="layer">Layer index (optional, -1 for rollout).</param>
    /// <param name="head">Head index (optional, -1 for average).</param>
    /// <returns>Dictionary mapping key positions to attention weights.</returns>
    public Dictionary<int, T> GetAttentionFromPosition(
        AttentionExplanation<T> explanation,
        int queryPosition,
        int layer = -1,
        int head = -1)
    {
        if (queryPosition < 0 || queryPosition >= explanation.SequenceLength)
            throw new ArgumentOutOfRangeException(nameof(queryPosition),
                $"Query position {queryPosition} must be in range [0, {explanation.SequenceLength}).");
        if (layer < -1 || layer >= explanation.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(layer),
                $"Layer {layer} must be in range [-1, {explanation.NumLayers}).");
        if (head < -1 || head >= explanation.NumHeads)
            throw new ArgumentOutOfRangeException(nameof(head),
                $"Head {head} must be in range [-1, {explanation.NumHeads}).");

        Vector<T> weights;
        if (layer < 0)
        {
            // Use rollout
            weights = explanation.AttentionRollout.GetRow(queryPosition);
        }
        else if (head < 0)
        {
            // Use layer average
            weights = explanation.AverageAttentionPerLayer[layer].GetRow(queryPosition);
        }
        else
        {
            // Use specific head
            var engine = AiDotNetEngine.Current;
            weights = engine.TensorSliceAxis(engine.TensorSliceAxis(explanation.LayerAttention[layer], 0, head), 0, queryPosition)
                .ToVector();
        }

        return Enumerable.Select(weights, (weight, position) => (position, weight))
            .Take(explanation.SequenceLength)
            .ToDictionary(entry => entry.position, entry => entry.weight);
    }

    /// <summary>
    /// Gets the most attended positions for a query position.
    /// </summary>
    public List<(int position, string label, T weight)> GetTopAttendedPositions(
        AttentionExplanation<T> explanation,
        int queryPosition,
        int topK = 5,
        int layer = -1)
    {
        var attention = GetAttentionFromPosition(explanation, queryPosition, layer);

        return attention
            .OrderByDescending(kvp => NumOps.ToDouble(kvp.Value))
            .Take(topK)
            .Select(kvp => (kvp.Key, explanation.TokenLabels[kvp.Key], kvp.Value))
            .ToList();
    }

    /// <summary>
    /// Extracts attention matrix from tensor.
    /// </summary>
    private Tensor<T> ExtractAttentionMatrix(Tensor<T> attnTensor)
    {
        // Validate tensor shape: expect [batch, heads, seq, seq] or [heads, seq, seq]
        if (attnTensor.Rank < 3 || attnTensor.Rank > 4)
            throw new ArgumentException($"Attention tensor must have rank 3 or 4, but got {attnTensor.Rank}.", nameof(attnTensor));

        int expectedElements = _numHeads * _sequenceLength * _sequenceLength;
        if (attnTensor.Length < expectedElements)
            throw new ArgumentException(
                $"Attention tensor has {attnTensor.Length} elements but expected at least {expectedElements} " +
                $"(numHeads={_numHeads}, sequenceLength={_sequenceLength}).", nameof(attnTensor));

        // The leading [heads, seq, seq] block in row-major order (the first batch entry for rank 4)
        return new Tensor<T>(new[] { _numHeads, _sequenceLength, _sequenceLength },
            attnTensor.ToVector().GetSubVector(0, expectedElements));
    }

    /// <summary>
    /// Computes average attention across heads.
    /// </summary>
    private Matrix<T> ComputeAverageAcrossHeads(Tensor<T> headAttention)
    {
        return AiDotNetEngine.Current.ReduceMean(headAttention, new[] { 0 }, false).ToMatrix();
    }

    /// <summary>
    /// Computes attention rollout across all layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Attention rollout tracks how attention propagates through layers.
    /// It's computed by multiplying attention matrices across layers, accounting for
    /// residual connections (which add identity to attention).
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeAttentionRollout(List<Matrix<T>> layerAttention)
    {
        var engine = AiDotNetEngine.Current;

        // Start with identity matrix (residual connection)
        var identity = engine.TensorEye<T>(_sequenceLength);
        var rollout = identity;
        T half = NumOps.FromDouble(0.5);

        // Multiply through layers
        foreach (var layerAttn in layerAttention)
        {
            // Add identity for residual connection: (attention + I) / 2
            var attnWithResidual = engine.TensorMultiplyScalar(engine.TensorAdd(Tensor<T>.FromMatrix(layerAttn), identity), half);

            // Re-normalize rows to sum to 1 (rows with a non-positive sum stay as they are)
            var rowSums = engine.ReduceSum(attnWithResidual, new[] { 1 }, true);
            var positive = engine.TensorGreaterThan(rowSums, NumOps.Zero);
            var divisors = engine.TensorWhere(positive, rowSums, engine.TensorFullLike(rowSums, NumOps.One));
            attnWithResidual = engine.TensorDivide(attnWithResidual, engine.TensorBroadcastTo(divisors, attnWithResidual.Shape));

            // Matrix multiplication: rollout = attnWithResidual * rollout
            rollout = engine.TensorMatMul(attnWithResidual, rollout);
        }

        return rollout;
    }

    /// <summary>
    /// Computes overall token importance from attention rollout.
    /// </summary>
    private Tensor<T> ComputeTokenImportance(Tensor<T> rollout)
    {
        var engine = AiDotNetEngine.Current;

        // Mean attention received from all positions (column mean)
        var importance = engine.ReduceMean(rollout, new[] { 0 }, false);

        // Normalize
        T maxVal = engine.TensorMaxValue(importance);
        return NumOps.GreaterThan(maxVal, NumOps.Zero)
            ? engine.TensorDivideScalar(importance, maxVal)
            : importance;
    }
}

/// <summary>
/// Represents the result of an Attention Visualization analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AttentionExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets attention weights for each layer [layer][head, query, key].
    /// </summary>
    public List<Tensor<T>> LayerAttention { get; set; } = new();

    /// <summary>
    /// Gets or sets average attention per layer (across heads) [layer][query, key].
    /// </summary>
    public List<Matrix<T>> AverageAttentionPerLayer { get; set; } = new();

    /// <summary>
    /// Gets or sets the attention rollout matrix [query, key].
    /// Shows accumulated attention through all layers.
    /// </summary>
    public Matrix<T> AttentionRollout { get; set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets or sets the overall token importance (derived from rollout).
    /// </summary>
    public T[] TokenImportance { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the number of layers.
    /// </summary>
    public int NumLayers { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the sequence length.
    /// </summary>
    public int SequenceLength { get; set; }

    /// <summary>
    /// Gets or sets labels for tokens/positions.
    /// </summary>
    public string[] TokenLabels { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets tokens sorted by importance.
    /// </summary>
    public List<(int position, string label, T importance)> GetSortedTokenImportance()
    {
        var result = new List<(int, string, T)>();
        for (int i = 0; i < TokenImportance.Length; i++)
        {
            result.Add((i, TokenLabels[i], TokenImportance[i]));
        }
        return result.OrderByDescending(x => NumOps.ToDouble(x.Item3)).ToList();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            "Attention Visualization Explanation:",
            $"  Layers: {NumLayers}",
            $"  Heads per layer: {NumHeads}",
            $"  Sequence length: {SequenceLength}",
            "",
            "Top Important Tokens (from rollout):"
        };

        var sorted = GetSortedTokenImportance().Take(10);
        foreach (var (pos, label, imp) in sorted)
        {
            lines.Add($"  [{pos}] {label}: {imp:F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
