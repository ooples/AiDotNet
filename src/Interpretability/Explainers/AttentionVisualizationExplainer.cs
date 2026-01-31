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
        var inputTensor = new Tensor<T>(new[] { 1, _sequenceLength });
        var dataSpan = inputTensor.Data.Span;
        for (int i = 0; i < instance.Length && i < _sequenceLength; i++)
        {
            dataSpan[i] = instance[i];
        }

        return ExplainTensor(inputTensor);
    }

    /// <summary>
    /// Visualizes attention patterns for an input tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>Attention explanation with attention weights.</returns>
    public AttentionExplanation<T> ExplainTensor(Tensor<T> input)
    {
        // Collect attention weights from all layers
        var allLayerAttention = new List<T[,,]>();

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
                var layerAttn = new T[_numHeads, _sequenceLength, _sequenceLength];
                double uniformWeight = 1.0 / _sequenceLength;

                for (int h = 0; h < _numHeads; h++)
                {
                    for (int i = 0; i < _sequenceLength; i++)
                    {
                        for (int j = 0; j < _sequenceLength; j++)
                        {
                            layerAttn[h, i, j] = NumOps.FromDouble(uniformWeight);
                        }
                    }
                }
                allLayerAttention.Add(layerAttn);
            }
        }

        // Compute average attention across heads
        var avgAttentionPerLayer = allLayerAttention.Select(ComputeAverageAcrossHeads).ToList();

        // Compute attention rollout
        var rollout = ComputeAttentionRollout(avgAttentionPerLayer);

        // Compute token importance (sum of attention received)
        var tokenImportance = ComputeTokenImportance(rollout);

        return new AttentionExplanation<T>
        {
            LayerAttention = allLayerAttention,
            AverageAttentionPerLayer = avgAttentionPerLayer,
            AttentionRollout = rollout,
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
        var result = new Dictionary<int, T>();

        if (layer < 0)
        {
            // Use rollout
            for (int j = 0; j < explanation.SequenceLength; j++)
            {
                result[j] = explanation.AttentionRollout[queryPosition, j];
            }
        }
        else if (head < 0)
        {
            // Use layer average
            for (int j = 0; j < explanation.SequenceLength; j++)
            {
                result[j] = explanation.AverageAttentionPerLayer[layer][queryPosition, j];
            }
        }
        else
        {
            // Use specific head
            for (int j = 0; j < explanation.SequenceLength; j++)
            {
                result[j] = explanation.LayerAttention[layer][head, queryPosition, j];
            }
        }

        return result;
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
    private T[,,] ExtractAttentionMatrix(Tensor<T> attnTensor)
    {
        // Validate tensor shape: expect [batch, heads, seq, seq] or [heads, seq, seq]
        if (attnTensor.Rank < 3 || attnTensor.Rank > 4)
            throw new ArgumentException($"Attention tensor must have rank 3 or 4, but got {attnTensor.Rank}.", nameof(attnTensor));

        int expectedElements = _numHeads * _sequenceLength * _sequenceLength;
        if (attnTensor.Length < expectedElements)
            throw new ArgumentException(
                $"Attention tensor has {attnTensor.Length} elements but expected at least {expectedElements} " +
                $"(numHeads={_numHeads}, sequenceLength={_sequenceLength}).", nameof(attnTensor));

        var result = new T[_numHeads, _sequenceLength, _sequenceLength];
        var attnSpan = attnTensor.Data.Span;

        // Assuming shape is [batch, heads, seq, seq] or [heads, seq, seq]
        int offset = attnTensor.Shape.Length == 4 ? 1 : 0;

        for (int h = 0; h < _numHeads; h++)
        {
            for (int i = 0; i < _sequenceLength; i++)
            {
                for (int j = 0; j < _sequenceLength; j++)
                {
                    int idx = h * _sequenceLength * _sequenceLength + i * _sequenceLength + j;
                    result[h, i, j] = idx < attnSpan.Length
                        ? attnSpan[idx]
                        : NumOps.Zero;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Computes average attention across heads.
    /// </summary>
    private T[,] ComputeAverageAcrossHeads(T[,,] headAttention)
    {
        var avg = new T[_sequenceLength, _sequenceLength];

        for (int i = 0; i < _sequenceLength; i++)
        {
            for (int j = 0; j < _sequenceLength; j++)
            {
                double sum = 0;
                for (int h = 0; h < _numHeads; h++)
                {
                    sum += NumOps.ToDouble(headAttention[h, i, j]);
                }
                avg[i, j] = NumOps.FromDouble(sum / _numHeads);
            }
        }

        return avg;
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
    private T[,] ComputeAttentionRollout(List<T[,]> layerAttention)
    {
        // Start with identity matrix (residual connection)
        var rollout = new T[_sequenceLength, _sequenceLength];
        for (int i = 0; i < _sequenceLength; i++)
        {
            rollout[i, i] = NumOps.One;
        }

        // Multiply through layers
        foreach (var layerAttn in layerAttention)
        {
            // Add identity for residual connection: (attention + I) / 2
            var attnWithResidual = new T[_sequenceLength, _sequenceLength];
            for (int i = 0; i < _sequenceLength; i++)
            {
                for (int j = 0; j < _sequenceLength; j++)
                {
                    double val = NumOps.ToDouble(layerAttn[i, j]);
                    if (i == j) val = (val + 1) / 2;
                    else val = val / 2;
                    attnWithResidual[i, j] = NumOps.FromDouble(val);
                }
            }

            // Re-normalize rows to sum to 1
            for (int i = 0; i < _sequenceLength; i++)
            {
                double rowSum = 0;
                for (int j = 0; j < _sequenceLength; j++)
                {
                    rowSum += NumOps.ToDouble(attnWithResidual[i, j]);
                }
                if (rowSum > 0)
                {
                    for (int j = 0; j < _sequenceLength; j++)
                    {
                        attnWithResidual[i, j] = NumOps.FromDouble(NumOps.ToDouble(attnWithResidual[i, j]) / rowSum);
                    }
                }
            }

            // Matrix multiplication: rollout = attnWithResidual * rollout
            var newRollout = new T[_sequenceLength, _sequenceLength];
            for (int i = 0; i < _sequenceLength; i++)
            {
                for (int j = 0; j < _sequenceLength; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < _sequenceLength; k++)
                    {
                        sum += NumOps.ToDouble(attnWithResidual[i, k]) * NumOps.ToDouble(rollout[k, j]);
                    }
                    newRollout[i, j] = NumOps.FromDouble(sum);
                }
            }
            rollout = newRollout;
        }

        return rollout;
    }

    /// <summary>
    /// Computes overall token importance from attention rollout.
    /// </summary>
    private T[] ComputeTokenImportance(T[,] rollout)
    {
        var importance = new T[_sequenceLength];

        // Sum attention received from all positions (column sum)
        for (int j = 0; j < _sequenceLength; j++)
        {
            double sum = 0;
            for (int i = 0; i < _sequenceLength; i++)
            {
                sum += NumOps.ToDouble(rollout[i, j]);
            }
            importance[j] = NumOps.FromDouble(sum / _sequenceLength);
        }

        // Normalize
        double maxVal = importance.Max(x => NumOps.ToDouble(x));
        if (maxVal > 0)
        {
            for (int j = 0; j < _sequenceLength; j++)
            {
                importance[j] = NumOps.FromDouble(NumOps.ToDouble(importance[j]) / maxVal);
            }
        }

        return importance;
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
    public List<T[,,]> LayerAttention { get; set; } = new();

    /// <summary>
    /// Gets or sets average attention per layer (across heads) [layer][query, key].
    /// </summary>
    public List<T[,]> AverageAttentionPerLayer { get; set; } = new();

    /// <summary>
    /// Gets or sets the attention rollout matrix [query, key].
    /// Shows accumulated attention through all layers.
    /// </summary>
    public T[,] AttentionRollout { get; set; } = new T[0, 0];

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
            lines.Add($"  [{pos}] {label}: {NumOps.ToDouble(imp):F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
