using System.IO;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>
/// Transformer decoder for DETR-style object detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The DETR decoder transforms a set of learnable object queries
/// into object predictions by attending to image features from the encoder. Each query
/// learns to look for a specific type of object or region.</para>
///
/// <para>Architecture:
/// - Object queries (learnable embeddings)
/// - Self-attention among queries
/// - Cross-attention between queries and encoder features
/// - FFN (feed-forward network) for each query
/// </para>
/// </remarks>
internal partial class DETRDecoder<T> : CvParameterModule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numLayers;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _numQueries;
    private readonly List<DecoderLayer<T>> _layers;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T> _queryEmbed;  // Learnable query embeddings
    private readonly Dense<T> _classHead;
    private readonly Dense<T> _boxHead;

    /// <summary>
    /// Creates a new DETR decoder.
    /// </summary>
    /// <param name="hiddenDim">Hidden dimension size.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numLayers">Number of decoder layers.</param>
    /// <param name="numQueries">Number of object queries.</param>
    /// <param name="numClasses">Number of detection classes.</param>
    public DETRDecoder(
        int hiddenDim = 256,
        int numHeads = 8,
        int numLayers = 6,
        int numQueries = 100,
        int numClasses = 80)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _numLayers = numLayers;
        _numQueries = numQueries;

        // Initialize decoder layers
        _layers = new List<DecoderLayer<T>>();
        for (int i = 0; i < numLayers; i++)
        {
            _layers.Add(new DecoderLayer<T>(hiddenDim, numHeads));
        }

        // Learnable query embeddings
        _queryEmbed = InitializeQueryEmbeddings(numQueries, hiddenDim);

        // Prediction heads
        _classHead = new Dense<T>(hiddenDim, numClasses + 1); // +1 for background/no-object
        _boxHead = new Dense<T>(hiddenDim, 4); // Box coordinates (cx, cy, w, h)
    }

    /// <summary>
    /// Forward pass through the decoder.
    /// </summary>
    /// <param name="memory">Encoder output features [batch, seq_len, hidden_dim].</param>
    /// <param name="posEncoding">Positional encoding for memory [batch, seq_len, hidden_dim].</param>
    /// <returns>Class logits and box predictions for each query.</returns>
    public (Tensor<T> classLogits, Tensor<T> boxPreds) Forward(Tensor<T> memory, Tensor<T>? posEncoding = null)
    {
        int batch = memory.Shape[0];

        // Expand query embeddings for batch
        var queries = ExpandQueriesForBatch(_queryEmbed, batch);

        // Pass through decoder layers
        var output = queries;
        foreach (var layer in _layers)
        {
            output = layer.Forward(output, memory, posEncoding);
        }

        // Apply prediction heads
        var classLogits = ApplyClassHead(output);
        var boxPreds = ApplyBoxHead(output);

        return (classLogits, boxPreds);
    }

    /// <summary>
    /// Decodes raw outputs into detections.
    /// </summary>
    /// <remarks>
    /// Returns one result tuple per batch item to maintain proper batch separation.
    /// </remarks>
    public List<(float[] boxes, float[] scores, int[] classIds)> DecodeOutputs(
        Tensor<T> classLogits,
        Tensor<T> boxPreds,
        int imageHeight,
        int imageWidth)
    {
        var results = new List<(float[] boxes, float[] scores, int[] classIds)>();

        int batch = classLogits.Shape[0];
        int numQueries = classLogits.Shape[1];
        int numClasses = classLogits.Shape[2];

        for (int b = 0; b < batch; b++)
        {
            var batchBoxes = new List<float>();
            var batchScores = new List<float>();
            var batchClassIds = new List<int>();

            for (int q = 0; q < numQueries; q++)
            {
                // Get class scores via softmax
                double maxScore = 0;
                int maxClassId = 0;
                double sumExp = 0;

                // Compute softmax
                double maxLogit = double.NegativeInfinity;
                for (int c = 0; c < numClasses; c++)
                {
                    double logit = _numOps.ToDouble(classLogits[b, q, c]);
                    maxLogit = Math.Max(maxLogit, logit);
                }

                var probs = new double[numClasses];
                for (int c = 0; c < numClasses; c++)
                {
                    double logit = _numOps.ToDouble(classLogits[b, q, c]);
                    probs[c] = Math.Exp(logit - maxLogit);
                    sumExp += probs[c];
                }

                // Find max class (excluding background which is the last class)
                for (int c = 0; c < numClasses - 1; c++)
                {
                    double prob = probs[c] / sumExp;
                    if (prob > maxScore)
                    {
                        maxScore = prob;
                        maxClassId = c;
                    }
                }

                // Decode box (DETR outputs normalized cx, cy, w, h)
                double cx = Sigmoid(_numOps.ToDouble(boxPreds[b, q, 0]));
                double cy = Sigmoid(_numOps.ToDouble(boxPreds[b, q, 1]));
                double w = Sigmoid(_numOps.ToDouble(boxPreds[b, q, 2]));
                double h = Sigmoid(_numOps.ToDouble(boxPreds[b, q, 3]));

                // Convert to absolute coordinates
                cx *= imageWidth;
                cy *= imageHeight;
                w *= imageWidth;
                h *= imageHeight;

                // Convert to x1, y1, x2, y2
                float x1 = (float)Math.Max(0, cx - w / 2);
                float y1 = (float)Math.Max(0, cy - h / 2);
                float x2 = (float)Math.Min(imageWidth, cx + w / 2);
                float y2 = (float)Math.Min(imageHeight, cy + h / 2);

                batchBoxes.AddRange(new[] { x1, y1, x2, y2 });
                batchScores.Add((float)maxScore);
                batchClassIds.Add(maxClassId);
            }

            results.Add((batchBoxes.ToArray(), batchScores.ToArray(), batchClassIds.ToArray()));
        }

        return results;
    }

    /// <summary>
    /// Gets the number of parameters in the decoder.
    /// </summary>
    public long GetParameterCount()
    {
        long count = _numQueries * _hiddenDim; // Query embeddings

        foreach (var layer in _layers)
        {
            count += layer.GetParameterCount();
        }

        count += _classHead.GetParameterCount();
        count += _boxHead.GetParameterCount();

        return count;
    }

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numHeads);
        writer.Write(_numLayers);
        writer.Write(_numQueries);

        // Write query embeddings
        for (int i = 0; i < _queryEmbed.Length; i++)
        {
            writer.Write(_numOps.ToDouble(_queryEmbed[i]));
        }

        // Write decoder layers
        foreach (var layer in _layers)
        {
            layer.WriteParameters(writer);
        }

        // Write prediction heads
        _classHead.WriteParameters(writer);
        _boxHead.WriteParameters(writer);
    }

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numQueries = reader.ReadInt32();

        if (hiddenDim != _hiddenDim || numHeads != _numHeads ||
            numLayers != _numLayers || numQueries != _numQueries)
        {
            throw new InvalidOperationException(
                $"DETRDecoder configuration mismatch: expected hiddenDim={_hiddenDim}, numHeads={_numHeads}, " +
                $"numLayers={_numLayers}, numQueries={_numQueries}.");
        }

        // Read query embeddings
        for (int i = 0; i < _queryEmbed.Length; i++)
        {
            _queryEmbed[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        // Read decoder layers
        foreach (var layer in _layers)
        {
            layer.ReadParameters(reader);
        }

        // Read prediction heads
        _classHead.ReadParameters(reader);
        _boxHead.ReadParameters(reader);
    }

    private Tensor<T> InitializeQueryEmbeddings(int numQueries, int hiddenDim)
    {
        var embeddings = new Tensor<T>(new[] { numQueries, hiddenDim });

        // Xavier/Glorot initialization
        double scale = Math.Sqrt(2.0 / (numQueries + hiddenDim));
        var random = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < embeddings.Length; i++)
        {
            embeddings[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * scale);
        }

        return embeddings;
    }

    private Tensor<T> ExpandQueriesForBatch(Tensor<T> queries, int batch)
    {
        // Broadcast rather than copy: the learnable query embeddings must stay on the tape.
        int numQueries = queries.Shape[0];
        int hiddenDim = queries.Shape[1];
        return AiDotNetEngine.Current.TensorBroadcastTo(AiDotNetEngine.Current.Reshape(queries, new[] { 1, numQueries, hiddenDim }), new[] { batch, numQueries, hiddenDim });
    }

    private Tensor<T> ApplyClassHead(Tensor<T> output) => _classHead.ForwardTokens(output);

    private Tensor<T> ApplyBoxHead(Tensor<T> output) => _boxHead.ForwardTokens(output);

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    /// <inheritdoc />
    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        foreach (var child in _layers) yield return child;
        yield return _classHead;
        yield return _boxHead;
    }

    /// <inheritdoc />
    protected override IEnumerable<Tensor<T>> OwnParameterTensors()
    {
        yield return _queryEmbed;
    }
}

/// <summary>
/// Single decoder layer in DETR.
/// </summary>
internal class DecoderLayer<T> : CvParameterModule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly MultiHeadSelfAttention<T> _selfAttn;
    private readonly MultiHeadCrossAttention<T> _crossAttn;
    private readonly Dense<T> _ffn1;
    private readonly Dense<T> _ffn2;
    private readonly LayerNorm<T> _norm1;
    private readonly LayerNorm<T> _norm2;
    private readonly LayerNorm<T> _norm3;

    public DecoderLayer(int hiddenDim, int numHeads)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;

        _selfAttn = new MultiHeadSelfAttention<T>(hiddenDim, numHeads);
        _crossAttn = new MultiHeadCrossAttention<T>(hiddenDim, numHeads);
        _ffn1 = new Dense<T>(hiddenDim, hiddenDim * 4);
        _ffn2 = new Dense<T>(hiddenDim * 4, hiddenDim);
        _norm1 = new LayerNorm<T>(hiddenDim);
        _norm2 = new LayerNorm<T>(hiddenDim);
        _norm3 = new LayerNorm<T>(hiddenDim);
    }

    public Tensor<T> Forward(Tensor<T> queries, Tensor<T> memory, Tensor<T>? posEncoding)
    {
        // Self-attention among queries
        var selfAttnOut = _selfAttn.Forward(queries);
        var q1 = AddTensors(queries, selfAttnOut);
        q1 = _norm1.Forward(q1);

        // Cross-attention with encoder memory
        var crossAttnOut = _crossAttn.Forward(q1, memory, posEncoding);
        var q2 = AddTensors(q1, crossAttnOut);
        q2 = _norm2.Forward(q2);

        // FFN
        var ffnOut = ApplyFFN(q2);
        var output = AddTensors(q2, ffnOut);
        output = _norm3.Forward(output);

        return output;
    }

    public long GetParameterCount()
    {
        long count = 0;
        count += _selfAttn.GetParameterCount();
        count += _crossAttn.GetParameterCount();
        count += _ffn1.GetParameterCount();
        count += _ffn2.GetParameterCount();
        count += _norm1.GetParameterCount();
        count += _norm2.GetParameterCount();
        count += _norm3.GetParameterCount();
        return count;
    }

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numHeads);

        _selfAttn.WriteParameters(writer);
        _crossAttn.WriteParameters(writer);
        _ffn1.WriteParameters(writer);
        _ffn2.WriteParameters(writer);
        _norm1.WriteParameters(writer);
        _norm2.WriteParameters(writer);
        _norm3.WriteParameters(writer);
    }

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();

        if (hiddenDim != _hiddenDim || numHeads != _numHeads)
        {
            throw new InvalidOperationException(
                $"DecoderLayer configuration mismatch: expected hiddenDim={_hiddenDim}, numHeads={_numHeads}.");
        }

        _selfAttn.ReadParameters(reader);
        _crossAttn.ReadParameters(reader);
        _ffn1.ReadParameters(reader);
        _ffn2.ReadParameters(reader);
        _norm1.ReadParameters(reader);
        _norm2.ReadParameters(reader);
        _norm3.ReadParameters(reader);
    }

    private Tensor<T> ApplyFFN(Tensor<T> x)
        => _ffn2.ForwardTokens(AiDotNetEngine.Current.GELU(_ffn1.ForwardTokens(x)));

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return AiDotNetEngine.Current.TensorAdd(a, b);
    }

    /// <inheritdoc />
    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _selfAttn;
        yield return _crossAttn;
        yield return _ffn1;
        yield return _ffn2;
        yield return _norm1;
        yield return _norm2;
        yield return _norm3;
    }
}

/// <summary>
/// Multi-head cross-attention for DETR decoder.
/// </summary>
internal class MultiHeadCrossAttention<T> : CvParameterModule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly Dense<T> _queryProj;
    private readonly Dense<T> _keyProj;
    private readonly Dense<T> _valueProj;
    private readonly Dense<T> _outputProj;
    private readonly double _scale;

    public MultiHeadCrossAttention(int hiddenDim, int numHeads)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _headDim = hiddenDim / numHeads;
        _scale = 1.0 / Math.Sqrt(_headDim);

        _queryProj = new Dense<T>(hiddenDim, hiddenDim);
        _keyProj = new Dense<T>(hiddenDim, hiddenDim);
        _valueProj = new Dense<T>(hiddenDim, hiddenDim);
        _outputProj = new Dense<T>(hiddenDim, hiddenDim);
    }

    public Tensor<T> Forward(Tensor<T> queries, Tensor<T> memory, Tensor<T>? posEncoding)
    {
        // Keys see the positional encoding; values do not (DETR convention).
        var memoryWithPos = posEncoding is not null ? AiDotNetEngine.Current.TensorAdd(memory, posEncoding) : memory;

        var q = _queryProj.ForwardTokens(queries);
        var k = _keyProj.ForwardTokens(memoryWithPos);
        var v = _valueProj.ForwardTokens(memory);

        var attended = CvTensorOps<T>.MultiHeadAttention(q, k, v, _numHeads, _scale);
        return _outputProj.ForwardTokens(attended);
    }

    public long GetParameterCount()
    {
        return _queryProj.GetParameterCount() +
               _keyProj.GetParameterCount() +
               _valueProj.GetParameterCount() +
               _outputProj.GetParameterCount();
    }

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numHeads);

        _queryProj.WriteParameters(writer);
        _keyProj.WriteParameters(writer);
        _valueProj.WriteParameters(writer);
        _outputProj.WriteParameters(writer);
    }

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();

        if (hiddenDim != _hiddenDim || numHeads != _numHeads)
        {
            throw new InvalidOperationException(
                $"MultiHeadCrossAttention configuration mismatch: expected hiddenDim={_hiddenDim}, numHeads={_numHeads}.");
        }

        _queryProj.ReadParameters(reader);
        _keyProj.ReadParameters(reader);
        _valueProj.ReadParameters(reader);
        _outputProj.ReadParameters(reader);
    }

    /// <inheritdoc />
    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _queryProj;
        yield return _keyProj;
        yield return _valueProj;
        yield return _outputProj;
    }
}
