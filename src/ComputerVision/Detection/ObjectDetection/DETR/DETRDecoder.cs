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
internal class DETRDecoder<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numLayers;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _numQueries;
    private readonly List<DecoderLayer<T>> _layers;
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
        int numQueries = queries.Shape[0];
        int hiddenDim = queries.Shape[1];

        var expanded = new Tensor<T>(new[] { batch, numQueries, hiddenDim });

        for (int b = 0; b < batch; b++)
        {
            for (int q = 0; q < numQueries; q++)
            {
                for (int d = 0; d < hiddenDim; d++)
                {
                    expanded[b, q, d] = queries[q, d];
                }
            }
        }

        return expanded;
    }

    private Tensor<T> ApplyClassHead(Tensor<T> output)
    {
        int batch = output.Shape[0];
        int numQueries = output.Shape[1];
        int hiddenDim = output.Shape[2];
        int numClasses = _classHead.OutputSize;

        var result = new Tensor<T>(new[] { batch, numQueries, numClasses });

        for (int b = 0; b < batch; b++)
        {
            for (int q = 0; q < numQueries; q++)
            {
                // Extract query features
                var queryFeat = new Tensor<T>(new[] { 1, hiddenDim });
                for (int d = 0; d < hiddenDim; d++)
                {
                    queryFeat[0, d] = output[b, q, d];
                }

                // Apply class head
                var classOut = _classHead.Forward(queryFeat);

                // Copy to result
                for (int c = 0; c < numClasses; c++)
                {
                    result[b, q, c] = classOut[0, c];
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyBoxHead(Tensor<T> output)
    {
        int batch = output.Shape[0];
        int numQueries = output.Shape[1];
        int hiddenDim = output.Shape[2];

        var result = new Tensor<T>(new[] { batch, numQueries, 4 });

        for (int b = 0; b < batch; b++)
        {
            for (int q = 0; q < numQueries; q++)
            {
                // Extract query features
                var queryFeat = new Tensor<T>(new[] { 1, hiddenDim });
                for (int d = 0; d < hiddenDim; d++)
                {
                    queryFeat[0, d] = output[b, q, d];
                }

                // Apply box head
                var boxOut = _boxHead.Forward(queryFeat);

                // Copy to result
                for (int i = 0; i < 4; i++)
                {
                    result[b, q, i] = boxOut[0, i];
                }
            }
        }

        return result;
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
}

/// <summary>
/// Single decoder layer in DETR.
/// </summary>
internal class DecoderLayer<T>
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

    private Tensor<T> ApplyFFN(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int hiddenDim = x.Shape[2];
        int ffnDim = _ffn1.OutputSize;

        var result = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                // Extract features
                var feat = new Tensor<T>(new[] { 1, hiddenDim });
                for (int d = 0; d < hiddenDim; d++)
                {
                    feat[0, d] = x[b, s, d];
                }

                // FFN1 with GELU
                var h = _ffn1.Forward(feat);
                for (int d = 0; d < ffnDim; d++)
                {
                    double val = _numOps.ToDouble(h[0, d]);
                    h[0, d] = _numOps.FromDouble(GELU(val));
                }

                // FFN2
                var output = _ffn2.Forward(h);

                // Copy to result
                for (int d = 0; d < hiddenDim; d++)
                {
                    result[b, s, d] = output[0, d];
                }
            }
        }

        return result;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = _numOps.Add(a[i], b[i]);
        }
        return result;
    }
    private static double GELU(double x)
    {
        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        double c = Math.Sqrt(2.0 / Math.PI);
        return 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
    }
}

/// <summary>
/// Multi-head cross-attention for DETR decoder.
/// </summary>
internal class MultiHeadCrossAttention<T>
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
        int batch = queries.Shape[0];
        int queryLen = queries.Shape[1];
        int memoryLen = memory.Shape[1];

        // Add positional encoding to memory if provided
        var memoryWithPos = memory;
        if (posEncoding is not null)
        {
            memoryWithPos = new Tensor<T>(memory.Shape);
            for (int i = 0; i < memory.Length; i++)
            {
                memoryWithPos[i] = _numOps.Add(memory[i], posEncoding[i]);
            }
        }

        // Project queries, keys, values
        var q = ProjectSequence(queries, _queryProj);
        var k = ProjectSequence(memoryWithPos, _keyProj);
        var v = ProjectSequence(memory, _valueProj);

        // Compute attention
        var attnOutput = ComputeAttention(q, k, v, batch, queryLen, memoryLen);

        // Project output
        var output = ProjectSequence(attnOutput, _outputProj);

        return output;
    }

    public long GetParameterCount()
    {
        return _queryProj.GetParameterCount() +
               _keyProj.GetParameterCount() +
               _valueProj.GetParameterCount() +
               _outputProj.GetParameterCount();
    }

    private Tensor<T> ProjectSequence(Tensor<T> x, Dense<T> proj)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int dim = x.Shape[2];
        int outDim = proj.OutputSize;

        var result = new Tensor<T>(new[] { batch, seqLen, outDim });

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                var feat = new Tensor<T>(new[] { 1, dim });
                for (int d = 0; d < dim; d++)
                {
                    feat[0, d] = x[b, s, d];
                }

                var projected = proj.Forward(feat);

                for (int d = 0; d < outDim; d++)
                {
                    result[b, s, d] = projected[0, d];
                }
            }
        }

        return result;
    }

    private Tensor<T> ComputeAttention(Tensor<T> q, Tensor<T> k, Tensor<T> v, int batch, int queryLen, int keyLen)
    {
        // Simplified attention computation for each head
        var output = new Tensor<T>(new[] { batch, queryLen, _hiddenDim });

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                int headOffset = h * _headDim;

                // Compute attention scores for this head
                var scores = new double[queryLen, keyLen];
                for (int i = 0; i < queryLen; i++)
                {
                    for (int j = 0; j < keyLen; j++)
                    {
                        double score = 0;
                        for (int d = 0; d < _headDim; d++)
                        {
                            score += _numOps.ToDouble(q[b, i, headOffset + d]) *
                                     _numOps.ToDouble(k[b, j, headOffset + d]);
                        }
                        scores[i, j] = score * _scale;
                    }
                }

                // Softmax over keys
                for (int i = 0; i < queryLen; i++)
                {
                    double maxScore = double.NegativeInfinity;
                    for (int j = 0; j < keyLen; j++)
                    {
                        maxScore = Math.Max(maxScore, scores[i, j]);
                    }

                    double sumExp = 0;
                    for (int j = 0; j < keyLen; j++)
                    {
                        scores[i, j] = Math.Exp(scores[i, j] - maxScore);
                        sumExp += scores[i, j];
                    }

                    for (int j = 0; j < keyLen; j++)
                    {
                        scores[i, j] /= sumExp;
                    }
                }

                // Apply attention to values
                for (int i = 0; i < queryLen; i++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        double value = 0;
                        for (int j = 0; j < keyLen; j++)
                        {
                            value += scores[i, j] * _numOps.ToDouble(v[b, j, headOffset + d]);
                        }
                        output[b, i, headOffset + d] = _numOps.FromDouble(value);
                    }
                }
            }
        }

        return output;
    }
}
