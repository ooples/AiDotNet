using System.IO;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Simple 2D convolution layer that works with variable input sizes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Unlike ConvolutionalLayer which requires input dimensions at construction,
/// this class computes convolution dynamically based on input tensor shape.
/// </remarks>
internal class Conv2D<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Tensor<T> _weights;
    private readonly Tensor<T>? _bias;
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;

    /// <summary>
    /// Gets the weight tensor for this convolutional layer.
    /// </summary>
    public Tensor<T> Weights => _weights;

    /// <summary>
    /// Gets the bias tensor for this convolutional layer, or null if bias is not used.
    /// </summary>
    public Tensor<T>? Bias => _bias;

    public Conv2D(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, bool useBias = true)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        // Initialize weights [outChannels, inChannels, kernelSize, kernelSize]
        _weights = new Tensor<T>(new[] { outChannels, inChannels, kernelSize, kernelSize });
        InitializeWeights();

        if (useBias)
        {
            _bias = new Tensor<T>(new[] { outChannels });
        }
    }

    private void InitializeWeights()
    {
        // He initialization
        double scale = Math.Sqrt(2.0 / (_inChannels * _kernelSize * _kernelSize));
        var random = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < _weights.Length; i++)
        {
            double val = random.NextDouble() * 2 * scale - scale;
            _weights[i] = _numOps.FromDouble(val);
        }
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // input: [batch, channels, height, width]
        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        if (inChannels != _inChannels)
        {
            throw new ArgumentException(
                $"Input has {inChannels} channels but layer expects {_inChannels} channels.",
                nameof(input));
        }

        int outHeight = (inHeight + 2 * _padding - _kernelSize) / _stride + 1;
        int outWidth = (inWidth + 2 * _padding - _kernelSize) / _stride + 1;

        var output = new Tensor<T>(new[] { batch, _outChannels, outHeight, outWidth });

        for (int n = 0; n < batch; n++)
        {
            for (int oc = 0; oc < _outChannels; oc++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        double sum = 0;

                        for (int ic = 0; ic < _inChannels; ic++)
                        {
                            for (int kh = 0; kh < _kernelSize; kh++)
                            {
                                for (int kw = 0; kw < _kernelSize; kw++)
                                {
                                    int ih = oh * _stride - _padding + kh;
                                    int iw = ow * _stride - _padding + kw;

                                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                    {
                                        double inputVal = _numOps.ToDouble(input[n, ic, ih, iw]);
                                        double weightVal = _numOps.ToDouble(_weights[oc, ic, kh, kw]);
                                        sum += inputVal * weightVal;
                                    }
                                }
                            }
                        }

                        if (_bias is not null)
                        {
                            sum += _numOps.ToDouble(_bias[oc]);
                        }

                        output[n, oc, oh, ow] = _numOps.FromDouble(sum);
                    }
                }
            }
        }

        return output;
    }

    public long GetParameterCount()
    {
        long count = _outChannels * _inChannels * _kernelSize * _kernelSize;
        if (_bias is not null)
        {
            count += _outChannels;
        }
        return count;
    }

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    public void WriteParameters(BinaryWriter writer)
    {
        // Write layer configuration
        writer.Write(_inChannels);
        writer.Write(_outChannels);
        writer.Write(_kernelSize);
        writer.Write(_stride);
        writer.Write(_padding);
        writer.Write(_bias is not null);

        // Write weights
        for (int i = 0; i < _weights.Length; i++)
        {
            writer.Write(_numOps.ToDouble(_weights[i]));
        }

        // Write bias if present
        if (_bias is not null)
        {
            for (int i = 0; i < _bias.Length; i++)
            {
                writer.Write(_numOps.ToDouble(_bias[i]));
            }
        }
    }

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    public void ReadParameters(BinaryReader reader)
    {
        // Read and verify configuration (already set in constructor)
        int inChannels = reader.ReadInt32();
        int outChannels = reader.ReadInt32();
        int kernelSize = reader.ReadInt32();
        int stride = reader.ReadInt32();
        int padding = reader.ReadInt32();
        bool hasBias = reader.ReadBoolean();

        if (inChannels != _inChannels || outChannels != _outChannels ||
            kernelSize != _kernelSize || stride != _stride || padding != _padding)
        {
            throw new InvalidOperationException("Conv2D configuration mismatch during deserialization.");
        }

        // Read weights
        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        // Read bias if present
        if (hasBias && _bias is not null)
        {
            for (int i = 0; i < _bias.Length; i++)
            {
                _bias[i] = _numOps.FromDouble(reader.ReadDouble());
            }
        }
    }
}

/// <summary>
/// Simple dense (fully connected) layer that works with variable input sizes.
/// </summary>
internal class Dense<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Tensor<T> _weights;
    private readonly Tensor<T> _bias;
    private readonly int _inFeatures;
    private readonly int _outFeatures;

    /// <summary>
    /// Gets the weight tensor for this dense layer.
    /// </summary>
    public Tensor<T> Weights => _weights;

    /// <summary>
    /// Gets the bias tensor for this dense layer.
    /// </summary>
    public Tensor<T> Bias => _bias;

    public Dense(int inFeatures, int outFeatures)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _inFeatures = inFeatures;
        _outFeatures = outFeatures;

        // Initialize weights [outFeatures, inFeatures]
        _weights = new Tensor<T>(new[] { outFeatures, inFeatures });
        _bias = new Tensor<T>(new[] { outFeatures });
        InitializeWeights();
    }

    /// <summary>
    /// Gets the output feature dimension.
    /// </summary>
    public int OutputSize => _outFeatures;

    /// <summary>
    /// Gets the total parameter count for this layer.
    /// </summary>
    public long GetParameterCount()
    {
        return (long)_inFeatures * _outFeatures + _outFeatures; // weights + bias
    }

    private void InitializeWeights()
    {
        // He initialization
        double scale = Math.Sqrt(2.0 / _inFeatures);
        var random = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < _weights.Length; i++)
        {
            double val = random.NextDouble() * 2 * scale - scale;
            _weights[i] = _numOps.FromDouble(val);
        }
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // input: [batch, ..., inFeatures] - operates on last dimension
        int[] inputShape = input.Shape;
        int lastDim = inputShape[^1];

        if (lastDim != _inFeatures)
        {
            throw new ArgumentException(
                $"Input has {lastDim} features but layer expects {_inFeatures} features.",
                nameof(input));
        }

        int totalElements = input.Length / lastDim;

        // Flatten to [batch_total, inFeatures]
        var output = new Tensor<T>(ReplaceLastDim(inputShape, _outFeatures));

        for (int i = 0; i < totalElements; i++)
        {
            for (int o = 0; o < _outFeatures; o++)
            {
                double sum = _numOps.ToDouble(_bias[o]);

                for (int j = 0; j < _inFeatures; j++)
                {
                    int inputIdx = i * lastDim + j;
                    double inputVal = _numOps.ToDouble(input[inputIdx]);
                    double weightVal = _numOps.ToDouble(_weights[o, j]);
                    sum += inputVal * weightVal;
                }

                int outputIdx = i * _outFeatures + o;
                output[outputIdx] = _numOps.FromDouble(sum);
            }
        }

        return output;
    }

    private static int[] ReplaceLastDim(int[] shape, int newLastDim)
    {
        var newShape = new int[shape.Length];
        Array.Copy(shape, newShape, shape.Length);
        newShape[^1] = newLastDim;
        return newShape;
    }

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    public void WriteParameters(BinaryWriter writer)
    {
        // Write layer configuration
        writer.Write(_inFeatures);
        writer.Write(_outFeatures);

        // Write weights
        for (int i = 0; i < _weights.Length; i++)
        {
            writer.Write(_numOps.ToDouble(_weights[i]));
        }

        // Write bias
        for (int i = 0; i < _bias.Length; i++)
        {
            writer.Write(_numOps.ToDouble(_bias[i]));
        }
    }

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    public void ReadParameters(BinaryReader reader)
    {
        // Read and verify configuration
        int inFeatures = reader.ReadInt32();
        int outFeatures = reader.ReadInt32();

        if (inFeatures != _inFeatures || outFeatures != _outFeatures)
        {
            throw new InvalidOperationException("Dense configuration mismatch during deserialization.");
        }

        // Read weights
        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        // Read bias
        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = _numOps.FromDouble(reader.ReadDouble());
        }
    }
}

/// <summary>
/// Simple multi-head self-attention for transformers.
/// </summary>
internal class MultiHeadSelfAttention<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Dense<T> _queryProj;
    private readonly Dense<T> _keyProj;
    private readonly Dense<T> _valueProj;
    private readonly Dense<T> _outProj;
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly double _scale;

    public MultiHeadSelfAttention(int dim, int numHeads)
    {
        if (dim % numHeads != 0)
        {
            throw new ArgumentException(
                $"Dimension {dim} must be divisible by number of heads {numHeads}.",
                nameof(dim));
        }

        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _dim = dim;
        _numHeads = numHeads;
        _headDim = dim / numHeads;
        _scale = 1.0 / Math.Sqrt(_headDim);

        _queryProj = new Dense<T>(dim, dim);
        _keyProj = new Dense<T>(dim, dim);
        _valueProj = new Dense<T>(dim, dim);
        _outProj = new Dense<T>(dim, dim);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // input: [batch, seq_len, dim]
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];

        // Project Q, K, V
        var q = _queryProj.Forward(input);
        var k = _keyProj.Forward(input);
        var v = _valueProj.Forward(input);

        // Reshape for multi-head attention: [batch, seq_len, dim] -> [batch, numHeads, seq_len, headDim]
        // Compute attention scores per head: [batch, numHeads, seq_len, seq_len]
        var scores = new Tensor<T>(new[] { batch, _numHeads, seqLen, seqLen });

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                int headOffset = h * _headDim;
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        double dot = 0;
                        for (int d = 0; d < _headDim; d++)
                        {
                            double qi = _numOps.ToDouble(q[b, i, headOffset + d]);
                            double kj = _numOps.ToDouble(k[b, j, headOffset + d]);
                            dot += qi * kj;
                        }
                        scores[b, h, i, j] = _numOps.FromDouble(dot * _scale);
                    }
                }
            }
        }

        // Softmax per head over last dimension
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    double maxScore = double.NegativeInfinity;
                    for (int j = 0; j < seqLen; j++)
                    {
                        maxScore = Math.Max(maxScore, _numOps.ToDouble(scores[b, h, i, j]));
                    }

                    double sumExp = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        double exp = Math.Exp(_numOps.ToDouble(scores[b, h, i, j]) - maxScore);
                        sumExp += exp;
                    }

                    for (int j = 0; j < seqLen; j++)
                    {
                        double exp = Math.Exp(_numOps.ToDouble(scores[b, h, i, j]) - maxScore);
                        scores[b, h, i, j] = _numOps.FromDouble(exp / sumExp);
                    }
                }
            }
        }

        // Apply attention to values per head and concatenate
        var attnOut = new Tensor<T>(new[] { batch, seqLen, _dim });
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                int headOffset = h * _headDim;
                for (int i = 0; i < seqLen; i++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        double sum = 0;
                        for (int j = 0; j < seqLen; j++)
                        {
                            double attn = _numOps.ToDouble(scores[b, h, i, j]);
                            double vj = _numOps.ToDouble(v[b, j, headOffset + d]);
                            sum += attn * vj;
                        }
                        attnOut[b, i, headOffset + d] = _numOps.FromDouble(sum);
                    }
                }
            }
        }

        // Output projection
        return _outProj.Forward(attnOut);
    }

    public long GetParameterCount()
    {
        // Each Dense layer has dim*dim weights plus dim bias terms
        return 4 * (_dim * _dim + _dim); // Q, K, V, Out projections with biases
    }

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    public void WriteParameters(BinaryWriter writer)
    {
        // Write configuration
        writer.Write(_dim);
        writer.Write(_numHeads);

        // Write projection layer parameters
        _queryProj.WriteParameters(writer);
        _keyProj.WriteParameters(writer);
        _valueProj.WriteParameters(writer);
        _outProj.WriteParameters(writer);
    }

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    public void ReadParameters(BinaryReader reader)
    {
        // Read and verify configuration
        int dim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();

        if (dim != _dim || numHeads != _numHeads)
        {
            throw new InvalidOperationException(
                $"MultiHeadSelfAttention configuration mismatch: expected dim={_dim}, numHeads={_numHeads}, got dim={dim}, numHeads={numHeads}.");
        }

        // Read projection layer parameters
        _queryProj.ReadParameters(reader);
        _keyProj.ReadParameters(reader);
        _valueProj.ReadParameters(reader);
        _outProj.ReadParameters(reader);
    }
}
