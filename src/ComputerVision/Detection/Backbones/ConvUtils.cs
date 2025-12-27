using AiDotNet.Tensors;

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
        var random = new Random(42);

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
        var random = new Random(42);

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

        // Compute attention scores [batch, seq_len, seq_len]
        var scores = new Tensor<T>(new[] { batch, seqLen, seqLen });

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    double dot = 0;
                    for (int d = 0; d < _dim; d++)
                    {
                        double qi = _numOps.ToDouble(q[b, i, d]);
                        double kj = _numOps.ToDouble(k[b, j, d]);
                        dot += qi * kj;
                    }
                    scores[b, i, j] = _numOps.FromDouble(dot * _scale);
                }
            }
        }

        // Softmax over last dimension
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                double maxScore = double.NegativeInfinity;
                for (int j = 0; j < seqLen; j++)
                {
                    maxScore = Math.Max(maxScore, _numOps.ToDouble(scores[b, i, j]));
                }

                double sumExp = 0;
                for (int j = 0; j < seqLen; j++)
                {
                    double exp = Math.Exp(_numOps.ToDouble(scores[b, i, j]) - maxScore);
                    sumExp += exp;
                }

                for (int j = 0; j < seqLen; j++)
                {
                    double exp = Math.Exp(_numOps.ToDouble(scores[b, i, j]) - maxScore);
                    scores[b, i, j] = _numOps.FromDouble(exp / sumExp);
                }
            }
        }

        // Apply attention to values
        var attnOut = new Tensor<T>(new[] { batch, seqLen, _dim });
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                for (int d = 0; d < _dim; d++)
                {
                    double sum = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        double attn = _numOps.ToDouble(scores[b, i, j]);
                        double vj = _numOps.ToDouble(v[b, j, d]);
                        sum += attn * vj;
                    }
                    attnOut[b, i, d] = _numOps.FromDouble(sum);
                }
            }
        }

        // Output projection
        return _outProj.Forward(attnOut);
    }

    public long GetParameterCount()
    {
        return 4 * _dim * _dim; // Q, K, V, Out projections
    }
}
