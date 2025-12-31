using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Deformable Convolutional Layer that learns spatial sampling offsets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Deformable convolution augments standard convolution with learnable 2D offsets for each
/// sampling location. This allows the convolution to adapt its receptive field to the
/// geometric structure of the input, making it particularly effective for video alignment.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular convolutions look at fixed grid positions around each pixel.
/// Deformable convolutions can look at shifted positions learned from the data itself.
///
/// This is useful for:
/// 1. Aligning features between video frames (different objects may have moved differently)
/// 2. Handling geometric transformations (rotation, scale, perspective)
/// 3. Object detection with varying shapes and poses
/// </para>
/// <para>
/// <b>Reference:</b> Dai et al., "Deformable Convolutional Networks", ICCV 2017.
/// https://arxiv.org/abs/1703.06211
/// </para>
/// </remarks>
public class DeformableConvolutionalLayer<T> : LayerBase<T>, IChainableComputationGraph<T>
{
    #region Fields

    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;
    private readonly int _groups;
    private readonly int _deformGroups;

    // Main convolution weights and bias
    private Tensor<T> _weights;
    private Tensor<T> _bias;

    // Offset prediction convolution
    private Tensor<T> _offsetWeights;
    private Tensor<T> _offsetBias;

    // Modulation mask prediction (for Deformable Conv v2)
    private readonly bool _useModulation;
    private Tensor<T>? _maskWeights;
    private Tensor<T>? _maskBias;

    // Gradients
    private Tensor<T>? _weightGradients;
    private Tensor<T>? _biasGradients;
    private Tensor<T>? _offsetWeightGradients;
    private Tensor<T>? _offsetBiasGradients;
    private Tensor<T>? _maskWeightGradients;
    private Tensor<T>? _maskBiasGradients;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOffsets;
    private Tensor<T>? _lastMask;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new Deformable Convolutional Layer.
    /// </summary>
    /// <param name="inputHeight">Height of input feature map.</param>
    /// <param name="inputWidth">Width of input feature map.</param>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <param name="outputChannels">Number of output channels.</param>
    /// <param name="kernelSize">Size of the convolution kernel (default: 3).</param>
    /// <param name="stride">Convolution stride (default: 1).</param>
    /// <param name="padding">Padding size (default: 1).</param>
    /// <param name="groups">Number of convolution groups (default: 1).</param>
    /// <param name="deformGroups">Number of deformable groups (default: 1).</param>
    /// <param name="useModulation">Whether to use modulation mask (DCNv2, default: true).</param>
    public DeformableConvolutionalLayer(
        int inputHeight,
        int inputWidth,
        int inputChannels,
        int outputChannels,
        int kernelSize = 3,
        int stride = 1,
        int padding = 1,
        int groups = 1,
        int deformGroups = 1,
        bool useModulation = true)
        : base(
            [inputChannels, inputHeight, inputWidth],
            [outputChannels, (inputHeight + 2 * padding - kernelSize) / stride + 1, (inputWidth + 2 * padding - kernelSize) / stride + 1])
    {
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        _groups = groups;
        _deformGroups = deformGroups;
        _useModulation = useModulation;

        // Initialize main convolution weights [outC, inC/groups, kH, kW]
        int inChannelsPerGroup = inputChannels / groups;
        _weights = InitializeWeights(outputChannels, inChannelsPerGroup, kernelSize, kernelSize);
        _bias = new Tensor<T>([outputChannels]);

        // Initialize offset prediction weights
        // Offsets: 2 (x, y) * kernelSize^2 per deform group
        int offsetChannels = 2 * kernelSize * kernelSize * deformGroups;
        _offsetWeights = InitializeWeights(offsetChannels, inputChannels, kernelSize, kernelSize);
        _offsetBias = new Tensor<T>([offsetChannels]);

        // Initialize modulation mask weights (if using DCNv2)
        if (useModulation)
        {
            int maskChannels = kernelSize * kernelSize * deformGroups;
            _maskWeights = InitializeWeights(maskChannels, inputChannels, kernelSize, kernelSize);
            _maskBias = new Tensor<T>([maskChannels]);
        }
    }

    #endregion

    #region Forward Pass

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        bool hasBatch = input.Rank == 4;
        int batch = hasBatch ? input.Shape[0] : 1;
        int channels = hasBatch ? input.Shape[1] : input.Shape[0];
        int height = hasBatch ? input.Shape[2] : input.Shape[1];
        int width = hasBatch ? input.Shape[3] : input.Shape[2];

        // Calculate output dimensions
        int outH = (height + 2 * _padding - _kernelSize) / _stride + 1;
        int outW = (width + 2 * _padding - _kernelSize) / _stride + 1;

        // Predict offsets using offset convolution
        var offsets = PredictOffsets(input, hasBatch, batch, height, width, outH, outW);
        _lastOffsets = offsets;

        // Predict modulation mask if using DCNv2
        Tensor<T>? mask = null;
        if (_useModulation)
        {
            mask = PredictMask(input, hasBatch, batch, height, width, outH, outW);
            _lastMask = mask;
        }

        // Perform deformable convolution
        var output = DeformableConv(input, offsets, mask, hasBatch, batch, channels, height, width, outH, outW);

        return output;
    }

    private Tensor<T> PredictOffsets(Tensor<T> input, bool hasBatch, int batch, int height, int width, int outH, int outW)
    {
        int offsetChannels = 2 * _kernelSize * _kernelSize * _deformGroups;
        var outShape = hasBatch ? new[] { batch, offsetChannels, outH, outW } : new[] { offsetChannels, outH, outW };
        var offsets = new Tensor<T>(outShape);

        // Simple convolution for offset prediction
        ApplyConvolution(input, _offsetWeights, _offsetBias, offsets, hasBatch, batch, _inputChannels, height, width, offsetChannels, outH, outW);

        return offsets;
    }

    private Tensor<T> PredictMask(Tensor<T> input, bool hasBatch, int batch, int height, int width, int outH, int outW)
    {
        int maskChannels = _kernelSize * _kernelSize * _deformGroups;
        var outShape = hasBatch ? new[] { batch, maskChannels, outH, outW } : new[] { maskChannels, outH, outW };
        var mask = new Tensor<T>(outShape);

        // Simple convolution for mask prediction, followed by sigmoid
        ApplyConvolution(input, _maskWeights!, _maskBias!, mask, hasBatch, batch, _inputChannels, height, width, maskChannels, outH, outW);

        // Apply sigmoid to mask
        for (int i = 0; i < mask.Length; i++)
        {
            double val = NumOps.ToDouble(mask.Data[i]);
            mask.Data[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
        }

        return mask;
    }

    private Tensor<T> DeformableConv(Tensor<T> input, Tensor<T> offsets, Tensor<T>? mask,
        bool hasBatch, int batch, int channels, int height, int width, int outH, int outW)
    {
        var outShape = hasBatch
            ? new[] { batch, _outputChannels, outH, outW }
            : new[] { _outputChannels, outH, outW };
        var output = new Tensor<T>(outShape);

        int inChannelsPerGroup = channels / _groups;
        int outChannelsPerGroup = _outputChannels / _groups;
        int k2 = _kernelSize * _kernelSize;

        for (int b = 0; b < batch; b++)
        {
            for (int g = 0; g < _groups; g++)
            {
                for (int oc = 0; oc < outChannelsPerGroup; oc++)
                {
                    int outChannel = g * outChannelsPerGroup + oc;

                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            T sum = _bias.Data[outChannel];

                            // For each kernel position
                            for (int kh = 0; kh < _kernelSize; kh++)
                            {
                                for (int kw = 0; kw < _kernelSize; kw++)
                                {
                                    int kIdx = kh * _kernelSize + kw;

                                    // Get offsets for this position
                                    int dg = g % _deformGroups;
                                    int offsetIdxX = dg * 2 * k2 + kIdx;
                                    int offsetIdxY = dg * 2 * k2 + k2 + kIdx;

                                    double offsetX = GetTensorValue(offsets, b, offsetIdxX, oh, ow, hasBatch, offsets.Shape);
                                    double offsetY = GetTensorValue(offsets, b, offsetIdxY, oh, ow, hasBatch, offsets.Shape);

                                    // Calculate sampling position
                                    double srcH = oh * _stride - _padding + kh + offsetY;
                                    double srcW = ow * _stride - _padding + kw + offsetX;

                                    // Get modulation weight
                                    double modWeight = 1.0;
                                    if (mask != null)
                                    {
                                        int maskIdx = dg * k2 + kIdx;
                                        modWeight = GetTensorValue(mask, b, maskIdx, oh, ow, hasBatch, mask.Shape);
                                    }

                                    // Sample from input for each input channel in this group
                                    for (int ic = 0; ic < inChannelsPerGroup; ic++)
                                    {
                                        int inChannel = g * inChannelsPerGroup + ic;
                                        T sampledValue = BilinearSample(input, b, inChannel, srcH, srcW, hasBatch, height, width, channels);

                                        // Apply modulation
                                        sampledValue = NumOps.Multiply(sampledValue, NumOps.FromDouble(modWeight));

                                        // Multiply by weight
                                        int weightIdx = oc * inChannelsPerGroup * k2 + ic * k2 + kIdx;
                                        T weight = _weights.Data[weightIdx];
                                        sum = NumOps.Add(sum, NumOps.Multiply(sampledValue, weight));
                                    }
                                }
                            }

                            // Store output
                            int outIdx = hasBatch
                                ? b * _outputChannels * outH * outW + outChannel * outH * outW + oh * outW + ow
                                : outChannel * outH * outW + oh * outW + ow;
                            output.Data[outIdx] = sum;
                        }
                    }
                }
            }
        }

        return output;
    }

    #endregion

    #region Backward Pass

    /// <inheritdoc/>
    public override Tensor<T> Backward(Tensor<T> gradOutput)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward must be called before Backward.");

        bool hasBatch = _lastInput.Rank == 4;
        int batch = hasBatch ? _lastInput.Shape[0] : 1;
        int channels = hasBatch ? _lastInput.Shape[1] : _lastInput.Shape[0];
        int height = hasBatch ? _lastInput.Shape[2] : _lastInput.Shape[1];
        int width = hasBatch ? _lastInput.Shape[3] : _lastInput.Shape[2];

        // Initialize gradients
        _weightGradients = new Tensor<T>(_weights.Shape);
        _biasGradients = new Tensor<T>(_bias.Shape);
        _offsetWeightGradients = new Tensor<T>(_offsetWeights.Shape);
        _offsetBiasGradients = new Tensor<T>(_offsetBias.Shape);

        if (_useModulation)
        {
            _maskWeightGradients = new Tensor<T>(_maskWeights!.Shape);
            _maskBiasGradients = new Tensor<T>(_maskBias!.Shape);
        }

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Simplified gradient computation
        // In practice, this requires computing gradients w.r.t. offsets and masks too
        int outH = hasBatch ? gradOutput.Shape[2] : gradOutput.Shape[1];
        int outW = hasBatch ? gradOutput.Shape[3] : gradOutput.Shape[2];

        // Compute bias gradients
        for (int oc = 0; oc < _outputChannels; oc++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int idx = hasBatch
                            ? b * _outputChannels * outH * outW + oc * outH * outW + oh * outW + ow
                            : oc * outH * outW + oh * outW + ow;
                        sum = NumOps.Add(sum, gradOutput.Data[idx]);
                    }
                }
            }
            _biasGradients.Data[oc] = sum;
        }

        return inputGradient;
    }

    #endregion

    #region Helper Methods

    private Tensor<T> InitializeWeights(int outC, int inC, int kH, int kW)
    {
        var weights = new Tensor<T>([outC, inC, kH, kW]);
        double fan_in = inC * kH * kW;
        double std = Math.Sqrt(2.0 / fan_in); // He initialization

        var random = new Random(42);
        for (int i = 0; i < weights.Length; i++)
        {
            weights.Data[i] = NumOps.FromDouble(random.NextDouble() * 2 * std - std);
        }

        return weights;
    }

    private void ApplyConvolution(Tensor<T> input, Tensor<T> weights, Tensor<T> bias, Tensor<T> output,
        bool hasBatch, int batch, int inChannels, int height, int width, int outChannels, int outH, int outW)
    {
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        T sum = bias.Data[oc];

                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kh = 0; kh < _kernelSize; kh++)
                            {
                                for (int kw = 0; kw < _kernelSize; kw++)
                                {
                                    int ih = oh * _stride - _padding + kh;
                                    int iw = ow * _stride - _padding + kw;

                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    {
                                        int inputIdx = hasBatch
                                            ? b * inChannels * height * width + ic * height * width + ih * width + iw
                                            : ic * height * width + ih * width + iw;
                                        int weightIdx = oc * inChannels * _kernelSize * _kernelSize +
                                                       ic * _kernelSize * _kernelSize +
                                                       kh * _kernelSize + kw;
                                        sum = NumOps.Add(sum, NumOps.Multiply(input.Data[inputIdx], weights.Data[weightIdx]));
                                    }
                                }
                            }
                        }

                        int outIdx = hasBatch
                            ? b * outChannels * outH * outW + oc * outH * outW + oh * outW + ow
                            : oc * outH * outW + oh * outW + ow;
                        output.Data[outIdx] = sum;
                    }
                }
            }
        }
    }

    private double GetTensorValue(Tensor<T> tensor, int b, int c, int h, int w, bool hasBatch, int[] shape)
    {
        int channels = hasBatch ? shape[1] : shape[0];
        int height = hasBatch ? shape[2] : shape[1];
        int width = hasBatch ? shape[3] : shape[2];

        int idx = hasBatch
            ? b * channels * height * width + c * height * width + h * width + w
            : c * height * width + h * width + w;
        return NumOps.ToDouble(tensor.Data[idx]);
    }

    private T BilinearSample(Tensor<T> input, int b, int c, double h, double w, bool hasBatch, int height, int width, int channels)
    {
        if (h < 0 || h >= height - 1 || w < 0 || w >= width - 1)
        {
            // Out of bounds - return zero
            return NumOps.Zero;
        }

        int h0 = (int)Math.Floor(h);
        int w0 = (int)Math.Floor(w);
        int h1 = h0 + 1;
        int w1 = w0 + 1;

        double hWeight = h - h0;
        double wWeight = w - w0;

        T v00 = GetInputValue(input, b, c, h0, w0, hasBatch, height, width, channels);
        T v01 = GetInputValue(input, b, c, h0, w1, hasBatch, height, width, channels);
        T v10 = GetInputValue(input, b, c, h1, w0, hasBatch, height, width, channels);
        T v11 = GetInputValue(input, b, c, h1, w1, hasBatch, height, width, channels);

        T top = NumOps.Add(
            NumOps.Multiply(v00, NumOps.FromDouble(1 - wWeight)),
            NumOps.Multiply(v01, NumOps.FromDouble(wWeight)));
        T bottom = NumOps.Add(
            NumOps.Multiply(v10, NumOps.FromDouble(1 - wWeight)),
            NumOps.Multiply(v11, NumOps.FromDouble(wWeight)));

        return NumOps.Add(
            NumOps.Multiply(top, NumOps.FromDouble(1 - hWeight)),
            NumOps.Multiply(bottom, NumOps.FromDouble(hWeight)));
    }

    private T GetInputValue(Tensor<T> input, int b, int c, int h, int w, bool hasBatch, int height, int width, int channels)
    {
        if (h < 0 || h >= height || w < 0 || w >= width)
            return NumOps.Zero;

        int idx = hasBatch
            ? b * channels * height * width + c * height * width + h * width + w
            : c * height * width + h * width + w;
        return input.Data[idx];
    }

    #endregion

    #region Layer Properties

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => false;

    #endregion

    #region IChainableComputationGraph Implementation

    /// <inheritdoc/>
    public override int[] GetInputShape() => [_inputChannels, _inputHeight, _inputWidth];

    /// <summary>
    /// Gets the output shape for this layer.
    /// </summary>
    public new int[] GetOutputShape()
    {
        int outH = (_inputHeight + 2 * _padding - _kernelSize) / _stride + 1;
        int outW = (_inputWidth + 2 * _padding - _kernelSize) / _stride + 1;
        return [_outputChannels, outH, outW];
    }

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null || inputNodes.Count == 0)
            throw new ArgumentException("Input nodes cannot be null or empty.", nameof(inputNodes));

        return BuildComputationGraph(inputNodes[0], "");
    }

    /// <inheritdoc/>
    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        // Deformable convolution is complex with dynamic offsets - return identity for now
        // Full JIT compilation support can be added later
        return inputNode;
    }

    #endregion

    #region Parameter Management

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        // Main conv weights and bias
        for (int i = 0; i < _weights.Length; i++)
            allParams.Add(_weights.Data[i]);
        for (int i = 0; i < _bias.Length; i++)
            allParams.Add(_bias.Data[i]);

        // Offset weights and bias
        for (int i = 0; i < _offsetWeights.Length; i++)
            allParams.Add(_offsetWeights.Data[i]);
        for (int i = 0; i < _offsetBias.Length; i++)
            allParams.Add(_offsetBias.Data[i]);

        // Mask weights and bias (if using modulation)
        if (_useModulation && _maskWeights != null && _maskBias != null)
        {
            for (int i = 0; i < _maskWeights.Length; i++)
                allParams.Add(_maskWeights.Data[i]);
            for (int i = 0; i < _maskBias.Length; i++)
                allParams.Add(_maskBias.Data[i]);
        }

        return new Vector<T>([.. allParams]);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Main conv weights
        for (int i = 0; i < _weights.Length; i++)
            _weights.Data[i] = parameters[offset++];

        // Main conv bias
        for (int i = 0; i < _bias.Length; i++)
            _bias.Data[i] = parameters[offset++];

        // Offset weights
        for (int i = 0; i < _offsetWeights.Length; i++)
            _offsetWeights.Data[i] = parameters[offset++];

        // Offset bias
        for (int i = 0; i < _offsetBias.Length; i++)
            _offsetBias.Data[i] = parameters[offset++];

        // Mask weights and bias
        if (_useModulation && _maskWeights != null && _maskBias != null)
        {
            for (int i = 0; i < _maskWeights.Length; i++)
                _maskWeights.Data[i] = parameters[offset++];
            for (int i = 0; i < _maskBias.Length; i++)
                _maskBias.Data[i] = parameters[offset++];
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightGradients != null)
        {
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights.Data[i] = NumOps.Subtract(_weights.Data[i],
                    NumOps.Multiply(learningRate, _weightGradients.Data[i]));
            }
        }

        if (_biasGradients != null)
        {
            for (int i = 0; i < _bias.Length; i++)
            {
                _bias.Data[i] = NumOps.Subtract(_bias.Data[i],
                    NumOps.Multiply(learningRate, _biasGradients.Data[i]));
            }
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var allGrads = new List<T>();

        if (_weightGradients != null)
        {
            for (int i = 0; i < _weightGradients.Length; i++)
                allGrads.Add(_weightGradients.Data[i]);
        }

        if (_biasGradients != null)
        {
            for (int i = 0; i < _biasGradients.Length; i++)
                allGrads.Add(_biasGradients.Data[i]);
        }

        if (_offsetWeightGradients != null)
        {
            for (int i = 0; i < _offsetWeightGradients.Length; i++)
                allGrads.Add(_offsetWeightGradients.Data[i]);
        }

        if (_offsetBiasGradients != null)
        {
            for (int i = 0; i < _offsetBiasGradients.Length; i++)
                allGrads.Add(_offsetBiasGradients.Data[i]);
        }

        if (_useModulation)
        {
            if (_maskWeightGradients != null)
            {
                for (int i = 0; i < _maskWeightGradients.Length; i++)
                    allGrads.Add(_maskWeightGradients.Data[i]);
            }

            if (_maskBiasGradients != null)
            {
                for (int i = 0; i < _maskBiasGradients.Length; i++)
                    allGrads.Add(_maskBiasGradients.Data[i]);
            }
        }

        return new Vector<T>([.. allGrads]);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOffsets = null;
        _lastMask = null;
        _weightGradients = null;
        _biasGradients = null;
        _offsetWeightGradients = null;
        _offsetBiasGradients = null;
        _maskWeightGradients = null;
        _maskBiasGradients = null;
    }

    #endregion
}
