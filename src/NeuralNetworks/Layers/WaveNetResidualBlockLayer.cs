using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A single WaveNet / Parallel WaveGAN residual block (van den Oord et al. 2016;
/// Yamamoto et al. 2020) for 1-D waveform/feature data <c>[B, C, T]</c>.
/// Implements the paper's gated-activation residual unit:
/// <code>
/// f = Conv1d_dilated(x)          # filter branch
/// g = Conv1d_dilated(x)          # gate branch
/// z = tanh(f) * sigmoid(g)       # gated activation
/// out = Conv1d_1x1(z)            # residual projection
/// y = x + out                    # residual connection
/// </code>
/// </summary>
/// <remarks>
/// <para>
/// The dual filter/gate dilated convolutions and the <c>tanh·sigmoid</c> product are
/// the defining WaveNet gated activation — a plain dilated-conv stack (no gating, no
/// residual) is NOT WaveNet. The residual connection carries the signal forward
/// through the deep dilation stack exactly as in the paper's residual path.
/// </para>
/// <para>
/// Built from three inner <see cref="Conv1DLayer{T}"/> instances (filter, gate, 1×1
/// projection), so the gradient tape and the fused conv kernels are reused — no
/// hand-written backward. Channel width is constant across the block (the residual
/// add requires <c>C_out == C_in</c>); the block is reconstructable purely from
/// <c>(channels, kernelSize, dilation)</c> for Clone/Deserialize.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 3, Cost = ComputeCost.Medium, TestInputShape = "1, 8, 16", TestConstructorArgs = "8, 3, 1")]
public partial class WaveNetResidualBlockLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _kernelSize;
    private readonly int _dilation;

    private readonly Conv1DLayer<T> _filterConv;
    private readonly Conv1DLayer<T> _gateConv;
    private readonly Conv1DLayer<T> _outConv;

    /// <summary>Constructs a WaveNet gated-residual block of constant channel width.</summary>
    /// <param name="channels">Residual channel width (input and output, <c>C</c>).</param>
    /// <param name="kernelSize">Dilated-conv kernel width (WaveNet uses 3). Defaults to 3.</param>
    /// <param name="dilation">Dilation factor for this block (WaveNet cycles <c>2^i</c>). Defaults to 1.</param>
    public WaveNetResidualBlockLayer(int channels, int kernelSize = 3, int dilation = 1)
        : base(new[] { channels, -1 }, new[] { channels, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));

        _channels = channels;
        _kernelSize = kernelSize;
        _dilation = dilation;

        // Filter and gate share shape but learn distinct kernels; "same" padding keeps T
        // so the residual add lines up. Tanh on the filter, sigmoid on the gate — the
        // gated activation is realized by multiplying the two activated branches.
        _filterConv = new Conv1DLayer<T>(channels, channels, kernelSize, dilation, 1, null, new TanhActivation<T>());
        _gateConv = new Conv1DLayer<T>(channels, channels, kernelSize, dilation, 1, null, new SigmoidActivation<T>());
        // 1×1 residual projection (identity activation).
        _outConv = new Conv1DLayer<T>(channels, channels, 1, 1, 1, null, new IdentityActivation<T>());
    }

    private IEnumerable<Conv1DLayer<T>> InnerConvs()
    {
        yield return _filterConv;
        yield return _gateConv;
        yield return _outConv;
    }

    public override bool SupportsTraining => true;

    public override long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var c in InnerConvs()) total += c.ParameterCount;
            return total;
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Gated activation: z = tanh(W_f * x) ⊙ sigmoid(W_g * x).
        var f = _filterConv.Forward(input);
        var g = _gateConv.Forward(input);
        var gated = Engine.TensorMultiply(f, g);

        // Residual projection + skip-add. Engine.TensorAdd records the residual on the
        // tape so the gradient flows through both the block and the identity shortcut.
        var projected = _outConv.Forward(gated);
        return Engine.TensorAdd(input, projected);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var c in InnerConvs()) c.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        Vector<T> all = Vector<T>.Empty();
        foreach (var c in InnerConvs())
            all = Vector<T>.Concatenate(all, c.GetParameters());
        return all;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var c in InnerConvs())
        {
            int len = (int)c.ParameterCount;
            var slice = new Vector<T>(parameters.AsSpan().Slice(offset, len).ToArray());
            c.SetParameters(slice);
            offset += len;
        }
        if (offset != parameters.Length)
        {
            throw new ArgumentException(
                $"Expected {offset} parameters for WaveNetResidualBlockLayer, but got {parameters.Length}.");
        }
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var c in InnerConvs()) c.SetTrainingMode(isTraining);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var c in InnerConvs()) c.ResetState();
    }

    /// <summary>Serialization metadata — the block is fully reconstructable from these.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString();
        metadata["KernelSize"] = _kernelSize.ToString();
        metadata["Dilation"] = _dilation.ToString();
        return metadata;
    }
}
