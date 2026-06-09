using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// HiFi-GAN Multi-Receptive Field (MRF) fusion module (Kong et al. 2020, §2.2) for
/// 1-D waveform/feature data <c>[B, C, T]</c>. After each upsampling stage the
/// generator runs the input through several residual blocks with different kernel
/// sizes and dilation patterns IN PARALLEL and returns their averaged sum, so the
/// network observes patterns over diverse receptive fields simultaneously:
/// <code>
/// MRF(x) = (1/K) * Σ_k ResBlock_k(x)
/// ResBlock_k(x): for d in dilations: x = x + Conv1d(LeakyReLU(Conv1d_dilated_d(x)))
/// </code>
/// </summary>
/// <remarks>
/// <para>
/// The official <c>jik876/hifi-gan</c> v1 config uses
/// <c>resblock_kernel_sizes=[3,7,11]</c> and
/// <c>resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]</c> — the defaults here. The
/// parallel-branch SUM is the defining MRF behaviour; a single sequential dilated-conv
/// chain is NOT MRF.
/// </para>
/// <para>
/// Built from inner <see cref="Conv1DLayer{T}"/> instances (two per dilation per
/// kernel — the leaky-pre-activated dilated conv and the dilation-1 projection), so
/// the tape handles backward and the fused conv kernels are reused. "Same" padding
/// keeps T constant across the block (required for the per-branch residual adds and
/// the cross-branch sum). Reconstructable from <c>(channels, kernelSizes, dilations)</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 8, 16", TestConstructorArgs = "8")]
public partial class HiFiGANResBlockLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int[] _kernelSizes;
    private readonly int[] _dilations;

    // Per (kernel, dilation): conv1 = LeakyReLU dilated conv, conv2 = dilation-1 projection.
    private readonly List<Conv1DLayer<T>> _convs1;
    private readonly List<Conv1DLayer<T>> _convs2;

    /// <summary>Constructs a HiFi-GAN MRF block over the given kernel sizes / dilations.</summary>
    /// <param name="channels">Channel width (constant; input == output).</param>
    /// <param name="kernelSizes">Residual-block kernel sizes (official v1: [3,7,11]).</param>
    /// <param name="dilations">Dilations applied within each residual block (official v1: [1,3,5]).</param>
    public HiFiGANResBlockLayer(int channels, int[]? kernelSizes = null, int[]? dilations = null)
        : base(new[] { channels, -1 }, new[] { channels, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        _channels = channels;
        _kernelSizes = kernelSizes is { Length: > 0 } ? kernelSizes : new[] { 3, 7, 11 };
        _dilations = dilations is { Length: > 0 } ? dilations : new[] { 1, 3, 5 };

        _convs1 = new List<Conv1DLayer<T>>(_kernelSizes.Length * _dilations.Length);
        _convs2 = new List<Conv1DLayer<T>>(_kernelSizes.Length * _dilations.Length);
        foreach (int k in _kernelSizes)
        {
            foreach (int d in _dilations)
            {
                // conv1: dilated, LeakyReLU; conv2: dilation 1, identity projection. Both
                // "same"-padded (default) so T is preserved for the residual adds.
                _convs1.Add(new Conv1DLayer<T>(channels, channels, k, d, 1, null, new LeakyReLUActivation<T>()));
                _convs2.Add(new Conv1DLayer<T>(channels, channels, k, 1, 1, null, new IdentityActivation<T>()));
            }
        }
    }

    private IEnumerable<Conv1DLayer<T>> InnerConvs() => _convs1.Concat(_convs2);

    public override bool SupportsTraining => true;

    public override long ParameterCount => InnerConvs().Sum(c => c.ParameterCount);

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int numDil = _dilations.Length;
        Tensor<T>? sum = null;

        for (int ki = 0; ki < _kernelSizes.Length; ki++)
        {
            // ResBlock_k: chained dilated residual adds.
            var xk = input;
            for (int di = 0; di < numDil; di++)
            {
                int idx = ki * numDil + di;
                var xt = _convs2[idx].Forward(_convs1[idx].Forward(xk));
                xk = Engine.TensorAdd(xk, xt);
            }
            sum = sum is null ? xk : Engine.TensorAdd(sum, xk);
        }

        // MRF returns the AVERAGE over the kernel-size branches (Kong 2020 §2.2).
        T inv = NumOps.FromDouble(1.0 / _kernelSizes.Length);
        return Engine.TensorMultiplyScalar(sum!, inv);
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
                $"Expected {offset} parameters for HiFiGANResBlockLayer, but got {parameters.Length}.");
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
        metadata["KernelSizes"] = string.Join(",", _kernelSizes);
        metadata["Dilations"] = string.Join(",", _dilations);
        return metadata;
    }
}
