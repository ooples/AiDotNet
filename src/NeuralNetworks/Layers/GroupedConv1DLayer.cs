using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A grouped 1-D convolution (PyTorch <c>Conv1d(groups = g)</c>): the channels are split into g groups and each output
/// group sees only its input group.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The kernel is <c>[out, in / g, 1, k]</c> with a bias per output channel, initialized as PyTorch does:
/// U(−1/√fan_in, 1/√fan_in) with fan_in = (in / g) · k. Each group runs its own convolution over its slice of the
/// channels, so the parameter count is a g-th of a full convolution's. Input and output are <c>[batch, channels, time]</c>.</para>
/// <para><b>For Beginners:</b> Splits the channels into independent bundles and convolves each bundle separately, which
/// is cheaper than mixing every channel with every other.</para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4, 6", TestConstructorArgs = "4, 4, 3, 2, 1")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GroupedConv1DLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _kernelSize;
    private readonly int _groups;
    private readonly int _padding;
    private readonly int _dilation;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _kernel;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer.</summary>
    /// <param name="inChannels">Input channels (divisible by <paramref name="groups"/>).</param>
    /// <param name="outChannels">Output channels (divisible by <paramref name="groups"/>).</param>
    /// <param name="kernelSize">Kernel length.</param>
    /// <param name="groups">Number of channel groups.</param>
    /// <param name="padding">Zero padding on each side of the time axis.</param>
    /// <param name="dilation">Kernel dilation (1).</param>
    public GroupedConv1DLayer([LayerState] int inChannels, [LayerState] int outChannels, [LayerState] int kernelSize,
        [LayerState] int groups, [LayerState] int padding, [LayerState] int dilation = 1)
        : base(new[] { inChannels }, new[] { outChannels })
    {
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));
        _dilation = dilation;
        if (groups <= 0 || inChannels <= 0 || outChannels <= 0 || inChannels % groups != 0 || outChannels % groups != 0)
            throw new ArgumentException($"Groups ({groups}) must divide the input ({inChannels}) and output ({outChannels}) channels.", nameof(groups));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding));
        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _groups = groups;
        _padding = padding;
        int fanIn = inChannels / groups * kernelSize;
        double bound = 1.0 / Math.Sqrt(fanIn);
        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        _kernel = new Tensor<T>(new[] { outChannels, inChannels / groups, 1, kernelSize });
        _bias = new Tensor<T>(new[] { outChannels });
        for (int i = 0; i < _kernel.Length; i++) _kernel[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        for (int i = 0; i < outChannels; i++) _bias[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        RegisterTrainableParameter(_kernel, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[1] != _inChannels)
            throw new ArgumentException($"Expected [batch, {_inChannels}, time], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], time = input.Shape[2], inPer = _inChannels / _groups, outPer = _outChannels / _groups;
        var x4 = Engine.Reshape(input, new[] { batch, _inChannels, 1, time });
        var outputs = new Tensor<T>[_groups];
        for (int g = 0; g < _groups; g++)
        {
            var slice = Engine.TensorSlice(x4, new[] { 0, g * inPer, 0, 0 }, new[] { batch, inPer, 1, time });
            var kernel = Engine.TensorSlice(_kernel, new[] { g * outPer, 0, 0, 0 }, new[] { outPer, inPer, 1, _kernelSize });
            outputs[g] = Engine.Conv2D(slice, kernel, new[] { 1, 1 }, new[] { 0, _padding }, new[] { 1, _dilation });
        }
        var y = _groups == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 1);
        int outTime = y.Shape[3];
        var bias = Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, _outChannels, 1, 1 }), new[] { batch, 1, 1, outTime });
        return Engine.Reshape(Engine.TensorAdd(y, bias), new[] { batch, _outChannels, outTime });
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InChannels"] = _inChannels.ToString(inv);
        metadata["OutChannels"] = _outChannels.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        metadata["Groups"] = _groups.ToString(inv);
        metadata["Padding"] = _padding.ToString(inv);
        metadata["Dilation"] = _dilation.ToString(inv);
        return metadata;
    }
}
