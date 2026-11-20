using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

public partial class NormedConv1DLayer<T>
{
    private delegate bool ConvBiasResidual(DirectGpuTensorEngine engine, Tensor<T> input, Tensor<T> kernel,
        Tensor<T> bias, Tensor<T> residual, int strideH, int strideW, int padH, int padW, int dilationH,
        int dilationW, T alpha, bool writeActivation, out Tensor<T> sum, out Tensor<T>? activated);
    private delegate bool ChannelBiasResidual(DirectGpuTensorEngine engine, Tensor<T> input, Tensor<T> bias,
        Tensor<T> residual, T alpha, bool writeActivation, out Tensor<T> sum, out Tensor<T>? activated);
    private delegate bool ChannelBiasLeaky(DirectGpuTensorEngine engine, Tensor<T> input, Tensor<T> bias,
        T alpha, out Tensor<T> activated);

    private static readonly ConvBiasResidual? FusedConvResidual = OptionalEngineOperation.Find<ConvBiasResidual>(
        typeof(DirectGpuTensorEngine), "TryConvBiasResidualInference", typeof(T));
    private static readonly ChannelBiasResidual? FusedBiasResidual = OptionalEngineOperation.Find<ChannelBiasResidual>(
        typeof(DirectGpuTensorEngine), "TryChannelBiasResidualInference", typeof(T));
    private static readonly ChannelBiasLeaky? FusedBiasLeaky = OptionalEngineOperation.Find<ChannelBiasLeaky>(
        typeof(DirectGpuTensorEngine), "TryChannelBiasLeakyInference", typeof(T));
    private static readonly Action<CpuEngine, Tensor<float>, Tensor<float>?, Tensor<float>?>? PrivateCpuEpilogue =
        OptionalEngineOperation.Find<Action<CpuEngine, Tensor<float>, Tensor<float>?, Tensor<float>?>>(
            typeof(CpuEngine), "AddPrivateConvolutionBiasResidual");
    private static readonly Func<IEngine, Tensor<T>, Tensor<T>, int[], int[], int[], int, Tensor<T>>? NativeGroupedConvolution =
        OptionalEngineOperation.Find<Func<IEngine, Tensor<T>, Tensor<T>, int[], int[], int[], int, Tensor<T>>>(
            typeof(IEngine).Assembly.GetType("AiDotNet.Tensors.Engines.GroupedConvolution"), "Forward", typeof(T));

    private Tensor<T> GroupedConvolutionForward(Tensor<T> input, Tensor<T> kernel)
    {
        int[] stride = { 1, _stride }, padding = { 0, _padding }, dilation = { 1, _dilation };
        if (NativeGroupedConvolution is { } grouped)
            return grouped(Engine, input, kernel, stride, padding, dilation, _groups);

        int batch = input.Shape[0], time = input.Shape[3], inPer = _in / _groups, outPer = _out / _groups;
        var parts = new Tensor<T>[_groups];
        for (int g = 0; g < _groups; g++)
        {
            var channels = Engine.TensorSlice(input, new[] { 0, g * inPer, 0, 0 }, new[] { batch, inPer, 1, time });
            var filter = Engine.TensorSlice(kernel, new[] { g * outPer, 0, 0, 0 }, new[] { outPer, inPer, 1, _kernel });
            parts[g] = Engine.Conv2D(channels, filter, stride, padding, dilation);
        }
        return Engine.TensorConcatenate(parts, 1);
    }

    private void AddPrivateCpuBiasResidual(CpuEngine cpu, Tensor<T> output, Tensor<T>? residual)
    {
        if (PrivateCpuEpilogue is { } epilogue)
        {
            epilogue(cpu, (Tensor<float>)(object)output, _useBias ? (Tensor<float>)(object)_bias : null,
                residual is null ? null : (Tensor<float>)(object)residual);
            return;
        }
        if (_useBias)
        {
            using var biased = cpu.TensorChannelBiasAdd(output, _bias);
            cpu.TensorCopy(biased, output);
        }
        if (residual is not null)
        {
            using var carry = cpu.Reshape(residual, output.Shape.ToArray());
            cpu.TensorAddInPlace(output, carry);
        }
    }
}
