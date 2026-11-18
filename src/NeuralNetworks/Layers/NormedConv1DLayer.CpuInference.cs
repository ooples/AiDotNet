using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

public partial class NormedConv1DLayer<T>
{
    internal bool SupportsPrivateCpuInference => GetType() == typeof(NormedConv1DLayer<T>)
        && typeof(T) == typeof(float) && !IsTrainingMode && !IsInferringShapes && _groups == 1
        && Engine.Unwrap().GetType() == typeof(CpuEngine) && ReferenceEquals(Engine, Engine.Unwrap());

    internal bool TryPrivateCpuOutputTime(int inputTime, out int outputTime)
    {
        long time = _transposed
            ? ((long)inputTime - 1) * _stride - 2L * _padding + _kernel + _outputPadding
            : ((long)inputTime + 2L * _padding - (long)_dilation * (_kernel - 1) - 1) / _stride + 1;
        outputTime = time > 0 && time <= int.MaxValue ? (int)time : 0;
        return outputTime != 0;
    }

    // Destination ownership is supplied by the generator's exclusive workspace lease.
    // Kernel() retains the ordinary inference normalization cache and parameter-version keys.
    internal void ForwardPrivateCpuInto(Tensor<T> input, Tensor<T> output4, Tensor<T>? residual,
        bool activate, T alpha)
    {
        if (!SupportsPrivateCpuInference || input.Rank != 3 || input.Shape[1] != _in
            || !TryPrivateCpuOutputTime(input.Shape[2], out int time)
            || output4.Rank != 4 || output4.Shape[0] != input.Shape[0] || output4.Shape[1] != _out
            || output4.Shape[2] != 1 || output4.Shape[3] != time)
            throw new InvalidOperationException("Private CPU convolution geometry/ownership contract mismatch.");
        BindInputContract(input.Shape.ToArray()).Validate(input);
        var cpu = Engine.Cast<CpuEngine>();
        var kernel = Kernel();
        using var input4 = Engine.Reshape(input, new[] { input.Shape[0], _in, 1, input.Shape[2] });
        if (_transposed)
            cpu.ConvTranspose2DInto(output4, input4, kernel, new[] { 1, _stride }, new[] { 0, _padding }, new[] { 0, _outputPadding });
        else
            cpu.Conv2DInto(output4, input4, kernel, new[] { 1, _stride }, new[] { 0, _padding }, new[] { 1, _dilation });
        cpu.AddPrivateConvolutionBiasResidual((Tensor<float>)(object)output4,
            _useBias ? (Tensor<float>)(object)_bias : null, residual is null ? null : (Tensor<float>)(object)residual);
        if (activate) cpu.LeakyReLUInto(output4, output4, alpha);
    }
}
