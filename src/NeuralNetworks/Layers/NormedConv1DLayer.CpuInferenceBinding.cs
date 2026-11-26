using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

public partial class NormedConv1DLayer<T>
{
    // Only the exclusive private generator workspace supplies these paired views.
    // Keep input validation and live Kernel() lookup on every execution; bind the
    // immutable geometry and borrowed NCHW views once for the complete batch.
    internal CpuInferenceBinding BindPrivateCpuInference(Tensor<T> input3, Tensor<T> input4,
        Tensor<T> output4, Tensor<T>? residual, bool activate, T alpha)
    {
        if (!SupportsPrivateCpuInference || input3.Rank != 3 || input3.Shape[1] != _in
            || !TryPrivateCpuOutputTime(input3.Shape[2], out int time)
            || input4.Rank != 4 || input4.Shape[0] != input3.Shape[0] || input4.Shape[1] != _in
            || input4.Shape[2] != 1 || input4.Shape[3] != input3.Shape[2]
            || output4.Rank != 4 || output4.Shape[0] != input3.Shape[0] || output4.Shape[1] != _out
            || output4.Shape[2] != 1 || output4.Shape[3] != time)
            throw new InvalidOperationException("Private CPU binding geometry/ownership contract mismatch.");
        if (!MemoryMarshal.TryGetArray(input3.ReadOnlyDataUnmarked, out var logical)
            || !MemoryMarshal.TryGetArray(input4.ReadOnlyDataUnmarked, out var nchw)
            || !ReferenceEquals(logical.Array, nchw.Array) || logical.Offset != nchw.Offset || logical.Count != nchw.Count)
            throw new InvalidOperationException("Private CPU binding requires paired views of the same owned workspace.");
        return new CpuInferenceBinding(this, input3, input4, output4, residual, activate, alpha,
            BindInputContract(input3.Shape.ToArray()));
    }

    internal sealed class CpuInferenceBinding
    {
        private readonly NormedConv1DLayer<T> _layer;
        private readonly CpuEngine _cpu;
        private readonly Tensor<T> _input3, _input4, _output4;
        private readonly Tensor<T>? _residual;
        private readonly bool _activate;
        private readonly T _alpha;
        private readonly BoundInputContract _contract;
        private readonly int[] _stride, _padding, _third;

        internal CpuInferenceBinding(NormedConv1DLayer<T> layer, Tensor<T> input3, Tensor<T> input4,
            Tensor<T> output4, Tensor<T>? residual, bool activate, T alpha, BoundInputContract contract)
        {
            _layer = layer; _cpu = layer.Engine.Cast<CpuEngine>();
            _input3 = input3; _input4 = input4; _output4 = output4; _residual = residual;
            _activate = activate; _alpha = alpha; _contract = contract;
            _stride = new[] { 1, layer._stride }; _padding = new[] { 0, layer._padding };
            _third = layer._transposed ? new[] { 0, layer._outputPadding } : new[] { 1, layer._dilation };
        }

        internal bool Matches(Tensor<T> input3, Tensor<T> input4, Tensor<T> output4,
            Tensor<T>? residual, bool activate, T alpha)
            => ReferenceEquals(_input3, input3) && ReferenceEquals(_input4, input4)
                && ReferenceEquals(_output4, output4) && ReferenceEquals(_residual, residual)
                && _activate == activate && EqualityComparer<T>.Default.Equals(_alpha, alpha);

        internal void Execute()
        {
            if (!_layer.SupportsPrivateCpuInference)
                throw new InvalidOperationException("Private CPU binding context changed.");
            _contract.Validate(_input3);
            var kernel = _layer.Kernel();
            if (_layer._transposed)
                _cpu.ConvTranspose2DInto(_output4, _input4, kernel, _stride, _padding, _third);
            else _cpu.Conv2DInto(_output4, _input4, kernel, _stride, _padding, _third);
            _layer.AddPrivateCpuBiasResidual(_cpu, _output4, _residual);
            if (_activate) _cpu.LeakyReLUInto(_output4, _output4, _alpha);
        }
    }
}
