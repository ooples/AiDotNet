using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralRadianceFields.Helpers;

/// <summary>
/// NeRF's frequency encoding γ(p) (Mildenhall et al. 2020, eq. 4) in recorded engine operations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// The layout is the engine kernel's: per input dimension, [sin(2^0·π·x), cos(2^0·π·x), sin(2^1·π·x), ...]. The
/// kernel is captured only by inference traces, so a compiled training plan evaluated it on the host at trace time and
/// replayed the first batch's encoding for every later batch. Built from multiply, sin and cos, the encoding replays
/// with each step's own positions and is differentiable in them.
/// </remarks>
internal static class PositionalEncoder<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Encodes <c>[N, D]</c> inputs into <c>[N, D·2·levels]</c>.</summary>
    public static Tensor<T> Encode(IEngine engine, Tensor<T> input, int levels)
    {
        int points = input.Shape[0];
        int dims = input.Shape[1];
        var frequencies = new Tensor<T>(new[] { 1, 1, levels });
        for (int l = 0; l < levels; l++) frequencies[l] = Ops.FromDouble(Math.Pow(2.0, l) * Math.PI);

        var angles = engine.TensorMultiply(engine.Reshape(input, new[] { points, dims, 1 }), frequencies); // [N, D, L]
        var pairs = engine.TensorConcatenate(new[]
        {
            engine.Reshape(engine.TensorSin(angles), new[] { points, dims, levels, 1 }),
            engine.Reshape(engine.TensorCos(angles), new[] { points, dims, levels, 1 }),
        }, axis: 3);
        return engine.Reshape(pairs, new[] { points, dims * 2 * levels });
    }
}
