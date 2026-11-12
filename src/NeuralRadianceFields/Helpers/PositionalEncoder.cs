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
        var frequencies = Frequencies(engine, levels).Reshape(1, 1, levels);

        var angles = engine.TensorMultiply(engine.Reshape(input, new[] { points, dims, 1 }), frequencies); // [N, D, L]
        var pairs = engine.TensorConcatenate(new[]
        {
            engine.Reshape(engine.TensorSin(angles), new[] { points, dims, levels, 1 }),
            engine.Reshape(engine.TensorCos(angles), new[] { points, dims, levels, 1 }),
        }, axis: 3);
        return engine.Reshape(pairs, new[] { points, dims * 2 * levels });
    }

    /// <summary>
    /// The octave frequencies 2^l * pi for l = 0..levels-1. Powers of two are built by an exact running product, so
    /// each frequency equals pi scaled by 2^l with no rounding beyond pi's own.
    /// </summary>
    private static Tensor<T> Frequencies(IEngine engine, int levels)
    {
        if (levels == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        var factors = engine.TensorFullLike(new Tensor<T>(new[] { levels }), Ops.FromDouble(2.0));
        factors[0] = Ops.One;
        var powers = engine.TensorCumProd(factors, 0);
        return engine.TensorMultiplyScalar(powers, Ops.FromDouble(Math.PI));
    }
}
