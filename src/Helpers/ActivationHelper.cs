using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;

namespace AiDotNet.Helpers;

/// <summary>
/// Provides centralized helper methods for applying activation functions with optimal performance.
/// Uses Engine methods (GPU/SIMD) for known activation types, falls back to standard activation otherwise.
/// </summary>
/// <remarks>
/// This class consolidates activation type-checking logic in one place, following DRY (Don't Repeat Yourself)
/// and SOLID principles. All layers should use these methods instead of duplicating if-else chains.
/// </remarks>
public static class ActivationHelper
{
    /// <summary>
    /// The exact GELU, <c>x · Φ(x) = ½ x erfc(−x / √2)</c>, as PyTorch's <c>F.gelu</c> computes it by default.
    /// </summary>
    /// <remarks><see cref="IEngine.GELU{T}"/> is the tanh approximation, which differs by up to about 1e-3; models
    /// matched to a PyTorch reference use this form.</remarks>
    internal static Tensor<T> ExactGelu<T>(Tensor<T> x, IEngine engine)
    {
        // float on the plain CPU engine, outside a graph trace: one fused pass (and one fused backward) instead of
        // the engine's erfc, which evaluates a double-precision series per element through a delegate and took a
        // third of a wav2vec 2.0 BASE training step.
        if (typeof(T) == typeof(float) && engine.GetType() == typeof(AiDotNet.Tensors.Engines.CpuEngine)
            && !AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive)
        {
            // A strided view is copied first when nothing records gradients (a frozen encoder, inference); on a tape
            // it keeps the engine path, whose ops carry the view's gradient.
            if (!x.IsContiguous && GradientTape<T>.Current is not null && !NoGradScope<T>.IsSuppressed)
                return SlowExactGelu(x, engine);
            return (Tensor<T>)(object)ExactGeluFloat((Tensor<float>)(object)(x.IsContiguous ? x : x.Contiguous()), engine);
        }
        return SlowExactGelu(x, engine);
    }

    private static Tensor<T> SlowExactGelu<T>(Tensor<T> x, IEngine engine)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorMultiplyScalar(
            engine.TensorMultiply(x, engine.TensorErfc(engine.TensorMultiplyScalar(x, ops.FromDouble(-1 / Math.Sqrt(2))))),
            ops.FromDouble(0.5));
    }

    private const int GeluChunk = 1 << 15;

    // x · Φ(x) for float, with erf from Abramowitz & Stegun 7.1.26 (absolute error below 1.5e-7, the form PyTorch's
    // vectorized CPU erf uses). The backward is Φ(x) + x · φ(x).
    private static Tensor<float> ExactGeluFloat(Tensor<float> x, IEngine engine)
    {
        var input = x.Data;
        var output = new float[x.Length];
        RunChunks(x.Length, (start, end) =>
        {
            var src = input.Span;
            for (int i = start; i < end; i++)
            {
                float v = src[i];
                output[i] = 0.5f * v * (1f + Erf(v * 0.70710678f));
            }
        });
        var result = new Tensor<float>(output, x.Shape.ToArray());
        DifferentiableOps.RecordUnary("ExactGelu", result, x, (gradOutput, inputs, _, _, eng, grads) =>
        {
            var g = gradOutput.Contiguous().Data;
            var xs = inputs[0].Contiguous().Data;
            var gradient = new float[g.Length];
            RunChunks(g.Length, (start, end) =>
            {
                var gs = g.Span;
                var vs = xs.Span;
                for (int i = start; i < end; i++)
                {
                    float v = vs[i];
                    float cdf = 0.5f * (1f + Erf(v * 0.70710678f));
                    float pdf = 0.39894228f * (float)Math.Exp(-0.5f * v * v);
                    gradient[i] = gs[i] * (cdf + v * pdf);
                }
            });
            DifferentiableOps.AccumulateGrad(grads, inputs[0], new Tensor<float>(gradient, gradOutput.Shape.ToArray()), eng);
        });
        return result;
    }

    private static void RunChunks(int length, Action<int, int> body)
    {
        int chunks = (length + GeluChunk - 1) / GeluChunk;
        if (chunks <= 1)
        {
            body(0, length);
            return;
        }
        System.Threading.Tasks.Parallel.For(0, chunks, c => body(c * GeluChunk, Math.Min(length, (c + 1) * GeluChunk)));
    }

    private static float Erf(float x)
    {
        float a = Math.Abs(x);
        float t = 1f / (1f + 0.3275911f * a);
        float r = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
        float y = 1f - r * (float)Math.Exp(-a * a);
        return x < 0 ? -y : y;
    }

    /// <summary>
    /// Applies a vector activation function to a tensor using Engine methods when possible.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="activation">The vector activation function to apply.</param>
    /// <param name="input">The input tensor to activate.</param>
    /// <param name="engine">The engine to use for optimized operations.</param>
    /// <returns>The activated tensor, or input unchanged if activation is null.</returns>
    public static Tensor<T> ApplyActivation<T>(IVectorActivationFunction<T>? activation, Tensor<T> input, IEngine engine)
    {
        if (activation == null)
            return input;

        // Use Engine methods for optimized activations (CPU SIMD / GPU kernels)
        if (activation is TanhActivation<T>)
            return engine.Tanh(input);
        else if (activation is SigmoidActivation<T>)
            return engine.Sigmoid(input);
        else if (activation is ReLUActivation<T>)
            return engine.ReLU(input);
        else if (activation is GELUActivation<T>)
            return engine.GELU(input);
        else if (activation is MishActivation<T>)
            return engine.Mish(input);
        else if (activation is SwishActivation<T> || activation is SiLUActivation<T>)
            return engine.Swish(input);
        else if (activation is ELUActivation<T>)
            return engine.ELU(input);
        else
            // Fall back to activation function for custom types
            return activation.Activate(input);
    }

    /// <summary>
    /// Applies a scalar (element-wise) activation function to a tensor, routing known
    /// activations through tape-connected Engine kernels and falling back to the
    /// activation's own <see cref="IActivationFunction{T}.Activate(Tensor{T})"/> for
    /// custom types. This is the single dispatch point for element-wise activations so
    /// individual layers don't each re-implement the engine mapping.
    /// </summary>
    public static Tensor<T> ApplyActivation<T>(IActivationFunction<T>? activation, Tensor<T> input, IEngine engine)
    {
        if (activation == null)
            return input;

        // Identity is a no-op: return the input tensor unchanged. This both avoids an
        // allocation/element-wise copy AND preserves GPU residency — the fall-through
        // activation.Activate(input) rebuilds the tensor on the host, which silently
        // de-residents a buffer and breaks CUDA-graph capture for layers (e.g. diffusion
        // Conv with IdentityActivation) whose inference fast path runs ApplyActivation.
        if (activation is IdentityActivation<T>)
            return input;

        // Use Engine methods for optimized activations (CPU SIMD / GPU kernels);
        // these are tape-connected so gradients flow through the activation.
        if (activation is TanhActivation<T>)
            return engine.Tanh(input);
        else if (activation is SigmoidActivation<T>)
            return engine.Sigmoid(input);
        else if (activation is ReLUActivation<T>)
            return engine.ReLU(input);
        else if (activation is GELUActivation<T>)
            return engine.GELU(input);
        else if (activation is MishActivation<T>)
            return engine.Mish(input);
        else if (activation is SwishActivation<T> || activation is SiLUActivation<T>)
            return engine.Swish(input);
        else if (activation is ELUActivation<T>)
            return engine.ELU(input);
        else
            // Fall back to the activation's own tensor application for custom types.
            return activation.Activate(input);
    }

    /// <summary>
    /// Applies a vector activation function to a vector using Engine methods when possible.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="activation">The vector activation function to apply.</param>
    /// <param name="input">The input vector to activate.</param>
    /// <param name="engine">The engine to use for optimized operations.</param>
    /// <returns>The activated vector, or input unchanged if activation is null.</returns>
    public static Vector<T> ApplyActivation<T>(IVectorActivationFunction<T>? activation, Vector<T> input, IEngine engine)
    {
        if (activation == null)
            return input;

        // Use Engine methods for optimized activations (CPU SIMD / GPU kernels)
        if (activation is TanhActivation<T>)
            return engine.Tanh(input);
        else if (activation is SigmoidActivation<T>)
            return engine.Sigmoid(input);
        else if (activation is ReLUActivation<T>)
            return engine.ReLU(input);
        else if (activation is GELUActivation<T>)
            return engine.GELU(input);
        else if (activation is MishActivation<T>)
            return engine.Mish(input);
        else if (activation is SwishActivation<T> || activation is SiLUActivation<T>)
            return engine.Swish(input);
        else if (activation is ELUActivation<T>)
            return engine.ELU(input);
        else
            // Fall back to activation function for custom types
            return activation.Activate(input);
    }
}
