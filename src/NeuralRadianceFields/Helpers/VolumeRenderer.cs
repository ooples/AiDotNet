using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralRadianceFields.Helpers;

/// <summary>
/// NeRF's volume-rendering quadrature (Mildenhall et al. 2020, eq. 3) in engine operations, so the
/// photometric loss differentiates through it into the radiance field.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// For a ray with samples i = 1..S at spacings δᵢ, densities σᵢ and colors cᵢ:
/// αᵢ = 1 − exp(−σᵢδᵢ), Tᵢ = exp(−Σ_{j&lt;i} σⱼδⱼ), and the pixel is C = Σᵢ Tᵢ αᵢ cᵢ.
/// The transmittance is an exclusive cumulative sum, and the composite is one batched matrix product
/// per ray. Every step is a tape-recorded engine op, so dC/dσ and dC/dc reach the field's weights.
/// The previous per-model renderers accumulated in host scalars, so the photometric gradient stopped
/// at the rendered color and image-space training updated nothing (#1834).
/// </para>
/// <para>
/// The sample spacings are constants of the sampling, not of the field, so they are computed on the
/// host: the distance to the next sample, or to the far bound for the last sample or a decreasing
/// pair; coincident samples get zero width. Without explicit sample positions the spacing is uniform,
/// (far − near) / S.
/// </para>
/// </remarks>
internal static class VolumeRenderer<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Composites per-sample colors and densities into one color per ray.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="rgb">Sample colors, <c>[rays·samples, 3]</c>, ray-major.</param>
    /// <param name="density">Sample densities, <c>[rays·samples]</c> or <c>[rays·samples, 1]</c>.</param>
    /// <param name="numRays">The number of rays.</param>
    /// <param name="numSamples">The samples per ray.</param>
    /// <param name="rayNear">Each ray's near bound.</param>
    /// <param name="rayFar">Each ray's far bound.</param>
    /// <param name="sampleTs">Each sample's distance along its ray, or null for uniform spacing.</param>
    /// <returns>The rendered colors, <c>[rays, 3]</c>.</returns>
    public static Tensor<T> Render(IEngine engine, Tensor<T> rgb, Tensor<T> density, int numRays, int numSamples,
        double[] rayNear, double[] rayFar, double[]? sampleTs)
    {
        if (numRays <= 0 || numSamples <= 0) return new Tensor<T>(new[] { Math.Max(numRays, 0), 3 });

        var deltas = new Tensor<T>(new[] { numRays, numSamples });
        for (int r = 0; r < numRays; r++)
        {
            double uniform = Math.Max(0.0, (rayFar[r] - rayNear[r]) / numSamples);
            for (int s = 0; s < numSamples; s++)
            {
                int idx = r * numSamples + s;
                double delta = uniform;
                if (sampleTs is not null)
                {
                    double t0 = sampleTs[idx];
                    double t1 = s + 1 < numSamples ? sampleTs[idx + 1] : rayFar[r];
                    if (t1 < t0) t1 = rayFar[r];
                    delta = Math.Max(0.0, t1 - t0);
                }

                deltas[idx] = Ops.FromDouble(delta);
            }
        }

        var sigma = engine.Reshape(density, new[] { numRays, numSamples });
        var opticalDepth = engine.TensorMultiply(sigma, deltas);
        var alpha = engine.TensorAddScalar(engine.TensorNegate(engine.TensorExp(engine.TensorNegate(opticalDepth))), Ops.One);

        // Exclusive cumulative optical depth: the light that reaches sample i has crossed samples 1..i-1.
        var before = engine.TensorSubtract(engine.TensorCumSum(opticalDepth, 1), opticalDepth);
        var transmittance = engine.TensorExp(engine.TensorNegate(before));
        var weights = engine.TensorMultiply(transmittance, alpha);

        var composite = engine.BatchMatMul(
            engine.Reshape(weights, new[] { numRays, 1, numSamples }),
            engine.Reshape(rgb, new[] { numRays, numSamples, 3 }));
        return engine.Reshape(composite, new[] { numRays, 3 });
    }
}
