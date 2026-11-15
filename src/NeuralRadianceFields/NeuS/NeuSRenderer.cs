namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>The differentiable outputs of a NeuS render of a batch of rays.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal sealed class NeuSRenderResult<T>
{
    /// <summary>The rendered colors, <c>[R, 3]</c>.</summary>
    public Tensor<T> Color { get; init; } = new Tensor<T>(new[] { 0, 3 });

    /// <summary>The per-sample weights <c>wᵢ = Tᵢ αᵢ</c>, <c>[R, S]</c>.</summary>
    public Tensor<T> Weights { get; init; } = new Tensor<T>(new[] { 0, 0 });

    /// <summary>The accumulated opacity <c>Σ wᵢ</c>, <c>[R, 1]</c>.</summary>
    public Tensor<T> WeightSum { get; init; } = new Tensor<T>(new[] { 0, 1 });

    /// <summary>The expected depth <c>Σ wᵢ tᵢ</c> at the section midpoints, <c>[R, 1]</c>.</summary>
    public Tensor<T> Depth { get; init; } = new Tensor<T>(new[] { 0, 1 });

    /// <summary>The rendered normal <c>Σ wᵢ ∇f/‖∇f‖</c>, <c>[R, 3]</c> (unnormalized).</summary>
    public Tensor<T> Normal { get; init; } = new Tensor<T>(new[] { 0, 3 });

    /// <summary>The masked mean Eikonal residual <c>(‖∇f‖ − 1)²</c>, a one-element tensor.</summary>
    public Tensor<T> GradientError { get; init; } = new Tensor<T>(new[] { 1 });

    /// <summary>The SDF at the samples, <c>[R, S]</c>.</summary>
    public Tensor<T> Sdf { get; init; } = new Tensor<T>(new[] { 0, 0 });

    /// <summary>The section midpoint depths, <c>[R·S]</c> on the host.</summary>
    public double[] MidDepths { get; init; } = Array.Empty<double>();

    /// <summary>The samples per ray (with marching, the longest ray's; shorter rays are padded).</summary>
    public int SamplesPerRay { get; init; }

    /// <summary>The samples evaluated (instant-nsr-pl's <c>num_samples</c>, which drives the dynamic ray batch).</summary>
    public int SampleCount { get; init; }
}

/// <summary>
/// NeuS's unbiased, occlusion-aware volume rendering of an SDF (Wang et al. 2021, §3.2-3.3), in engine operations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// With the logistic CDF <c>Φ_s(x) = sigmoid(s·x)</c> (the S-density's integral), the discrete opacity of section i is
/// <c>αᵢ = max((Φ_s(f(pᵢ)) − Φ_s(f(pᵢ₊₁))) / Φ_s(f(pᵢ)), 0)</c> (eq. 13), and the weight is <c>wᵢ = Tᵢ αᵢ</c> with
/// <c>Tᵢ = Πⱼ&lt;ᵢ (1 − αⱼ)</c>. The reference evaluates f at the section ends from the midpoint SDF and the slope
/// along the ray, <c>f ± cos·δ/2</c>, with the slope annealed from <c>−relu(−cos/2 + 1/2)</c> (never zero, so early
/// training gets gradient everywhere) to <c>−relu(−cos)</c> by <c>cos_anneal_ratio</c>. This weight is unbiased: for a
/// plane it peaks exactly at the zero crossing, which <c>NeuSRendererTests</c> checks.
/// </para>
/// <para>
/// The transmittance's product is taken as <c>exp(Σ log(1 − α + 10⁻⁷))</c> with an exclusive cumulative sum, the
/// same quantity the reference's <c>cumprod</c> forms, without a product whose gradient divides by the factors.
/// </para>
/// </remarks>
internal static class NeuSRenderer<T>
{
    /// <summary>
    /// The S-density sharpness <c>s = clamp(exp(10·v), 10⁻⁶, 10⁶)</c> from the variance parameter <c>v</c>
    /// (reference <c>SingleVarianceNetwork</c> and the <c>clip(1e-6, 1e6)</c> in <c>render_core</c>).
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="variance">The one-element variance parameter.</param>
    /// <returns>A <c>[1, 1]</c> tensor.</returns>
    public static Tensor<T> InverseS(IEngine engine, Tensor<T> variance)
    {
        var s = engine.TensorExp(engine.TensorMultiplyScalar(engine.Reshape(variance, new[] { 1, 1 }), NeuSHost<T>.Scalar(10.0)));
        return engine.TensorClamp(s, NeuSHost<T>.Epsilon(1e-6), NeuSHost<T>.Scalar(1e6));
    }

    /// <summary>
    /// The discrete opacity of every sample (reference <c>render_core</c>, <c>alpha_type = 'div'</c>).
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="sdf">The SDF at the sample midpoints, <c>[N, 1]</c>.</param>
    /// <param name="trueCos">The ray direction's dot product with ∇f, <c>[N, 1]</c>.</param>
    /// <param name="dists">The section lengths, <c>[N, 1]</c>.</param>
    /// <param name="inverseS">The sharpness, <c>[1, 1]</c> or <c>[N, 1]</c>.</param>
    /// <param name="cosAnnealRatio">The annealing ratio in [0, 1].</param>
    /// <param name="cosClip">SparseNeuS clips the slope to ±10; null leaves it unclipped (NeuS).</param>
    /// <param name="pointMask">A <c>[N, 1]</c> 0/1 mask applied to the slope and the opacity (SparseNeuS), or null.</param>
    /// <returns>The opacities, <c>[N, 1]</c>, in [0, 1].</returns>
    public static Tensor<T> Alpha(IEngine engine, Tensor<T> sdf, Tensor<T> trueCos, Tensor<T> dists, Tensor<T> inverseS,
        double cosAnnealRatio, double? cosClip, Tensor<T>? pointMask)
    {
        var half = NeuSHost<T>.Scalar(0.5);
        var early = engine.TensorReLU(engine.TensorAddScalar(engine.TensorMultiplyScalar(trueCos, NeuSHost<T>.Scalar(-0.5)), half));
        var late = engine.TensorReLU(engine.TensorNegate(trueCos));
        var iterCos = engine.TensorNegate(engine.TensorAdd(
            engine.TensorMultiplyScalar(early, NeuSHost<T>.Scalar(1.0 - cosAnnealRatio)),
            engine.TensorMultiplyScalar(late, NeuSHost<T>.Scalar(cosAnnealRatio))));
        if (pointMask is not null) iterCos = engine.TensorMultiply(iterCos, pointMask);
        if (cosClip.HasValue)
            iterCos = engine.TensorClamp(iterCos, NeuSHost<T>.Scalar(-cosClip.Value), NeuSHost<T>.Scalar(cosClip.Value));

        var step = engine.TensorMultiplyScalar(engine.TensorMultiply(iterCos, dists), half);
        var prevCdf = engine.TensorSigmoid(engine.TensorMultiply(engine.TensorSubtract(sdf, step), inverseS));
        var nextCdf = engine.TensorSigmoid(engine.TensorMultiply(engine.TensorAdd(sdf, step), inverseS));
        var eps = NeuSHost<T>.Epsilon(1e-5);
        var alpha = engine.TensorDivide(
            engine.TensorAddScalar(engine.TensorSubtract(prevCdf, nextCdf), eps),
            engine.TensorAddScalar(prevCdf, eps));
        alpha = engine.TensorClamp(alpha, NeuSHost<T>.Scalar(0.0), NeuSHost<T>.Scalar(1.0));
        return pointMask is null ? alpha : engine.TensorMultiply(alpha, pointMask);
    }

    /// <summary>The weights <c>wᵢ = αᵢ Πⱼ&lt;ᵢ (1 − αⱼ + 10⁻⁷)</c>, <c>[R, S]</c>.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="alpha">The opacities, <c>[R, S]</c>.</param>
    public static Tensor<T> Weights(IEngine engine, Tensor<T> alpha)
    {
        var survive = engine.TensorAddScalar(engine.TensorNegate(alpha), NeuSHost<T>.Scalar(1.0));
        survive = engine.TensorAddScalar(survive, NeuSHost<T>.Epsilon(1e-7));
        var logSurvive = engine.TensorLog(survive);
        var before = engine.TensorSubtract(engine.TensorCumSum(logSurvive, 1), logSurvive);
        return engine.TensorMultiply(alpha, engine.TensorExp(before));
    }

    /// <summary>The weighted sum over each ray's samples, <c>[R, C]</c>.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="weights">The weights, <c>[R, S]</c>.</param>
    /// <param name="values">Per-sample values, <c>[R·S, C]</c>, ray-major.</param>
    public static Tensor<T> Composite(IEngine engine, Tensor<T> weights, Tensor<T> values)
    {
        int rays = weights.Shape[0], samples = weights.Shape[1];
        int channels = values.Shape[values.Shape.Length - 1];
        var weighted = engine.TensorMultiply(
            engine.Reshape(values, new[] { rays, samples, channels }),
            engine.Reshape(weights, new[] { rays, samples, 1 }));
        return engine.ReduceSum(weighted, new[] { 1 }, false);
    }

    /// <summary>
    /// <c>Σ mᵢ (‖∇fᵢ‖ − 1)² / (Σ mᵢ + 10⁻⁵)</c>, the Eikonal term (IGR, Gropp et al. 2020) over the masked samples.
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="gradients">∇f, <c>[N, 3]</c>.</param>
    /// <param name="mask">Per-sample 0/1 weights (NeuS: inside the radius-1.2 sphere).</param>
    /// <param name="perSampleEpsilon">Whether the 10⁻⁵ is added to every sample of the denominator,
    /// <c>sum(m + 1e-5)</c> (SyncDreamer), rather than once, <c>sum(m) + 1e-5</c> (NeuS).</param>
    public static Tensor<T> EikonalError(IEngine engine, Tensor<T> gradients, double[] mask, bool perSampleEpsilon = false)
    {
        int n = gradients.Shape[0];
        double count = perSampleEpsilon ? (n - 1) * 1e-5 : 0.0;
        foreach (double m in mask) count += m;
        var norm = Norm(engine, gradients);
        var residual = engine.TensorAddScalar(norm, NeuSHost<T>.Scalar(-1.0));
        var masked = engine.TensorMultiply(engine.TensorMultiply(residual, residual), NeuSHost<T>.FromDoubles(mask, n, 1));
        return engine.TensorMultiplyScalar(engine.ReduceSum(masked, new[] { 0, 1 }, false), NeuSHost<T>.Scalar(1.0 / (count + 1e-5)));
    }

    /// <summary>The row norms <c>‖v‖</c>, <c>[N, 1]</c>, with the squared norm floored so the root's gradient stays finite.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="vectors">The vectors, <c>[N, C]</c>.</param>
    public static Tensor<T> Norm(IEngine engine, Tensor<T> vectors)
    {
        var squared = engine.ReduceSum(engine.TensorMultiply(vectors, vectors), new[] { 1 }, true);
        return engine.TensorSqrt(engine.TensorMax(squared, NeuSHost<T>.Epsilon(1e-12)));
    }

    /// <summary>The section lengths and midpoints of sorted depths, the last section <paramref name="sampleDist"/> long.</summary>
    /// <param name="z">The depths, <c>[R, S]</c>.</param>
    /// <param name="samples">S.</param>
    /// <param name="sampleDist">The last section's length, or null to repeat the previous section's
    /// (<c>torch.cat([dists, dists[..., -1:]], -1)</c>, SyncDreamer).</param>
    public static (double[] Dists, double[] Mid) Sections(double[] z, int samples, double? sampleDist)
    {
        var dists = new double[z.Length];
        var mid = new double[z.Length];
        int rays = z.Length / samples;
        for (int r = 0; r < rays; r++)
            for (int s = 0; s < samples; s++)
            {
                int i = r * samples + s;
                dists[i] = s + 1 < samples ? z[i + 1] - z[i]
                    : sampleDist ?? (samples > 1 ? z[i] - z[i - 1] : 0.0);
                mid[i] = z[i] + dists[i] * 0.5;
            }

        return (dists, mid);
    }
}
