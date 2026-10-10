using AiDotNet.Tensors.Engines;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// The structural similarity index (SSIM, Wang et al. 2004) of two spectrograms treated as single-channel images, as
/// SpeedySpeech's loss computes it.
/// </summary>
/// <remarks>
/// <para>
/// Matches <c>pytorch_ssim.ssim</c> in the SpeedySpeech implementation: local statistics through an 11 × 11 Gaussian
/// window (σ = 1.5, normalized) applied as a convolution with 5 frames/bins of zero padding, constants
/// <c>C1 = 0.01²</c> and <c>C2 = 0.03²</c>, and the SSIM map averaged over every position.
/// </para>
/// <para>
/// The Gaussian window is separable, so the zero-padded 2-D convolution is <c>G_t · X · G_f</c> with banded matrices
/// <c>G[i, j] = g(j − i)</c> for <c>|j − i| ≤ 5</c>; rows near an edge simply lose the weights that fall on the padding.
/// Written with matrix products, the whole index is differentiable.
/// </para>
/// </remarks>
internal static class SpectrogramSsim
{
    private const int WindowSize = 11;
    private const double Sigma = 1.5;
    private const double C1 = 0.01 * 0.01;
    private const double C2 = 0.03 * 0.03;

    /// <summary>SSIM of <paramref name="a"/> and <paramref name="b"/>, both <c>[frames, bins]</c>, as a scalar tensor.</summary>
    public static Tensor<T> Ssim<T>(IEngine engine, Tensor<T> a, Tensor<T> b)
    {
        if (a.Rank != 2 || b.Rank != 2 || a.Shape[0] != b.Shape[0] || a.Shape[1] != b.Shape[1])
            throw new ArgumentException($"Expected two [frames, bins] spectrograms of one shape, got [{string.Join(", ", a.Shape)}] and [{string.Join(", ", b.Shape)}].");
        int frames = a.Shape[0], bins = a.Shape[1];
        var ops = MathHelper.GetNumericOperations<T>();
        var timeWindow = Band<T>(frames);
        var binWindow = Band<T>(bins);
        Tensor<T> Blur(Tensor<T> x) => engine.TensorMatMul(engine.TensorMatMul(timeWindow, x), binWindow);

        var muA = Blur(a);
        var muB = Blur(b);
        var muA2 = engine.TensorMultiply(muA, muA);
        var muB2 = engine.TensorMultiply(muB, muB);
        var muAB = engine.TensorMultiply(muA, muB);
        var sigmaA2 = engine.TensorSubtract(Blur(engine.TensorMultiply(a, a)), muA2);
        var sigmaB2 = engine.TensorSubtract(Blur(engine.TensorMultiply(b, b)), muB2);
        var sigmaAB = engine.TensorSubtract(Blur(engine.TensorMultiply(a, b)), muAB);

        T two = ops.FromDouble(2.0), c1 = ops.FromDouble(C1), c2 = ops.FromDouble(C2);
        var numerator = engine.TensorMultiply(
            engine.TensorAddScalar(engine.TensorMultiplyScalar(muAB, two), c1),
            engine.TensorAddScalar(engine.TensorMultiplyScalar(sigmaAB, two), c2));
        var denominator = engine.TensorMultiply(
            engine.TensorAddScalar(engine.TensorAdd(muA2, muB2), c1),
            engine.TensorAddScalar(engine.TensorAdd(sigmaA2, sigmaB2), c2));
        var map = engine.TensorDivide(numerator, denominator);
        return engine.ReduceMean(map, new[] { 0, 1 }, keepDims: false);
    }

    /// <summary>The <c>n × n</c> symmetric band matrix applying the normalized 1-D Gaussian window with zero padding.</summary>
    private static Tensor<T> Band<T>(int n)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int half = WindowSize / 2;
        var weights = new double[WindowSize];
        double total = 0;
        for (int i = 0; i < WindowSize; i++)
        {
            weights[i] = Math.Exp(-(i - half) * (i - half) / (2 * Sigma * Sigma));
            total += weights[i];
        }
        var band = new Tensor<T>(new[] { n, n });
        for (int i = 0; i < n; i++)
            for (int j = Math.Max(0, i - half); j <= Math.Min(n - 1, i + half); j++)
                band[i, j] = ops.FromDouble(weights[j - i + half] / total);
        return band;
    }
}
