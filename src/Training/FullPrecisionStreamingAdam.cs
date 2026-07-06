namespace AiDotNet.Training;

/// <summary>
/// #1662 lever #1: full-precision per-parameter Adam for the streaming optimizer-in-backward
/// path. Bit-identical to the classic <c>AdamOptimizer</c> whole-vector step (same formula,
/// bias-correction, epsilon placement, and once-per-step <c>t</c>) — NO update clamping and NO
/// 8-bit moment quantization, unlike <see cref="StreamingAdam8Bit{T}"/>. Used when the streaming
/// fused path engages for a model that FITS in memory (the common-case throughput/locality win),
/// where matching the classic trajectory exactly is required.
/// </summary>
internal sealed class FullPrecisionStreamingAdam<T> : FullPrecisionStreamingOptimizer<T>
{
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private readonly double _weightDecay;

    public FullPrecisionStreamingAdam(
        double learningRate,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8,
        double weightDecay = 0.0)
        : base(nameof(FullPrecisionStreamingAdam<T>), learningRate, momentCount: 2)
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
    }

    protected override double UpdateElement(
        double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        // m = beta1*m + (1-beta1)*g ;  v = beta2*v + (1-beta2)*g^2
        double m = _beta1 * moments[0] + (1.0 - _beta1) * gradient;
        double v = _beta2 * moments[1] + (1.0 - _beta2) * gradient * gradient;
        nextMoments[0] = m;
        nextMoments[1] = v;

        // Bias correction with the global step t (advanced once per BeginStep, matching
        // AdamOptimizer._t). For t >= 1 these are always > 0, so no guard is needed.
        double biasCorr1 = 1.0 - Math.Pow(_beta1, Step);
        double biasCorr2 = 1.0 - Math.Pow(_beta2, Step);

        parameter = ApplyDecoupledWeightDecay(parameter, _weightDecay);
        // update = lr * mHat / (sqrt(vHat) + eps) ; NO clamping (classic Adam does not clamp).
        double update = LearningRate * (m / biasCorr1) / (Math.Sqrt(v / biasCorr2) + _epsilon);
        return parameter - update;
    }
}
