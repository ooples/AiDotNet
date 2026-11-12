using AiDotNet.Interfaces;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>AudioCraft's subsampled diffusion controller. The final listed timestep is a
/// boundary, not another model evaluation. Gaussian draws are explicit for reproducible audits.</summary>
internal sealed class MetaVoiceDiffusionSchedule<T>
{
    private readonly IEngine _engine;
    private readonly double[] _betas;
    public MetaVoiceDiffusionSchedule(IEngine engine, IReadOnlyList<double> betas)
    {
        if (betas.Count < 2 || betas.Any(x => double.IsNaN(x) || x <= 0 || x >= 1))
            throw new ArgumentException("Diffusion betas must lie strictly between zero and one.", nameof(betas));
        _engine = engine; _betas = betas.ToArray();
    }

    public IReadOnlyList<Tensor<T>> Generate(Tensor<T> initial, Tensor<T> condition,
        Func<Tensor<T>, int, Tensor<T>, Tensor<T>> estimate, Func<int, Tensor<T>> gaussian,
        IReadOnlyList<int>? stepList = null, double noiseScale = 1, double clip = 5)
    {
        if (noiseScale <= 0 || double.IsInfinity(noiseScale) || double.IsNaN(noiseScale) || clip < 0 || double.IsNaN(clip))
            throw new ArgumentOutOfRangeException(nameof(noiseScale));
        int[] steps = stepList?.ToArray() ?? Enumerable.Range(0, 20).Select(i => 999 - i * 50).Concat(new[] { 0 }).ToArray();
        if (steps.Length < 2 || steps.Any(x => x < 0 || x >= _betas.Length) ||
            steps.Zip(steps.Skip(1), (a, b) => a > b).Any(valid => !valid))
            throw new ArgumentException("Sampling timesteps must be valid and strictly decreasing.", nameof(stepList));
        var alphaBars = new double[_betas.Length]; double product = 1;
        for (int i = 0; i < _betas.Length; i++) alphaBars[i] = product *= 1 - _betas[i];
        double alphaBar = alphaBars[alphaBars.Length - 1];
        var current = Scale(initial, noiseScale);
        var trajectory = new List<Tensor<T>> { current };
        for (int index = 0; index < steps.Length - 1; index++)
        {
            int step = steps[index];
            // Reversed selected alpha-bars -> betas_from_alpha_bar -> reversed beta indexing.
            double alpha = alphaBars[step] / alphaBars[steps[index + 1]];
            var prediction = Scale(estimate(current, step, condition), noiseScale);
            var previous = Scale(_engine.TensorSubtract(current,
                Scale(prediction, (1 - alpha) / Math.Sqrt(1 - alphaBar))), 1 / Math.Sqrt(alpha));
            double previousAlphaBar = alphaBars[steps[index + 1]];
            double variance = index == steps.Length - 2 ? 0 : (1 - previousAlphaBar) / (1 - alphaBar) * (1 - alpha);
            if (variance > 0)
            {
                var draw = gaussian(index);
                if (!draw.Shape.ToArray().SequenceEqual(initial.Shape.ToArray())) throw new ArgumentException("Gaussian draw shape must match the audio.", nameof(gaussian));
                previous = _engine.TensorAdd(previous, Scale(draw, Math.Sqrt(variance) * noiseScale));
            }
            if (clip > 0)
            {
                var numeric = MathHelper.GetNumericOperations<T>();
                previous = _engine.TensorClamp(previous, numeric.FromDouble(-clip), numeric.FromDouble(clip));
            }
            current = previous;
            alphaBar = index == steps.Length - 2 ? 1 : previousAlphaBar;
            trajectory.Add(current);
        }
        return trajectory;
    }

    private Tensor<T> Scale(Tensor<T> value, double scale)
        => _engine.TensorMultiplyScalar(value, MathHelper.GetNumericOperations<T>().FromDouble(scale));
}
