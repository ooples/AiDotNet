namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>TorToiSe's linear-beta, learned-range Gaussian diffusion, including spaced DDPM transitions.</summary>
internal sealed class TortoiseDiffusionSchedule<T>
{
    private readonly IEngine _engine;
    private readonly int _steps, _channels;
    private readonly double[] _alpha;
    private static readonly INumericOperations<T> Num = TortoiseOps<T>.NumOps;

    internal TortoiseDiffusionSchedule(IEngine engine, int steps, int channels)
    {
        _engine = engine; _steps = steps; _channels = channels;
        _alpha = new double[steps]; double product = 1;
        for (int i = 0; i < steps; i++)
        {
            // OpenAI's linear schedule is defined for 1,000 steps and rescaled to the trained step count.
            double beta = 1000.0 / steps * (0.0001 + (0.02 - 0.0001) * i / Math.Max(1, steps - 1));
            product *= 1 - beta; _alpha[i] = product;
        }
    }

    internal double Alpha(int step) => step < 0 ? 1 : _alpha[step];

    internal (Tensor<T> Mean, Tensor<T> LogVariance, Tensor<T> Clean) Statistics(Tensor<T> noisy,
        Tensor<T> prediction, int step, int previous, double clippedPosteriorVariance,
        Tensor<T>? unconditioned = null, double guidance = 2, bool clipClean = true,
        int? guidanceIndex = null, int? guidanceSteps = null)
    {
        double alpha = Alpha(step), priorAlpha = Alpha(previous), beta = 1 - alpha / priorAlpha;
        var epsilon = _engine.TensorNarrow(prediction, 1, 0, _channels);
        if (unconditioned is not null)
        {
            // The release enables ramp_conditioning_free by default. SpacedDiffusion
            // ramps by the index in the selected schedule, not the original model timestep.
            guidance *= 1 - (guidanceIndex ?? step) / (double)(guidanceSteps ?? _steps);
            epsilon = _engine.TensorSubtract(_engine.TensorMultiplyScalar(epsilon, Num.FromDouble(1 + guidance)),
                _engine.TensorMultiplyScalar(_engine.TensorNarrow(unconditioned, 1, 0, _channels), Num.FromDouble(guidance)));
        }
        var variance = _engine.TensorNarrow(prediction, 1, _channels, _channels);
        double minLog = Math.Log(Math.Max(clippedPosteriorVariance, 1e-20)), maxLog = Math.Log(beta);
        var logVariance = _engine.TensorAddScalar(_engine.TensorMultiplyScalar(
            _engine.TensorAddScalar(variance, Num.One), Num.FromDouble(0.5 * (maxLog - minLog))), Num.FromDouble(minLog));
        var clean = _engine.TensorMultiplyScalar(_engine.TensorSubtract(noisy,
            _engine.TensorMultiplyScalar(epsilon, Num.FromDouble(Math.Sqrt(1 - alpha)))), Num.FromDouble(1 / Math.Sqrt(alpha)));
        if (clipClean) clean = _engine.TensorClamp(clean, Num.FromDouble(-1), Num.One);
        var mean = _engine.TensorAdd(_engine.TensorMultiplyScalar(clean, Num.FromDouble(beta * Math.Sqrt(priorAlpha) / (1 - alpha))),
            _engine.TensorMultiplyScalar(noisy, Num.FromDouble((1 - priorAlpha) * Math.Sqrt(1 - beta) / (1 - alpha))));
        return (mean, logVariance, clean);
    }

    internal double PosteriorVariance(int step, int previous) =>
        (1 - Alpha(previous)) / (1 - Alpha(step)) * (1 - Alpha(step) / Alpha(previous));

    internal Tensor<T> Noisy(Tensor<T> clean, Tensor<T> noise, int step) =>
        _engine.TensorAdd(_engine.TensorMultiplyScalar(clean, Num.FromDouble(Math.Sqrt(Alpha(step)))),
            _engine.TensorMultiplyScalar(noise, Num.FromDouble(Math.Sqrt(1 - Alpha(step)))));

    internal (Tensor<T> Mse, Tensor<T> Variational, Tensor<T> Total) TrainingTerms(
        Tensor<T> clean, Tensor<T> noise, Tensor<T> prediction, int step)
    {
        var noisy = Noisy(clean, noise, step);
        var epsilon = _engine.TensorNarrow(prediction, 1, 0, _channels);
        var variance = _engine.TensorNarrow(prediction, 1, _channels, _channels);
        var frozen = _engine.TensorConcatenate(new[] {
            new Tensor<T>(epsilon.Shape.ToArray(), new Vector<T>(epsilon.ToArray())), variance }, 1);
        int previous = step - 1;
        double clipped = step == 0 ? PosteriorVariance(1, 0) : PosteriorVariance(step, previous);
        var stats = Statistics(noisy, frozen, step, previous, clipped, clipClean: false);
        double alpha = Alpha(step), priorAlpha = Alpha(previous), beta = 1 - alpha / priorAlpha;
        var trueMean = _engine.TensorAdd(_engine.TensorMultiplyScalar(clean, Num.FromDouble(beta * Math.Sqrt(priorAlpha) / (1 - alpha))),
            _engine.TensorMultiplyScalar(noisy, Num.FromDouble((1 - priorAlpha) * Math.Sqrt(1 - beta) / (1 - alpha))));
        Tensor<T> Mean(Tensor<T> value) => _engine.ReduceMean(value, new[] { 0, 1, 2 }, true);
        var difference = _engine.TensorSubtract(epsilon, noise);
        var mse = Mean(_engine.TensorMultiply(difference, difference));
        Tensor<T> vb;
        if (step > 0)
        {
            var delta = _engine.TensorSubtract(trueMean, stats.Mean);
            var kl = _engine.TensorAdd(_engine.TensorAddScalar(stats.LogVariance, Num.FromDouble(-Math.Log(clipped) - 1)),
                _engine.TensorMultiply(_engine.TensorAddScalar(_engine.TensorMultiply(delta, delta), Num.FromDouble(clipped)),
                    _engine.TensorExp(_engine.TensorNegate(stats.LogVariance))));
            vb = _engine.TensorMultiplyScalar(Mean(kl), Num.FromDouble(0.5 / Math.Log(2)));
        }
        else
        {
            // The release uses a discretized Gaussian likelihood (8-bit bins) at t=0.
            var centered = _engine.TensorSubtract(clean, stats.Mean);
            var inverseStd = _engine.TensorExp(_engine.TensorMultiplyScalar(stats.LogVariance, Num.FromDouble(-0.5)));
            Tensor<T> Cdf(Tensor<T> x)
            {
                var cube = _engine.TensorMultiply(_engine.TensorMultiply(x, x), x);
                var inner = _engine.TensorMultiplyScalar(_engine.TensorAdd(x,
                    _engine.TensorMultiplyScalar(cube, Num.FromDouble(0.044715))), Num.FromDouble(Math.Sqrt(2 / Math.PI)));
                return _engine.TensorMultiplyScalar(_engine.TensorAddScalar(_engine.Tanh(inner), Num.One), Num.FromDouble(0.5));
            }
            var plus = Cdf(_engine.TensorMultiply(inverseStd, _engine.TensorAddScalar(centered, Num.FromDouble(1.0 / 255))));
            var minus = Cdf(_engine.TensorMultiply(inverseStd, _engine.TensorAddScalar(centered, Num.FromDouble(-1.0 / 255))));
            var low = new Tensor<T>(clean.Shape.ToArray()); var high = new Tensor<T>(clean.Shape.ToArray());
            var middle = new Tensor<T>(clean.Shape.ToArray());
            for (int i = 0; i < clean.Length; i++)
            {
                double value = Num.ToDouble(clean[i]);
                if (value < -0.999) low[i] = Num.One;
                else if (value > 0.999) high[i] = Num.One;
                else middle[i] = Num.One;
            }
            Tensor<T> Log(Tensor<T> x) => _engine.TensorLog(_engine.TensorClampMin(x, Num.FromDouble(1e-12)));
            var logProbability = _engine.TensorAdd(_engine.TensorAdd(_engine.TensorMultiply(low, Log(plus)),
                _engine.TensorMultiply(high, Log(_engine.TensorAddScalar(_engine.TensorNegate(minus), Num.One)))),
                _engine.TensorMultiply(middle, Log(_engine.TensorSubtract(plus, minus))));
            vb = _engine.TensorMultiplyScalar(Mean(logProbability), Num.FromDouble(-1 / Math.Log(2)));
        }
        return (mse, vb, _engine.TensorAdd(mse, vb));
    }
}
