using AiDotNet.LearningRateSchedulers;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// fairseq's <c>polynomial_decay</c> schedule: the first update runs at peak / W, update k (k updates done, 1 ≤ k ≤ W) at
/// peak · k / W, then the rate decays as (1 − (k − W) / (T − W))^power to the end rate, which it holds from update T.
/// </summary>
internal sealed class FairseqPolynomialDecaySchedule : LearningRateSchedulerBase
{
    private readonly int _warmup;
    private readonly int _total;
    private readonly double _power;
    private readonly double _end;

    public FairseqPolynomialDecaySchedule(double peak, int warmup, int total, double power = 1.0, double end = 0.0)
        : base(peak)
    {
        if (warmup < 0) throw new ArgumentOutOfRangeException(nameof(warmup));
        if (total <= warmup) throw new ArgumentOutOfRangeException(nameof(total), "The schedule runs past its warm-up.");
        _warmup = warmup;
        _total = total;
        _power = power;
        _end = end;
        _currentLearningRate = ComputeLearningRate(0);
    }

    protected override double ComputeLearningRate(int step)
    {
        if (_warmup > 0 && step == 0) return _baseLearningRate / _warmup;
        if (_warmup > 0 && step <= _warmup) return _baseLearningRate * step / _warmup;
        if (step >= _total) return _end;
        double remaining = 1.0 - (double)(step - _warmup) / (_total - _warmup);
        return (_baseLearningRate - _end) * Math.Pow(remaining, _power) + _end;
    }
}
