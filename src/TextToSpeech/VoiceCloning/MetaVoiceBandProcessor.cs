using AiDotNet.Interfaces;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>AudioCraft's fitted multiband sample processor, including its original
/// second-moment target statistic (which is not square-rooted).</summary>
internal sealed class MetaVoiceBandProcessor<T>
{
    private readonly IEngine _engine;
    private readonly MetaVoiceBandFilters<T> _split;
    private readonly double _power;
    private readonly int _bands, _fitSamples;
    private double _count;
    private Tensor<T> _sum, _squares, _targetSquares;
    internal double FittedCount => _count;

    public MetaVoiceBandProcessor(IEngine engine, double power, int bands = 8, int fitSamples = 10000)
    {
        if (double.IsNaN(power) || double.IsInfinity(power) || bands < 2 || fitSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(power));
        _engine = engine; _power = power; _bands = bands; _fitSamples = fitSamples;
        _split = new MetaVoiceBandFilters<T>(engine, bands);
        _sum = new Tensor<T>(new[] { bands }); _squares = new Tensor<T>(new[] { bands }); _targetSquares = new Tensor<T>(new[] { bands });
    }

    public void LoadTorch(Func<string, int[], double[]> read)
    {
        _count = read("counts", new[] { 1 })[0];
        Tensor<T> Read(string name) => new(new[] { _bands }, new Vector<T>(read(name, new[] { _bands })
            .Select(x => MathHelper.GetNumericOperations<T>().FromDouble(x)).ToArray()));
        _sum = Read("sum_x"); _squares = Read("sum_x2"); _targetSquares = Read("sum_target_x2");
        _split.LoadFilters(read("split_bands.lowpass.filters", _split.TorchFilterShape));
        if (_count <= 0 || double.IsNaN(_count) || double.IsInfinity(_count)) throw new ArgumentException("Fitted band processor requires a positive count.");
    }

    private Tensor<T> Scalar(Tensor<T> vector, int band) => _engine.Reshape(
        _engine.TensorSlice(vector, new[] { band }, new[] { 1 }), new[] { 1, 1, 1 });
    private Tensor<T> Scale(Tensor<T> value, double scale) => _engine.TensorMultiplyScalar(value, MathHelper.GetNumericOperations<T>().FromDouble(scale));

    private (Tensor<T> Mean, Tensor<T> Std, Tensor<T> Target) Statistics(int band)
    {
        if (_count <= 0) throw new InvalidOperationException("The band processor must be fitted before inversion.");
        var mean = Scale(Scalar(_sum, band), 1 / _count);
        var variance = _engine.TensorSubtract(Scale(Scalar(_squares, band), 1 / _count), _engine.TensorMultiply(mean, mean));
        var std = _engine.TensorSqrt(_engine.TensorClampMin(variance, MathHelper.GetNumericOperations<T>().Zero));
        return (mean, std, Scale(Scalar(_targetSquares, band), 1 / _count));
    }

    public Tensor<T> ReturnSample(Tensor<T> input)
    {
        var bands = _split.Split(input); var output = new Tensor<T>(input.Shape.ToArray());
        for (int i = 0; i < _bands; i++)
        {
            var (mean, std, target) = Statistics(i);
            var scale = _engine.TensorPow(_engine.TensorDivide(std, target), MathHelper.GetNumericOperations<T>().FromDouble(_power));
            output = _engine.TensorAdd(output, _engine.TensorAdd(_engine.TensorMultiply(bands[i],
                _engine.TensorTile(scale, input.Shape.ToArray())), _engine.TensorTile(mean, input.Shape.ToArray())));
        }
        return output;
    }

    public Tensor<T> ProjectSample(Tensor<T> input, Func<Tensor<T>> referenceNoise)
    {
        var bands = _split.Split(input);
        if (_count < _fitSamples)
        {
            var noise = referenceNoise();
            if (!noise.Shape.ToArray().SequenceEqual(input.Shape.ToArray())) throw new ArgumentException("Fitting noise must match the input.", nameof(referenceNoise));
            var target = _split.Split(noise);
            Tensor<T> Aggregate(bool square, Tensor<T>[] values)
            {
                var entries = values.Select(x => _engine.Reshape(Scale(_engine.ReduceSum(square ? _engine.TensorMultiply(x, x) : x,
                    new[] { 0, 1, 2 }, true), 1.0 / (x.Shape[1] * x.Shape[2])), new[] { 1 })).ToArray();
                return _engine.TensorConcatenate(entries, 0);
            }
            _count += input.Shape[0];
            _sum = _engine.TensorAdd(_sum, Aggregate(false, bands));
            _squares = _engine.TensorAdd(_squares, Aggregate(true, bands));
            _targetSquares = _engine.TensorAdd(_targetSquares, Aggregate(true, target));
        }
        var output = new Tensor<T>(input.Shape.ToArray());
        for (int i = 0; i < _bands; i++)
        {
            var (mean, std, target) = Statistics(i);
            var scale = _engine.TensorPow(_engine.TensorDivide(target,
                _engine.TensorClampMin(std, MathHelper.GetNumericOperations<T>().FromDouble(1e-12))), MathHelper.GetNumericOperations<T>().FromDouble(_power));
            output = _engine.TensorAdd(output, _engine.TensorMultiply(_engine.TensorSubtract(bands[i],
                _engine.TensorTile(mean, input.Shape.ToArray())), _engine.TensorTile(scale, input.Shape.ToArray())));
        }
        return output;
    }
}
