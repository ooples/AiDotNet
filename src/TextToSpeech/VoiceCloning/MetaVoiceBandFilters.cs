using AiDotNet.Interfaces;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Julius 0.2.7's mel-spaced, Hann-windowed sinc band decomposition.
/// Replicate padding and telescoping low-pass differences preserve the complete input.</summary>
internal sealed class MetaVoiceBandFilters<T>
{
    private readonly IEngine _engine;
    private Tensor<T> _filters;
    private readonly int _bands, _half;

    public MetaVoiceBandFilters(IEngine engine, int bands = 32, int sampleRate = 24000)
    {
        if (bands < 2 || sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(bands));
        _engine = engine; _bands = bands;
        double highMel = 2595 * Math.Log10(1 + sampleRate / 1400.0);
        var cutoffs = Enumerable.Range(1, bands - 1)
            .Select(i => 700 * (Math.Pow(10, highMel * i / bands / 2595) - 1) / sampleRate).ToArray();
        _half = (int)(4 / cutoffs[0]);
        int kernel = 2 * _half + 1;
        _filters = new Tensor<T>(new[] { bands - 1, 1, 1, kernel });
        var numeric = MathHelper.GetNumericOperations<T>();
        for (int band = 0; band < bands - 1; band++)
        {
            var values = new double[kernel]; double sum = 0;
            for (int tap = 0; tap < kernel; tap++)
            {
                double phase = 2 * cutoffs[band] * Math.PI * (tap - _half);
                double window = .5 * (1 - Math.Cos(2 * Math.PI * tap / (kernel - 1)));
                values[tap] = 2 * cutoffs[band] * window * (phase == 0 ? 1 : Math.Sin(phase) / phase);
                sum += values[tap];
            }
            for (int tap = 0; tap < kernel; tap++) _filters[band * kernel + tap] = numeric.FromDouble(values[tap] / sum);
        }
    }

    public int[] TorchFilterShape => new[] { _bands - 1, 1, 2 * _half + 1 };

    public void LoadFilters(double[] values)
    {
        if (values.Length != _filters.Length) throw new ArgumentException("Julius low-pass buffer size differs.", nameof(values));
        var numeric = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < values.Length; i++) _filters[i] = numeric.FromDouble(values[i]);
        _engine.InvalidatePersistentTensor(_filters);
    }

    public Tensor<T>[] Split(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[2] <= 0) throw new ArgumentException("Expected BCT audio.", nameof(input));
        int batch = input.Shape[0], channels = input.Shape[1], time = input.Shape[2];
        var flattened = _engine.Reshape(input, new[] { batch * channels, 1, 1, time });
        Tensor<T> Edge(int offset) => _engine.TensorTile(_engine.TensorSlice(flattened,
            new[] { 0, 0, 0, offset }, new[] { batch * channels, 1, 1, 1 }), new[] { 1, 1, 1, _half });
        var padded = _engine.TensorConcatenate(new[] { Edge(0), flattened, Edge(time - 1) }, 3);
        // Direct convolution implements the same linear filter as Julius's optional FFT convolution.
        var lows = _engine.Conv2D(padded, _filters, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var result = new Tensor<T>[_bands]; Tensor<T>? previous = null;
        for (int i = 0; i < _bands - 1; i++)
        {
            var low = _engine.Reshape(_engine.TensorSlice(lows, new[] { 0, i, 0, 0 },
                new[] { batch * channels, 1, 1, time }), input.Shape.ToArray());
            result[i] = previous is null ? low : _engine.TensorSubtract(low, previous);
            previous = low;
        }
        result[_bands - 1] = _engine.TensorSubtract(input, previous!);
        return result;
    }

    public Tensor<T> MatchEqualization(Tensor<T> waveform, Tensor<T> reference, double strictness = 1)
    {
        if (!waveform.Shape.ToArray().SequenceEqual(reference.Shape.ToArray()) || waveform.Length < 2)
            throw new ArgumentException("Equalization requires equal audio shapes with at least two samples.", nameof(reference));
        var bands = Split(waveform); var target = Split(reference);
        var numeric = MathHelper.GetNumericOperations<T>();
        Tensor<T> Std(Tensor<T> x)
        {
            var centered = _engine.TensorSubtract(x, _engine.TensorTile(_engine.ReduceMean(x, new[] { 0, 1, 2 }, true), x.Shape.ToArray()));
            return _engine.TensorSqrt(_engine.TensorMultiplyScalar(_engine.ReduceSum(
                _engine.TensorMultiply(centered, centered), new[] { 0, 1, 2 }, true), numeric.FromDouble(1.0 / (x.Length - 1))));
        }
        var output = new Tensor<T>(reference.Shape.ToArray());
        for (int i = 0; i < bands.Length; i++)
        {
            // Author uses global unbiased std over the whole band, without an epsilon or per-item reduction.
            var ratio = _engine.TensorPow(_engine.TensorDivide(Std(target[i]), Std(bands[i])), numeric.FromDouble(strictness));
            output = _engine.TensorAdd(output, _engine.TensorMultiply(bands[i], _engine.TensorTile(ratio, waveform.Shape.ToArray())));
        }
        return output;
    }
}
