namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Real, centered, periodic-Hann DFT for Kokoro's detached harmonic excitation.
/// Pairing opposite samples before summation preserves the very small imaginary components
/// of nearly even frames; discarding those components changes the learned phase conditioning.</summary>
internal sealed class KokoroHarmonicStft<T>
{
    private readonly int _fft, _hop;
    private readonly double[] _window;
    public KokoroHarmonicStft(int fft = 20, int hop = 5, double[]? window = null)
    {
        if (fft < 2 || fft % 2 != 0 || hop <= 0) throw new ArgumentOutOfRangeException(nameof(fft));
        _fft = fft; _hop = hop;
        if (window is not null && window.Length != fft) throw new ArgumentException("STFT window length differs.", nameof(window));
        _window = window is null ? Enumerable.Range(0, fft).Select(n => .5 - .5 * Math.Cos(2 * Math.PI * n / fft)).ToArray()
            : (double[])window.Clone();
    }

    public Tensor<T> Forward(Tensor<T> source)
    {
        var operations = MathHelper.GetNumericOperations<T>();
        var signal = source.ToArray();
        int length = signal.Length, half = _fft / 2, bins = half + 1, frames = 1 + length / _hop;
        if (length <= half) throw new ArgumentException("Harmonic source is too short for centered reflection padding.", nameof(source));
        var result = new Tensor<T>(new[] { 1, 2 * bins, frames });
        var framed = new double[_fft];
        for (int frame = 0; frame < frames; frame++)
        {
            for (int n = 0; n < _fft; n++)
            {
                int index = frame * _hop + n - half;
                if (index < 0) index = -index;
                if (index >= length) index = 2 * length - 2 - index;
                framed[n] = operations.ToDouble(signal[index]) * _window[n];
            }
            for (int k = 0; k < bins; k++)
            {
                double real = framed[0] + (k % 2 == 0 ? framed[half] : -framed[half]), imaginary = 0;
                for (int n = 1; n < half; n++)
                {
                    double angle = 2 * Math.PI * k * n / _fft;
                    real += (framed[n] + framed[_fft - n]) * Math.Cos(angle);
                    imaginary -= (framed[n] - framed[_fft - n]) * Math.Sin(angle);
                }
                if (k == 0 || k == half) imaginary = 0;
                result[0, k, frame] = operations.FromDouble(Math.Sqrt(real * real + imaginary * imaginary));
                result[0, bins + k, frame] = operations.FromDouble(Math.Atan2(imaginary, real));
            }
        }
        return result;
    }
}
