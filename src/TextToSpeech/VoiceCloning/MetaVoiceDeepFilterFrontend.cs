using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>DeepFilterNet 0.5.6's 48 kHz libdf analysis, Vorbis window,
/// exponential feature normalization and overlap-add synthesis.</summary>
internal sealed class MetaVoiceDeepFilterFrontend
{
    internal const int Fft = 960, Hop = 480, Bins = 481, Bands = 32, DeepBins = 96;
    private readonly IEngine _engine;
    private readonly int[] _widths;
    private readonly float[] _window = new float[Fft];

    internal MetaVoiceDeepFilterFrontend(IEngine engine, int[] erbWidths)
    {
        if (erbWidths.Length != Bands || erbWidths.Any(x => x <= 0) || erbWidths.Sum() != Bins)
            throw new ArgumentException("Expected the released DeepFilterNet3 ERB partition.", nameof(erbWidths));
        _engine = engine; _widths = (int[])erbWidths.Clone();
        for (int i = 0; i < Fft; i++)
        {
            double sine = Math.Sin(Math.PI * (i + .5) / Fft);
            _window[i] = (float)Math.Sin(.5 * Math.PI * sine * sine);
        }
    }

    internal (Tensor<float> Spectrum, Tensor<float> Erb, Tensor<float> Features) Analyze(float[] audio, bool pad = false)
    {
        if (audio.Length == 0) throw new ArgumentException("A nonempty mono waveform is required.", nameof(audio));
        int frames = (audio.Length + (pad ? Fft : 0)) / Hop;
        if (frames == 0) throw new ArgumentException("Unpadded libdf analysis needs one complete hop.", nameof(audio));
        var windowed = new Tensor<float>(new[] { frames, Fft });
        for (int frame = 0; frame < frames; frame++) for (int n = 0; n < Fft; n++)
        {
            int index = frame * Hop + n - Hop;
            windowed[frame, n] = (index >= 0 && index < audio.Length ? audio[index] : 0) * _window[n];
        }
        // libdf uses an unnormalized forward FFT and a float32 1/960 analysis gain.
        var spectrum = new Tensor<float>(new[] { 1, 1, frames, Bins, 2 });
        float gain = 1f / Fft;
        var transform = new FastFourierTransform<float>();
        for (int frame = 0; frame < frames; frame++)
        {
            var samples = new Vector<float>(Fft);
            for (int n = 0; n < Fft; n++) samples[n] = windowed[frame, n];
            // IEngine.RFFT pads 960 to 1024; libdf requires the exact 960-bin grid.
            var fft = transform.Forward(samples);
            for (int k = 0; k < Bins; k++)
            {
                spectrum[0, 0, frame, k, 0] = fft[k].Real * gain;
                spectrum[0, 0, frame, k, 1] = fft[k].Imaginary * gain;
            }
        }
        var erb = new Tensor<float>(new[] { 1, 1, frames, Bands });
        var features = new Tensor<float>(new[] { 1, 1, frames, DeepBins, 2 });
        var mean = new float[Bands]; var unit = new float[DeepBins];
        float meanStep = -30f / (Bands - 1), unitStep = (.0001f - .001f) / (DeepBins - 1);
        for (int i = 0; i < Bands; i++) mean[i] = -60f + i * meanStep;
        for (int i = 0; i < DeepBins; i++) unit[i] = .001f + i * unitStep;
        // get_norm_alpha rounds exp(-hop/sr/tau) to three decimal places.
        const float alpha = .99f;
        for (int frame = 0; frame < frames; frame++)
        {
            int start = 0;
            for (int band = 0; band < Bands; band++)
            {
                float energy = 0, inverseWidth = 1f / _widths[band];
                for (int k = 0; k < _widths[band]; k++)
                {
                    float real = spectrum[0, 0, frame, start + k, 0], imaginary = spectrum[0, 0, frame, start + k, 1];
                    energy += (real * real + imaginary * imaginary) * inverseWidth;
                }
                float decibels = (float)Math.Log10(energy + 1e-10f) * 10f;
                mean[band] = decibels * (1f - alpha) + mean[band] * alpha;
                erb[0, 0, frame, band] = (decibels - mean[band]) / 40f;
                start += _widths[band];
            }
            for (int k = 0; k < DeepBins; k++)
            {
                float real = spectrum[0, 0, frame, k, 0], imaginary = spectrum[0, 0, frame, k, 1];
                float magnitude = (float)Math.Sqrt(real * real + imaginary * imaginary);
                unit[k] = magnitude * (1f - alpha) + unit[k] * alpha;
                float divisor = (float)Math.Sqrt(unit[k]);
                features[0, 0, frame, k, 0] = real / divisor;
                features[0, 0, frame, k, 1] = imaginary / divisor;
            }
        }
        return (spectrum, erb, features);
    }

    internal float[] Synthesize(Tensor<float> spectrum, int? originalLength = null)
    {
        if (spectrum.Rank != 5 || spectrum.Shape[0] != 1 || spectrum.Shape[1] != 1 || spectrum.Shape[3] != Bins || spectrum.Shape[4] != 2)
            throw new ArgumentException("Expected mono [1,1,time,481,2] spectrum.", nameof(spectrum));
        int frames = spectrum.Shape[2];
        var transform = new FastFourierTransform<float>();
        var result = new float[frames * Hop]; var memory = new float[Hop];
        for (int frame = 0; frame < frames; frame++)
        {
            var bins = new Vector<Complex<float>>(Fft);
            for (int k = 0; k < Bins; k++) bins[k] = new Complex<float>(spectrum[0, 0, frame, k, 0],
                k == 0 || k == Bins - 1 ? 0 : spectrum[0, 0, frame, k, 1]);
            for (int k = 1; k < Bins - 1; k++) bins[Fft - k] = new Complex<float>(bins[k].Real, -bins[k].Imaginary);
            var time = transform.Inverse(bins);
            for (int n = 0; n < Hop; n++)
            {
                // libdf's inverse is unnormalized; no additional envelope division.
                result[frame * Hop + n] = time[n] * Fft * _window[n] + memory[n];
                memory[n] = time[n + Hop] * Fft * _window[n + Hop];
            }
        }
        if (originalLength is not int length) return result;
        if (length <= 0 || length + Hop > result.Length) throw new ArgumentOutOfRangeException(nameof(originalLength));
        var trimmed = new float[length]; Array.Copy(result, Hop, trimmed, 0, length); return trimmed;
    }
}
