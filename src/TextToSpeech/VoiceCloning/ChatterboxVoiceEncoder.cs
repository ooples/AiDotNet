using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>The CorentinJ-style voice encoder shipped in <c>ve.safetensors</c>.</summary>
internal sealed class ChatterboxVoiceEncoder<T>
{
    private const int SampleRate = 16000;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly ChatterboxOptions _options;
    private readonly List<LSTMCellLayer<T>> _cells = new();
    private readonly DenseLayer<T> _projection;
    private readonly TiedEmbeddingLayer<T> _similarityWeight;
    private readonly TiedEmbeddingLayer<T> _similarityBias;
    private readonly CenteredLogMel<T> _spectrogram;

    public ChatterboxVoiceEncoder(IEngine engine, List<LayerBase<T>> layers, ChatterboxOptions options)
    {
        _engine = engine;
        _options = options;
        int input = options.VoiceEncoderMelChannels;
        for (int i = 0; i < options.VoiceEncoderLayers; i++)
        {
            var cell = new LSTMCellLayer<T>(input, options.VoiceEncoderHiddenDim);
            _cells.Add(cell);
            layers.Add(cell);
            input = options.VoiceEncoderHiddenDim;
        }
        _projection = new DenseLayer<T>(options.T3SpeakerEmbeddingDim,
            new IdentityActivation<T>() as IActivationFunction<T>);
        layers.Add(_projection);
        _similarityWeight = new TiedEmbeddingLayer<T>(1, 1);
        _similarityBias = new TiedEmbeddingLayer<T>(1, 1);
        layers.Add(_similarityWeight);
        layers.Add(_similarityBias);
        // Librosa's 400-point, centered Hann STFT and Slaney power mel. The shared FFT path
        // does not reproduce a non-power-of-two transform, so use the verified Fourier matrices.
        _spectrogram = new CenteredLogMel<T>(engine, SampleRate, 400, 160, 400,
            options.VoiceEncoderMelChannels, 0, 8000, 1e-5, spectrumPower: 2, logarithm: false);
    }

    internal IReadOnlyList<LSTMCellLayer<T>> Cells => _cells;
    internal DenseLayer<T> Projection => _projection;

    public Tensor<T> Embed(Tensor<T> recording, int sampleRate)
    {
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (recording.Length == 0) throw new ArgumentException("The voice encoder requires a nonempty recording.", nameof(recording));
        var audio = sampleRate == SampleRate
            ? _engine.Reshape(recording, new[] { recording.Length })
            : AudioHelper<T>.ResampleBandLimited(_engine, _engine.Reshape(recording, new[] { recording.Length }),
                sampleRate, SampleRate);
        var values = new double[audio.Length];
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.ToDouble(audio[i]);
        values = TrimSilence(values, topDb: 20);
        // Reflection, including repeated reflection on very short clips, follows librosa;
        // extending a short waveform with zero samples would alter both its mel and frame count.
        var mel = _spectrogram.Forward(new Tensor<T>(new[] { values.Length },
            new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble))));
        int frames = mel.Shape[2];

        int step = (int)Math.Round((SampleRate / _options.VoiceEncoderPartialRate)
            / _options.VoiceEncoderPartialFrames);
        step = Math.Max(1, Math.Min(step, _options.VoiceEncoderPartialFrames));
        int adjusted = Math.Max(frames - _options.VoiceEncoderPartialFrames + step, 0);
        int partials = adjusted / step;
        int remainder = adjusted % step;
        if (partials == 0 || (double)(remainder + _options.VoiceEncoderPartialFrames - step)
                / _options.VoiceEncoderPartialFrames >= 0.8)
            partials++;
        int targetFrames = _options.VoiceEncoderPartialFrames + step * (partials - 1);

        var packed = new Tensor<T>(new[]
            { partials, _options.VoiceEncoderPartialFrames, _options.VoiceEncoderMelChannels });
        for (int partial = 0; partial < partials; partial++)
            for (int frame = 0; frame < _options.VoiceEncoderPartialFrames; frame++)
            {
                int source = partial * step + frame;
                if (source >= frames || source >= targetFrames) continue;
                for (int channel = 0; channel < _options.VoiceEncoderMelChannels; channel++)
                    packed[partial, frame, channel] = mel[0, channel, source];
            }

        var states = _cells.Select(_ => new Tensor<T>(
            new[] { partials, 2 * _options.VoiceEncoderHiddenDim })).ToArray();
        Tensor<T>? finalHidden = null;
        for (int frame = 0; frame < _options.VoiceEncoderPartialFrames; frame++)
        {
            var input = _engine.TensorSlice(packed,
                new[] { 0, frame, 0 },
                new[] { partials, 1, _options.VoiceEncoderMelChannels });
            input = _engine.Reshape(input, new[] { partials, _options.VoiceEncoderMelChannels });
            for (int layer = 0; layer < _cells.Count; layer++)
            {
                states[layer] = _cells[layer].Forward(input, states[layer]);
                finalHidden = _cells[layer].SplitState(states[layer]).Hidden;
                input = finalHidden;
            }
        }

        var projected = _engine.ReLU(_projection.Forward(finalHidden
            ?? throw new InvalidOperationException("The voice encoder has no LSTM layers.")));
        var partialEmbeddings = NormalizeRows(projected);
        var averaged = _engine.ReduceMean(partialEmbeddings, new[] { 0 }, keepDims: true);
        return _engine.Reshape(NormalizeRows(averaged), new[] { _options.T3SpeakerEmbeddingDim });
    }

    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        int hidden = _options.VoiceEncoderHiddenDim;
        for (int layer = 0; layer < _cells.Count; layer++)
        {
            int input = layer == 0 ? _options.VoiceEncoderMelChannels : hidden;
            _cells[layer].LoadTorchWeights(
                read($"lstm.weight_ih_l{layer}", new[] { 4 * hidden, input }),
                read($"lstm.weight_hh_l{layer}", new[] { 4 * hidden, hidden }),
                read($"lstm.bias_ih_l{layer}", new[] { 4 * hidden }),
                read($"lstm.bias_hh_l{layer}", new[] { 4 * hidden }));
        }
        TorchParameters.Linear(_engine, _projection, hidden, _options.T3SpeakerEmbeddingDim,
            read("proj.weight", new[] { _options.T3SpeakerEmbeddingDim, hidden }),
            read("proj.bias", new[] { _options.T3SpeakerEmbeddingDim }));
        _similarityWeight.LoadTable(read("similarity_weight", new[] { 1 }));
        _similarityBias.LoadTable(read("similarity_bias", new[] { 1 }));
    }

    private Tensor<T> NormalizeRows(Tensor<T> rows)
    {
        int count = rows.Shape[0], width = rows.Shape[1];
        var norm = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(rows, rows),
            new[] { 1 }, keepDims: true));
        norm = _engine.TensorClampMin(norm, NumOps.FromDouble(1e-12));
        return _engine.TensorDivide(rows, _engine.TensorTile(norm, new[] { 1, width }));
    }

    internal static double[] TrimSilence(double[] audio, double topDb)
    {
        const int frameLength = 2048;
        const int hop = 512;
        int frames = 1 + audio.Length / hop;
        var rms = new double[frames];
        double maximum = 0;
        for (int frame = 0; frame < frames; frame++)
        {
            int start = frame * hop - frameLength / 2;
            double sum = 0;
            for (int i = 0; i < frameLength; i++)
            {
                int index = start + i;
                double sample = index >= 0 && index < audio.Length ? audio[index] : 0;
                sum += sample * sample;
            }
            rms[frame] = Math.Sqrt(sum / frameLength);
            maximum = Math.Max(maximum, rms[frame]);
        }

        // librosa.amplitude_to_db's default amin applies to RMS amplitude, not power.
        const double floor = 1e-5;
        int first = -1, last = -1;
        for (int frame = 0; frame < frames; frame++)
        {
            double decibels = 20.0 * Math.Log10(Math.Max(rms[frame], floor))
                - 20.0 * Math.Log10(Math.Max(maximum, floor));
            if (decibels <= -topDb) continue;
            if (first < 0) first = frame;
            last = frame;
        }
        if (first < 0) return audio;
        int from = Math.Min(audio.Length, first * hop);
        int to = Math.Min(audio.Length, (last + 1) * hop);
        return audio.Skip(from).Take(Math.Max(0, to - from)).ToArray();
    }
}
