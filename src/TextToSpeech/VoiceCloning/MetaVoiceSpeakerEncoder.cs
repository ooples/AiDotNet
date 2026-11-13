using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>MetaVoice's released three-layer LSTM speaker encoder, with waveform padding
/// before power-mel analysis and the author's 0.75 partial-window coverage threshold.</summary>
internal sealed class MetaVoiceSpeakerEncoder<T>
{
    private readonly IEngine _engine;
    private readonly int _melChannels, _hidden, _dimension;
    private readonly DenseLayer<T>[] _input, _recurrent;
    private readonly DenseLayer<T> _projection;
    private readonly CenteredLogMel<T> _spectrogram;
    private readonly List<LayerBase<T>> _layers = new();
    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    public MetaVoiceSpeakerEncoder(IEngine engine, int melChannels = 40, int hidden = 256, int dimension = 256, int layers = 3,
        IReadOnlyList<LayerBase<T>>? restoredLayers = null)
    {
        if (melChannels <= 0 || hidden <= 0 || dimension <= 0 || layers <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        _engine = engine; _melChannels = melChannels; _hidden = hidden; _dimension = dimension;
        if (restoredLayers is not null && (restoredLayers.Count != 2 * layers + 1 || restoredLayers.Any(layer => layer is not DenseLayer<T>)))
            throw new InvalidDataException("The restored speaker encoder has a different topology.");
        _input = new DenseLayer<T>[layers]; _recurrent = new DenseLayer<T>[layers];
        for (int i = 0; i < layers; i++)
        {
            // Keep both Torch LSTM biases as independent parameters, including during optimization.
            _input[i] = restoredLayers is null ? new DenseLayer<T>(4 * hidden, new IdentityActivation<T>() as IActivationFunction<T>)
                : (DenseLayer<T>)restoredLayers[2 * i];
            _recurrent[i] = restoredLayers is null ? new DenseLayer<T>(4 * hidden, new IdentityActivation<T>() as IActivationFunction<T>)
                : (DenseLayer<T>)restoredLayers[2 * i + 1];
            _layers.Add(_input[i]); _layers.Add(_recurrent[i]);
        }
        _projection = restoredLayers is null ? new DenseLayer<T>(dimension, new IdentityActivation<T>() as IActivationFunction<T>)
            : (DenseLayer<T>)restoredLayers[2 * layers];
        _layers.Add(_projection);
        _spectrogram = new CenteredLogMel<T>(engine, 16000, 400, 160, 400, melChannels, 0, 8000, 1e-5,
            spectrumPower: 2, logarithm: false, constantPadding: true);
    }

    private Tensor<T> Normalize(Tensor<T> value)
    {
        var norm = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(value, value), new[] { 1 }, true));
        // The original does not clamp the norm: do not silently change zero-vector behavior.
        return _engine.TensorDivide(value, _engine.TensorTile(norm, new[] { 1, value.Shape[1] }));
    }

    public Tensor<T> ForwardMels(Tensor<T> mels)
    {
        if (mels.Rank != 3 || mels.Shape[2] != _melChannels || mels.Shape[1] <= 0)
            throw new ArgumentException("Expected [batch, frames, mel] input.", nameof(mels));
        int batch = mels.Shape[0];
        var hidden = _input.Select(_ => new Tensor<T>(new[] { batch, _hidden })).ToArray();
        var cells = _input.Select(_ => new Tensor<T>(new[] { batch, _hidden })).ToArray();
        Tensor<T>? final = null;
        for (int frame = 0; frame < mels.Shape[1]; frame++)
        {
            var input = _engine.Reshape(_engine.TensorSlice(mels, new[] { 0, frame, 0 }, new[] { batch, 1, _melChannels }), new[] { batch, _melChannels });
            for (int i = 0; i < _input.Length; i++)
            {
                var gates = _engine.TensorAdd(_input[i].Forward(input), _recurrent[i].Forward(hidden[i]));
                Tensor<T> Gate(int offset) => _engine.TensorSlice(gates, new[] { 0, offset * _hidden }, new[] { batch, _hidden });
                cells[i] = _engine.TensorAdd(_engine.TensorMultiply(_engine.Sigmoid(Gate(1)), cells[i]),
                    _engine.TensorMultiply(_engine.Sigmoid(Gate(0)), _engine.Tanh(Gate(2))));
                hidden[i] = _engine.TensorMultiply(_engine.Sigmoid(Gate(3)), _engine.Tanh(cells[i]));
                final = hidden[i]; input = final;
            }
        }
        return Normalize(_engine.ReLU(_projection.Forward(final!)));
    }

    internal static int[] PartialStarts(int samples, double rate = 1.3, double minimumCoverage = .75)
    {
        if (samples <= 0 || rate <= 0 || double.IsNaN(rate) || minimumCoverage < 0 || minimumCoverage > 1)
            throw new ArgumentOutOfRangeException(nameof(samples));
        int frames = (int)Math.Ceiling((samples + 1.0) / 160);
        int step = (int)Math.Round(100 / rate);
        if (step <= 0) throw new ArgumentOutOfRangeException(nameof(rate));
        int extent = Math.Max(1, frames - 160 + step + 1);
        var starts = new List<int>();
        for (int i = 0; i < extent; i += step) starts.Add(i);
        int last = starts[starts.Count - 1];
        double coverage = (samples - last * 160.0) / (160 * 160);
        if (coverage < minimumCoverage && starts.Count > 1) starts.RemoveAt(starts.Count - 1);
        return starts.ToArray();
    }

    /// <summary>Embeds already resampled, silence-trimmed mono 16 kHz audio.</summary>
    public Tensor<T> EmbedPreprocessed(Tensor<T> audio)
    {
        var embeddings = ForwardMels(PreparePartials(audio));
        return Normalize(_engine.ReduceMean(embeddings, new[] { 0 }, true));
    }

    internal Tensor<T> PreparePartials(Tensor<T> audio)
    {
        if (audio.Length <= 0) throw new ArgumentException("Speaker recording is empty.", nameof(audio));
        var starts = PartialStarts(audio.Length);
        int required = (starts[starts.Length - 1] + 160) * 160;
        var waveform = _engine.Reshape(audio, new[] { audio.Length });
        if (required > audio.Length) waveform = _engine.TensorConcatenate(new[] { waveform, new Tensor<T>(new[] { required - audio.Length }) }, 0);
        var mel = _spectrogram.Forward(waveform);
        var partials = starts.Select(start => _engine.TensorPermute(_engine.TensorSlice(mel,
            new[] { 0, 0, start }, new[] { 1, _melChannels, 160 }), new[] { 0, 2, 1 })).ToArray();
        return _engine.TensorConcatenate(partials, 0);
    }

    public void LoadTorch(Func<string, int[], double[]> read) => LoadTorch(TorchTensorSource.From(read));

    /// <summary>Loads the LSTM layers and projection straight from <paramref name="source"/>.</summary>
    internal void LoadTorch(TorchTensorSource source)
    {
        for (int i = 0; i < _input.Length; i++)
        {
            int input = i == 0 ? _melChannels : _hidden;
            TorchParameters.Linear(_engine, _input[i], input, 4 * _hidden, source, $"lstm.weight_ih_l{i}", $"lstm.bias_ih_l{i}");
            TorchParameters.Linear(_engine, _recurrent[i], _hidden, 4 * _hidden, source, $"lstm.weight_hh_l{i}", $"lstm.bias_hh_l{i}");
        }
        TorchParameters.Linear(_engine, _projection, _hidden, _dimension, source, "linear.weight", "linear.bias");
    }
}
