using System.Collections.Generic;
using AiDotNet.Audio.Resampling;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Shared waveform analysis and neural tone-color conversion for both OpenVoice releases.</summary>
internal sealed class OpenVoiceNative<T>
{
    private readonly IEngine _engine;
    private readonly int _fft, _hop, _latent, _speakerChannels;
    private readonly DifferentiableMel<T> _analysis;
    private readonly OpenVoiceReferenceEncoder<T> _reference;
    private readonly OpenVoiceConverterCore<T> _converter;
    private readonly Random _random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(2294);
    private readonly List<LayerBase<T>> _layers = new();
    public IReadOnlyList<LayerBase<T>> Layers => _layers;
    internal int FftSize => _fft;
    internal int HopSize => _hop;
    internal int LatentChannels => _latent;
    internal int DecoderScaleFactor { get; }

    public OpenVoiceNative(IEngine engine, int fft, int hop, int hidden, int latent, int speakerChannels, bool versionTwo,
        int[]? upsampleRates = null, int[]? upsampleKernels = null, int initialChannels = 512,
        int[]? residualKernels = null, int[][]? residualDilations = null)
    {
        if (fft < 2 || hop <= 0 || hop > fft) throw new ArgumentOutOfRangeException(nameof(fft));
        _engine = engine; _fft = fft; _hop = hop; _latent = latent; _speakerChannels = speakerChannels;
        _analysis = new DifferentiableMel<T>(engine, 22050, fft, hop, fft, 80, 0, 11025);
        _reference = new OpenVoiceReferenceEncoder<T>(engine, fft / 2 + 1, speakerChannels);
        _converter = new OpenVoiceConverterCore<T>(engine, fft / 2 + 1, latent, hidden, speakerChannels, versionTwo,
            upsampleRates, upsampleKernels, initialChannels, residualKernels, residualDilations);
        DecoderScaleFactor = (upsampleRates ?? new[] { 8, 8, 2, 2 }).Aggregate(1, (a, b) => a * b);
        _layers.AddRange(_reference.Layers); _layers.AddRange(_converter.Layers);
    }

    public Tensor<T> Embedding(Tensor<T> audio)
    {
        ValidateAudio(audio);
        var rows = _analysis.Magnitude(_engine.Reshape(audio, new[] { audio.Length }), 1e-6);
        return _engine.Reshape(_reference.Forward(_engine.Reshape(rows,
            new[] { 1, rows.Shape[0], _fft / 2 + 1 })), new[] { _speakerChannels });
    }

    internal Tensor<T> TrainingSpectrogram(Tensor<T> audio)
    {
        ValidateAudio(audio);
        return _analysis.Magnitude(_engine.Reshape(audio, new[] { audio.Length }), 1e-6);
    }

    internal (Tensor<T> Waveform, Tensor<T> Posterior, Tensor<T> Prior, Tensor<T> Mean, Tensor<T> LogScale) TrainingForward(
        Tensor<T> spectrumRows, Tensor<T> referenceRows, Tensor<T> noise, int start, int frames)
    {
        int time = spectrumRows.Shape[0];
        var speaker = _reference.Forward(_engine.Reshape(referenceRows, new[] { 1, referenceRows.Shape[0], _fft / 2 + 1 }));
        var spectrum = _engine.Reshape(_engine.TensorTranspose(spectrumRows), new[] { 1, _fft / 2 + 1, time });
        var mask = new Tensor<T>(new[] { 1, 1, time }); var numeric = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < time; i++) mask[i] = numeric.One;
        return _converter.TrainingForward(spectrum, mask, _engine.Reshape(speaker, new[] { 1, _speakerChannels, 1 }), noise, start, frames);
    }

    public Tensor<T> Convert(Tensor<T> audio, Tensor<T> source, Tensor<T> target, double tau = .3, Tensor<T>? noise = null)
    {
        ValidateAudio(audio);
        if (source.Length != _speakerChannels || target.Length != _speakerChannels)
            throw new ArgumentException("Source and target must be converter speaker embeddings.");
        var rows = _analysis.Magnitude(_engine.Reshape(audio, new[] { audio.Length }), 1e-6);
        int frames = rows.Shape[0];
        var spectrogram = _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, _fft / 2 + 1, frames });
        var mask = new Tensor<T>(new[] { 1, 1, frames });
        var operations = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < mask.Length; i++) mask[i] = operations.One;
        noise ??= tau == 0 ? new Tensor<T>(new[] { 1, _latent, frames })
            : new Tensor<T>(new[] { 1, _latent, frames }, TorchParameters.Gaussian<T>(_random, _latent * frames, 1));
        var result = _converter.Convert(spectrogram, mask,
            _engine.Reshape(source, new[] { 1, _speakerChannels, 1 }),
            _engine.Reshape(target, new[] { 1, _speakerChannels, 1 }), noise, tau).Waveform;
        return _engine.Reshape(result, new[] { result.Length });
    }

    public void LoadCheckpoint(string path)
    {
        using var file = AiDotNet.ModelLoading.TorchCheckpointFile.Open(path, strictAxes: true);
        var state = file.Names.Any(name => name.StartsWith("model.", StringComparison.Ordinal)) ? file.WithPrefix("model.") : file;
        LoadTorch(state, state);
    }

    internal void LoadTorch(Func<string, int[], double[]> read, Func<string, double[]?> optional) =>
        LoadTorch(AiDotNet.ModelLoading.TorchTensorSource.From(read), OpenVoiceConverterCore<T>.OnePerName(optional));

    /// <summary>Loads the reference encoder and converter straight from <paramref name="source"/>, the decoder from
    /// <paramref name="decoder"/>.</summary>
    internal void LoadTorch(AiDotNet.ModelLoading.TorchTensorSource source, AiDotNet.ModelLoading.TorchTensorSource decoder)
    {
        _reference.LoadTorch(source); _converter.LoadTorch(source, decoder);
    }

    internal static Tensor<T> Resample(Tensor<T> audio, int inputRate)
    {
        if (inputRate == 22050) return audio;
        var operations = MathHelper.GetNumericOperations<T>();
        var values = SoxrResampler.LibrosaResample(audio.ToArray().Select(x => operations.ToDouble(x)).ToArray(), inputRate, 22050);
        return new Tensor<T>(new[] { values.Length }, new Vector<T>(values.Select(x => operations.FromDouble(x))));
    }

    private void ValidateAudio(Tensor<T> audio)
    {
        if (audio.Rank > 3 || audio.Length <= (_fft - _hop) / 2 || audio.Length < _hop)
            throw new ArgumentException("The mono waveform must be long enough for reflect-padded STFT analysis.", nameof(audio));
        if (audio.Rank > 1 && audio.Shape.ToArray().Take(audio.Rank - 1).Any(dimension => dimension != 1))
            throw new ArgumentException("Expected a single mono waveform.", nameof(audio));
    }
}
