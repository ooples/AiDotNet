using AiDotNet.Audio.Whisper;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

public abstract partial class CosyVoiceBase<T>
{
    private OnnxModel<T>? _releasedSpeechTokenizer, _releasedSpeakerEncoder;

    protected void InitializeReleasedFrontends()
    {
        if (Settings.MaxReferenceSeconds <= 0) throw new ArgumentOutOfRangeException(nameof(Settings.MaxReferenceSeconds));
        if (!string.IsNullOrWhiteSpace(Settings.SpeechTokenizerPath)) OpenReleasedSpeechTokenizer(Settings.SpeechTokenizerPath!);
        if (!string.IsNullOrWhiteSpace(Settings.SpeakerEncoderOnnxPath)) OpenReleasedSpeakerEncoder(Settings.SpeakerEncoderOnnxPath!);
    }

    private void OpenReleasedSpeechTokenizer(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException("The original CosyVoice speech tokenizer is missing.", path);
        _releasedSpeechTokenizer?.Dispose();
        _releasedSpeechTokenizer = new OnnxModel<T>(path, Settings.OnnxOptions);
        Settings.SpeechTokenizerPath = path;
    }

    private void OpenReleasedSpeakerEncoder(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException("The original CosyVoice CAM++ encoder is missing.", path);
        _releasedSpeakerEncoder?.Dispose();
        _releasedSpeakerEncoder = new OnnxModel<T>(path, Settings.OnnxOptions);
        Settings.SpeakerEncoderOnnxPath = path;
    }

    private int[]? SpeechTokensFromReleasedFrontend(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        if (_releasedSpeechTokenizer is null) return null;
        var audio = ResampleAudio(recording, sampleRate, 16000);
        if (audio.Length > Settings.MaxReferenceSeconds * 16000L)
            throw new ArgumentException($"The reference recording may be at most {Settings.MaxReferenceSeconds} seconds.", nameof(recording));
        var logMel = OpenAiWhisperTranscriber<T>.LogMelSpectrogram(audio, 128);
        int bands = logMel.Shape[0], frames = logMel.Shape[1];
        var features = Engine.Reshape(logMel, new[] { 1, bands, frames });
        var length = new Tensor<T>(new[] { 1 }); length[0] = NumOps.FromDouble(frames);
        var inputs = _releasedSpeechTokenizer.Metadata.Inputs.Select(x => x.Name).ToArray();
        if (inputs.Length != 2) throw new InvalidDataException("The speech tokenizer requires features and feature lengths.");
        using var scope = new NoGradScope<T>();
        return IdsOf(_releasedSpeechTokenizer.Run(new Dictionary<string, Tensor<T>>
            { [inputs[0]] = features, [inputs[1]] = length }).Values.First());
    }

    private Tensor<T>? ForwardReleasedSpeakerEncoder(Tensor<T> features)
    {
        if (_releasedSpeakerEncoder is null) return null;
        var inputs = _releasedSpeakerEncoder.Metadata.Inputs;
        if (inputs.Count != 1) throw new InvalidDataException("campplus.onnx requires one Kaldi-feature input.");
        return _releasedSpeakerEncoder.Run(new Dictionary<string, Tensor<T>> { [inputs[0].Name] = features }).Values.First();
    }

    private bool TryLoadReleasedSpeechTokenizer(string modelDirectory, string speechTokenizerFile)
    {
        string converted = Path.Combine(modelDirectory, speechTokenizerFile);
        if (File.Exists(converted))
        {
            _releasedSpeechTokenizer?.Dispose(); _releasedSpeechTokenizer = null; Settings.SpeechTokenizerPath = null;
            return false;
        }
        string original = Path.Combine(modelDirectory, Path.ChangeExtension(speechTokenizerFile, ".onnx"));
        OpenReleasedSpeechTokenizer(File.Exists(original) || string.IsNullOrWhiteSpace(Settings.SpeechTokenizerPath)
            ? original : Settings.SpeechTokenizerPath!);
        return true;
    }

    private bool TryLoadReleasedSpeakerEncoder(string modelDirectory)
    {
        string converted = Path.Combine(modelDirectory, "campplus.safetensors");
        if (File.Exists(converted))
        {
            _releasedSpeakerEncoder?.Dispose(); _releasedSpeakerEncoder = null; Settings.SpeakerEncoderOnnxPath = null;
            return false;
        }
        string original = Path.Combine(modelDirectory, "campplus.onnx");
        OpenReleasedSpeakerEncoder(File.Exists(original) || string.IsNullOrWhiteSpace(Settings.SpeakerEncoderOnnxPath)
            ? original : Settings.SpeakerEncoderOnnxPath!);
        return true;
    }

    private void DisposeReleasedFrontends()
    {
        _releasedSpeechTokenizer?.Dispose(); _releasedSpeakerEncoder?.Dispose();
        _releasedSpeechTokenizer = null; _releasedSpeakerEncoder = null;
    }
}
