namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The files Vevo's release reads (<c>models/vc/vevo/infer_*.py</c>).</summary>
public sealed class VevoCheckpoints
{
    /// <summary>The <c>amphion/Vevo</c> directory: <c>tokenizer/vq8192</c>, <c>tokenizer/vq32</c>,
    /// <c>contentstyle_modeling/PhoneToVq8192</c> and <c>Vq32ToVq8192</c>, <c>acoustic_modeling/Vq8192ToMels</c> and
    /// <c>Vocoder</c>.</summary>
    public required string Directory { get; init; }

    /// <summary>HuBERT-Large's weights (<c>facebook/hubert-large-ll60k</c>'s <c>pytorch_model.bin</c> or
    /// <c>model.safetensors</c>; torchaudio's <c>HUBERT_LARGE</c> is the same fairseq model).</summary>
    public required string Hubert { get; init; }
}

public partial class Vevo<T>
{
    /// <summary>A checkpoint read once: each tensor becomes doubles when it is read and is then released.</summary>
    private sealed class Checkpoint
    {
        private readonly Dictionary<string, Tensor<float>> _tensors;

        public Checkpoint(string path)
        {
            _tensors = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
            Keys = new HashSet<string>(_tensors.Keys, StringComparer.Ordinal);
        }

        public HashSet<string> Keys { get; }

        public double[]? Take(string name)
        {
            if (!_tensors.TryGetValue(name, out var tensor)) return null;
            _tensors.Remove(name);
            var values = new double[tensor.Length];
            for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
            return values;
        }

        public Func<string, int[], double[]> Reader(string prefix, string what) => (name, _) =>
            Take(prefix + name) ?? throw new InvalidDataException($"The {what} checkpoint has no tensor '{prefix + name}'.");
    }

    /// <summary>
    /// Loads the release: HuBERT-Large (its first <see cref="VevoOptions.HubertLayer"/> layers), the content-style
    /// tokenizer with HuBERT's feature statistics, the autoregressive transformers, the content tokenizer (when voice
    /// conversion is on), the flow-matching transformer and Vocos.
    /// </summary>
    public void LoadReferenceWeights(VevoCheckpoints checkpoints)
    {
        ThrowIfDisposed();
        if (checkpoints is null) throw new ArgumentNullException(nameof(checkpoints));
        if (!HasPaperLayers) throw NotNative();
        string dir = checkpoints.Directory;
        string Path(params string[] parts) => System.IO.Path.Combine(new[] { dir }.Concat(parts).ToArray());

        LoadHubert(new Checkpoint(checkpoints.Hubert));
        var stats = new Checkpoint(Path("tokenizer", "vq32", "hubert_large_l18_mean_std.npz"));
        var std = stats.Take("std") ?? throw new InvalidDataException("The statistics hold no 'std'.");
        ContentStyleTokenizer.LoadStatistics(stats.Take("mean") ?? throw new InvalidDataException("The statistics hold no 'mean'."),
            std.Select(s => s * s).ToArray());
        var contentStyle = new Checkpoint(Path("tokenizer", "vq8192", "model.safetensors"));
        ContentStyleTokenizer.Load((name, _) => contentStyle.Take(name));
        TextToContentStyle.Load(new Checkpoint(Path("contentstyle_modeling", "PhoneToVq8192", "model.safetensors")).Reader("", "Vevo-TTS transformer"));
        if (_options.VoiceConversion)
        {
            var content = new Checkpoint(Path("tokenizer", "vq32", "hubert_large_l18_c32.pkl"));
            string prefix = content.Keys.Any(k => k.StartsWith("model.repcodec.", StringComparison.Ordinal)) ? "model.repcodec." : "";
            ContentTokenizer.Load(content.Reader(prefix, "content tokenizer"));
            ContentToContentStyle.Load(new Checkpoint(Path("contentstyle_modeling", "Vq32ToVq8192", "model.safetensors")).Reader("", "Vevo-Voice transformer"));
        }
        Flow.Load(new Checkpoint(Path("acoustic_modeling", "Vq8192ToMels", "model.safetensors")).Reader("", "flow"));
        var o = _options;
        Vocoder.LoadTorchWeights(new Checkpoint(Path("acoustic_modeling", "Vocoder", "model.safetensors")).Reader("", "Vocos"), o.MelChannels, o.VocoderIntermediate);
    }

    /// <summary>Loads HuBERT from Hugging Face's names; the layers above <see cref="VevoOptions.HubertLayer"/> are not
    /// read.</summary>
    internal void LoadHubert(Func<string, double[]?> take) => Hubert.LoadTorchWeights(take);

    private void LoadHubert(Checkpoint checkpoint) => LoadHubert(checkpoint.Take);
}
