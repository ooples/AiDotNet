using AiDotNet.ModelLoading;

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
    /// <summary>A state dict that is not a torch checkpoint (the <c>.npz</c> statistics, the content tokenizer's
    /// <c>.pkl</c>): each tensor becomes doubles when it is read and is then released.</summary>
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

        public TorchTensorSource Source(string prefix) => TorchTensorSource.From(_tensors, "the content tokenizer checkpoint").WithPrefix(prefix);
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

        // HuBERT's and MaskGCT's codec loaders still take each tensor's values.
        using (var hubert = TorchCheckpointFile.Open(checkpoints.Hubert)) LoadHubert(name => hubert.TryRead(name));
        var stats = new Checkpoint(Path("tokenizer", "vq32", "hubert_large_l18_mean_std.npz"));
        var std = stats.Take("std") ?? throw new InvalidDataException("The statistics hold no 'std'.");
        ContentStyleTokenizer.LoadStatistics(stats.Take("mean") ?? throw new InvalidDataException("The statistics hold no 'mean'."),
            std.Select(s => s * s).ToArray());
        using (var contentStyle = TorchCheckpointFile.Open(Path("tokenizer", "vq8192", "model.safetensors")))
            ContentStyleTokenizer.Load((name, _) => contentStyle.TryRead(name));
        using (var tts = TorchCheckpointFile.Open(Path("contentstyle_modeling", "PhoneToVq8192", "model.safetensors"))) TextToContentStyle.Load(tts);
        if (_options.VoiceConversion)
        {
            var content = new Checkpoint(Path("tokenizer", "vq32", "hubert_large_l18_c32.pkl"));
            string prefix = content.Keys.Any(k => k.StartsWith("model.repcodec.", StringComparison.Ordinal)) ? "model.repcodec." : "";
            ContentTokenizer.Load(content.Source(prefix));
            using var voice = TorchCheckpointFile.Open(Path("contentstyle_modeling", "Vq32ToVq8192", "model.safetensors"));
            ContentToContentStyle.Load(voice);
        }
        using (var flow = TorchCheckpointFile.Open(Path("acoustic_modeling", "Vq8192ToMels", "model.safetensors"))) Flow.Load(flow);
        var o = _options;
        // Vocos's loader still takes each tensor's values.
        using (var vocos = TorchCheckpointFile.Open(Path("acoustic_modeling", "Vocoder", "model.safetensors")))
            Vocoder.LoadTorchWeights(vocos.AsFunction(), o.MelChannels, o.VocoderIntermediate);
    }

    /// <summary>Loads HuBERT from Hugging Face's names; the layers above <see cref="VevoOptions.HubertLayer"/> are not
    /// read.</summary>
    internal void LoadHubert(Func<string, double[]?> take) => Hubert.LoadTorchWeights(take);
}
