using AiDotNet.Audio.Foundations;
using AiDotNet.ModelLoading;
using AiDotNet.Models.Options;

namespace AiDotNet.TextToSpeech.MultiModal;

public partial class SpiritLM<T>
{
    // A file's tensors under a prefix, each read as the loader asks for it.
    private static void LoadFile(string path, string prefix, Action<Func<string, double[]?>> load)
    {
        using var file = TorchCheckpointFile.Open(path);
        load(name => file.TryRead(prefix + name));
    }

    // The checkpoint's files: its safetensors (one source over the shards), else its pytorch_model.bin shards.
    private static TorchCheckpointFile[] OpenShards(string directory)
    {
        if (File.Exists(Path.Combine(directory, "model.safetensors.index.json")) || File.Exists(Path.Combine(directory, "model.safetensors")))
            return new[] { TorchCheckpointFile.OpenDirectory(directory) };
        string index = Path.Combine(directory, "pytorch_model.bin.index.json");
        if (File.Exists(index))
            return Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(index))["weight_map"]!.Values<string>().Distinct()
                .Select(f => TorchCheckpointFile.Open(Path.Combine(directory, f!))).ToArray();
        string single = Path.Combine(directory, "pytorch_model.bin");
        if (File.Exists(single)) return new[] { TorchCheckpointFile.Open(single) };
        throw new FileNotFoundException($"No checkpoint in {directory}.");
    }

    /// <summary>
    /// Loads a released SpiritLM: the language model of <paramref name="modelDirectory"/> (<c>spirit-lm-base-7b</c> or
    /// <c>spirit-lm-expressive-7b</c>; its <c>tokenizer.model</c> unless <see cref="SpiritLMOptions.TokenizerPath"/> is set)
    /// and, with <paramref name="speechTokenizerDirectory"/>, the release's <c>speech_tokenizer</c> directory
    /// (<see cref="LoadSpeechTokenizer"/>).
    /// </summary>
    public void LoadReleasedWeights(string modelDirectory, string? speechTokenizerDirectory = null)
    {
        ThrowIfDisposed();
        if (modelDirectory is null) throw new ArgumentNullException(nameof(modelDirectory));
        var files = OpenShards(modelDirectory);
        try
        {
            LanguageModel.LoadTorchWeights(files.Skip(1).Aggregate((TorchTensorSource)files[0], (all, file) => all.Or(file)), "model.", "lm_head.weight");
            foreach (var file in files) file.RequireAllUsed("language model", k => k.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal));
        }
        finally
        {
            foreach (var file in files) file.Dispose();
        }
        if (_options.TokenizerPath is null && File.Exists(Path.Combine(modelDirectory, "tokenizer.model")))
        {
            _options.TokenizerPath = Path.Combine(modelDirectory, "tokenizer.model");
            _tokenizer = null;
        }
        if (speechTokenizerDirectory is not null) LoadSpeechTokenizer(speechTokenizerDirectory);
    }

    /// <summary>
    /// Loads the release's <c>speech_tokenizer</c> directory: the 25 Hz mHuBERT and its linear quantizer
    /// (<c>hubert_25hz</c>), the variant's vocoder (<c>hifigan_spiritlm_base</c> or
    /// <c>hifigan_spiritlm_expressive_w2v2</c>) and, for Expressive, the F0 VQ-VAE (<c>vqvae_f0_quantizer</c>) and the style
    /// classifier (<c>style_encoder_w2v2</c>). FCPE ships with torchfcpe; load it with <see cref="LoadFcpe"/>.
    /// </summary>
    public void LoadSpeechTokenizer(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        LoadFile(Path.Combine(directory, "hubert_25hz", "mhubert_base_25hz.pt"), "model_weight.", UnitExtractor.LoadFairseqWeights);
        if (_options.UnitQuantizer == SpiritLmUnitQuantizer.LinearCtc)
            LoadFile(Path.Combine(directory, "hubert_25hz", "L11_quantizer_500.pt"), "model_weight.", UnitExtractor.Quantizer!.LoadTorchWeights);
        string vocoder = IsExpressive ? "hifigan_spiritlm_expressive_w2v2" : "hifigan_spiritlm_base";
        LoadFile(Path.Combine(directory, vocoder, "generator.pt"), "generator.", read => Vocoder.LoadTorchWeights(read));
        if (IsExpressive)
        {
            using (var pitch = TorchCheckpointFile.Open(Path.Combine(directory, "vqvae_f0_quantizer", "model.pt")))
                PitchQuantizer.LoadTorchWeights(pitch.WithPrefix("model."));
            // The released style classifier; SpeechProp has no released weights.
            if (!UsesSpeechProp)
            {
                using var style = TorchCheckpointFile.Open(Path.Combine(directory, "style_encoder_w2v2", "pytorch_model.bin"));
                StyleEncoder.LoadTorchWeights(style);
            }
        }
    }

    /// <summary>Loads FCPE's weights, torchfcpe's bundled <c>assets/fcpe_c_v001.pt</c> (SpiritLM Expressive with
    /// <see cref="SpiritLmPitchExtractor.Fcpe"/>).</summary>
    public void LoadFcpe(string path)
    {
        ThrowIfDisposed();
        var fcpe = Fcpe ?? throw new InvalidOperationException("FCPE is SpiritLM Expressive's pitch extractor with PitchExtractor = Fcpe.");
        LoadFile(path, "model.", fcpe.LoadTorchWeights);
    }

    /// <summary>
    /// Loads the k-means centroids of <see cref="SpiritLmUnitQuantizer.KMeans"/>: a joblib-pickled scikit-learn model (as
    /// TWIST's and textlesslib's tokenizers ship them) or a torch file with <c>cluster_centers</c> (the release's
    /// <c>KmeansModel</c> format).
    /// </summary>
    public void LoadUnitKMeans(string path)
    {
        ThrowIfDisposed();
        if (_options.UnitQuantizer != SpiritLmUnitQuantizer.KMeans) throw new InvalidOperationException("Centroids replace the linear quantizer only with UnitQuantizer = KMeans.");
        if (!File.Exists(path)) throw new FileNotFoundException($"k-means model not found: {path}", path);
        double[,] centers;
        try
        {
            // A small file of centroids: read whole.
            var tensors = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
            if (!tensors.TryGetValue("cluster_centers", out var table)) throw new InvalidDataException("The torch file has no 'cluster_centers'.");
            int clusters = table.Shape[0], width = table.Shape[1];
            centers = new double[clusters, width];
            for (int k = 0; k < clusters; k++)
                for (int i = 0; i < width; i++) centers[k, i] = table[k, i];
        }
        catch (Exception e) when (e is not InvalidDataException)
        {
            centers = JoblibKMeans.ReadClusterCenters(path);
        }
        if (centers.GetLength(0) != _options.Units || centers.GetLength(1) != _options.HubertWidth)
            throw new InvalidDataException($"The k-means has {centers.GetLength(0)} x {centers.GetLength(1)} centroids; the model reads {_options.Units} x {_options.HubertWidth}.");
        UnitExtractor.Centroids = centers;
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    /// <remarks>The unit k-means centroids, FCPE's cent grid and the pitch VQ-VAE's codebook and its moving averages are not
    /// trained parameters.</remarks>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        _units?.DeclareCentroids(state, "UnitCentroids");
        _fcpe?.DeclareCentTable(state, "FcpeCentTable");
        _pitchQuantizer?.DeclareState(state, "PitchQuantizer");
        DeclareSpeechPropState(state);
    }

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SpiritLM-Native" : "SpiritLM-ONNX",
            Description = "SpiRit-LM: interleaved spoken and written language model (Nguyen et al., 2024)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "SpiritLM";
        metadata.AdditionalInfo["Variant"] = _options.Variant.ToString();
        metadata.AdditionalInfo["UnitQuantizer"] = _options.UnitQuantizer.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SpiritLM<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
