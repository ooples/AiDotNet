using AiDotNet.Audio.Foundations;
using AiDotNet.Models.Options;

namespace AiDotNet.TextToSpeech.MultiModal;

public partial class SpiritLM<T>
{
    private static Dictionary<string, Tensor<float>> ReadAll(IEnumerable<string> paths)
    {
        var tensors = new Dictionary<string, Tensor<float>>(StringComparer.Ordinal);
        foreach (var path in paths)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
            foreach (var pair in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path)) tensors[pair.Key] = pair.Value;
        }
        return tensors;
    }

    private static double[] Values(Tensor<float> tensor)
    {
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
        return values;
    }

    private static Func<string, double[]?> Reader(Dictionary<string, Tensor<float>> tensors, string prefix) =>
        name => tensors.TryGetValue(prefix + name, out var t) ? Values(t) : null;

    private static IEnumerable<string> Shards(string directory)
    {
        foreach (var index in new[] { "model.safetensors.index.json", "pytorch_model.bin.index.json" })
        {
            string path = Path.Combine(directory, index);
            if (File.Exists(path))
                return Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(path))["weight_map"]!.Values<string>().Distinct().Select(f => Path.Combine(directory, f!)).ToList();
        }
        foreach (var single in new[] { "model.safetensors", "pytorch_model.bin" })
            if (File.Exists(Path.Combine(directory, single))) return new[] { Path.Combine(directory, single) };
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
        var tensors = ReadAll(Shards(modelDirectory));
        var used = new HashSet<string>(StringComparer.Ordinal);
        LanguageModel.LoadTorchWeights((name, _) =>
        {
            if (!tensors.TryGetValue(name, out var tensor)) throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            used.Add(name);
            return Values(tensor);
        }, "model.", "lm_head.weight");
        var left = tensors.Keys.Where(k => !used.Contains(k) && !k.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal)).ToList();
        if (left.Count > 0) throw new InvalidDataException($"The checkpoint has {left.Count} tensors this model has no place for (first: '{left[0]}').");
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
        var hubert = ReadAll(new[] { Path.Combine(directory, "hubert_25hz", "mhubert_base_25hz.pt") });
        UnitExtractor.LoadFairseqWeights(Reader(hubert, "model_weight."));
        if (_options.UnitQuantizer == SpiritLmUnitQuantizer.LinearCtc)
            UnitExtractor.Quantizer!.LoadTorchWeights(Reader(ReadAll(new[] { Path.Combine(directory, "hubert_25hz", "L11_quantizer_500.pt") }), "model_weight."));
        string vocoder = IsExpressive ? "hifigan_spiritlm_expressive_w2v2" : "hifigan_spiritlm_base";
        Vocoder.LoadTorchWeights(Reader(ReadAll(new[] { Path.Combine(directory, vocoder, "generator.pt") }), "generator."));
        if (IsExpressive)
        {
            PitchQuantizer.LoadTorchWeights(Reader(ReadAll(new[] { Path.Combine(directory, "vqvae_f0_quantizer", "model.pt") }), "model."));
            // The released style classifier; SpeechProp has no released weights.
            if (!UsesSpeechProp)
                StyleEncoder.LoadTorchWeights(Reader(ReadAll(new[] { Path.Combine(directory, "style_encoder_w2v2", "pytorch_model.bin") }), ""));
        }
    }

    /// <summary>Loads FCPE's weights, torchfcpe's bundled <c>assets/fcpe_c_v001.pt</c> (SpiritLM Expressive with
    /// <see cref="SpiritLmPitchExtractor.Fcpe"/>).</summary>
    public void LoadFcpe(string path)
    {
        ThrowIfDisposed();
        var fcpe = Fcpe ?? throw new InvalidOperationException("FCPE is SpiritLM Expressive's pitch extractor with PitchExtractor = Fcpe.");
        fcpe.LoadTorchWeights(Reader(ReadAll(new[] { path }), "model."));
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
            var tensors = ReadAll(new[] { path });
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
