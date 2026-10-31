namespace AiDotNet.TextToSpeech.MultiModal;

public partial class Moshi<T>
{
    private static Dictionary<string, double[]> ReadState(string path)
    {
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
            state[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        return state;
    }

    /// <summary>
    /// Loads the release (<c>kyutai/moshiko-pytorch-bf16</c> or moshika): the language model from
    /// <c>model.safetensors</c>, Mimi from <c>tokenizer-e351c8d8-checkpoint125.safetensors</c> and Helium's tokenizer
    /// <c>tokenizer_spm_32k_3.model</c> when <see cref="MoshiOptions.TokenizerPath"/> is unset. The release's Depth
    /// Transformer predicts Moshi's streams only: build with <see cref="MoshiOptions.Release"/>.
    /// </summary>
    public void LoadReleasedWeights(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        if (!HasPaperLayers) throw NotNative();
        string model = Path.Combine(directory, "model.safetensors");
        if (!File.Exists(model)) throw new FileNotFoundException($"No model.safetensors in {directory}.", model);
        LoadLanguageModel(model);
        string? mimi = Directory.GetFiles(directory, "tokenizer-*.safetensors").OrderBy(f => f, StringComparer.Ordinal).FirstOrDefault();
        if (mimi is not null) LoadMimi(mimi);
        string tokenizer = Path.Combine(directory, "tokenizer_spm_32k_3.model");
        if (_options.TokenizerPath is null && File.Exists(tokenizer))
        {
            _options.TokenizerPath = tokenizer;
            Core.ResetTokenizer();
        }
    }

    /// <summary>Loads the release's <c>LMModel</c> state dict (a <c>.safetensors</c> file).</summary>
    public void LoadLanguageModel(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var state = ReadState(path);
        var used = new HashSet<string>(StringComparer.Ordinal);
        Core.LanguageModel.LoadTorchWeights(name =>
        {
            if (!state.TryGetValue(name, out var v)) return null;
            used.Add(name);
            return v;
        });
        var unused = state.Keys.Where(k => !used.Contains(k)).ToList();
        if (unused.Count > 0) throw new InvalidDataException($"The language model checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}').");
    }

    /// <summary>Loads Mimi's state dict (the release's <c>MimiModel</c> names, a <c>.safetensors</c> file).</summary>
    public void LoadMimi(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var state = ReadState(path);
        var used = Core.Mimi.Load(name => state.TryGetValue(name, out var v) ? v : null);
        var unused = state.Keys.Where(k => !used.Contains(k)).ToList();
        if (unused.Count > 0) throw new InvalidDataException($"The Mimi checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}').");
    }

    /// <summary>Loads WavLM Large for Mimi's distillation (transformers' <c>microsoft/wavlm-large</c>: a directory with
    /// <c>pytorch_model.bin</c> or <c>model.safetensors</c>, or the file). Needs <see cref="MoshiOptions.IncludeMimiTraining"/>.</summary>
    public void LoadWavLm(string path)
    {
        ThrowIfDisposed();
        var wavLm = Core.MimiTraining?.WavLm ?? throw new InvalidOperationException("WavLM is built with MoshiOptions.IncludeMimiTraining.");
        if (Directory.Exists(path))
            path = new[] { "model.safetensors", "pytorch_model.bin" }.Select(f => Path.Combine(path, f)).FirstOrDefault(File.Exists)
                ?? throw new FileNotFoundException($"No model.safetensors or pytorch_model.bin in {path}.");
        var state = ReadState(path);
        string Key(string name) => state.ContainsKey(name) ? name : "wavlm." + name;
        wavLm.LoadTorchWeights((name, _) => state.TryGetValue(Key(name), out var v) ? v : throw new InvalidDataException($"The WavLM checkpoint has no tensor '{name}'."),
            name => state.ContainsKey(Key(name)));
    }

    /// <inheritdoc />
    /// <remarks>Mimi's codebooks are moving averages, not trained parameters: each one's sums, usages and whether it was
    /// initialized are model state.</remarks>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        if (_core is null) return;
        int q = 0;
        foreach (var book in _core.Mimi.AllCodebooks)
        {
            var b = book;
            double epsilon = _core.Mimi.Configuration.CodebookEpsilon;
            state.DeclareDoubleVector($"MimiCodebook{q}.Sum", () => new Vector<double>(b.Sum),
                v => { if (v is not null) b.Load(v.ToArray(), b.Usage, epsilon, b.Initialized); });
            state.DeclareDoubleVector($"MimiCodebook{q}.Usage", () => new Vector<double>(b.Usage),
                v => { if (v is not null) b.Load(b.Sum, v.ToArray(), epsilon, b.Initialized); });
            state.DeclareDoubleVector($"MimiCodebook{q}.Initialized", () => new Vector<double>(new[] { b.Initialized ? 1.0 : 0.0 }),
                v => { if (v is not null) b.Initialized = v[0] != 0; });
            q++;
        }
    }
}
