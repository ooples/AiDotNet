using AiDotNet.ModelLoading;

namespace AiDotNet.TextToSpeech.MultiModal;

public partial class Moshi<T>
{
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
        using var state = TorchCheckpointFile.Open(path);
        Core.LanguageModel.LoadTorchWeights(state);
        state.RequireAllUsed("language model");
    }

    /// <summary>Loads Mimi's state dict (the release's <c>MimiModel</c> names, a <c>.safetensors</c> file).</summary>
    public void LoadMimi(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        using var state = TorchCheckpointFile.Open(path);
        var used = Core.Mimi.Load(name => state.TryRead(name));
        var unused = state.Names.Where(k => !used.Contains(k)).ToList();
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
        using var file = TorchCheckpointFile.Open(path);
        var state = file.WithNames(name => file.Contains(name) ? name : "wavlm." + name);
        wavLm.LoadTorchWeights((name, _) => state.Read(name, Array.Empty<int>()), state.Contains);
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
