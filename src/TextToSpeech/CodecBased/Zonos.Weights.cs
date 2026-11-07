using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class Zonos<T>
{
    private readonly List<LayerBase<T>> _speakerLayers = new();

    /// <summary>Loads Zonos's weights (<c>Zyphra/Zonos-v0.1-transformer</c> or <c>-hybrid</c>): its
    /// <c>model.safetensors</c>, or a directory holding it. Build the model with the matching options
    /// (<see cref="ZonosOptions.Hybrid"/> for the hybrid).</summary>
    public void LoadReferenceWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        if (path is null) throw new ArgumentNullException(nameof(path));
        if (Directory.Exists(path)) path = Path.Combine(path, "model.safetensors");
        if (!File.Exists(path)) throw new FileNotFoundException($"The checkpoint {path} does not exist.", path);
        var state = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        // Each tensor is widened when it is read rather than the whole checkpoint at once.
        double[] Read(string name, int[] shape)
        {
            if (!state.TryGetValue(name, out var tensor)) throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            int count = shape.Aggregate(1, (a, s) => a * s);
            if (tensor.Length != count) throw new InvalidDataException($"'{name}' has {tensor.Length} values; this model expects {count}.");
            return Array.ConvertAll(tensor.ToArray(), x => (double)x);
        }
        var used = LoadTorchWeights(Read);
        var unused = state.Keys.Where(k => !used.Contains(k)).ToList();
        if (unused.Count > 0) throw new InvalidDataException($"The checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}').");
    }

    /// <summary>Loads a state dict in the release's names (the backbone, tables, heads and prefix conditioner); returns the
    /// names read.</summary>
    internal HashSet<string> LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var used = Network.LoadTorchWeights(read);
        used.UnionWith(Conditioner.LoadTorchWeights(read));
        return used;
    }

    /// <summary>
    /// Loads the release's speaker model (<c>Zyphra/Zonos-v0.1-speaker-embedding</c>): the directory holding
    /// <c>ResNet293_SimAM_ASP_base.pt</c> and <c>ResNet293_SimAM_ASP_base_LDA-128.pt</c>.
    /// </summary>
    public void LoadSpeakerModel(string directory)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        string model = Path.Combine(directory ?? throw new ArgumentNullException(nameof(directory)), "ResNet293_SimAM_ASP_base.pt");
        string lda = Path.Combine(directory, "ResNet293_SimAM_ASP_base_LDA-128.pt");
        foreach (var file in new[] { model, lda })
            if (!File.Exists(file)) throw new FileNotFoundException($"The speaker model file {file} does not exist.", file);
        var loader = new AiDotNet.ComputerVision.Weights.WeightLoader();
        var state = loader.LoadWeights(model);
        foreach (var entry in loader.LoadWeights(lda)) state["lda." + entry.Key] = entry.Value;
        LoadSpeakerWeights((name, shape) =>
        {
            if (!state.TryGetValue(name, out var tensor)) throw new InvalidDataException($"The speaker model has no tensor '{name}'.");
            int count = shape.Aggregate(1, (a, s) => a * s);
            if (tensor.Length != count) throw new InvalidDataException($"'{name}' has {tensor.Length} values; the speaker model expects {count}.");
            return Array.ConvertAll(tensor.ToArray(), x => (double)x);
        }, state.Keys.Where(k => !k.EndsWith("num_batches_tracked", StringComparison.Ordinal)));
    }

    /// <summary>Builds the speaker model and loads a state dict in the release's names (the ResNet's, and the LDA's under
    /// <c>lda.</c>); every name in <paramref name="names"/> must be read.</summary>
    internal void LoadSpeakerWeights(Func<string, int[], double[]> read, IEnumerable<string> names, int[]? blocks = null, int channels = 64)
    {
        _speakerLayers.Clear();
        var encoder = new ZonosSpeakerEncoder<T>(Engine, _speakerLayers, blocks, channels);
        var used = encoder.LoadTorchWeights(read);
        var unused = names.Where(k => !used.Contains(k)).ToList();
        if (unused.Count > 0) throw new InvalidDataException($"The speaker model has {unused.Count} tensors with no place (first: '{unused[0]}').");
        foreach (var layer in _speakerLayers) layer.SetTrainingMode(false);
        _speakerEncoder = encoder;
    }

    /// <summary>The speaker embedding Zonos conditions on (<c>make_speaker_embedding</c>): the speaker model's 128-value LDA
    /// projection of <paramref name="audio"/> (<c>[samples]</c>, or <c>[channels, samples]</c> averaged) at
    /// <paramref name="sampleRate"/>.</summary>
    public double[] SpeakerEmbedding(Tensor<T> audio, int sampleRate)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        var encoder = _speakerEncoder ?? throw new InvalidOperationException("Load the speaker model first (LoadSpeakerModel with Zyphra/Zonos-v0.1-speaker-embedding).");
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return encoder.Forward(encoder.Prepare(audio, sampleRate)).Lda;
    }

    internal ZonosSpeakerEncoder<T>? SpeakerEncoder => _speakerEncoder;
}
