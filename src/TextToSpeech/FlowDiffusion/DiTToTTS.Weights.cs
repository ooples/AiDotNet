using AiDotNet.ModelLoading;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// Files used to assemble DiTTo-TTS. SpeechT5 and BigVGAN are public upstream checkpoints. The DiTTo authors did not
/// release their DiT, length predictor, or privately obtained CLaM-TTS Mel-VAE; those three paths therefore name
/// safetensors converted to the stable AiDotNet names described by each module's loader.
/// </summary>
public sealed class DiTToCheckpointPaths
{
    public required string SpeechT5 { get; init; }
    public required string MelVae { get; init; }
    public required string DiffusionTransformer { get; init; }
    public required string LengthPredictor { get; init; }
    public required string BigVgan { get; init; }
}

public partial class DiTToTTS<T>
{
    /// <summary>
    /// Loads the public SpeechT5/BigVGAN state and AiDotNet-converted DiTTo/CLaM tensors. This deliberately does not call
    /// the converted files "official": as of the paper publication there is no author-released DiTTo repository or model.
    /// </summary>
    public void LoadConvertedWeights(DiTToCheckpointPaths checkpoints)
    {
        Validation.Guard.NotNull(checkpoints); ThrowIfDisposed(); RequirePaperModel();
        WithWeights(checkpoints.SpeechT5, source => TextEncoder!.LoadHuggingFace(source));
        WithWeights(checkpoints.MelVae, source => MelVae!.Load(source, ""));
        WithWeights(checkpoints.DiffusionTransformer, source => DiffusionTransformer!.Load(source, ""));
        WithWeights(checkpoints.LengthPredictor, source => LengthPredictor!.Load(source, ""));
        WithWeights(checkpoints.BigVgan, source =>
            BigVgan!.LoadTorch(source.WithNames(name => source.Contains(name) ? name : "generator." + name)));
    }

    // A file, or a directory whose *.safetensors files are one state (a later file's tensor wins a shared name), streamed
    // into the component and closed.
    private static void WithWeights(string path, Action<TorchTensorSource> load)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("A checkpoint path is required.", nameof(path));
        string[] files;
        if (Directory.Exists(path))
        {
            files = Directory.GetFiles(path, "*.safetensors").OrderBy(p => p, StringComparer.Ordinal).ToArray();
            if (files.Length == 0) throw new FileNotFoundException($"No safetensors files were found in {path}.", path);
        }
        else
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
            files = new[] { path };
        }
        var opened = new List<TorchCheckpointFile>();
        try
        {
            foreach (string file in files) opened.Add(TorchCheckpointFile.Open(file));
            TorchTensorSource source = opened[opened.Count - 1];
            for (int i = opened.Count - 2; i >= 0; i--) source = source.Or(opened[i]);
            load(source);
        }
        finally
        {
            foreach (var file in opened) file.Dispose();
        }
    }
}
