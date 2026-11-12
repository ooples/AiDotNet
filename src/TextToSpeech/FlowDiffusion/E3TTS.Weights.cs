using AiDotNet.ModelLoading;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Checkpoint inputs for E3's public BERT dependency and AiDotNet-converted private U-Net weights.</summary>
public sealed record E3TTSCheckpointPaths(string Bert, string ConvertedUnet);

public partial class E3TTS<T>
{
    /// <summary>Loads a Hugging Face/Google bert-base-uncased checkpoint into E3's text encoder.</summary>
    public void LoadBertWeights(string path)
    {
        ThrowIfDisposed();
        RequirePaperModel();
        using var file = Open(path);
        string prefix = file.Contains("bert.embeddings.word_embeddings.weight") ? "bert." : string.Empty;
        TextEncoder!.LoadHuggingFace(file, prefix);
    }

    /// <summary>
    /// Loads E3 U-Net weights converted to AiDotNet's documented module names. The E3 authors did not release source or
    /// checkpoints, so these names are deliberately not described as an official format.
    /// </summary>
    /// <remarks>
    /// Root names are <c>unet.time.0</c>, <c>unet.time.2</c>, <c>unet.speaker_embedding</c>,
    /// <c>unet.variance</c>, <c>unet.down.{i}</c>, <c>unet.up.{i}</c> and <c>unet.output</c>. Blocks contain
    /// <c>first.{expert}</c>, <c>second.{expert}</c> for adaptive blocks (otherwise <c>second</c>),
    /// optional <c>shortcut</c>, <c>norm1</c>/<c>norm2</c>, either
    /// <c>adaptive_logits</c> or <c>film</c>, and optional <c>self</c>/<c>cross</c> attention projections.
    /// </remarks>
    public void LoadConvertedWeights(string path)
    {
        ThrowIfDisposed();
        RequirePaperModel();
        using var file = Open(path);
        WaveformUnet!.Load(file, "unet.");
    }

    public void LoadCheckpoints(E3TTSCheckpointPaths checkpoints)
    {
        Validation.Guard.NotNull(checkpoints);
        LoadBertWeights(checkpoints.Bert);
        LoadConvertedWeights(checkpoints.ConvertedUnet);
    }

    private static TorchCheckpointFile Open(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Checkpoint path required.", nameof(path));
        if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
        return TorchCheckpointFile.Open(path);
    }
}
