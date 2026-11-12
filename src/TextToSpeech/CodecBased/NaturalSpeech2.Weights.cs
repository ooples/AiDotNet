using AiDotNet.ModelLoading;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class NaturalSpeech2<T>
{
    /// <summary>
    /// Loads Amphion's NaturalSpeech 2 checkpoint (<c>amphion/naturalspeech2_libritts</c>'s <c>pytorch_model.bin</c>). Build
    /// the model with <see cref="NaturalSpeech2Options.Amphion"/>, and load EnCodec's 24 kHz checkpoint (<c>facebook/encodec_24khz</c>'s <c>model.safetensors</c>) into
    /// <see cref="Codec"/> (<c>EnCodec.LoadPretrainedWeights</c>); the checkpoint's own copy of the quantizer's codebooks is
    /// loaded too.
    /// </summary>
    public void LoadAmphionCheckpoint(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        using var file = TorchCheckpointFile.Open(path);
        LoadAmphionState(file);
    }

    /// <summary>Loads Amphion's state dict through <paramref name="read"/> (name → values, or null when absent).</summary>
    internal void LoadAmphionState(Func<string, double[]?> read) => LoadAmphionState(TorchTensorSource.From(read));

    /// <summary>Loads Amphion's state dict from <paramref name="source"/>.</summary>
    internal void LoadAmphionState(TorchTensorSource source)
    {
        Core.Load(source);
        var quantizer = Quantizer;
        for (int q = 0; q < quantizer.Quantizers; q++)
        {
            string p = $"quantizer.vq.layers.{q}._codebook.";
            if (source.TryRead(p + "embed") is not { } embed) continue;
            quantizer.LoadCodebook(q, embed, source.TryRead(p + "embed_avg") ?? embed,
                source.TryRead(p + "cluster_size") ?? Enumerable.Repeat(1.0, quantizer.Bins).ToArray(), true);
        }
    }
}
