namespace AiDotNet.Audio.Generation;

public partial class EnCodec<T>
{
    /// <summary>The original summed RVQ embedding used to condition multiband diffusion.
    /// This is the same latent supplied to the EnCodec decoder, before waveform generation.</summary>
    internal Tensor<T> DecodeCodesToEmbedding(int[,] codes)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers || _quantizer is null) throw new NotSupportedException("Codec embeddings require native EnCodec.");
        return _quantizer.Decode(codes);
    }
}
