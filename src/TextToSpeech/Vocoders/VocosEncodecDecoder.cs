using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>The sizes of Vocos's EnCodec-token model (charactr/vocos-encodec-24khz, <c>config.yaml</c>).</summary>
/// <param name="Bandwidths">EnCodec bandwidths in kbps, one adaptive-norm class each (1.5, 3, 6, 12).</param>
/// <param name="CodebookSize">Codes per codebook (1024).</param>
/// <param name="LatentDim">EnCodec's latent width (128).</param>
/// <param name="FrameRate">Codec frames a second (75).</param>
/// <param name="Dim">Backbone width (384).</param>
/// <param name="IntermediateDim">ConvNeXt intermediate width (1152).</param>
/// <param name="Layers">ConvNeXt blocks (8).</param>
/// <param name="FftSize">ISTFT size (1280).</param>
/// <param name="HopSize">ISTFT hop (320).</param>
internal sealed record VocosEncodecConfiguration(
    double[] Bandwidths, int CodebookSize = 1024, int LatentDim = 128, int FrameRate = 75, int Dim = 384,
    int IntermediateDim = 1152, int Layers = 8, int FftSize = 1280, int HopSize = 320)
{
    /// <summary>The released model's configuration.</summary>
    public static VocosEncodecConfiguration Released() => new(new[] { 1.5, 3.0, 6.0, 12.0 });

    /// <summary>Codebooks at the highest bandwidth (the codebook table's blocks).</summary>
    public int MaxCodebooks => (int)Math.Round(Bandwidths.Max() * 1000 / (FrameRate * Math.Log(CodebookSize, 2)));
}

/// <summary>
/// Vocos decoding EnCodec codes (Siuzdak 2024, §4.2; gemelo-ai/vocos <c>EncodecFeatures</c> with the
/// <c>VocosBackbone</c> and <c>ISTFTHead</c>): each frame's feature is the sum of its codes' EnCodec codebook vectors (the
/// quantized latent, without EnCodec's decoder), and the backbone's layer norms adapt to the bandwidth.
/// </summary>
/// <remarks>VALL-E 2 decodes its codes with the released model (Chen et al. 2024, §4.1.1). The codebook vectors are
/// EnCodec's, stored in the checkpoint (<c>feature_extractor.codebook_weights</c>, frozen there).</remarks>
internal sealed class VocosEncodecDecoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly VocosGenerator<T> _generator;
    private readonly List<LayerBase<T>> _layers = new();

    public VocosEncodecDecoder(IEngine engine, Random initialization, VocosEncodecConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        Codebooks = new TiedEmbeddingLayer<T>(c.MaxCodebooks * c.CodebookSize, c.LatentDim);
        Codebooks.Reinitialize(() =>
        {
            double u1 = 1.0 - initialization.NextDouble(), u2 = initialization.NextDouble();
            return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        });
        _layers.Add(Codebooks);
        _generator = new VocosGenerator<T>(engine, initialization, c.LatentDim, c.Dim, c.IntermediateDim, c.Layers, c.FftSize, c.HopSize,
            adaptiveClasses: c.Bandwidths.Length, samePadding: true);
        _layers.AddRange(_generator.Layers);
    }

    public VocosEncodecConfiguration Configuration { get; }

    /// <summary>EnCodec's codebook vectors, codebook by codebook <c>[codebooks · codes, latent]</c>.</summary>
    public TiedEmbeddingLayer<T> Codebooks { get; }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    /// <summary>The bandwidth class of <paramref name="codebooks"/> codebooks.</summary>
    public int BandwidthId(int codebooks)
    {
        var c = Configuration;
        for (int i = 0; i < c.Bandwidths.Length; i++)
            if ((int)Math.Round(c.Bandwidths[i] * 1000 / (c.FrameRate * Math.Log(c.CodebookSize, 2))) == codebooks) return i;
        throw new ArgumentException($"Vocos decodes {string.Join(", ", c.Bandwidths)} kbps; {codebooks} codebooks is none of them.");
    }

    /// <summary>The waveform <c>[frames · hop]</c> of codes <c>[codebooks, frames]</c> (codes_to_features, then decode at
    /// that bandwidth).</summary>
    public Tensor<T> Decode(int[,] codes)
    {
        int codebooks = codes.GetLength(0), frames = codes.GetLength(1);
        if (codebooks > Configuration.MaxCodebooks)
            throw new ArgumentException($"Vocos's table holds {Configuration.MaxCodebooks} codebooks; got {codebooks}.", nameof(codes));
        // embedding(codes + offsets).sum(dim=0): codebook q's code k is row q · bins + k.
        var ids = new Tensor<T>(new[] { codebooks * frames });
        for (int q = 0; q < codebooks; q++)
            for (int f = 0; f < frames; f++) ids[q * frames + f] = NumOps.FromDouble(q * Configuration.CodebookSize + codes[q, f]);
        var vectors = _engine.Reshape(Codebooks.Forward(ids), new[] { codebooks, frames, Configuration.LatentDim });
        var summed = _engine.ReduceSum(vectors, new[] { 0 }, keepDims: false);                                     // [F, latent]
        var features = _engine.Reshape(_engine.TensorTranspose(summed), new[] { 1, Configuration.LatentDim, frames });
        var audio = _generator.Forward(features, BandwidthId(codebooks));
        return _engine.Reshape(audio, new[] { audio.Length });
    }

    /// <summary>Loads the released model's state dictionary (<c>pytorch_model.bin</c> of charactr/vocos-encodec-24khz).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        Codebooks.LoadTable(read("feature_extractor.codebook_weights", new[] { c.MaxCodebooks * c.CodebookSize, c.LatentDim }));
        _generator.LoadTorchWeights(read, c.LatentDim, c.IntermediateDim);
    }
}
