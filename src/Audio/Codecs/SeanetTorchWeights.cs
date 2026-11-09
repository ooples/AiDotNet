using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Codecs;

/// <summary>Loads original AudioCraft/EnCodec SEANet names without a flattened parameter-vector conversion.</summary>
internal static class SeanetTorchWeights
{
    internal static void Load<T>(Func<string, int[], double[]> read, SeanetEncoder<T> encoder, SeanetDecoder<T> decoder,
        ResidualVectorQuantizerLayer<T> quantizer, int quantizers, int codes, int dimension, string prefix = "", string sequentialName = "model") =>
        Load(TorchTensorSource.From(read), encoder, decoder, quantizer, quantizers, codes, dimension, prefix, sequentialName);

    /// <summary>Loads the AudioCraft SEANet names straight from <paramref name="source"/>.</summary>
    internal static void Load<T>(TorchTensorSource source, SeanetEncoder<T> encoder, SeanetDecoder<T> decoder,
        ResidualVectorQuantizerLayer<T> quantizer, int quantizers, int codes, int dimension, string prefix = "", string sequentialName = "model")
    {
        void Conv(string name, NormedConv1DLayer<T> conv, bool transposed)
        {
            string p = prefix + name + (transposed ? ".convtr.convtr" : ".conv.conv");
            conv.LoadTorchWeights(source, p + ".weight_v", conv.TorchWeightShape, p + ".weight_g", p + ".bias");
        }
        void Lstm(string name, SeanetLstm<T> lstm)
        {
            for (int i = 0; i < lstm.ForwardCells.Count; i++)
            {
                var cell = lstm.ForwardCells[i]; string p = prefix + name + ".lstm.";
                cell.LoadTorchWeights(source, p + "weight_ih_l" + i, p + "weight_hh_l" + i, p + "bias_ih_l" + i, p + "bias_hh_l" + i);
            }
        }
        foreach (var (name, conv) in encoder.NamedConvolutions("encoder." + sequentialName)) Conv(name, conv.Conv, false);
        if (encoder.NamedLstm("encoder." + sequentialName) is { } e) Lstm(e.Name, e.Lstm);
        foreach (var (name, conv) in decoder.NamedConvolutions("decoder." + sequentialName)) Conv(name, conv.Conv, false);
        foreach (var (name, conv) in decoder.NamedTransposedConvolutions("decoder." + sequentialName)) Conv(name, conv.Conv, true);
        if (decoder.NamedLstm("decoder." + sequentialName) is { } d) Lstm(d.Name, d.Lstm);
        for (int q = 0; q < quantizers; q++)
        {
            string p = prefix + $"quantizer.vq.layers.{q}._codebook.";
            if (quantizer.Bins != codes || quantizer.Codebook(q).Length != codes * dimension)
                throw new ArgumentException($"Expected codebooks [{codes}, {dimension}]; the quantizer's are [{quantizer.Bins}, {quantizer.Codebook(q).Length / quantizer.Bins}].");
            quantizer.LoadCodebook(q, source, p + "embed", p + "embed_avg", p + "cluster_size", source.Read(p + "inited", new[] { 1 })[0] != 0);
        }
    }
}
