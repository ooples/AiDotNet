using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Audio.Codecs;

namespace AiDotNet.Audio.Generation;

public partial class EnCodec<T>
{
    /// <summary>
    /// Loads the official EnCodec weights from a safetensors file in the Hugging Face layout (for example
    /// <c>facebook/encodec_24khz</c>'s <c>model.safetensors</c>) into this model's encoder, quantizer and decoder.
    /// </summary>
    /// <param name="path">The safetensors file.</param>
    /// <remarks>
    /// <para>The model must be built with the checkpoint's configuration — <see cref="EnCodecOptions.OfficialCheckpoint24kHz"/>
    /// or <see cref="EnCodecOptions.OfficialCheckpoint48kHz"/>: the released models use residual kernels (3, 1) where the
    /// paper's text says (3, 3). Every tensor's shape is checked; a mismatch names the tensor and both shapes.</para>
    /// <para>The discriminators are not part of the release and keep their initialization.</para>
    /// </remarks>
    public void LoadPretrainedWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("Pretrained weights load into the native encoder, quantizer and decoder.");
        using var stream = File.OpenRead(path);
        var file = SafetensorsReader.Read(stream);
        var used = new HashSet<string>();

        double[] Read(string name, params int[] shape)
        {
            var tensor = file.Get(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            var actual = tensor.Shape.Select(d => (int)d).ToArray();
            if (!actual.SequenceEqual(shape))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] in this " +
                    "model; build the model with the checkpoint's configuration (EnCodecOptions.OfficialCheckpoint24kHz or OfficialCheckpoint48kHz).");
            used.Add(name);
            return file.ReadAsDouble(name);
        }

        void LoadConv(string name, AiDotNet.NeuralNetworks.Layers.NormedConv1DLayer<T> conv,
            AiDotNet.NeuralNetworks.Layers.GroupNormalizationLayer<T>? norm)
        {
            var shape = conv.TorchWeightShape;
            var bias = Read($"{name}.conv.bias", conv.OutputChannels);
            if (file.Get($"{name}.conv.weight_g") is not null)
                conv.LoadTorchWeights(Read($"{name}.conv.weight_v", shape), Read($"{name}.conv.weight_g", shape[0], 1, 1), bias);
            else
                conv.LoadTorchWeights(Read($"{name}.conv.weight", shape), null, bias);
            if (norm is not null)
                norm.LoadAffine(Read($"{name}.norm.weight", conv.OutputChannels), Read($"{name}.norm.bias", conv.OutputChannels));
        }

        void LoadLstm(string name, SeanetLstm<T> lstm)
        {
            var cells = lstm.ForwardCells;
            for (int l = 0; l < cells.Count; l++)
            {
                int hidden = cells[l].HiddenSize, input = cells[l].InputSize;
                cells[l].LoadTorchWeights(Read($"{name}.lstm.weight_ih_l{l}", 4 * hidden, input), Read($"{name}.lstm.weight_hh_l{l}", 4 * hidden, hidden),
                    Read($"{name}.lstm.bias_ih_l{l}", 4 * hidden), Read($"{name}.lstm.bias_hh_l{l}", 4 * hidden));
            }
        }

        foreach (var (name, conv) in _encoder!.NamedConvolutions("encoder.layers")) LoadConv(name, conv.Conv, conv.Norm);
        if (_encoder.NamedLstm("encoder.layers") is { } encoderLstm) LoadLstm(encoderLstm.Name, encoderLstm.Lstm);
        foreach (var (name, conv) in _decoder!.NamedConvolutions("decoder.layers")) LoadConv(name, conv.Conv, conv.Norm);
        foreach (var (name, conv) in _decoder.NamedTransposedConvolutions("decoder.layers")) LoadConv(name, conv.Conv, conv.Norm);
        if (_decoder.NamedLstm("decoder.layers") is { } decoderLstm) LoadLstm(decoderLstm.Name, decoderLstm.Lstm);

        var o = PaperOptions;
        for (int q = 0; q < o.NumQuantizers; q++)
        {
            string prefix = $"quantizer.layers.{q}.codebook";
            _quantizer!.LoadCodebook(q, Read($"{prefix}.embed", o.CodebookSize, o.Dimension), Read($"{prefix}.embed_avg", o.CodebookSize, o.Dimension),
                Read($"{prefix}.cluster_size", o.CodebookSize), Read($"{prefix}.inited", 1)[0] != 0);
        }

        var unused = file.Names.Where(n => !used.Contains(n)).ToList();
        if (unused.Count > 0)
            throw new InvalidDataException($"The checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}'); " +
                "its configuration differs from the model's.");
    }
}
