using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Audio.Codecs;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Generation;

public partial class SpeechTokenizer<T>
{
    /// <summary>
    /// Loads the official SpeechTokenizer weights (the released <c>SpeechTokenizer.pt</c>, a PyTorch state dictionary of
    /// the reference model) into this model's encoder, quantizer, decoder and semantic projection.
    /// </summary>
    /// <param name="path">The checkpoint (.pt).</param>
    /// <remarks>Build the model with <see cref="SpeechTokenizerOptions.OfficialCheckpoint"/>: the release uses C = 64 where
    /// the paper states 32. Every tensor's shape is checked; a checkpoint tensor this model has no place for is an
    /// error.</remarks>
    public void LoadPretrainedWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("Pretrained weights load into the native encoder, quantizer and decoder.");
        var file = new WeightLoader().LoadWeights(path);
        var used = new HashSet<string>();

        double[] Read(string name, params int[] shape)
        {
            if (!file.TryGetValue(name, out var tensor)) throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            var actual = tensor.Shape.ToArray();
            if (!actual.SequenceEqual(shape))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] in this " +
                    "model; build the model with the checkpoint's configuration (SpeechTokenizerOptions.OfficialCheckpoint).");
            used.Add(name);
            return tensor.ToVector().Select(v => (double)v).ToArray();
        }

        void LoadConv(string name, NormedConv1DLayer<T> conv)
        {
            var shape = conv.TorchWeightShape;
            conv.LoadTorchWeights(Read($"{name}.weight_v", shape), Read($"{name}.weight_g", shape[0], 1, 1), Read($"{name}.bias", conv.OutputChannels));
        }

        void LoadLstm(string name, SeanetLstm<T> lstm)
        {
            var forward = lstm.ForwardCells;
            var backward = lstm.BackwardCells;
            for (int l = 0; l < forward.Count; l++)
            {
                foreach (var (cell, suffix) in new[] { (forward[l], ""), (backward[l], "_reverse") })
                {
                    if (cell is null) continue;
                    int hidden = cell.HiddenSize, input = cell.InputSize;
                    cell.LoadTorchWeights(Read($"{name}.lstm.weight_ih_l{l}{suffix}", 4 * hidden, input),
                        Read($"{name}.lstm.weight_hh_l{l}{suffix}", 4 * hidden, hidden),
                        Read($"{name}.lstm.bias_ih_l{l}{suffix}", 4 * hidden), Read($"{name}.lstm.bias_hh_l{l}{suffix}", 4 * hidden));
                }
            }
        }

        foreach (var (name, conv) in _encoder!.NamedConvolutions("encoder.model")) LoadConv($"{name}.conv.conv", conv.Conv);
        if (_encoder.NamedLstm("encoder.model") is { } encoderLstm) LoadLstm(encoderLstm.Name, encoderLstm.Lstm);
        foreach (var (name, conv) in _decoder!.NamedConvolutions("decoder.model")) LoadConv($"{name}.conv.conv", conv.Conv);
        foreach (var (name, conv) in _decoder.NamedTransposedConvolutions("decoder.model")) LoadConv($"{name}.convtr.convtr", conv.Conv);
        if (_decoder.NamedLstm("decoder.model") is { } decoderLstm) LoadLstm(decoderLstm.Name, decoderLstm.Lstm);

        var o = PaperOptions;
        for (int q = 0; q < o.NumQuantizers; q++)
        {
            string prefix = $"quantizer.vq.layers.{q}._codebook";
            _quantizer!.LoadCodebook(q, Read($"{prefix}.embed", o.CodebookSize, o.Dimension), Read($"{prefix}.embed_avg", o.CodebookSize, o.Dimension),
                Read($"{prefix}.cluster_size", o.CodebookSize), Read($"{prefix}.inited", 1)[0] != 0);
        }

        // transform: nn.Linear(dimension, semantic_dimension), weight [out, in].
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            _projection!.Forward(new Tensor<T>(new[] { 1, o.Dimension }));
        var weight = Read("transform.weight", o.SemanticDimension, o.Dimension);
        var bias = Read("transform.bias", o.SemanticDimension);
        var w = _projection!.GetWeights();                                                       // [in, out]
        var b = _projection.GetBiases();
        for (int r = 0; r < o.SemanticDimension; r++)
        {
            for (int c = 0; c < o.Dimension; c++) w[c, r] = NumOps.FromDouble(weight[r * o.Dimension + c]);
            b[r] = NumOps.FromDouble(bias[r]);
        }
        Engine.InvalidatePersistentTensor(w);
        Engine.InvalidatePersistentTensor(b);

        var unused = file.Keys.Where(n => !used.Contains(n)).ToList();
        if (unused.Count > 0)
            throw new InvalidDataException($"The checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}'); " +
                "its configuration differs from the model's.");
    }
}
