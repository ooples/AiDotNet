using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Audio.Codecs;
using AiDotNet.ModelLoading;

namespace AiDotNet.Audio.Generation;

public partial class EnCodec<T>
{
    /// <summary>Loads the original weight-normalized SEANet state dictionary (as safetensors or a supported PyTorch archive).</summary>
    public void LoadAuthorWeights(string path, string prefix = "")
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("Author weights require the native codec.");
        if (PaperOptions.TimeGroupNormalization ?? !PaperOptions.Causal)
            throw new NotSupportedException("Use LoadPretrainedWeights for the time-group-normalized EnCodec preset.");
        // A safetensors file or a .pt/.pth archive streams with each tensor's axes checked; another torch.save archive
        // (AudioCraft's .th) is unpickled whole.
        string extension = Path.GetExtension(path).ToLowerInvariant();
        bool streamed = extension is ".safetensors" or ".pt" or ".pth" or ".tar" or ".ckpt";
        using var file = streamed ? TorchCheckpointFile.Open(path, strictAxes: true) : null;
        TorchTensorSource source = file ?? TorchTensorSource.From(new ComputerVision.Weights.WeightLoader().LoadWeights(path), $"the checkpoint {path}");
        prefix = AuthorCheckpointPrefix(source, prefix);
        var o = PaperOptions;
        SeanetTorchWeights.Load(source, _encoder!, _decoder!, _quantizer!, o.NumQuantizers, o.CodebookSize, o.Dimension, prefix);
    }

    internal static string AuthorCheckpointPrefix(IReadOnlyDictionary<string, Tensor<float>> state, string requestedPrefix) =>
        AuthorCheckpointPrefix(TorchTensorSource.From(state), requestedPrefix);

    private static string AuthorCheckpointPrefix(TorchTensorSource state, string requestedPrefix)
    {
        if (!string.IsNullOrEmpty(requestedPrefix)) return requestedPrefix;
        const string firstWeight = "encoder.model.0.conv.conv.weight_v";
        if (state.Contains(firstWeight)) return string.Empty;
        // AudioCraft CompressionSolver.model_from_checkpoint selects best_state['model'] for inference.
        if (state.Contains("best_state.model." + firstWeight)) return "best_state.model.";
        return state.Contains("model." + firstWeight) ? "model." : string.Empty;
    }

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
        List<string> names;
        using (var header = File.OpenRead(path)) names = SafetensorsReader.Read(header).Names.ToList();
        using var file = TorchCheckpointFile.Open(path);
        var used = new HashSet<string>();

        // Checks the stored shape and returns the name.
        string Use(string name, params int[] shape)
        {
            var actual = file.ShapeOf(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            if (!actual.SequenceEqual(shape.Select(d => (long)d)))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] in this " +
                    "model; build the model with the checkpoint's configuration (EnCodecOptions.OfficialCheckpoint24kHz or OfficialCheckpoint48kHz).");
            used.Add(name);
            return name;
        }

        void LoadConv(string name, AiDotNet.NeuralNetworks.Layers.NormedConv1DLayer<T> conv,
            AiDotNet.NeuralNetworks.Layers.GroupNormalizationLayer<T>? norm)
        {
            var shape = conv.TorchWeightShape;
            string bias = Use($"{name}.conv.bias", conv.OutputChannels);
            if (file.Contains($"{name}.conv.weight_g"))
                conv.LoadTorchWeights(file, Use($"{name}.conv.weight_v", shape), shape, Use($"{name}.conv.weight_g", shape[0], 1, 1), bias);
            else
                conv.LoadTorchWeights(file, Use($"{name}.conv.weight", shape), shape, null, bias);
            if (norm is not null)
                norm.LoadAffine(file, Use($"{name}.norm.weight", conv.OutputChannels), Use($"{name}.norm.bias", conv.OutputChannels));
        }

        void LoadLstm(string name, SeanetLstm<T> lstm)
        {
            var cells = lstm.ForwardCells;
            for (int l = 0; l < cells.Count; l++)
            {
                int hidden = cells[l].HiddenSize, input = cells[l].InputSize;
                cells[l].LoadTorchWeights(file, Use($"{name}.lstm.weight_ih_l{l}", 4 * hidden, input), Use($"{name}.lstm.weight_hh_l{l}", 4 * hidden, hidden),
                    Use($"{name}.lstm.bias_ih_l{l}", 4 * hidden), Use($"{name}.lstm.bias_hh_l{l}", 4 * hidden));
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
            _quantizer!.LoadCodebook(q, file, Use($"{prefix}.embed", o.CodebookSize, o.Dimension), Use($"{prefix}.embed_avg", o.CodebookSize, o.Dimension),
                Use($"{prefix}.cluster_size", o.CodebookSize), file.Read(Use($"{prefix}.inited", 1), new[] { 1 })[0] != 0);
        }

        var unused = names.Where(n => !used.Contains(n)).ToList();
        if (unused.Count > 0)
            throw new InvalidDataException($"The checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}'); " +
                "its configuration differs from the model's.");
    }
}
