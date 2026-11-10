using AiDotNet.Audio.Codecs;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Effects;

public partial class DAC<T>
{
    /// <summary>
    /// Loads the official DAC weights from a safetensors file in the Hugging Face layout (for example
    /// <c>descript/dac_44khz</c>'s <c>model.safetensors</c>) into this model's encoder, quantizer and decoder.
    /// </summary>
    /// <param name="path">The safetensors file.</param>
    /// <remarks>
    /// <para>That layout stores each weight-normalized convolution's folded weight; it loads as the direction V with gain
    /// ‖V‖, which reproduces the same kernel. Every tensor's shape is checked, and a checkpoint tensor this model has no
    /// place for is an error. The discriminators are not part of the release.</para>
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
                    "model; build the model with the checkpoint's configuration.");
            used.Add(name);
            return name;
        }

        void Load(string name, LayerBase<T> layer)
        {
            switch (layer)
            {
                case NormedConv1DLayer<T> conv:
                    conv.LoadTorchWeights(file, Use($"{name}.weight", conv.TorchWeightShape), conv.TorchWeightShape, null,
                        Use($"{name}.bias", conv.OutputChannels));
                    break;
                case SnakeLayer<T> snake:
                    snake.LoadAlpha(file.Read(Use($"{name}.alpha", 1, snake.Channels, 1), new[] { 1, snake.Channels, 1 }));
                    break;
                default:
                    throw new InvalidOperationException($"No loader for {layer.GetType().Name} at '{name}'.");
            }
        }

        foreach (var (name, layer) in _encoder!.Named("encoder")) Load(name, layer);
        foreach (var (name, layer) in _decoder!.Named("decoder")) Load(name, layer);
        var o = PaperOptions;
        for (int q = 0; q < o.NumQuantizers; q++)
        {
            string prefix = $"quantizer.quantizers.{q}";
            Load($"{prefix}.in_proj", _quantizer!.InProjection(q));
            Load($"{prefix}.out_proj", _quantizer.OutProjection(q));
            var target = _quantizer.Codebook(q);
            file.CopyTo(Use($"{prefix}.codebook.weight", o.CodebookSize, o.CodebookDim), new[] { o.CodebookSize, o.CodebookDim }, target);
            Engine.InvalidatePersistentTensor(target);
        }

        var unused = names.Where(n => !used.Contains(n)).ToList();
        if (unused.Count > 0)
            throw new InvalidDataException($"The checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}'); " +
                "its configuration differs from the model's.");
    }

    /// <summary>
    /// Loads a checkpoint in descript-audio-codec's own layout (its <c>DAC.save</c> state dict, as
    /// <c>weights_*.pth</c> files and fine-tunes such as ibm-research/DAC.speech.v1.0 store it): <c>encoder.block.*</c>,
    /// <c>decoder.model.*</c> and <c>quantizer.quantizers.*</c>, each weight-normalized convolution as its
    /// <c>weight_g</c> and <c>weight_v</c>. Its convolutions are mapped to this model's (the layout Hugging Face's
    /// conversion script renames), and every tensor must be used.
    /// </summary>
    /// <param name="read">The checkpoint's tensors by name, null for a name it lacks.</param>
    internal void LoadDescriptWeights(Func<string, double[]?> read) => LoadDescriptWeights(TorchTensorSource.From(read));

    /// <summary>Loads descript-audio-codec's layout straight from <paramref name="source"/>.</summary>
    internal void LoadDescriptWeights(TorchTensorSource source)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("Pretrained weights load into the native encoder, quantizer and decoder.");
        var o = PaperOptions;
        _usedDescriptNames = LoadDescriptModules(Engine, _encoder!, _decoder!, _quantizer!, o.EncoderRates.Length, o.DecoderRates.Length, o.NumQuantizers,
            o.CodebookSize, o.CodebookDim, source);
    }

    /// <summary>Loads descript-audio-codec's layout into DAC's modules wherever they live (a codec, or a model that owns
    /// them); returns the names read.</summary>
    internal static HashSet<string> LoadDescriptModules(IEngine engine, DacEncoder<T> encoder, DacDecoder<T> decoder, FactorizedVectorQuantizerLayer<T> quantizer,
        int encoderBlocks, int decoderBlocks, int quantizers, int codebookSize, int codebookDim, Func<string, double[]?> read) =>
        LoadDescriptModules(engine, encoder, decoder, quantizer, encoderBlocks, decoderBlocks, quantizers, codebookSize, codebookDim,
            TorchTensorSource.From(read));

    /// <summary>Loads descript-audio-codec's layout straight from <paramref name="source"/>; returns the names read.</summary>
    internal static HashSet<string> LoadDescriptModules(IEngine engine, DacEncoder<T> encoder, DacDecoder<T> decoder, FactorizedVectorQuantizerLayer<T> quantizer,
        int encoderBlocks, int decoderBlocks, int quantizers, int codebookSize, int codebookDim, TorchTensorSource source)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        string Use(string name)
        {
            if (!source.Contains(name)) throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            used.Add(name);
            return name;
        }
        double[] Read(string name, int count)
        {
            var values = source.TryRead(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            if (values.Length != count) throw new InvalidDataException($"'{name}' has {values.Length} values; this model expects {count}.");
            used.Add(name);
            return values;
        }
        void Load(string descript, LayerBase<T> layer)
        {
            switch (layer)
            {
                case NormedConv1DLayer<T> conv:
                    conv.LoadTorchWeights(source, Use($"{descript}.weight_v"), conv.TorchWeightShape, Use($"{descript}.weight_g"), Use($"{descript}.bias"));
                    break;
                case SnakeLayer<T> snake:
                    snake.LoadAlpha(Read($"{descript}.alpha", snake.Channels));
                    break;
                default:
                    throw new InvalidOperationException($"No loader for {layer.GetType().Name} at '{descript}'.");
            }
        }
        foreach (var (name, layer) in encoder.Named("encoder")) Load(DescriptName(name, encoderBlocks, decoderBlocks), layer);
        foreach (var (name, layer) in decoder.Named("decoder")) Load(DescriptName(name, encoderBlocks, decoderBlocks), layer);
        for (int q = 0; q < quantizers; q++)
        {
            string prefix = $"quantizer.quantizers.{q}";
            Load($"{prefix}.in_proj", quantizer.InProjection(q));
            Load($"{prefix}.out_proj", quantizer.OutProjection(q));
            var target = quantizer.Codebook(q);
            source.CopyTo(Use($"{prefix}.codebook.weight"), new[] { codebookSize, codebookDim }, target);
            engine.InvalidatePersistentTensor(target);
        }
        return used;
    }

    private HashSet<string>? _usedDescriptNames;

    /// <summary>The descript-layout names the last <see cref="LoadDescriptWeights"/> read.</summary>
    internal IReadOnlyCollection<string> UsedDescriptNames => _usedDescriptNames ?? new HashSet<string>();

    /// <summary>Loads descript-audio-codec's <c>.pth</c> checkpoint (its <c>state_dict</c> entry, or a bare state dict)
    /// in descript's layout; see <see cref="LoadDescriptWeights"/>.</summary>
    public void LoadDescriptCheckpoint(string path)
    {
        // A .pth state dict is unpickled whole; its tensors are copied from there into the layers.
        var all = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        var state = new Dictionary<string, Tensor<float>>(StringComparer.Ordinal);
        foreach (var entry in all)
        {
            string name = entry.Key.StartsWith("state_dict.", StringComparison.Ordinal) ? entry.Key.Substring("state_dict.".Length) : entry.Key;
            state[name] = entry.Value;
        }
        LoadDescriptWeights(TorchTensorSource.From(state, $"the checkpoint {path}"));
        var unused = state.Keys.Where(n => !UsedDescriptNames.Contains(n)).ToList();
        if (unused.Count > 0)
            throw new InvalidDataException($"The checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}'); " +
                "its configuration differs from the model's.");
    }

    // Hugging Face's DacModel names to descript's: encoder.block.0 is the first convolution, encoder.block.{i+1} the i-th
    // block (block.0-2 its residual units, block.3 its Snake, block.4 its strided convolution), then the final Snake and
    // convolution; the decoder likewise under decoder.model (block.0 the Snake, block.1 the transposed convolution,
    // block.2-4 the residual units); a residual unit's block.0-3 are snake1, conv1, snake2, conv2.
    private static string DescriptName(string name, int encoderBlocks, int decoderBlocks)
    {
        var parts = name.Split('.');
        bool encoder = parts[0] == "encoder";
        string root = encoder ? "encoder.block" : "decoder.model";
        int blocks = encoder ? encoderBlocks : decoderBlocks;
        if (parts.Length == 2)
            return parts[1] switch
            {
                "conv1" => $"{root}.0",
                "snake1" => $"{root}.{blocks + 1}",
                "conv2" => $"{root}.{blocks + 2}",
                _ => throw new InvalidOperationException($"Unexpected DAC layer '{name}'."),
            };
        int block = int.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture) + 1;
        string member = parts[3];
        if (member.StartsWith("res_unit", StringComparison.Ordinal))
        {
            int unit = int.Parse(member.Substring("res_unit".Length), System.Globalization.CultureInfo.InvariantCulture) - 1;
            int inner = parts[4] switch { "snake1" => 0, "conv1" => 1, "snake2" => 2, "conv2" => 3, _ => throw new InvalidOperationException($"Unexpected DAC layer '{name}'.") };
            return $"{root}.{block}.block.{(encoder ? unit : unit + 2)}.block.{inner}";
        }
        return (encoder, member) switch
        {
            (true, "snake1") => $"{root}.{block}.block.3",
            (true, "conv1") => $"{root}.{block}.block.4",
            (false, "snake1") => $"{root}.{block}.block.0",
            (false, "conv_t1") => $"{root}.{block}.block.1",
            _ => throw new InvalidOperationException($"Unexpected DAC layer '{name}'."),
        };
    }
}
