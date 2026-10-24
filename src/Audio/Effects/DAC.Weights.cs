using AiDotNet.Audio.Codecs;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Agentic.Models.Local;
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
        using var stream = File.OpenRead(path);
        var file = SafetensorsReader.Read(stream);
        var used = new HashSet<string>();

        double[] Read(string name, params int[] shape)
        {
            var tensor = file.Get(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            var actual = tensor.Shape.Select(d => (int)d).ToArray();
            if (!actual.SequenceEqual(shape))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] in this " +
                    "model; build the model with the checkpoint's configuration.");
            used.Add(name);
            return file.ReadAsDouble(name);
        }

        void Load(string name, LayerBase<T> layer)
        {
            switch (layer)
            {
                case NormedConv1DLayer<T> conv:
                    conv.LoadTorchWeights(Read($"{name}.weight", conv.TorchWeightShape), null, Read($"{name}.bias", conv.OutputChannels));
                    break;
                case SnakeLayer<T> snake:
                    snake.LoadAlpha(Read($"{name}.alpha", 1, snake.Channels, 1));
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
            var codebook = Read($"{prefix}.codebook.weight", o.CodebookSize, o.CodebookDim);
            var target = _quantizer.Codebook(q);
            for (int i = 0; i < codebook.Length; i++) target[i] = NumOps.FromDouble(codebook[i]);
            Engine.InvalidatePersistentTensor(target);
        }

        var unused = file.Names.Where(n => !used.Contains(n)).ToList();
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
    internal void LoadDescriptWeights(Func<string, double[]?> read)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("Pretrained weights load into the native encoder, quantizer and decoder.");
        var o = PaperOptions;
        _usedDescriptNames = LoadDescriptModules(Engine, _encoder!, _decoder!, _quantizer!, o.EncoderRates.Length, o.DecoderRates.Length, o.NumQuantizers,
            o.CodebookSize, o.CodebookDim, read);
    }

    /// <summary>Loads descript-audio-codec's layout into DAC's modules wherever they live (a codec, or a model that owns
    /// them); returns the names read.</summary>
    internal static HashSet<string> LoadDescriptModules(IEngine engine, DacEncoder<T> encoder, DacDecoder<T> decoder, FactorizedVectorQuantizerLayer<T> quantizer,
        int encoderBlocks, int decoderBlocks, int quantizers, int codebookSize, int codebookDim, Func<string, double[]?> read)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var used = new HashSet<string>(StringComparer.Ordinal);
        double[] Read(string name, int count)
        {
            var values = read(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            if (values.Length != count) throw new InvalidDataException($"'{name}' has {values.Length} values; this model expects {count}.");
            used.Add(name);
            return values;
        }
        void Load(string descript, LayerBase<T> layer)
        {
            switch (layer)
            {
                case NormedConv1DLayer<T> conv:
                    var shape = conv.TorchWeightShape;
                    conv.LoadTorchWeights(Read($"{descript}.weight_v", shape.Aggregate(1, (a, b) => a * b)), Read($"{descript}.weight_g", shape[0]),
                        Read($"{descript}.bias", conv.OutputChannels));
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
            var codebook = Read($"{prefix}.codebook.weight", codebookSize * codebookDim);
            var target = quantizer.Codebook(q);
            for (int i = 0; i < codebook.Length; i++) target[i] = ops.FromDouble(codebook[i]);
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
        var all = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in all)
        {
            string name = entry.Key.StartsWith("state_dict.", StringComparison.Ordinal) ? entry.Key.Substring("state_dict.".Length) : entry.Key;
            state[name] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        }
        LoadDescriptWeights(name => state.TryGetValue(name, out var v) ? v : null);
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
