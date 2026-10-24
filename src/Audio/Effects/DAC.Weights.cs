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
}
