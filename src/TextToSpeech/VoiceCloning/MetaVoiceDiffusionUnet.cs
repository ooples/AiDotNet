using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>AudioCraft 1.2's conditioned multiband diffusion denoiser, as used by MetaVoice.
/// The released preset has four stages, two residual blocks, learned time embeddings,
/// nearest-neighbor codec conditioning and no transformer or recurrent bottleneck.</summary>
internal sealed class MetaVoiceDiffusionUnet<T>
{
    private sealed record Residual(GroupNormalizationLayer<T> FirstNorm, NormedConv1DLayer<T> First,
        GroupNormalizationLayer<T> SecondNorm, NormedConv1DLayer<T> Second);
    private sealed record Stage(NormedConv1DLayer<T> Convolution, GroupNormalizationLayer<T> Norm,
        Residual[] Residuals);
    private readonly IEngine _engine;
    private readonly int _stride, _codecDimension, _steps;
    private readonly Stage[] _encoders, _decoders;
    private readonly TiedEmbeddingLayer<T>[] _time;
    private readonly NormedConv1DLayer<T> _codec;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly Dictionary<string, (LayerBase<T> Layer, int[] Shape)> _weights = new();
    public IReadOnlyList<LayerBase<T>> Layers => _layers;
    private readonly int[] _configuration;
    private readonly IReadOnlyList<LayerBase<T>>? _restoredLayers;
    private int _restoredIndex;
    internal int[] Configuration => (int[])_configuration.Clone();

    public MetaVoiceDiffusionUnet(IEngine engine, int hidden = 48, int depth = 4, int growth = 4,
        int codecDimension = 128, int steps = 1000, int groups = 4, int residualBlocks = 2,
        int kernel = 8, int stride = 4, int maximumChannels = 10000,
        IReadOnlyList<LayerBase<T>>? restoredLayers = null)
    {
        if (hidden <= 0 || depth <= 0 || growth <= 0 || codecDimension <= 0 || steps <= 0 || groups <= 0 ||
            residualBlocks < 0 || kernel < stride || stride <= 0 || (kernel - stride) % 2 != 0 || maximumChannels <= 0)
            throw new ArgumentException("Invalid AudioCraft diffusion denoiser dimensions.");
        _engine = engine; _stride = stride; _codecDimension = codecDimension; _steps = steps;
        _configuration = new[] { hidden, depth, growth, codecDimension, steps, groups, residualBlocks, kernel, stride, maximumChannels };
        _restoredLayers = restoredLayers;
        _encoders = new Stage[depth]; _decoders = new Stage[depth]; _time = new TiedEmbeddingLayer<T>[depth];
        int input = 1, width = hidden;
        for (int stage = 0; stage < depth; stage++)
        {
            if (width % groups != 0) throw new ArgumentException("Group count must divide every stage width.");
            string encoder = "encoders." + stage, decoder = "decoders." + (depth - 1 - stage);
            _encoders[stage] = new Stage(Conv(encoder + ".conv", input, width, kernel, stride, (kernel - stride) / 2, false, false),
                Norm(encoder + ".norm", width, groups), Blocks(encoder, width, groups, residualBlocks));
            _decoders[depth - 1 - stage] = new Stage(
                Conv(decoder + ".convtr", width, input, kernel, stride, (kernel - stride) / 2, true, false),
                Norm(decoder + ".norm", width, groups), Blocks(decoder, width, groups, residualBlocks));
            var table = ReuseOrCreate(() => new TiedEmbeddingLayer<T>(steps, width), (long)steps * width);
            _time[stage] = table; _layers.Add(table);
            _weights.Add(stage == 0 ? "embedding" : "embeddings." + (stage - 1), (table, new[] { steps, width }));
            input = width; width = Math.Min(checked(width * growth), maximumChannels);
        }
        _codec = Conv("conv_codec", codecDimension, input, 1, 1, 0, false, true);
        if (_restoredLayers is not null && _restoredIndex != _restoredLayers.Count)
            throw new InvalidDataException("The restored multiband graph has unconsumed layers.");
    }

    private TLayer ReuseOrCreate<TLayer>(Func<TLayer> create, long parameters) where TLayer : LayerBase<T>
    {
        if (_restoredLayers is null) return create();
        if (_restoredIndex >= _restoredLayers.Count)
            throw new InvalidDataException("The restored multiband graph is missing layer " + _restoredIndex + ".");
        var restored = _restoredLayers[_restoredIndex++];
        // LayerBase's flat vector also contains persistent buffers (for example
        // spectral-normalization singular vectors). Compare neural weights only.
        long trainableCount = restored.GetTrainableParametersWithoutMaterialization().Sum(tensor => (long)tensor.Length);
        if (restored is not TLayer layer || trainableCount != parameters)
            throw new InvalidDataException($"Restored multiband layer {_restoredIndex - 1}: expected {typeof(TLayer).Name} " +
                $"with {parameters} trainable values, found {restored.GetType().Name} with {trainableCount}.");
        return layer;
    }

    private NormedConv1DLayer<T> Conv(string name, int input, int output, int kernel, int stride, int padding,
        bool transposed, bool bias, int dilation = 1)
    {
        var layer = ReuseOrCreate(() => new NormedConv1DLayer<T>(input, output, kernel, stride, dilation, 1, padding, transposed,
            ConvolutionNormalization.None, bias), (long)input * output * kernel + (bias ? output : 0));
        _layers.Add(layer);
        _weights.Add(name, (layer, transposed ? new[] { input, output, kernel } : new[] { output, input, kernel }));
        return layer;
    }

    private GroupNormalizationLayer<T> Norm(string name, int channels, int groups)
    {
        var layer = ReuseOrCreate(() => new GroupNormalizationLayer<T>(groups, channels, 1e-5), 2L * channels);
        _layers.Add(layer); _weights.Add(name, (layer, new[] { channels })); return layer;
    }

    private Residual[] Blocks(string name, int channels, int groups, int count)
    {
        var result = new Residual[count];
        for (int i = 0; i < count; i++)
        {
            string prefix = name + ".res_blocks." + i;
            int dilation = 1 << i;
            result[i] = new Residual(Norm(prefix + ".norm1", channels, groups),
                Conv(prefix + ".conv1", channels, channels, 3, 1, dilation, false, true, dilation),
                Norm(prefix + ".norm2", channels, groups),
                Conv(prefix + ".conv2", channels, channels, 3, 1, dilation, false, true, dilation));
        }
        return result;
    }

    private Tensor<T> Normalize(GroupNormalizationLayer<T> layer, Tensor<T> input)
    {
        // The shared GroupNorm interprets rank three as unbatched CHW; preserve BCT explicitly.
        var output = layer.Forward(_engine.Reshape(input, new[] { input.Shape[0], input.Shape[1], 1, input.Shape[2] }));
        return _engine.Reshape(output, input.Shape.ToArray());
    }

    private Tensor<T> Residuals(Residual[] blocks, Tensor<T> input)
    {
        foreach (var block in blocks)
        {
            var branch = block.First.Forward(_engine.ReLU(Normalize(block.FirstNorm, input)));
            branch = block.Second.Forward(_engine.ReLU(Normalize(block.SecondNorm, branch)));
            input = _engine.TensorAdd(input, branch);
        }
        return input;
    }

    private Tensor<T> Crop(Tensor<T> input, int frames) => _engine.TensorSlice(input,
        new[] { 0, 0, 0 }, new[] { input.Shape[0], input.Shape[1], frames });

    public Tensor<T> Forward(Tensor<T> waveform, int[] steps, Tensor<T> condition)
    {
        if (waveform.Rank != 3 || waveform.Shape[1] != 1 || waveform.Shape[2] <= 0 ||
            condition.Rank != 3 || condition.Shape[0] != waveform.Shape[0] || condition.Shape[1] != _codecDimension ||
            condition.Shape[2] <= 0 || steps.Length != waveform.Shape[0] || steps.Any(step => step < 0 || step >= _steps))
            throw new ArgumentException("Expected mono BCT audio, matching codec features and one valid timestep per batch item.");
        var skips = new Tensor<T>[_encoders.Length];
        var z = waveform;
        for (int i = 0; i < _encoders.Length; i++)
        {
            int pad = (_stride - z.Shape[2] % _stride) % _stride;
            if (pad > 0) z = _engine.TensorConcatenate(new[] { z, new Tensor<T>(new[] { z.Shape[0], z.Shape[1], pad }) }, 2);
            var stage = _encoders[i];
            z = Residuals(stage.Residuals, _engine.ReLU(Normalize(stage.Norm, stage.Convolution.Forward(z))));
            var ids = new Tensor<T>(new[] { steps.Length });
            for (int b = 0; b < steps.Length; b++) ids[b] = MathHelper.GetNumericOperations<T>().FromDouble(steps[b]);
            var embedding = _time[i].Forward(ids);
            z = _engine.TensorAdd(z, _engine.TensorTile(_engine.Reshape(embedding,
                new[] { z.Shape[0], z.Shape[1], 1 }), new[] { 1, 1, z.Shape[2] }));
            skips[i] = z;
        }
        var projected = _codec.Forward(condition);
        if (projected.Shape[2] > 2 * z.Shape[2]) throw new ArgumentException("Codec conditioning downsamples by more than two.", nameof(condition));
        var frames = new Tensor<T>[z.Shape[2]];
        for (int i = 0; i < frames.Length; i++)
        {
            int source = (int)((long)i * projected.Shape[2] / frames.Length);
            frames[i] = _engine.TensorSlice(projected, new[] { 0, 0, source }, new[] { projected.Shape[0], projected.Shape[1], 1 });
        }
        // Author uses z += condition_emb: skips[-1] aliases z and receives the condition too.
        skips[skips.Length - 1] = _engine.TensorAdd(z, _engine.TensorConcatenate(frames, 2));
        z = new Tensor<T>(z.Shape.ToArray());
        for (int i = 0; i < _decoders.Length; i++)
        {
            var skip = skips[skips.Length - 1 - i];
            z = _engine.TensorAdd(Crop(z, skip.Shape[2]), skip);
            var stage = _decoders[i];
            z = stage.Convolution.Forward(_engine.ReLU(Normalize(stage.Norm, Residuals(stage.Residuals, z))));
        }
        return Crop(z, waveform.Shape[2]);
    }

    public void LoadTorch(Func<string, int[], double[]> read)
    {
        foreach (var entry in _weights)
        {
            var (layer, shape) = entry.Value;
            var weight = read(entry.Key + ".weight", shape);
            if (layer is GroupNormalizationLayer<T> norm)
                norm.LoadAffine(weight, read(entry.Key + ".bias", shape));
            else if (layer is TiedEmbeddingLayer<T> embedding)
                embedding.LoadTable(weight);
            else if (layer is NormedConv1DLayer<T> convolution)
            {
                int channels = entry.Key.EndsWith(".convtr", StringComparison.Ordinal) ? shape[1] : shape[0];
                bool bias = entry.Key == "conv_codec" || entry.Key.Contains(".res_blocks.");
                convolution.LoadTorchWeights(weight, null, bias ? read(entry.Key + ".bias", new[] { channels }) : new double[channels]);
            }
        }
    }
}
