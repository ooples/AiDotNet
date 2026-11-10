using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>The released TextEncoder architecture for the paper's auxiliary IPA prior:
/// learned phoneme embeddings, relative-attention Transformer and diagonal-Gaussian projection.
/// No speaker, style or language condition enters this branch.</summary>
internal sealed class OpenVoiceIpaPrior<T>
{
    private readonly IEngine _engine;
    private readonly int _vocabulary, _hidden, _latent, _filter, _heads, _kernel;
    private readonly TiedEmbeddingLayer<T> _embedding;
    private readonly RelativePositionTransformerBlock<T>[] _blocks;
    private readonly NormedConv1DLayer<T> _projection;
    private readonly List<LayerBase<T>> _layers = new();
    internal IReadOnlyList<LayerBase<T>> Layers => _layers;

    internal OpenVoiceIpaPrior(IEngine engine, int vocabulary, int hidden = 192, int latent = 192,
        int filter = 768, int heads = 2, int layers = 6, int kernel = 3, double dropout = .1)
    {
        if (vocabulary <= 0 || hidden <= 0 || latent <= 0 || filter <= 0 || heads <= 0 || hidden % heads != 0 || layers <= 0)
            throw new ArgumentException("Invalid OpenVoice IPA-prior dimensions.");
        _engine = engine; _vocabulary = vocabulary; _hidden = hidden; _latent = latent; _filter = filter; _heads = heads; _kernel = kernel;
        _embedding = new TiedEmbeddingLayer<T>(vocabulary, hidden); _layers.Add(_embedding);
        _blocks = new RelativePositionTransformerBlock<T>[layers];
        for (int i = 0; i < layers; i++)
        {
            _blocks[i] = new RelativePositionTransformerBlock<T>(hidden, heads, filter, kernel, dropout, 4, normalizationEpsilon: 1e-5);
            _layers.Add(_blocks[i]);
        }
        _projection = new NormedConv1DLayer<T>(hidden, 2 * latent, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None);
        _layers.Add(_projection);
    }

    internal (Tensor<T> Hidden, Tensor<T> Mean, Tensor<T> LogScale) Forward(Tensor<T> tokens, bool training = false,
        Func<string, int[], double[]>? dropoutMasks = null)
    {
        if (tokens.Rank != 1 || tokens.Length == 0) throw new ArgumentException("Expected one nonempty IPA-token sequence.", nameof(tokens));
        var numeric = MathHelper.GetNumericOperations<T>();
        foreach (var value in tokens.ToArray())
        {
            double id = numeric.ToDouble(value);
            if (double.IsNaN(id) || double.IsInfinity(id) || id != Math.Truncate(id) || id < 0 || id >= _vocabulary)
                throw new ArgumentException("IPA-token IDs must be integers in the configured vocabulary.", nameof(tokens));
        }
        var hidden = _engine.TensorMultiplyScalar(_embedding.Forward(tokens), numeric.FromDouble(Math.Sqrt(_hidden)));
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].SetTrainingMode(training); int index = i;
            hidden = dropoutMasks is null ? _blocks[i].Forward(hidden) : _blocks[i].ForwardWithDropoutMasks(hidden,
                (site, shape) => dropoutMasks("block." + index + "." + site, shape));
        }
        hidden = _engine.Reshape(_engine.TensorTranspose(hidden), new[] { 1, _hidden, tokens.Length });
        var stats = _projection.Forward(hidden);
        var mean = _engine.TensorSlice(stats, new[] { 0, 0, 0 }, new[] { 1, _latent, tokens.Length });
        var logs = _engine.TensorSlice(stats, new[] { 0, _latent, 0 }, new[] { 1, _latent, tokens.Length });
        return (hidden, mean, logs);
    }

    internal void LoadTorch(Func<string, int[], double[]> read) => LoadTorch(AiDotNet.ModelLoading.TorchTensorSource.From(read));

    /// <summary>Loads the released TextEncoder straight from <paramref name="source"/>.</summary>
    internal void LoadTorch(AiDotNet.ModelLoading.TorchTensorSource source)
    {
        _embedding.LoadTable(source, "emb.weight");
        for (int i = 0; i < _blocks.Length; i++)
        {
            var block = _blocks[i]; string attention = "encoder.attn_layers." + i;
            // The 1x1 convolutions are [out, in, 1]: small, read and transposed into the dense layers.
            void Dense(DenseLayer<T> layer, string name) => TorchParameters.Linear(_engine, layer, _hidden, _hidden,
                source.Read(attention + "." + name + ".weight", new[] { _hidden, _hidden, 1 }),
                source.Read(attention + "." + name + ".bias", new[] { _hidden }));
            Dense(block.QueryProjection, "conv_q"); Dense(block.KeyProjection, "conv_k");
            Dense(block.ValueProjection, "conv_v"); Dense(block.OutputProjection, "conv_o");
            void Relative(Tensor<T> target, string name)
            {
                source.CopyTo(attention + "." + name, new[] { 1, 9, _hidden / _heads }, target);
                _engine.InvalidatePersistentTensor(target);
            }
            Relative(block.RelativeKeys, "emb_rel_k"); Relative(block.RelativeValues, "emb_rel_v");
            void Norm(LayerNormalizationLayer<T> layer, string name) =>
                TorchParameters.LayerNorm(_engine, layer, source, name + ".gamma", name + ".beta");
            Norm(block.AttentionNormalization, "encoder.norm_layers_1." + i);
            Norm(block.FeedForwardNormalization, "encoder.norm_layers_2." + i);
            void Conv(Conv1DLayer<T> layer, string name, int input, int output) =>
                layer.LoadTorchWeights(source, name + ".weight", new[] { output, input, _kernel }, name + ".bias");
            Conv(block.FeedForwardInput, "encoder.ffn_layers." + i + ".conv_1", _hidden, _filter);
            Conv(block.FeedForwardOutput, "encoder.ffn_layers." + i + ".conv_2", _filter, _hidden);
        }
        _projection.LoadTorchWeights(source, "proj.weight", new[] { 2 * _latent, _hidden, 1 }, null, "proj.bias");
    }
}
