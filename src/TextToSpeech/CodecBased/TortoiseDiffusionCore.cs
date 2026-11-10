using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.VoiceCloning;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The original DiffusionTts residual block, with GroupNorm and timestep scale/shift.</summary>
internal sealed class TortoiseDiffusionResidual<T>
{
    private readonly IEngine _engine;
    private readonly int _dim;
    private readonly GroupNormalizationLayer<T> _inputNorm, _outputNorm;
    private readonly Conv1DLayer<T> _input, _output;
    private readonly DenseLayer<T> _time;

    internal TortoiseDiffusionResidual(IEngine engine, List<LayerBase<T>> owner, int dimension)
    {
        _engine = engine; _dim = dimension;
        _inputNorm = TortoiseDiffusionCore<T>.Norm(owner, dimension);
        _outputNorm = TortoiseDiffusionCore<T>.Norm(owner, dimension);
        _input = TortoiseDiffusionCore<T>.Conv(owner, dimension, dimension, 1);
        _output = TortoiseDiffusionCore<T>.Conv(owner, dimension, dimension, 3);
        _time = TortoiseOps<T>.Own(owner, new DenseLayer<T>(2 * dimension,
            new IdentityActivation<T>() as IActivationFunction<T>, biasMode: BiasMode.Always));
    }

    internal Tensor<T> Forward(Tensor<T> input, Tensor<T> time)
    {
        int frames = input.Shape[2];
        var x = _input.Forward(_engine.Swish(TortoiseDiffusionCore<T>.Normalize(_engine, _inputNorm, input)));
        var affine = _time.Forward(_engine.Swish(time));
        Tensor<T> Part(int part) => _engine.TensorTile(_engine.Reshape(_engine.TensorNarrow(affine, 1,
            part * _dim, _dim), new[] { 1, _dim, 1 }), new[] { 1, 1, frames });
        x = _engine.TensorAdd(_engine.TensorMultiply(TortoiseDiffusionCore<T>.Normalize(_engine, _outputNorm, x),
            _engine.TensorAddScalar(Part(0), TortoiseOps<T>.NumOps.One)), Part(1));
        return _engine.TensorAdd(input, _output.Forward(_engine.Swish(x)));
    }

    internal void Load(TorchTensorSource source, string prefix)
    {
        _inputNorm.LoadAffine(source, prefix + "in_layers.0.weight", prefix + "in_layers.0.bias");
        _outputNorm.LoadAffine(source, prefix + "out_layers.0.weight", prefix + "out_layers.0.bias");
        XttsOps<T>.LoadConv(source, _input, prefix + "in_layers.2", _dim, _dim, 1);
        XttsOps<T>.LoadConv(source, _output, prefix + "out_layers.3", _dim, _dim, 3);
        TorchParameters.Linear(_engine, _time, _dim, 2 * _dim, source, prefix + "emb_layers.1.weight", prefix + "emb_layers.1.bias");
    }
}

/// <summary>Named-weight-compatible TorToiSe DiffusionTts, including both epsilon and learned variance channels.</summary>
internal sealed class TortoiseDiffusionCore<T>
{
    private readonly IEngine _engine;
    private readonly TortoiseTTSOptions _options;
    private readonly Conv1DLayer<T> _input, _latentInput, _context1, _context2, _integrating, _melHead, _output;
    private readonly DenseLayer<T> _time1, _time2;
    private readonly TiedEmbeddingLayer<T> _codeEmbedding, _unconditioned;
    private readonly GroupNormalizationLayer<T> _codeNorm, _outputNorm;
    private readonly XttsConditioningAttention<T>[] _codeAttention, _latentAttention, _contextAttention;
    private readonly (TortoiseDiffusionResidual<T> Residual, XttsConditioningAttention<T> Attention)[] _integrators, _layers;
    private readonly TortoiseDiffusionResidual<T>[] _finalResiduals;

    internal static Conv1DLayer<T> Conv(List<LayerBase<T>> owner, int input, int output, int kernel, int stride = 1) =>
        TortoiseOps<T>.Own(owner, new Conv1DLayer<T>(input, output, kernel, dilation: 1, stride: stride, padding: kernel / 2));

    internal static GroupNormalizationLayer<T> Norm(List<LayerBase<T>> owner, int dimension)
    {
        int groups = dimension <= 16 ? 8 : dimension <= 64 ? 16 : 32;
        while (dimension % groups != 0) groups /= 2;
        return TortoiseOps<T>.Own(owner, new GroupNormalizationLayer<T>(groups, dimension));
    }

    internal static Tensor<T> Normalize(IEngine engine, GroupNormalizationLayer<T> norm, Tensor<T> input) =>
        engine.Reshape(norm.Forward(engine.Reshape(input, new[] { input.Shape[1], 1, input.Shape[2] })), input.Shape.ToArray());

    internal TortoiseDiffusionCore(IEngine engine, List<LayerBase<T>> owner, TortoiseTTSOptions options)
    {
        _engine = engine; _options = options; int d = options.DiffusionDim;
        XttsConditioningAttention<T> Attention(int dimension) => new(engine, owner, dimension,
            options.DiffusionHeads, normalizeResidual: false, relativePosition: true);
        XttsConditioningAttention<T>[] Stack(int dimension, int count) => Enumerable.Range(0, count).Select(_ => Attention(dimension)).ToArray();
        (TortoiseDiffusionResidual<T>, XttsConditioningAttention<T>)[] Blocks(int count) => Enumerable.Range(0, count)
            .Select(_ => (new TortoiseDiffusionResidual<T>(engine, owner, d), Attention(d))).ToArray();
        _input = Conv(owner, options.MelChannels, d, 3);
        _latentInput = Conv(owner, options.LLMDim, d, 3);
        _codeEmbedding = TortoiseOps<T>.Own(owner, new TiedEmbeddingLayer<T>(options.CodebookSize + 1, d));
        _codeAttention = Stack(d, 3); _latentAttention = Stack(d, 4);
        _context1 = Conv(owner, options.MelChannels, d, 3, 2);
        _context2 = Conv(owner, d, 2 * d, 3, 2);
        _contextAttention = Stack(2 * d, 5);
        _codeNorm = Norm(owner, d); _outputNorm = Norm(owner, d);
        _unconditioned = TortoiseOps<T>.Own(owner, new TiedEmbeddingLayer<T>(1, d));
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _time1 = TortoiseOps<T>.Own(owner, new DenseLayer<T>(d, identity, biasMode: BiasMode.Always));
        _time2 = TortoiseOps<T>.Own(owner, new DenseLayer<T>(d, identity, biasMode: BiasMode.Always));
        _integrators = Blocks(3); _layers = Blocks(options.DiffusionLayers);
        _finalResiduals = Enumerable.Range(0, 3).Select(_ => new TortoiseDiffusionResidual<T>(engine, owner, d)).ToArray();
        _integrating = Conv(owner, 2 * d, d, 1); _melHead = Conv(owner, d, options.MelChannels, 3);
        _output = Conv(owner, d, 2 * options.MelChannels, 3);
    }

    private Tensor<T> Channels(Tensor<T> rows) => _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], rows.Shape[0] });

    internal Tensor<T> Conditioning(Tensor<T> melRows)
    {
        var x = _context2.Forward(_context1.Forward(Channels(melRows)));
        foreach (var block in _contextAttention) x = block.Forward(x);
        return _engine.Reshape(_engine.ReduceMean(x, new[] { 2 }, true), new[] { 1, 2 * _options.DiffusionDim });
    }

    internal Tensor<T> Aligned(Tensor<T> latents, Tensor<T> condition, int frames, IReadOnlyList<int>? codes = null)
    {
        Tensor<T> x;
        if (codes is null)
        {
            x = _latentInput.Forward(Channels(latents));
            foreach (var block in _latentAttention) x = block.Forward(x);
        }
        else
        {
            x = Channels(_codeEmbedding.Forward(TortoiseOps<T>.Ids(codes)));
            foreach (var block in _codeAttention) x = block.Forward(x);
        }
        int d = _options.DiffusionDim, length = x.Shape[2];
        Tensor<T> Part(int part) => _engine.TensorTile(_engine.Reshape(_engine.TensorNarrow(condition, 1,
            part * d, d), new[] { 1, d, 1 }), new[] { 1, 1, length });
        x = _engine.TensorAdd(_engine.TensorMultiply(Normalize(_engine, _codeNorm, x),
            _engine.TensorAddScalar(Part(0), TortoiseOps<T>.NumOps.One)), Part(1));
        var indices = new int[frames];
        for (int i = 0; i < frames; i++) indices[i] = Math.Min(length - 1, i * length / frames);
        return _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(x, new[] { d, length }),
            new Tensor<int>(new[] { frames }, new Vector<int>(indices)), 1), new[] { 1, d, frames });
    }

    internal Tensor<T> Forward(Tensor<T> noisy, double timestep, Tensor<T> latents, Tensor<T>? speakerMel,
        bool conditioningFree = false, IReadOnlyList<int>? codes = null, Tensor<T>? precomputedAligned = null)
    {
        int d = _options.DiffusionDim, frames = noisy.Shape[2];
        var embedding = new Tensor<T>(new[] { 1, d });
        for (int i = 0; i < d / 2; i++)
        {
            double angle = timestep * Math.Exp(-Math.Log(10000) * i / Math.Truncate(d / 2.0));
            embedding[0, i] = TortoiseOps<T>.NumOps.FromDouble(Math.Cos(angle));
            embedding[0, i + d / 2] = TortoiseOps<T>.NumOps.FromDouble(Math.Sin(angle));
        }
        var time = _time2.Forward(_engine.Swish(_time1.Forward(embedding)));
        var aligned = conditioningFree
            ? _engine.TensorTile(_engine.Reshape(_unconditioned.Forward(TortoiseOps<T>.Ids(new[] { 0 })), new[] { 1, d, 1 }), new[] { 1, 1, frames })
            : precomputedAligned ?? Aligned(latents, Conditioning(speakerMel ?? throw new ArgumentNullException(nameof(speakerMel))), frames, codes);
        foreach (var block in _integrators) aligned = block.Attention.Forward(block.Residual.Forward(aligned, time));
        var x = _integrating.Forward(_engine.TensorConcatenate(new[] { _input.Forward(noisy), aligned }, 1));
        foreach (var block in _layers) x = block.Attention.Forward(block.Residual.Forward(x, time));
        foreach (var block in _finalResiduals) x = block.Forward(x, time);
        return _output.Forward(_engine.Swish(Normalize(_engine, _outputNorm, x)));
    }

    internal void LoadTorchWeights(Func<string, int[], double[]> read) => LoadTorchWeights(TorchTensorSource.From(read));

    internal void LoadTorchWeights(TorchTensorSource source)
    {
        int d = _options.DiffusionDim, mel = _options.MelChannels;
        XttsOps<T>.LoadConv(source, _input, "inp_block", mel, d, 3);
        XttsOps<T>.LoadConv(source, _latentInput, "latent_conditioner.0", _options.LLMDim, d, 3);
        XttsOps<T>.LoadConv(source, _context1, "contextual_embedder.0", mel, d, 3);
        XttsOps<T>.LoadConv(source, _context2, "contextual_embedder.1", d, 2 * d, 3);
        _codeEmbedding.LoadTable(source, "code_embedding.weight");
        _unconditioned.LoadTable(source, "unconditioned_embedding", new[] { 1, d, 1 });
        for (int i = 0; i < _codeAttention.Length; i++) _codeAttention[i].LoadTorchWeights(source, $"code_converter.{i}.");
        for (int i = 0; i < _latentAttention.Length; i++) _latentAttention[i].LoadTorchWeights(source, $"latent_conditioner.{i + 1}.");
        for (int i = 0; i < _contextAttention.Length; i++) _contextAttention[i].LoadTorchWeights(source, $"contextual_embedder.{i + 2}.");
        _codeNorm.LoadAffine(source, "code_norm.weight", "code_norm.bias");
        _outputNorm.LoadAffine(source, "out.0.weight", "out.0.bias");
        TorchParameters.Linear(_engine, _time1, d, d, source, "time_embed.0.weight", "time_embed.0.bias");
        TorchParameters.Linear(_engine, _time2, d, d, source, "time_embed.2.weight", "time_embed.2.bias");
        for (int i = 0; i < _integrators.Length; i++)
        {
            _integrators[i].Residual.Load(source, $"conditioning_timestep_integrator.{i}.resblk.");
            _integrators[i].Attention.LoadTorchWeights(source, $"conditioning_timestep_integrator.{i}.attn.");
        }
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].Residual.Load(source, $"layers.{i}.resblk.");
            _layers[i].Attention.LoadTorchWeights(source, $"layers.{i}.attn.");
        }
        for (int i = 0; i < _finalResiduals.Length; i++) _finalResiduals[i].Load(source, $"layers.{i + _layers.Length}.");
        XttsOps<T>.LoadConv(source, _integrating, "integrating_conv", 2 * d, d, 1);
        XttsOps<T>.LoadConv(source, _melHead, "mel_head", d, mel, 3);
        XttsOps<T>.LoadConv(source, _output, "out.2", d, 2 * mel, 3);
    }
}
