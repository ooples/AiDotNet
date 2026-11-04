using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.StyleEmotion;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>The released Kokoro-82M neural graph: shared ALBERT, duration/prosody predictors,
/// phoneme encoder and style-conditioned harmonic iSTFTNet. This model has no style diffusion.</summary>
internal sealed class KokoroCore<T>
{
    private readonly IEngine _engine;
    private readonly AlbertNetwork<T> _bert;
    private readonly DenseLayer<T> _bertProjection;
    private readonly StyleTtsTextEncoder<T> _text;
    private readonly StyleTtsProsodyPredictor<T> _prosody;
    private readonly StyleTts2Decoder<T> _decoder;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly Random _random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(2294);
    public IReadOnlyList<LayerBase<T>> Layers => _layers;
    internal int? SynthesisSeed { get; set; }

    internal KokoroCore(IEngine engine, bool originalFloat32Buffers = true)
    {
        _engine = engine;
        _bert = new AlbertNetwork<T>(engine, _layers, new AlbertConfiguration
        {
            VocabSize = 178, EmbeddingSize = 128, Hidden = 768, Layers = 12, Heads = 12,
            Intermediate = 2048, MaxPositions = 512, Dropout = 0
        });
        _bertProjection = StyleTtsOps.Own(_layers, StyleTtsOps.Linear<T>(512));
        _prosody = new StyleTtsProsodyPredictor<T>(engine, _layers, 128, 512, 3, .2, 50);
        _text = new StyleTtsTextEncoder<T>(engine, _layers, 512, 5, 3, 178);
        _decoder = new StyleTts2Decoder<T>(engine, _layers, StyleTts2DecoderType.IstftNet,
            512, 128, 1024, 64, new[] { 3, 7, 11 }, new[] { 10, 6 }, 512,
            new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } }, new[] { 20, 12 }, 20, 5, 24000);
        _decoder.Generator.Source.UseFloat32NoiseAmplitude = originalFloat32Buffers;
        var window = originalFloat32Buffers ? KokoroSignalBuffers.HannFloat32() : null;
        _decoder.Generator.SourceSpectrumTransform = new KokoroHarmonicStft<T>(window: window).Forward;
        if (window is not null) _decoder.Generator.LoadSynthesisWindow(window);
    }

    internal bool StochasticExcitation
    {
        get => _decoder.Generator.Source.Stochastic;
        set => _decoder.Generator.Source.Stochastic = value;
    }

    internal void SetExcitationDraws(double[] initialPhases, double[] gaussianNoise)
    {
        _decoder.Generator.Source.InitialPhases = initialPhases;
        _decoder.Generator.Source.GaussianNoise = gaussianNoise;
    }

    public (Tensor<T> Audio, int[] Durations, Tensor<T> Bert, Tensor<T> Projected,
        Tensor<T> DurationLogits, Tensor<T> Pitch, Tensor<T> Energy, Tensor<T> Text) Forward(
        IReadOnlyList<int> tokens, Tensor<T> referenceStyle, double speed = 1)
    {
        if (tokens.Count == 0 || tokens.Count > 512 || tokens.Any(token => token < 0 || token >= 178))
            throw new ArgumentException("Kokoro accepts 1–512 phoneme IDs in [0,178).", nameof(tokens));
        if (referenceStyle.Length != 256) throw new ArgumentException("Kokoro requires its 256-value voice style.", nameof(referenceStyle));
        if (double.IsNaN(speed) || double.IsInfinity(speed) || speed <= 0) throw new ArgumentOutOfRangeException(nameof(speed));
        int length = tokens.Count;
        var random = SynthesisSeed is { } seed ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed) : _random;
        var style = _engine.Reshape(referenceStyle, new[] { 1, 256 });
        var acoustic = _engine.TensorSlice(style, new[] { 0, 0 }, new[] { 1, 128 });
        var prosodic = _engine.TensorSlice(style, new[] { 0, 128 }, new[] { 1, 128 });
        var bert = _bert.Encode(tokens, false, random);
        var projected = _bertProjection.Forward(bert);
        var encoded = _prosody.EncodeText(_engine.Reshape(_engine.TensorTranspose(projected),
            new[] { 1, 512, length }), prosodic, false, random);
        var logits = _prosody.Durations(encoded, false, random);
        var probabilities = _engine.Sigmoid(logits);
        var operations = MathHelper.GetNumericOperations<T>();
        var durations = new int[length];
        for (int i = 0; i < length; i++)
        {
            double sum = 0;
            for (int bin = 0; bin < 50; bin++) sum += operations.ToDouble(probabilities[i, bin]);
            durations[i] = Math.Max(1, checked((int)Math.Round(sum / speed, MidpointRounding.ToEven)));
        }
        int frames = checked(durations.Sum());
        var alignment = new Tensor<T>(new[] { length, frames });
        int frame = 0;
        for (int token = 0; token < length; token++)
            for (int i = 0; i < durations[token]; i++) alignment[token, frame++] = operations.One;
        var aligned = _engine.Reshape(_engine.TensorMatMul(_engine.TensorTranspose(encoded), alignment),
            new[] { 1, 640, frames });
        var (pitch, energy) = _prosody.Prosody(aligned, prosodic, false, random);
        var ids = new Tensor<T>(new[] { length });
        for (int i = 0; i < length; i++) ids[i] = operations.FromDouble(tokens[i]);
        var text = _text.Forward(ids, false, random);
        var asr = _engine.Reshape(_engine.TensorMatMul(_engine.Reshape(text, new[] { 512, length }), alignment),
            new[] { 1, 512, frames });
        var audio = _decoder.Forward(asr, pitch, energy, acoustic, false, random);
        return (audio, durations, bert, projected, logits, pitch, energy, text);
    }

    public void LoadTorch(Func<string, int[], double[]> read)
    {
        _bert.LoadTorchWeights(read, "bert.");
        TorchParameters.Linear(_engine, _bertProjection, 768, 512,
            read("bert_encoder.weight", new[] { 512, 768 }), read("bert_encoder.bias", new[] { 512 }));
        _prosody.Load((name, shape) => read(name, shape), "predictor");
        _text.Load((name, shape) => read(name, shape), "text_encoder");
        _decoder.Load((name, shape) => read(name, shape), "decoder");
    }
}
