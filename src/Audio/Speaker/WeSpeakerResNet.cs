using AiDotNet.Audio.Features;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// WeSpeaker's ResNet speaker embedding network (Wang et al. 2023; reference wenet-e2e/wespeaker
/// <c>wespeaker/models/resnet.py</c>) with its front end: 80-bin Kaldi fbank of the 16 kHz waveform at int16 scale
/// (Hamming window, 25/10 ms), mean-normalized over time; a 3×3 stem and four stages of basic residual blocks
/// (channels m, 2m, 4m, 8m; stride 2 from the second), temporal statistics pooling (mean and standard deviation) and a
/// linear embedding. ResNet34 is blocks {3, 4, 6, 3} with m = 32 and a 256-dimensional embedding.
/// </summary>
/// <remarks>For frozen use (StyleTTS-ZS's speaker-embedding feature matching, §III-A): batch norms read their running
/// statistics. Its layers are added to the owner's list.</remarks>
internal sealed class WeSpeakerResNet<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly KaldiFbank<T> _fbank;

    public WeSpeakerResNet(IEngine engine, List<LayerBase<T>> layers, int[]? blocks = null, int channels = 32, int melBins = 80,
        int embedding = 256, double waveScale = 32768.0)
    {
        _engine = engine;
        blocks ??= new[] { 3, 4, 6, 3 };
        if (blocks.Length != 4) throw new ArgumentException("A WeSpeaker ResNet has four stages.", nameof(blocks));
        if (melBins % 8 != 0) throw new ArgumentException("The stages halve the mel axis three times; use a multiple of 8.", nameof(melBins));
        WaveScale = waveScale;
        MelBins = melBins;
        _fbank = new KaldiFbank<T>(engine, 16000, melBins, WindowType.Hamming);
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        NormedConv2DLayer<T> Conv(int input, int output, int kernel, int stride) =>
            Own(new NormedConv2DLayer<T>(input, output, kernel, kernel, stride, stride, kernel / 2, kernel / 2, 1, ConvolutionNormalization.None, useBias: false));
        BatchNormalizationLayer<T> Norm(int c) => Own(new BatchNormalizationLayer<T>(c, 1e-5, 0.9));
        StemConv = Conv(1, channels, 3, 1);
        StemNorm = Norm(channels);
        int input = channels;
        for (int stage = 0; stage < 4; stage++)
        {
            int planes = channels << stage;
            var list = new List<Block>();
            for (int b = 0; b < blocks[stage]; b++)
            {
                int stride = b == 0 && stage > 0 ? 2 : 1;
                bool project = stride != 1 || input != planes;
                list.Add(new Block(Conv(input, planes, 3, stride), Norm(planes), Conv(planes, planes, 3, 1), Norm(planes),
                    project ? Conv(input, planes, 1, stride) : null, project ? Norm(planes) : null));
                input = planes;
            }
            Stages.Add(list);
        }
        StatisticsDim = melBins / 8 * channels * 8 * 2;
        Embedding = Own(new DenseLayer<T>(embedding, (IActivationFunction<T>)new IdentityActivation<T>()));
        Blocks = blocks;
        EmbeddingDim = embedding;
    }

    /// <summary>A basic residual block: two 3×3 convolutions with batch norms, and a projected shortcut when the shape
    /// changes.</summary>
    internal sealed record Block(NormedConv2DLayer<T> Conv1, BatchNormalizationLayer<T> Norm1, NormedConv2DLayer<T> Conv2,
        BatchNormalizationLayer<T> Norm2, NormedConv2DLayer<T>? Shortcut, BatchNormalizationLayer<T>? ShortcutNorm);

    public double WaveScale { get; }
    public int MelBins { get; }
    public int[] Blocks { get; }
    public int StatisticsDim { get; }
    public int EmbeddingDim { get; }
    public NormedConv2DLayer<T> StemConv { get; }
    public BatchNormalizationLayer<T> StemNorm { get; }
    public List<List<Block>> Stages { get; } = new();
    public DenseLayer<T> Embedding { get; }

    /// <summary>The mean-normalized fbank <c>[frames, mels]</c> of a 16 kHz waveform in [−1, 1].</summary>
    public Tensor<T> Features(Tensor<T> audio)
    {
        var fbank = _fbank.Forward(_engine.TensorMultiplyScalar(_engine.Reshape(audio, new[] { audio.Length }), NumOps.FromDouble(WaveScale)));
        int frames = fbank.Shape[0];
        var mean = _engine.TensorTile(_engine.ReduceMean(fbank, new[] { 0 }, keepDims: true), new[] { frames, 1 });
        return _engine.TensorSubtract(fbank, mean);
    }

    /// <summary>The feature maps <c>[1, C, mels', frames']</c> of the stem and of each stage, and the embedding
    /// <c>[1, embedding]</c>, of a 16 kHz waveform.</summary>
    public (IReadOnlyList<Tensor<T>> Maps, Tensor<T> Embedding) Forward(Tensor<T> audio)
    {
        var features = Features(audio);
        int frames = features.Shape[0];
        // (B, T, F) → (B, 1, F, T).
        var x = _engine.Reshape(_engine.TensorTranspose(features), new[] { 1, 1, MelBins, frames });
        x = _engine.ReLU(StemNorm.Forward(StemConv.Forward(x)));
        var maps = new List<Tensor<T>> { x };
        foreach (var stage in Stages)
        {
            foreach (var b in stage)
            {
                var y = _engine.ReLU(b.Norm1.Forward(b.Conv1.Forward(x)));
                y = b.Norm2.Forward(b.Conv2.Forward(y));
                var shortcut = b.Shortcut is null ? x : b.ShortcutNorm!.Forward(b.Shortcut.Forward(x));
                x = _engine.ReLU(_engine.TensorAdd(y, shortcut));
            }
            maps.Add(x);
        }
        // TSTP: per channel and frequency, the mean and sqrt(var + 1e-7) over time (unbiased variance).
        int c = x.Shape[1], f = x.Shape[2], t = x.Shape[3];
        var rows = _engine.Reshape(x, new[] { c * f, t });
        var mean = _engine.ReduceMean(rows, new[] { 1 }, keepDims: true);
        var centered = _engine.TensorSubtract(rows, _engine.TensorTile(mean, new[] { 1, t }));
        var variance = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(centered, centered), new[] { 1 }, keepDims: true),
            NumOps.FromDouble(1.0 / (t - 1)));
        var std = _engine.TensorSqrt(_engine.TensorAddScalar(variance, NumOps.FromDouble(1e-7)));
        var stats = _engine.Reshape(_engine.TensorConcatenate(new[] { _engine.TensorTranspose(mean), _engine.TensorTranspose(std) }, 1),
            new[] { 1, 2 * c * f });
        return (maps, Embedding.Forward(stats));
    }

    /// <summary>Loads a WeSpeaker ResNet state dict (<c>avg_model.pt</c>'s names, after an optional prefix).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        void Conv(NormedConv2DLayer<T> conv, string name) => conv.LoadTorchWeights(read(name + ".weight", conv.TorchWeightShape), null);
        void Norm(BatchNormalizationLayer<T> norm, string name, int width) => TorchParameters.BatchNorm(norm, read(name + ".weight", new[] { width }),
            read(name + ".bias", new[] { width }), read(name + ".running_mean", new[] { width }), read(name + ".running_var", new[] { width }));
        int width = StemConv.TorchWeightShape[0];
        Conv(StemConv, "conv1");
        Norm(StemNorm, "bn1", width);
        for (int s = 0; s < Stages.Count; s++)
            for (int b = 0; b < Stages[s].Count; b++)
            {
                var block = Stages[s][b];
                string p = $"layer{s + 1}.{b}";
                int planes = block.Conv1.TorchWeightShape[0];
                Conv(block.Conv1, p + ".conv1");
                Norm(block.Norm1, p + ".bn1", planes);
                Conv(block.Conv2, p + ".conv2");
                Norm(block.Norm2, p + ".bn2", planes);
                if (block.Shortcut is not null)
                {
                    Conv(block.Shortcut, p + ".shortcut.0");
                    Norm(block.ShortcutNorm!, p + ".shortcut.1", planes);
                }
            }
        int outputs = EmbeddingDim;
        TorchParameters.Linear(_engine, Embedding, StatisticsDim, outputs, read("seg_1.weight", new[] { outputs, StatisticsDim }),
            read("seg_1.bias", new[] { outputs }));
    }
}
