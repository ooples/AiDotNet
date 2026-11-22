using AiDotNet.Audio.Classification;
using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// The "panns-cnn14" row: the shipped <see cref="PANNs{T}"/> CNN14 (Kong et al. 2020) at the paper's AudioSet setting:
/// 10-second clips at 32 kHz as [1001 frames, 64 mels] log-mel features (hop 320), bn0 over the mel bins, six
/// conv blocks of two 3x3 conv + BN + ReLU (64 -> 2048 channels, 2x2 average pooling in blocks 1-5, dropout 0.2),
/// mean over frequency, max + mean over time, a 2048 ReLU embedding and 527 logits, trained with binary
/// cross-entropy on logits against multi-hot labels (two tags per clip). Paper batch 32. The features are computed
/// outside the timed region on both sides (the model takes log-mel features). benchmark.py's <c>PannsCnn14</c> is
/// Kong's released Cnn14 minus its spectrogram front end, with the convolution biases AiDotNet's layers carry (Kong's
/// convolutions have none), so the trainable parameter counts are equal.
/// </summary>
internal sealed class AiDotNetPannsCnn14Model : IBenchmarkModel
{
    internal const int Frames = 1001;
    internal const int Mels = 64;
    internal const int Classes = 527;
    internal const int TagsPerClip = 2;

    private readonly PANNs<float> _model;
    private readonly Random _random;
    private Tensor<float> _input = Tensor<float>.Empty();
    private Tensor<float> _label = Tensor<float>.Empty();
    private bool _trained;

    public AiDotNetPannsCnn14Model(int seed)
    {
        _random = new Random(seed);
        _model = new PANNs<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiLabelClassification,
                inputHeight: Frames, inputWidth: Mels, inputDepth: 1, outputSize: Classes),
            new PANNsOptions { NumMels = Mels, Seed = seed },
            // The optimizer every row uses (torch.optim.AdamW defaults, lr 1e-3) with the same global-norm clip.
            optimizer: new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 1e-3, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
                    EnableGradientClipping = true, MaxGradientNorm = 1.0,
                }));

        LoadSyntheticBatch(1);
        Forward();
        ParameterCount = _model.TrainableParameterCount;
    }

    public long ParameterCount { get; }

    public (Tensor<float> Input, Tensor<float> Label) CurrentBatch
    {
        get => (_input, _label);
        set => (_input, _label) = value;
    }

    /// <summary>Standard-normal log-mel features [B, 1, 1001, 64] and multi-hot labels with two random tags per clip.</summary>
    public void LoadSyntheticBatch(int batchSize)
    {
        var features = new float[batchSize * Frames * Mels];
        for (var i = 0; i < features.Length; i++)
        {
            double u1 = 1.0 - _random.NextDouble(), u2 = _random.NextDouble();
            features[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        _input = new Tensor<float>(features, [batchSize, 1, Frames, Mels]);
        var labels = new float[batchSize * Classes];
        for (var b = 0; b < batchSize; b++)
            for (var t = 0; t < TagsPerClip; t++)
                labels[b * Classes + _random.Next(Classes)] = 1f;
        _label = new Tensor<float>(labels, [batchSize, Classes]);
    }

    public void Forward() => _ = _model.Predict(_input);

    public void Backward()
    {
        // PANNs' Train is the whole step: forward, BCE-with-logits, backward and the update.
        _model.Train(_input, _label);
        _trained = true;
    }

    public void Step()
    {
        // The optimizer step ran inside Backward's Train call.
    }

    public double? LastLoss => _trained ? _model.GetLastLoss() : null;
}

/// <summary>
/// The "wav2vec2-base" row: the shipped <see cref="Wav2Vec2Model{T}"/> at the BASE size (Baevski et al. 2020: a 7-layer
/// convolutional feature encoder with group norm, 768 wide, 12 post-norm Transformer layers, 12 heads, FFN 3072, a
/// grouped convolutional positional embedding) with a CTC head over a 34-symbol character vocabulary (blank 0),
/// fine-tuned the way transformers' Wav2Vec2ForCTC is: feature encoder frozen, dropout 0.1 everywhere, SpecAugment
/// time masking (p 0.05, span 10). LayerDrop is 0 on both sides (the paper fine-tunes with 0.1 or 0.05; it skips whole
/// layers at random, which a compiled graph cannot express without a graph break per layer). Input: 2-second clips of
/// raw 16 kHz audio (32,000 samples, 99 frames) with 24-character transcripts; loss: log-softmax + CTC per utterance,
/// averaged over the batch; optimizer AdamW at the model's fine-tuning rate 5e-5 (no weight decay) with the global-norm
/// clip at 1. benchmark.py's <c>Wav2Vec2BaseCtc</c> is transformers' Wav2Vec2ForCTC at the same configuration; the
/// trainable parameter counts are equal.
/// </summary>
internal sealed class AiDotNetWav2Vec2BaseModel : IBenchmarkModel
{
    internal const int Samples = 32000;
    internal const int TranscriptLength = 24;
    internal const int Vocabulary = 34;

    private readonly Wav2Vec2Model<float> _model;
    private readonly Random _random;
    private Tensor<float> _input = Tensor<float>.Empty();
    private Tensor<float> _label = Tensor<float>.Empty();
    private bool _trained;

    public AiDotNetWav2Vec2BaseModel(int seed)
    {
        _random = new Random(seed);
        _model = new Wav2Vec2Model<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.SequenceToSequence,
                inputSize: Samples, outputSize: Vocabulary),
            // The defaults are the BASE configuration and Wav2Vec2ForCTC's fine-tuning settings; the row spells out
            // the ones it depends on.
            new Wav2Vec2ModelOptions
            {
                HiddenDim = 768, NumTransformerLayers = 12, NumHeads = 12, FfDim = 3072, MaxAudioLengthSeconds = 30,
                FreezeFeatureEncoder = true, HiddenDropout = 0.1, AttentionDropout = 0.1, ActivationDropout = 0.1,
                FinalDropout = 0.1, MaskTimeProbability = 0.05, LayerDrop = 0.0, Seed = seed,
            },
            optimizer: new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 5e-5, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.0,
                    EnableGradientClipping = true, MaxGradientNorm = 1.0,
                }));

        LoadSyntheticBatch(1);
        Forward();
        ParameterCount = _model.TrainableParameterCount;
    }

    public long ParameterCount { get; }

    public (Tensor<float> Input, Tensor<float> Label) CurrentBatch
    {
        get => (_input, _label);
        set => (_input, _label) = value;
    }

    /// <summary>
    /// Zero-mean, unit-variance waveforms (the feature extractor's normalization), as a 220 Hz-ish tone with noise,
    /// and transcripts of 24 random non-blank symbols.
    /// </summary>
    public void LoadSyntheticBatch(int batchSize)
    {
        var audio = new float[batchSize * Samples];
        for (var b = 0; b < batchSize; b++)
        {
            double frequency = 100 + 300 * _random.NextDouble();
            for (var i = 0; i < Samples; i++)
            {
                double u1 = 1.0 - _random.NextDouble(), u2 = _random.NextDouble();
                double noise = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                audio[b * Samples + i] = (float)(Math.Sin(2.0 * Math.PI * frequency * i / 16000.0) + 0.5 * noise);
            }
        }
        _input = new Tensor<float>(audio, [batchSize, Samples]);
        var labels = new float[batchSize * TranscriptLength];
        for (var i = 0; i < labels.Length; i++) labels[i] = _random.Next(1, Vocabulary);
        _label = new Tensor<float>(labels, [batchSize, TranscriptLength]);
    }

    public void Forward() => _ = _model.Predict(_input);

    public void Backward()
    {
        // Wav2Vec2Model.Train is the whole step: forward, log-softmax + CTC, backward and the update.
        _model.Train(_input, _label);
        _trained = true;
    }

    public void Step()
    {
        // The optimizer step ran inside Backward's Train call.
    }

    public double? LastLoss => _trained ? _model.GetLastLoss() : null;
}

/// <summary>
/// The "hifigan-v1" row: the shipped <see cref="HiFiGAN{T}"/> at Kong et al.'s V1 configuration (22.05 kHz, 80 mels,
/// hop 256; generator: initial width 512, upsampling 8, 8, 2, 2 with kernels 16, 16, 4, 4, multi-receptive-field
/// fusion with kernels 3, 7, 11 and dilations 1, 3, 5; multi-period discriminator, periods 2, 3, 5, 7, 11; multi-scale
/// discriminator, 3 scales, spectral norm on the first; weight norm elsewhere). One training step is the model's GAN
/// iteration on a random 8192-sample segment (32 mel frames) of a 256-frame utterance: a discriminator update (least
/// squares) on a detached generation, then a generator update (least-squares adversarial + 2 x feature matching +
/// 45 x mel L1), each under its own AdamW (lr 2e-4, betas 0.8 / 0.99, weight decay 0.01), no clipping. The model trains
/// one utterance per step (it crops and scores a single waveform), so the row runs at --batch-size 1; the paper's batch
/// is 16. Inference is the generator on the whole 256-frame mel (65,536 samples, about 3 s). The mel of each synthetic
/// utterance is computed outside the timed region with the model's own analysis. benchmark.py's <c>HifiGanV1</c> is the
/// authors' reference generator and discriminators (jik876/hifi-gan) with the same step.
/// </summary>
internal sealed class AiDotNetHifiGanV1Model : IBenchmarkModel
{
    internal const int MelChannels = 80;
    internal const int Hop = 256;
    internal const int UtteranceFrames = 256;

    private readonly HiFiGAN<float> _model;
    private readonly Random _random;
    private Tensor<float> _input = Tensor<float>.Empty();
    private Tensor<float> _label = Tensor<float>.Empty();
    private bool _trained;

    public AiDotNetHifiGanV1Model(int seed)
    {
        _random = new Random(seed);
        // The defaults are the V1 configuration and the paper's optimizer; SamplingSeed seeds initialization and crops.
        _model = new HiFiGAN<float>(new NeuralNetworkArchitecture<float>(inputFeatures: MelChannels, outputSize: 1),
            new HiFiGANOptions { SamplingSeed = seed });

        LoadSyntheticBatch(1);
        Forward();
        ParameterCount = _model.TrainableParameterCount;
    }

    public long ParameterCount { get; }

    /// <summary>The input is the mel [B, 80, 256]; the label is the waveform [B, 65536] it was computed from.</summary>
    public (Tensor<float> Input, Tensor<float> Label) CurrentBatch
    {
        get => (_input, _label);
        set => (_input, _label) = value;
    }

    /// <summary>Waveforms of a few random partials plus noise, peak 0.95 (the reference loader's normalization).</summary>
    public void LoadSyntheticBatch(int batchSize)
    {
        int samples = UtteranceFrames * Hop;
        var audio = new float[batchSize * samples];
        var mels = new List<Tensor<float>>();
        for (var b = 0; b < batchSize; b++)
        {
            var row = new double[samples];
            double peak = 0;
            for (var partial = 0; partial < 3; partial++)
            {
                double frequency = 100 + 900 * _random.NextDouble(), phase = 2 * Math.PI * _random.NextDouble();
                for (var i = 0; i < samples; i++) row[i] += Math.Sin(2 * Math.PI * frequency * i / 22050.0 + phase);
            }
            for (var i = 0; i < samples; i++)
            {
                row[i] += 0.3 * (2 * _random.NextDouble() - 1);
                peak = Math.Max(peak, Math.Abs(row[i]));
            }
            var waveform = new float[samples];
            for (var i = 0; i < samples; i++) waveform[i] = (float)(row[i] / peak * 0.95);
            Array.Copy(waveform, 0, audio, b * samples, samples);
            // The model's own input analysis: [1, 80, frames]; a centred STFT's extra last frame is dropped.
            var mel = _model.ComputeMel(new Tensor<float>(waveform, [samples]));
            mels.Add(new Tensor<float>(mel.ToArray().AsSpan(0, MelChannels * mel.Shape[2]).ToArray(), [1, MelChannels, mel.Shape[2]]));
        }
        int frames = Math.Min(UtteranceFrames, mels[0].Shape[2]);
        var melValues = new float[batchSize * MelChannels * frames];
        for (var b = 0; b < batchSize; b++)
        {
            var values = mels[b].ToArray();
            int have = mels[b].Shape[2];
            for (var c = 0; c < MelChannels; c++)
                Array.Copy(values, c * have, melValues, (b * MelChannels + c) * frames, frames);
        }
        _input = new Tensor<float>(melValues, [batchSize, MelChannels, frames]);
        _label = new Tensor<float>(audio, [batchSize, samples]);
    }

    public void Forward() => _ = _model.Predict(_input);

    public void Backward()
    {
        if (_input.Shape[0] != 1)
            throw new InvalidOperationException(
                "HiFiGAN trains one utterance per step (it crops and scores a single waveform); run the hifigan-v1 row " +
                "with --batch-size 1.");
        // HiFiGAN.Train is the whole GAN iteration: segment crop, discriminator update, generator update.
        _model.Train(_input, _label);
        _trained = true;
    }

    public void Step()
    {
        // Both optimizer steps ran inside Backward's Train call.
    }

    public double? LastLoss => _trained ? _model.GetLastLoss() : null;
}
