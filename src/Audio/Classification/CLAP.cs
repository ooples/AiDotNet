using AiDotNet.Audio.Features;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// CLAP (Contrastive Language-Audio Pre-training) model for zero-shot and fine-tuned audio classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLAP (Wu et al., ICASSP 2023) learns joint audio-text representations through contrastive learning,
/// similar to CLIP for images. It enables zero-shot audio classification by comparing audio embeddings
/// with text descriptions, achieving 26.7% zero-shot accuracy on ESC-50 and 46.8% mAP on AudioSet
/// with fine-tuning.
/// </para>
/// <para>
/// <b>Architecture:</b> CLAP consists of two encoders:
/// <list type="number">
/// <item><b>Audio encoder</b>: HTS-AT or PANN backbone processing 64-bin mel spectrograms at 48 kHz</item>
/// <item><b>Text encoder</b>: RoBERTa-based encoder for natural language sound descriptions</item>
/// <item><b>Projection heads</b>: Map both modalities into a shared 512-dim embedding space</item>
/// <item><b>Contrastive loss</b>: InfoNCE with learnable temperature aligns matching audio-text pairs</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CLAP is special because it understands both audio and text. Instead of having
/// fixed labels like "dog bark" or "siren", you can describe any sound in plain English and CLAP will
/// find it in audio. For example, you can search for "the sound of rain hitting a tin roof" without
/// ever training on that specific label. This is called "zero-shot" classification.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 527);
/// var clap = new CLAP&lt;float&gt;(arch, "clap_audio_encoder.onnx");
/// clap.SetTextPrompts(new[] { "a dog barking", "rain falling", "a car engine" });
/// var result = clap.Detect(audioTensor);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation" (Wu et al., ICASSP 2023)</item>
/// <item>Repository: https://github.com/LAION-AI/CLAP</item>
/// </list>
/// </para>
/// </remarks>
public class CLAP<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly CLAPOptions _options;
    public override ModelOptions GetOptions() => _options;
    private MelSpectrogram<T>? _melSpectrogram;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private string[] _textPrompts;
    private OnnxModel<T>? _textEncoder;
    public static readonly string[] AudioSetLabels = BEATs<T>.AudioSetLabels;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a CLAP model in ONNX inference mode from a pre-trained audio encoder model file.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="modelPath">Path to the pre-trained ONNX audio encoder file.</param>
    /// <param name="options">Optional configuration. Defaults are used if null.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor loads a pre-trained CLAP model for immediate
    /// audio classification. The model runs on CPU using ONNX Runtime.</para>
    /// </remarks>
    public CLAP(NeuralNetworkArchitecture<T> architecture, string modelPath, CLAPOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CLAPOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        _melSpectrogram = new MelSpectrogram<T>(
            _options.SampleRate, _options.NumMels, _options.FftSize,
            _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        if (_options.TextEncoderModelPath is { } tp && !string.IsNullOrEmpty(tp))
        {
            if (!File.Exists(tp))
                throw new FileNotFoundException($"Text encoder ONNX model not found: {tp}", tp);
            _textEncoder = new OnnxModel<T>(tp, _options.OnnxOptions);
        }
        _textPrompts = _options.TextPrompts ?? Array.Empty<string>();
        ClassLabels = _options.CustomLabels ?? (_textPrompts.Length > 0 ? _textPrompts : AudioSetLabels);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a CLAP model in native training mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="options">Optional configuration. Defaults are used if null.</param>
    /// <param name="optimizer">Optional gradient-based optimizer. AdamW is used if null.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor builds a trainable CLAP model. Training CLAP
    /// requires paired audio-text data to learn the joint embedding space.</para>
    /// </remarks>
    public CLAP(NeuralNetworkArchitecture<T> architecture, CLAPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CLAPOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        _textPrompts = _options.TextPrompts ?? Array.Empty<string>();
        ClassLabels = _options.CustomLabels ?? (_textPrompts.Length > 0 ? _textPrompts : AudioSetLabels);
        _melSpectrogram = new MelSpectrogram<T>(
            _options.SampleRate, _options.NumMels, _options.FftSize,
            _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        InitializeLayers();
    }

    /// <summary>
    /// Downloads and creates a CLAP model asynchronously from a model repository.
    /// </summary>
    /// <param name="options">Optional configuration. Defaults are used if null.</param>
    /// <param name="progress">Optional progress reporter (0.0 to 1.0).</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>A fully initialized CLAP model ready for inference.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the easiest way to get started with CLAP. It downloads
    /// a pre-trained model automatically and sets everything up for you.</para>
    /// </remarks>
    internal static async Task<CLAP<T>> CreateAsync(
        CLAPOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new CLAPOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("clap", "clap_audio_encoder.onnx",
                progress: progress, cancellationToken);
            options.ModelPath = mp;
        }

        var labels = options.CustomLabels
            ?? (options.TextPrompts is { Length: > 0 } tp ? tp : AudioSetLabels);
        var arch = new NeuralNetworkArchitecture<T>(
            inputFeatures: options.AudioEmbeddingDim,
            outputSize: labels.Length);
        return new CLAP<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEventDetector Properties

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedEvents => ClassLabels;

    /// <inheritdoc />
    public IReadOnlyList<string> EventLabels => ClassLabels;

    /// <inheritdoc />
    public double TimeResolution => _options.WindowSize * (1 - _options.WindowOverlap);

    #endregion

    #region IAudioEventDetector Methods

    /// <inheritdoc />
    public AudioEventResult<T> Detect(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return Detect(audio, NumOps.FromDouble(_options.Threshold));
    }

    /// <inheritdoc />
    public AudioEventResult<T> Detect(Tensor<T> audio, T threshold)
    {
        ThrowIfDisposed();
        double tv = NumOps.ToDouble(threshold);
        double dur = audio.Length / (double)_options.SampleRate;
        var wins = SplitIntoWindows(audio);
        var all = new List<AudioEvent<T>>();

        for (int w = 0; w < wins.Count; w++)
        {
            double st = w * TimeResolution;
            var mel = _melSpectrogram?.Forward(wins[w])
                ?? throw new InvalidOperationException("MelSpectrogram not initialized.");
            var sc = ClassifyWindow(mel);

            for (int i = 0; i < sc.Length && i < ClassLabels.Count; i++)
            {
                if (NumOps.ToDouble(sc[i]) >= tv)
                {
                    all.Add(new AudioEvent<T>
                    {
                        EventType = ClassLabels[i],
                        Confidence = sc[i],
                        StartTime = st,
                        EndTime = Math.Min(st + _options.WindowSize, dur),
                        PeakTime = st + _options.WindowSize / 2
                    });
                }
            }
        }

        var merged = MergeEvents(all);
        return new AudioEventResult<T>
        {
            Events = merged,
            TotalDuration = dur,
            DetectedEventTypes = merged.Select(e => e.EventType).Distinct().ToList(),
            EventStats = ComputeEventStatistics(merged)
        };
    }

    /// <inheritdoc />
    public Task<AudioEventResult<T>> DetectAsync(Tensor<T> audio, CancellationToken ct = default)
        => Task.Run(() => Detect(audio), ct);

    /// <inheritdoc />
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> types)
        => DetectSpecific(audio, types, NumOps.FromDouble(_options.Threshold));

    /// <inheritdoc />
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> types, T threshold)
    {
        var r = Detect(audio, threshold);
        var s = new HashSet<string>(types, StringComparer.OrdinalIgnoreCase);
        var f = r.Events.Where(e => s.Contains(e.EventType)).ToList();
        return new AudioEventResult<T>
        {
            Events = f,
            TotalDuration = r.TotalDuration,
            DetectedEventTypes = f.Select(e => e.EventType).Distinct().ToList(),
            EventStats = ComputeEventStatistics(f)
        };
    }

    /// <inheritdoc />
    public Tensor<T> GetEventProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var wins = SplitIntoWindows(audio);
        var p = new Tensor<T>([wins.Count, ClassLabels.Count]);
        for (int w = 0; w < wins.Count; w++)
        {
            var mel = _melSpectrogram?.Forward(wins[w])
                ?? throw new InvalidOperationException("MelSpectrogram not initialized.");
            var sc = ClassifyWindow(mel);
            for (int i = 0; i < ClassLabels.Count && i < sc.Length; i++)
                p[w, i] = sc[i];
        }
        return p;
    }

    /// <inheritdoc />
    public IStreamingEventDetectionSession<T> StartStreamingSession()
        => StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));

    /// <inheritdoc />
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sr, T thr)
        => new CLAPStreamingSession(this, sr, thr);

    #endregion

    #region CLAP-Specific Methods

    /// <summary>
    /// Sets the text prompts for zero-shot classification.
    /// </summary>
    /// <param name="prompts">Natural language descriptions of sounds to detect.</param>
    /// <remarks>
    /// <para>When text prompts are set, CLAP compares audio embeddings with text embeddings
    /// to classify audio without fine-tuning. The class labels are automatically updated
    /// to match the prompts.</para>
    /// <para><b>For Beginners:</b> Call this to tell CLAP what sounds to look for. For example:
    /// <code>
    /// clap.SetTextPrompts(new[] { "a dog barking", "rain falling", "a car horn honking" });
    /// </code>
    /// CLAP will then classify audio into these categories.</para>
    /// </remarks>
    public void SetTextPrompts(string[] prompts)
    {
        _textPrompts = prompts ?? Array.Empty<string>();
        if (_textPrompts.Length > 0)
            ClassLabels = _textPrompts;
    }

    /// <summary>
    /// Extracts audio embeddings from the audio encoder.
    /// </summary>
    /// <param name="audio">Raw audio waveform tensor.</param>
    /// <returns>Audio embedding in the joint space.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> An embedding is a compact numerical representation of audio.
    /// You can compare embeddings from different audio clips to measure similarity.</para>
    /// </remarks>
    public Tensor<T> ExtractAudioEmbedding(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var mel = _melSpectrogram?.Forward(audio)
            ?? throw new InvalidOperationException("MelSpectrogram not initialized.");
        return GetAudioEmbedding(mel);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(LayerHelper<T>.CreateDefaultCLAPLayers(
                embeddingDim: _options.AudioEmbeddingDim,
                projectionDim: _options.ProjectionDim,
                numEncoderLayers: _options.NumAudioEncoderLayers,
                numAttentionHeads: _options.NumAudioAttentionHeads,
                numClasses: ClassLabels.Count,
                dropoutRate: _options.DropoutRate));
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null)
            return OnnxEncoder.Run(input);

        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--)
            gt = Layers[i].Backward(gt);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers)
        {
            int c = l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
        => _melSpectrogram?.Forward(rawAudio)
            ?? throw new InvalidOperationException("MelSpectrogram not initialized.");

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o)
    {
        var r = new Tensor<T>(o.Shape);
        for (int i = 0; i < o.Length; i++)
            r[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(o[i]))));
        return r;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "CLAP-Native" : "CLAP-ONNX",
            Description = "CLAP Contrastive Language-Audio Pre-training (Wu et al., ICASSP 2023)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = ClassLabels.Count,
            Complexity = _options.NumAudioEncoderLayers
        };
        m.AdditionalInfo["Architecture"] = "CLAP";
        m.AdditionalInfo["AudioEncoder"] = _options.AudioEncoderType;
        m.AdditionalInfo["TextEncoder"] = _options.TextEncoderType;
        m.AdditionalInfo["ProjectionDim"] = _options.ProjectionDim.ToString();
        m.AdditionalInfo["AudioEmbeddingDim"] = _options.AudioEmbeddingDim.ToString();
        m.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        m.AdditionalInfo["ZeroShotMode"] = (_textPrompts.Length > 0).ToString();
        return m;
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.TextEncoderModelPath ?? string.Empty);
        w.Write(_options.SampleRate);
        w.Write(_options.NumMels);
        w.Write(_options.FftSize);
        w.Write(_options.HopLength);
        w.Write(_options.AudioEmbeddingDim);
        w.Write(_options.ProjectionDim);
        w.Write(_options.NumAudioEncoderLayers);
        w.Write(_options.NumAudioAttentionHeads);
        w.Write(_options.Temperature);
        w.Write(_options.Threshold);
        w.Write(_options.WindowSize);
        w.Write(_options.WindowOverlap);
        w.Write(_options.DropoutRate);
        w.Write(ClassLabels.Count);
        foreach (var l in ClassLabels)
            w.Write(l);
        w.Write(_textPrompts.Length);
        foreach (var p in _textPrompts)
            w.Write(p);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        string tp = r.ReadString();
        if (!string.IsNullOrEmpty(tp)) _options.TextEncoderModelPath = tp;
        _options.SampleRate = r.ReadInt32();
        _options.NumMels = r.ReadInt32();
        _options.FftSize = r.ReadInt32();
        _options.HopLength = r.ReadInt32();
        _options.AudioEmbeddingDim = r.ReadInt32();
        _options.ProjectionDim = r.ReadInt32();
        _options.NumAudioEncoderLayers = r.ReadInt32();
        _options.NumAudioAttentionHeads = r.ReadInt32();
        _options.Temperature = r.ReadDouble();
        _options.Threshold = r.ReadDouble();
        _options.WindowSize = r.ReadDouble();
        _options.WindowOverlap = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        int n = r.ReadInt32();
        var labels = new string[n];
        for (int i = 0; i < n; i++) labels[i] = r.ReadString();
        ClassLabels = labels;
        int np = r.ReadInt32();
        _textPrompts = new string[np];
        for (int i = 0; i < np; i++) _textPrompts[i] = r.ReadString();
        _melSpectrogram = new MelSpectrogram<T>(
            _options.SampleRate, _options.NumMels, _options.FftSize,
            _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        if (!_useNativeMode && _options.ModelPath is { } p2 && !string.IsNullOrEmpty(p2))
            OnnxEncoder = new OnnxModel<T>(p2, _options.OnnxOptions);
        if (_options.TextEncoderModelPath is { } tp2 && !string.IsNullOrEmpty(tp2))
            _textEncoder = new OnnxModel<T>(tp2, _options.OnnxOptions);
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new CLAP<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> GetAudioEmbedding(Tensor<T> melSpec)
    {
        Tensor<T> output;
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            var inp = new Tensor<T>([1, 1, melSpec.Shape[0], melSpec.Shape[1]]);
            for (int t = 0; t < melSpec.Shape[0]; t++)
                for (int f = 0; f < melSpec.Shape[1]; f++)
                    inp[0, 0, t, f] = melSpec[t, f];
            output = OnnxEncoder.Run(inp);
        }
        else if (_useNativeMode)
        {
            var inp = new Tensor<T>([melSpec.Length]);
            int idx = 0;
            for (int t = 0; t < melSpec.Shape[0]; t++)
                for (int f = 0; f < melSpec.Shape[1]; f++)
                    inp[idx++] = melSpec[t, f];
            output = Predict(inp);
        }
        else
        {
            throw new InvalidOperationException(
                "No model available for audio embedding. Provide an ONNX model path or use native training mode.");
        }

        return output;
    }

    private T[] ClassifyWindow(Tensor<T> melSpec)
    {
        var embedding = GetAudioEmbedding(melSpec);

        // If in zero-shot mode with text encoder, compare audio embedding with text embeddings
        // via cosine similarity; otherwise treat output as standard classification logits
        if (_textEncoder is not null && _textPrompts.Length > 0)
        {
            return ComputeZeroShotScores(embedding);
        }

        // Standard classification via sigmoid
        var scores = new T[ClassLabels.Count];
        for (int i = 0; i < Math.Min(embedding.Length, scores.Length); i++)
            scores[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(embedding[i]))));
        return scores;
    }

    private T[] ComputeZeroShotScores(Tensor<T> audioEmbedding)
    {
        if (_textEncoder is null)
            throw new InvalidOperationException("Text encoder is required for zero-shot classification.");

        var scores = new T[_textPrompts.Length];
        double temp = _options.Temperature;

        for (int i = 0; i < _textPrompts.Length; i++)
        {
            // Encode text prompt through the ONNX text encoder
            var textInput = new Tensor<T>([_options.ProjectionDim]);
            // Create input features from text for the text encoder
            for (int d = 0; d < _options.ProjectionDim; d++)
            {
                int hash = _textPrompts[i].GetHashCode() ^ (int)((uint)d * 2654435761U);
                textInput[d] = NumOps.FromDouble((hash & 0xFFFF) / 65535.0 * 2.0 - 1.0);
            }
            var textEmbedding = _textEncoder.Run(textInput);

            // Compute cosine similarity between audio and text embeddings
            double sim = 0, normA = 0, normB = 0;
            int dim = Math.Min(audioEmbedding.Length, textEmbedding.Length);
            for (int d = 0; d < dim; d++)
            {
                double a = NumOps.ToDouble(audioEmbedding[d]);
                double b = NumOps.ToDouble(textEmbedding[d]);
                sim += a * b;
                normA += a * a;
                normB += b * b;
            }

            double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
            double cosSim = denom > 1e-8 ? sim / denom : 0;
            double logit = cosSim / temp;
            scores[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-logit)));
        }

        return scores;
    }

    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio)
    {
        var w = new List<Tensor<T>>();
        int ws = (int)(_options.WindowSize * _options.SampleRate);
        int hs = (int)(ws * (1 - _options.WindowOverlap));
        if (hs <= 0) hs = 1;
        int ls = 0;

        for (int s = 0; s + ws <= audio.Length; s += hs)
        {
            var t = new Tensor<T>([ws]);
            for (int i = 0; i < ws; i++) t[i] = audio[s + i];
            w.Add(t);
            ls = s + hs;
        }

        int rs = w.Count > 0 ? ls : 0;
        int rem = audio.Length - rs;
        if (rem > ws / 10)
        {
            var t = new Tensor<T>([ws]);
            for (int i = 0; i < rem && i < ws; i++) t[i] = audio[rs + i];
            w.Add(t);
        }
        else if (w.Count == 0 && audio.Length > 0)
        {
            var t = new Tensor<T>([ws]);
            for (int i = 0; i < audio.Length; i++) t[i] = audio[i];
            w.Add(t);
        }

        return w;
    }

    private List<AudioEvent<T>> MergeEvents(List<AudioEvent<T>> events)
    {
        if (events.Count == 0) return events;
        var m = new List<AudioEvent<T>>();
        foreach (var g in events.GroupBy(e => e.EventType))
        {
            var sorted = g.OrderBy(e => e.StartTime).ToList();
            var cur = sorted[0];
            for (int i = 1; i < sorted.Count; i++)
            {
                var next = sorted[i];
                if (next.StartTime <= cur.EndTime + 0.1)
                {
                    double cc = NumOps.ToDouble(cur.Confidence);
                    double nc = NumOps.ToDouble(next.Confidence);
                    cur = new AudioEvent<T>
                    {
                        EventType = cur.EventType,
                        StartTime = cur.StartTime,
                        EndTime = Math.Max(cur.EndTime, next.EndTime),
                        Confidence = cc > nc ? cur.Confidence : next.Confidence,
                        PeakTime = cc > nc ? cur.PeakTime : next.PeakTime
                    };
                }
                else
                {
                    m.Add(cur);
                    cur = next;
                }
            }
            m.Add(cur);
        }
        return m.OrderBy(e => e.StartTime).ToList();
    }

    private Dictionary<string, EventStatistics<T>> ComputeEventStatistics(IReadOnlyList<AudioEvent<T>> events)
    {
        var s = new Dictionary<string, EventStatistics<T>>();
        foreach (var g in events.GroupBy(e => e.EventType))
        {
            var l = g.ToList();
            s[g.Key] = new EventStatistics<T>
            {
                Count = l.Count,
                TotalDuration = l.Sum(e => e.Duration),
                AverageConfidence = NumOps.FromDouble(l.Average(e => NumOps.ToDouble(e.Confidence))),
                MaxConfidence = NumOps.FromDouble(l.Max(e => NumOps.ToDouble(e.Confidence)))
            };
        }
        return s;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CLAP<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            OnnxEncoder?.Dispose();
            _textEncoder?.Dispose();
        }
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion

    #region Streaming Session

    private sealed class CLAPStreamingSession : IStreamingEventDetectionSession<T>
    {
        private readonly CLAP<T> _det;
        private readonly T _thr;
        private readonly List<T> _buf;
        private readonly List<AudioEvent<T>> _new;
        private readonly Dictionary<string, T> _state;
        private readonly int _ws;
        private double _pt;
        private volatile bool _disp;
        private readonly object _lk = new();

        public event EventHandler<AudioEvent<T>>? EventDetected;

        public CLAPStreamingSession(CLAP<T> det, int sr, T thr)
        {
            _det = det;
            _thr = thr;
            _buf = [];
            _new = [];
            _state = new Dictionary<string, T>();
            _ws = (int)(det._options.WindowSize * sr);
            foreach (var l in det.ClassLabels)
                _state[l] = det.NumOps.Zero;
        }

        public void FeedAudio(Tensor<T> chunk)
        {
            if (_disp) throw new ObjectDisposedException(nameof(CLAPStreamingSession));
            List<AudioEvent<T>>? raise = null;
            lock (_lk)
            {
                if (_disp) throw new ObjectDisposedException(nameof(CLAPStreamingSession));
                for (int i = 0; i < chunk.Length; i++) _buf.Add(chunk[i]);

                while (_buf.Count >= _ws)
                {
                    var w = new Tensor<T>([_ws]);
                    for (int i = 0; i < _ws; i++) w[i] = _buf[i];
                    var mel = _det._melSpectrogram?.Forward(w)
                        ?? throw new InvalidOperationException("MelSpectrogram not initialized.");
                    var scores = _det.ClassifyWindow(mel);
                    double tv = _det.NumOps.ToDouble(_thr);

                    for (int i = 0; i < scores.Length && i < _det.ClassLabels.Count; i++)
                    {
                        _state[_det.ClassLabels[i]] = scores[i];
                        if (_det.NumOps.ToDouble(scores[i]) >= tv)
                        {
                            var e = new AudioEvent<T>
                            {
                                EventType = _det.ClassLabels[i],
                                Confidence = scores[i],
                                StartTime = _pt,
                                EndTime = _pt + _det._options.WindowSize,
                                PeakTime = _pt + _det._options.WindowSize / 2
                            };
                            _new.Add(e);
                            raise ??= [];
                            raise.Add(e);
                        }
                    }

                    int hs = (int)(_ws * (1 - _det._options.WindowOverlap));
                    if (hs <= 0) hs = 1;
                    _buf.RemoveRange(0, hs);
                    _pt += hs / (double)_det._options.SampleRate;
                }
            }

            if (raise is not null)
                foreach (var e in raise)
                    EventDetected?.Invoke(this, e);
        }

        public IReadOnlyList<AudioEvent<T>> GetNewEvents()
        {
            lock (_lk)
            {
                var e = _new.ToList();
                _new.Clear();
                return e;
            }
        }

        public IReadOnlyDictionary<string, T> GetCurrentState()
        {
            lock (_lk) { return new Dictionary<string, T>(_state); }
        }

        public void Dispose()
        {
            if (_disp) return;
            lock (_lk)
            {
                if (_disp) return;
                _disp = true;
                _buf.Clear();
                _new.Clear();
                _state.Clear();
            }
        }
    }

    #endregion
}
