using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// HiFi-GAN neural vocoder (Kong et al., 2020).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// HiFi-GAN is a GAN-based vocoder that converts mel-spectrograms to high-fidelity waveforms.
/// Using multi-period and multi-scale discriminators, it achieves MOS 4.23 at 13.9x real-time
/// on CPU (V1 variant). HiFi-GAN is the most widely-used vocoder in production TTS systems.
/// </para>
/// <para>
/// <b>For Beginners:</b> A vocoder is the final step in text-to-speech. After a model like
/// Tacotron or StyleTTS creates a spectrogram (a picture of sound), the vocoder turns it
/// into actual audio. HiFi-GAN is like a highly skilled musician who can read sheet music
/// (the spectrogram) and play it perfectly at super speed.
///
/// Why "GAN"? It was trained using two competing networks:
/// - A Generator that creates audio
/// - Discriminators that try to tell real audio from generated audio
/// This competition pushes the generator to make increasingly realistic audio.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 1);
/// var vocoder = new HiFiGAN&lt;float&gt;(arch, "hifigan_v1.onnx");
/// var audio = vocoder.Synthesize("This text is converted elsewhere; pass mel-spec directly.");
/// // Or use directly with mel-spectrogram:
/// var waveform = vocoder.Predict(melSpectrogram);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.GAN)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.TextToSpeech)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis", "https://arxiv.org/abs/2010.05646", Year = 2020, Authors = "Jungil Kong, Jaehyeon Kim, Jaekyoung Bae")]
public class HiFiGAN<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
{
    #region Fields

    private readonly HiFiGANOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region ITextToSpeech Properties

    /// <inheritdoc />
    public IReadOnlyList<VoiceInfo<T>> AvailableVoices { get; } = new[]
    {
        new VoiceInfo<T> { Id = "default", Name = "Default", Language = "en", Gender = VoiceGender.Neutral, Style = "neutral" }
    };

    /// <inheritdoc />
    public bool SupportsVoiceCloning => false;

    /// <inheritdoc />
    public bool SupportsEmotionControl => false;

    /// <inheritdoc />
    public bool SupportsStreaming => true;

    #endregion

    #region Constructors

    /// <summary>Creates a HiFi-GAN vocoder in ONNX inference mode.</summary>
    public HiFiGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, HiFiGANOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new HiFiGANOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a HiFi-GAN vocoder in native training mode.</summary>
    public HiFiGAN(NeuralNetworkArchitecture<T> architecture, HiFiGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new HiFiGANOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<HiFiGAN<T>> CreateAsync(HiFiGANOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new HiFiGANOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("hifigan", $"hifigan_{options.Variant.ToLowerInvariant()}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: 1);
        return new HiFiGAN<T>(arch, mp, options);
    }

    #endregion

    #region ITextToSpeech

    /// <inheritdoc />
    public Tensor<T> Synthesize(string text, string? voiceId = null, double speakingRate = 1.0, double pitch = 0.0)
    {
        ThrowIfDisposed();
        // HiFi-GAN is a vocoder; it expects mel-spectrogram input, not text.
        // For text input, encode as mel-shaped features.
        var melFeatures = TextToMelFeatures(text);
        return VocoderForward(melFeatures);
    }

    /// <inheritdoc />
    public Task<Tensor<T>> SynthesizeAsync(string text, string? voiceId = null, double speakingRate = 1.0,
        double pitch = 0.0, CancellationToken cancellationToken = default)
        => Task.Run(() => Synthesize(text, voiceId, speakingRate, pitch), cancellationToken);

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithVoiceCloning(string text, Tensor<T> referenceAudio, double speakingRate = 1.0, double pitch = 0.0)
        => throw new NotSupportedException("HiFi-GAN is a vocoder and does not support voice cloning. Use StyleTTS 2 + HiFi-GAN.");

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithEmotion(string text, string emotion, double emotionIntensity = 0.5,
        string? voiceId = null, double speakingRate = 1.0)
        => throw new NotSupportedException("HiFi-GAN is a vocoder and does not support emotion control.");

    /// <inheritdoc />
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
        => throw new NotSupportedException("HiFi-GAN is a vocoder and does not extract speaker embeddings.");

    /// <inheritdoc />
    public IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0)
        => new HiFiGANStreamingSession(this);

    /// <summary>
    /// Converts a mel-spectrogram to waveform audio (the primary vocoder function).
    /// </summary>
    /// <param name="melSpectrogram">Mel-spectrogram tensor [frames, numMels] or [numMels, frames].</param>
    /// <returns>Audio waveform tensor [samples].</returns>
    public Tensor<T> VocoderForward(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(melSpectrogram) : Predict(melSpectrogram);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultHiFiGANLayers(
            numMels: _options.NumMels, upsampleInitialChannel: _options.UpsampleInitialChannel,
            upsampleRates: _options.UpsampleRates, numResBlocks: _options.NumResBlocks,
            dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o)
    {
        // Tanh clamp output to [-1, 1]
        for (int i = 0; i < o.Length; i++)
        {
            double v = NumOps.ToDouble(o[i]);
            o[i] = NumOps.FromDouble(Math.Tanh(v));
        }
        return o;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "HiFi-GAN-Native" : "HiFi-GAN-ONNX",
            Description = $"HiFi-GAN {_options.Variant} neural vocoder (Kong et al., 2020)",
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.NumMels); w.Write(_options.HopLength);
        w.Write(_options.UpsampleInitialChannel); w.Write(_options.NumResBlocks);
        w.Write(_options.UpsampleRates.Length);
        foreach (int r in _options.UpsampleRates) w.Write(r);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.NumMels = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.UpsampleInitialChannel = r.ReadInt32(); _options.NumResBlocks = r.ReadInt32();
        int numRates = r.ReadInt32();
        _options.UpsampleRates = new int[numRates];
        for (int i = 0; i < numRates; i++) _options.UpsampleRates[i] = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new HiFiGAN<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    /// <summary>
    /// Converts text to a mel-spectrogram suitable for the HiFi-GAN vocoder
    /// via a phonetic-class projection. Each character is classified into a
    /// coarse phoneme category (vowel/nasal/liquid/voiced-stop/unvoiced-stop/
    /// sibilant/silence), and the category's typical spectral envelope is
    /// laid into the mel-bin axis using formant frequencies from standard
    /// phonetics references (Ladefoged &amp; Johnson 2010 "A Course in Phonetics"
    /// Chs. 4 and 8). Output is on the log-mel scale (~[-12, 4]) that the
    /// LibriTTS-trained HiFi-GAN was conditioned on.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is NOT a pretrained TTS acoustic model — it does not reproduce
    /// natural speech. It IS a real, deterministic, structurally-correct
    /// text→mel mapping that the vocoder can consume: vowel-heavy text
    /// concentrates energy in low-mid bins (around the first three formants),
    /// sibilants concentrate energy in the 4–6 kHz mel range, plosives appear
    /// as brief broadband bursts, and whitespace renders as silent floor.
    /// Output dimensionality and dynamic range match what the HiFi-GAN
    /// generator expects, so the vocoder pipeline is end-to-end functional
    /// even without an upstream Tacotron2 / FastSpeech2 / VITS acoustic model.
    /// </para>
    /// <para>
    /// For natural-sounding speech, wire a real acoustic model upstream and
    /// call <see cref="VocoderForward"/> directly with its mel output.
    /// </para>
    /// </remarks>
    private Tensor<T> TextToMelFeatures(string text)
    {
        const int framesPerChar = 5;                  // ~50 ms at 10 ms hop (LibriTTS HiFi-GAN config)
        const double silentDb = -11.5;                // log of the ~1e-5 mel floor used in extraction
        int numFrames = Math.Max(1, text.Length * framesPerChar);
        int numMels = _options.NumMels;
        var mel = new Tensor<T>(new[] { numFrames, numMels });

        // Seed every frame to the silence floor (silentDb) up front. When
        // text is empty the character loop below never runs and the tensor
        // would otherwise stay zero-filled — which is NOT silence on the
        // log-mel scale (zero is amplitude 1.0 in the linear mel space)
        // and would feed a non-silent frame into the vocoder.
        var silentT = NumOps.FromDouble(silentDb);
        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int m = 0; m < numMels; m++)
            {
                mel[frame, m] = silentT;
            }
        }

        // Deterministic per-text pitch jitter so identical successive letters
        // produce slightly different frames (matches natural prosody at ±5%).
        // Seed via a stable FNV-1a hash of the text — string.GetHashCode is
        // randomized per-process in .NET Core+ and would make the jitter
        // pattern differ across processes / versions / 32-bit vs 64-bit
        // hosts, breaking reproducibility of the synthesised mel frames.
        var rng = RandomHelper.CreateSeededRandom(StableFnv1aHash(text));
        var classes = HiFiGANPhoneticTable.Instance;

        for (int i = 0; i < text.Length; i++)
        {
            var cls = classes.Classify(text[i]);
            double pitchJitter = 1.0 + (rng.NextDouble() - 0.5) * 0.1;

            for (int f = 0; f < framesPerChar; f++)
            {
                // Smooth sin attack/release across the character's frames
                // avoids clicky transients between adjacent phonemes.
                double envelope = Math.Sin(Math.PI * (f + 0.5) / framesPerChar);
                int frameIdx = i * framesPerChar + f;

                for (int m = 0; m < numMels; m++)
                {
                    // Mel-bin distance to each spectral peak (Gaussian).
                    // Peak coordinates are on an 80-bin grid and rescale to
                    // numMels so configurations like 64- or 128-mel still
                    // produce the right relative formant pattern.
                    double activation = 0.0;
                    foreach (var peak in cls.Peaks)
                    {
                        double scaledPeak = peak.Bin * pitchJitter * numMels / 80.0;
                        double scaledWidth = peak.Width * numMels / 80.0;
                        double dz = (m - scaledPeak) / scaledWidth;
                        activation += peak.Amplitude * Math.Exp(-0.5 * dz * dz);
                    }
                    double logMel = silentDb + activation * envelope * cls.Energy;
                    mel[frameIdx, m] = NumOps.FromDouble(logMel);
                }
            }
        }

        return mel;
    }

    /// <summary>
    /// Deterministic 32-bit FNV-1a hash. Unlike <see cref="string.GetHashCode()"/>,
    /// which is randomized per-process in .NET Core+ and can differ across
    /// .NET versions / 32-bit vs 64-bit hosts (per
    /// https://learn.microsoft.com/dotnet/api/system.string.gethashcode),
    /// FNV-1a is stable across runs / processes so the per-text pitch
    /// jitter pattern stays reproducible.
    /// </summary>
    private static int StableFnv1aHash(string s)
    {
        const uint FnvOffsetBasis = 2166136261;
        const uint FnvPrime = 16777619;
        uint hash = FnvOffsetBasis;
        for (int i = 0; i < s.Length; i++)
        {
            hash ^= s[i];
            hash *= FnvPrime;
        }
        return unchecked((int)hash);
    }

    /// <summary>
    /// Phonetic-class spectral envelope table. Peaks are in mel-bin
    /// coordinates on an 80-bin grid mapped to 0–8 kHz (LibriTTS HiFi-GAN
    /// config); values derive from formant frequencies in Ladefoged &amp; Johnson
    /// 2010 (vowels, Ch. 8; fricatives, Ch. 4).
    /// </summary>
    private sealed class HiFiGANPhoneticTable
    {
        public readonly struct Peak
        {
            public readonly double Bin;
            public readonly double Width;
            public readonly double Amplitude;
            public Peak(double bin, double width, double amp) { Bin = bin; Width = width; Amplitude = amp; }
        }

        public sealed class PhoneticClass
        {
            public Peak[] Peaks { get; }
            public double Energy { get; }
            public PhoneticClass(double energy, Peak[] peaks) { Energy = energy; Peaks = peaks; }
        }

        private static readonly PhoneticClass Silence       = new(0.0,  Array.Empty<Peak>());
        private static readonly PhoneticClass Vowel         = new(11.0, [new(10, 4, 1.0), new(24, 5, 0.8), new(40, 6, 0.4)]);   // F1≈500Hz, F2≈1500Hz, F3≈2500Hz
        private static readonly PhoneticClass Nasal         = new(8.0,  [new(8,  3, 0.9), new(32, 5, 0.4)]);
        private static readonly PhoneticClass VoicedStop    = new(7.0,  [new(6,  3, 1.0)]);                                       // low-bin voice bar
        private static readonly PhoneticClass UnvoicedStop  = new(6.0,  [new(55, 8, 1.0)]);                                       // brief broadband burst high
        private static readonly PhoneticClass Sibilant      = new(10.0, [new(62, 6, 1.0), new(70, 4, 0.6)]);                      // 4–6 kHz fricative peak
        private static readonly PhoneticClass Liquid        = new(9.0,  [new(12, 4, 0.9), new(28, 5, 0.6)]);

        public static readonly HiFiGANPhoneticTable Instance = new();

        public PhoneticClass Classify(char c)
        {
            if (char.IsWhiteSpace(c) || char.IsPunctuation(c) || char.IsControl(c))
                return Silence;
            char lower = char.ToLowerInvariant(c);
            return lower switch
            {
                'a' or 'e' or 'i' or 'o' or 'u' or 'y' => Vowel,
                'm' or 'n'                              => Nasal,
                'l' or 'r' or 'w'                       => Liquid,
                'b' or 'd' or 'g' or 'v' or 'z' or 'j' => VoicedStop,
                'p' or 't' or 'k' or 'c' or 'q'         => UnvoicedStop,
                's' or 'f' or 'h' or 'x'                => Sibilant,
                _                                       => Vowel  // best default for digits / unknown letters
            };
        }
    }

    #endregion

    #region Streaming Session

    private sealed class HiFiGANStreamingSession : IStreamingSynthesisSession<T>
    {
        private readonly HiFiGAN<T> _model;
        private readonly List<Tensor<T>> _pendingMels = new();
        private bool _disposed;

        public HiFiGANStreamingSession(HiFiGAN<T> model) { _model = model; }

        public void FeedText(string textChunk)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HiFiGANStreamingSession));
            var mel = _model.TextToMelFeatures(textChunk);
            _pendingMels.Add(mel);
        }

        public IEnumerable<Tensor<T>> GetAvailableAudio()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HiFiGANStreamingSession));
            foreach (var mel in _pendingMels) yield return _model.VocoderForward(mel);
            _pendingMels.Clear();
        }

        public IEnumerable<Tensor<T>> Finalize()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HiFiGANStreamingSession));
            var results = new List<Tensor<T>>(GetAvailableAudio());
            _disposed = true;
            return results;
        }

        public void Dispose() { _disposed = true; }
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(HiFiGAN<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
