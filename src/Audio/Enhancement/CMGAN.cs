using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// CMGAN (Conformer-based Metric GAN) for speech enhancement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CMGAN (Cao et al., INTERSPEECH 2022) combines a conformer-based generator with a metric
/// discriminator for high-quality speech enhancement, achieving PESQ 3.41 and STOI 0.97
/// on the VoiceBank-DEMAND dataset.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <list type="number">
/// <item><b>U-Net encoder</b>: Compresses the noisy spectrogram with convolutional blocks</item>
/// <item><b>Conformer bottleneck</b>: Self-attention + convolution for global context</item>
/// <item><b>U-Net decoder</b>: Reconstructs clean spectrogram with skip connections</item>
/// <item><b>Metric discriminator</b>: Judges enhancement quality during training</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CMGAN uses a competition between two networks: a "generator" that
/// cleans audio and a "discriminator" that judges quality. The generator uses Conformer
/// layers that combine attention (understanding context) with convolution (local patterns).
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 201, outputSize: 201);
/// var model = new CMGAN&lt;float&gt;(arch, "cmgan_voicebank.onnx");
/// var clean = model.Enhance(noisyAudio);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "CMGAN: Conformer-based Metric GAN for Speech Enhancement" (Cao et al., INTERSPEECH 2022)</item>
/// <item>Repository: https://github.com/ruizhecao96/CMGAN</item>
/// </list>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.GAN)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Enhancement)]
[ModelTask(ModelTask.Denoising)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("CMGAN: Conformer-based Metric GAN for Speech Enhancement", "https://arxiv.org/abs/2203.15149", Year = 2022, Authors = "Ruizhe Cao, Sherif Abdulatif, Bin Yang")]
public partial class CMGAN<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// DERIVED, not stored: <c>PredictCore</c> folds <c>Layers</c> and <c>PostprocessOutput</c> is the
    /// identity, so the width is the final <c>DenseLayer&lt;T&gt;(numFreqBins * 3)</c> of
    /// <c>CreateDefaultCMGANLayers</c>. The 3x is the paper's DECOUPLED head packed into one
    /// projection - bins [0, F) are the magnitude mask and [F, 3F) the interleaved real/imaginary
    /// pair - so the width is three times <c>_options.NumFreqBins</c>, not <c>NumFreqBins</c> itself.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.NumFreqBins * 3;

    #region Fields

    private const double ExponentComparisonTolerance = 1e-12;
    private readonly CMGANOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private ShortTimeFourierTransform<T> _stft;
    [Scratch]
    private Tensor<T>? _lastPhase;
    [Buffer]
    private Tensor<T>? _noiseProfile;
    private bool _useNativeMode;
    private bool _disposed;
    private List<T>? _streamingBuffer;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a CMGAN model in ONNX inference mode.
    /// </summary>
    public CMGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, CMGANOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CMGANOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a CMGAN model in native training mode.
    /// </summary>
    public CMGAN(NeuralNetworkArchitecture<T> architecture, CMGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CMGANOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    /// <summary>
    /// Downloads and creates a CMGAN model asynchronously.
    /// </summary>
    internal static async Task<CMGAN<T>> CreateAsync(
        CMGANOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new CMGANOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("cmgan", "cmgan_voicebank.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumFreqBins, outputSize: options.NumFreqBins);
        return new CMGAN<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc />
    public int NumChannels { get; } = 1;

    /// <inheritdoc />
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc />
    public int LatencySamples => _options.FftSize;

    #endregion

    #region IAudioEnhancer Methods

    /// <inheritdoc />
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var stft = ComputeSTFT(audio);

        // Apply spectral subtraction if noise profile is available (per-bin, handles length differences)
        if (_noiseProfile is not null)
        {
            int len = Math.Min(stft.Length, _noiseProfile.Length);
            for (int i = 0; i < len; i++)
            {
                T subtracted = NumOps.Subtract(stft[i], _noiseProfile[i]);
                stft[i] = NumOps.GreaterThan(subtracted, NumOps.Zero) ? subtracted : NumOps.Zero;
            }
        }

        Tensor<T> decoded;
        if (IsOnnxMode && OnnxEncoder is not null)
            decoded = OnnxEncoder.Run(stft);
        else
            decoded = Predict(stft);

        // Reconstruct from BOTH decoder heads (mask + complex), per the paper. This replaces the
        // previous ApplyMask -> ComputeISTFT pair, which used the magnitude branch alone and fed
        // the untouched noisy phase into the inverse transform.
        var result = ReconstructFromDecoupledHeads(stft, decoded, audio.Length);

        // Apply enhancement strength blending: output = strength * enhanced + (1 - strength) * original
        double strength = _options.EnhancementStrength;
        if (strength < 1.0)
        {
            T s = NumOps.FromDouble(strength);
            T inv = NumOps.FromDouble(1.0 - strength);
            for (int i = 0; i < result.Length && i < audio.Length; i++)
            {
                result[i] = NumOps.Add(NumOps.Multiply(s, result[i]), NumOps.Multiply(inv, audio[i]));
            }
        }

        return result;
    }

    /// <inheritdoc />
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
    {
        // Use the reference signal as a noise profile estimate before enhancement
        EstimateNoiseProfile(reference);
        return Enhance(audio);
    }

    /// <inheritdoc />
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        ThrowIfDisposed();
        _streamingBuffer ??= [];
        for (int i = 0; i < audioChunk.Length; i++) _streamingBuffer.Add(audioChunk[i]);
        int frameSize = _options.FftSize;
        if (_streamingBuffer.Count < frameSize) return new Tensor<T>([0]);
        var frame = new Tensor<T>([frameSize]);
        for (int i = 0; i < frameSize; i++) frame[i] = _streamingBuffer[i];
        _streamingBuffer.RemoveRange(0, _options.HopLength);
        return Enhance(frame);
    }

    /// <inheritdoc />
    public override void ResetState() { base.ResetState(); _streamingBuffer = null; }

    /// <inheritdoc />
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        // Compute STFT of noise-only audio to get spectral noise floor.
        // Compressed with the same power law ComputeSTFT uses, because Enhance subtracts this
        // profile from the compressed input spectrogram — mixing a raw profile into a
        // compressed spectrogram would subtract wildly mismatched magnitudes.
        _stft.MagnitudeAndPhase(noiseOnlyAudio, out var magnitude, out _);
        _noiseProfile = ApplyPowerLaw(magnitude, _options.PowerLawCompressionExponent);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(LayerHelper<T>.CreateDefaultCMGANLayers(
                numFreqBins: _options.NumFreqBins,
                conformerDim: _options.ConformerDim,
                numConformerLayers: _options.NumConformerLayers,
                numAttentionHeads: _options.NumAttentionHeads,
                dropoutRate: _options.DropoutRate));
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input;
        foreach (var l in Layers) c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => ComputeSTFT(rawAudio);
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "CMGAN-Native" : "CMGAN-ONNX",
            Description = "CMGAN Conformer-based Metric GAN (Cao et al., INTERSPEECH 2022)",
            FeatureCount = _options.NumFreqBins,
            Complexity = _options.NumConformerLayers
        };
        m.AdditionalInfo["Architecture"] = "CMGAN";
        m.AdditionalInfo["ConformerDim"] = _options.ConformerDim.ToString();
        m.AdditionalInfo["NumConformerLayers"] = _options.NumConformerLayers.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.FftSize); w.Write(_options.HopLength);
        w.Write(_options.NumFreqBins); w.Write(_options.ConformerDim); w.Write(_options.NumConformerLayers);
        w.Write(_options.NumAttentionHeads); w.Write(_options.EnhancementStrength); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.NumFreqBins = r.ReadInt32(); _options.ConformerDim = r.ReadInt32(); _options.NumConformerLayers = r.ReadInt32();
        _options.NumAttentionHeads = r.ReadInt32(); _options.EnhancementStrength = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        _lastPhase = null;
        _noiseProfile = null;
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new CMGAN<T>(Architecture, mp, _options);
        return new CMGAN<T>(Architecture, _options);
    }

    #endregion

    #region Private Helpers

    private Tensor<T> ComputeSTFT(Tensor<T> audio)
    {
        _stft.MagnitudeAndPhase(audio, out var magnitude, out var phase);
        _lastPhase = phase;

        // Power-law compression |X|^c before the encoder, per Cao et al., INTERSPEECH 2022
        // (arXiv:2203.15149), which uses c = 0.3. Speech magnitudes span a very large dynamic
        // range, so an uncompressed spectrogram lets loud bins dominate the objective; the
        // matching inverse is applied in ReconstructFromDecoupledHeads. The exponent is a public
        // option (PowerLawCompressionExponent) and 1.0 disables compression.
        return ApplyPowerLaw(magnitude, _options.PowerLawCompressionExponent);
    }

    /// <summary>
    /// Raises every element to <paramref name="exponent"/>, preserving sign. Used for CMGAN's
    /// power-law compression and its inverse.
    /// </summary>
    private Tensor<T> ApplyPowerLaw(Tensor<T> values, double exponent)
    {
        if (Math.Abs(exponent - 1.0) <= ExponentComparisonTolerance) return values;

        var result = new Tensor<T>(values._shape);
        for (int i = 0; i < values.Length; i++)
        {
            double v = NumOps.ToDouble(values.Data.Span[i]);
            double magnitude = Math.Pow(Math.Abs(v), exponent);
            result.Data.Span[i] = NumOps.FromDouble(v < 0 ? -magnitude : magnitude);
        }

        return result;
    }

    private Tensor<T> ApplyMask(Tensor<T> stft, Tensor<T> mask)
    {
        // Vectorised mask application: Engine.TensorMultiply is SIMD when
        // shapes match. Fall back to the prior min-length scalar loop only
        // for the (rare) mismatched-shape path.
        if (stft.Length == mask.Length && stft._shape.SequenceEqual(mask._shape))
            return Engine.TensorMultiply(stft, mask);

        var result = new Tensor<T>(stft._shape);
        for (int i = 0; i < Math.Min(stft.Length, mask.Length); i++)
            result[i] = NumOps.Multiply(stft[i], mask[i]);
        return result;
    }

    private Tensor<T> ComputeISTFT(Tensor<T> magnitude, int originalLength)
    {
        if (_lastPhase is null)
            throw new InvalidOperationException("Phase not available. Call ComputeSTFT first.");
        return _stft.InverseFromMagnitudeAndPhase(magnitude, _lastPhase, originalLength);
    }

    /// <summary>
    /// Reconstructs the enhanced waveform from the decoder's two decoupled heads, per
    /// Cao et al., INTERSPEECH 2022 (arXiv:2203.15149).
    /// </summary>
    /// <remarks>
    /// <para>The paper estimates the magnitude and the complex spectrogram in SEPARATE decoder
    /// branches and then jointly incorporates them:</para>
    /// <code>
    ///   S_mag = (mask ⊙ |X|) · e^{jθx}      // magnitude branch, noisy phase
    ///   S_cpx = Ŝr + j·Ŝi                    // complex branch, directly predicted
    ///   Ŝ     = S_mag + S_cpx                // jointly incorporated
    /// </code>
    /// <para>Summing the branches is what lets the model correct phase. The previous
    /// implementation only ever produced the magnitude branch and reconstructed with the
    /// unmodified NOISY phase, so phase was never enhanced at all — which discards the paper's
    /// central contribution.</para>
    /// <para>Decoder layout is [mask(F) | interleaved real/imag(2F)] per frame, matching
    /// <see cref="LayerHelper{T}.CreateDefaultCMGANLayers"/>.</para>
    /// </remarks>
    private Tensor<T> ReconstructFromDecoupledHeads(Tensor<T> noisyMagnitude, Tensor<T> decoded, int originalLength)
    {
        if (_lastPhase is null)
            throw new InvalidOperationException("Phase not available. Call ComputeSTFT first.");

        int f = _options.NumFreqBins;
        int frames = f > 0 ? noisyMagnitude.Length / f : 0;

        // Fall back to the magnitude-only path when the decoder output is not the expected
        // 3F-per-frame layout (e.g. an ONNX graph exporting only a mask).
        if (frames == 0 || decoded.Length < frames * 3 * f)
        {
            var maskedOnly = ApplyMask(noisyMagnitude, decoded);
            // noisyMagnitude is power-law compressed, so expand before the inverse transform.
            double fallbackExponent = _options.PowerLawCompressionExponent;
            if (Math.Abs(fallbackExponent) > ExponentComparisonTolerance)
                maskedOnly = ApplyPowerLaw(maskedOnly, 1.0 / fallbackExponent);
            return ComputeISTFT(maskedOnly, originalLength);
        }

        // Magnitude branch: mask the noisy magnitude, keep the noisy phase.
        var maskedMagnitude = new Tensor<T>(noisyMagnitude._shape);
        for (int t = 0; t < frames; t++)
        {
            int outBase = t * f;
            int decBase = t * 3 * f;
            for (int k = 0; k < f; k++)
                maskedMagnitude[outBase + k] = NumOps.Multiply(decoded[decBase + k], noisyMagnitude[outBase + k]);
        }

        var spectrogram = ShortTimeFourierTransform<T>.PolarToComplex(maskedMagnitude, _lastPhase);

        // Complex branch: add the directly-predicted real/imaginary pair.
        for (int t = 0; t < frames; t++)
        {
            int specBase = t * f;
            int decBase = t * 3 * f + f;
            for (int k = 0; k < f; k++)
            {
                var current = spectrogram[specBase + k];
                spectrogram[specBase + k] = new Complex<T>(
                    NumOps.Add(current.Real, decoded[decBase + (2 * k)]),
                    NumOps.Add(current.Imaginary, decoded[decBase + (2 * k) + 1]));
            }
        }

        // Undo the power-law compression applied in ComputeSTFT. Everything above operates in
        // the compressed domain (the mask scales a compressed magnitude and the complex head is
        // trained against compressed targets), so the magnitude must be expanded by 1/c before
        // the inverse transform while the phase angle is left untouched.
        double c = _options.PowerLawCompressionExponent;
        if (Math.Abs(c - 1.0) > ExponentComparisonTolerance &&
            Math.Abs(c) > ExponentComparisonTolerance)
        {
            double inverseExponent = 1.0 / c;
            for (int i = 0; i < spectrogram.Length; i++)
            {
                var bin = spectrogram[i];
                double re = NumOps.ToDouble(bin.Real);
                double im = NumOps.ToDouble(bin.Imaginary);
                double compressedMagnitude = Math.Sqrt((re * re) + (im * im));
                if (compressedMagnitude <= 0) continue;

                double gain = Math.Pow(compressedMagnitude, inverseExponent) / compressedMagnitude;
                spectrogram[i] = new Complex<T>(
                    NumOps.FromDouble(re * gain),
                    NumOps.FromDouble(im * gain));
            }
        }

        return _stft.Inverse(spectrogram, originalLength);
    }

    private static int NextPowerOfTwo(int v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CMGAN<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
