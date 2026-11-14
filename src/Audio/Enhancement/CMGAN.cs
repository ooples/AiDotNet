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
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private ShortTimeFourierTransform<T> _stft;
    [Scratch]
    private Tensor<T>? _lastPhase;
    [Buffer(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Conditional)]
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
        Options = _options;
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
        Options = _options;
        _useNativeMode = true;
        // The rate this model publishes on its own options. Built bare, the optimizer
        // would use its own default instead and LearningRate would be configuration that
        // nothing reads — the defect that diverged MusicFlamingo's training.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            { InitialLearningRate = _options.LearningRate });
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
            stft = AudioEngineOps.SubtractNoiseClampPrefix(Engine, stft, _noiseProfile);

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
            result = AudioEngineOps.BlendPrefix(Engine, result, audio, s, inv);
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
        _streamingBuffer.AddRange(audioChunk.ToArray());
        int frameSize = _options.FftSize;
        if (_streamingBuffer.Count < frameSize) return new Tensor<T>([0]);
        var frame = new Tensor<T>(_streamingBuffer.GetRange(0, frameSize).ToArray(), [frameSize]);
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

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input;
        foreach (var l in Layers) c = l.Forward(c);
        return c;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => ComputeSTFT(rawAudio);
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    /// <inheritdoc />
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

        return AudioEngineOps.SignedPower(Engine, values, exponent);
    }

    private Tensor<T> ApplyMask(Tensor<T> stft, Tensor<T> mask)
    {
        // Vectorised mask application: Engine.TensorMultiply is SIMD when
        // shapes match. Fall back to the prior min-length scalar loop only
        // for the (rare) mismatched-shape path.
        if (stft.Length == mask.Length && stft._shape.SequenceEqual(mask._shape))
            return Engine.TensorMultiply(stft, mask);

        return AudioEngineOps.MultiplyPrefix(Engine, stft, mask);
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
        // Decoder layout per frame: [mask (F) | interleaved real/imag pairs (2F)].
        int span = frames * f;
        var decodedRows = Engine.Reshape(
            Engine.TensorSlice(AudioEngineOps.Flatten(Engine, decoded), [0], [frames * 3 * f]), [frames, 3 * f]);
        var maskHead = Engine.Reshape(Engine.TensorSlice(decodedRows, [0, 0], [frames, f]), [span]);
        var pairs = Engine.Reshape(Engine.TensorSlice(decodedRows, [0, f], [frames, 2 * f]), [frames, f, 2]);
        var realHead = Engine.Reshape(Engine.TensorSlice(pairs, [0, 0, 0], [frames, f, 1]), [span]);
        var imagHead = Engine.Reshape(Engine.TensorSlice(pairs, [0, 0, 1], [frames, f, 1]), [span]);

        // Bins past frames * F (if any) stay zero, as before.
        var maskedMagnitude = Engine.Reshape(
            AudioEngineOps.FitLength(Engine, Engine.TensorMultiply(maskHead, AudioEngineOps.FitLength(Engine, noisyMagnitude, span)), noisyMagnitude.Length),
            noisyMagnitude.Shape.ToArray());

        var spectrogram = ShortTimeFourierTransform<T>.PolarToComplex(maskedMagnitude, _lastPhase);

        // Complex branch: add the directly-predicted real/imaginary pair.
        var real = AudioEngineOps.MapPrefix(Engine, Engine.TensorReal(spectrogram), span, head => Engine.TensorAdd(head, realHead));
        var imag = AudioEngineOps.MapPrefix(Engine, Engine.TensorImag(spectrogram), span, head => Engine.TensorAdd(head, imagHead));

        // Undo the power-law compression applied in ComputeSTFT. Everything above operates in
        // the compressed domain (the mask scales a compressed magnitude and the complex head is
        // trained against compressed targets), so the magnitude must be expanded by 1/c before
        // the inverse transform while the phase angle is left untouched.
        double c = _options.PowerLawCompressionExponent;
        if (Math.Abs(c - 1.0) > ExponentComparisonTolerance &&
            Math.Abs(c) > ExponentComparisonTolerance)
        {
            double inverseExponent = 1.0 / c;
            var compressedMagnitude = Engine.TensorSqrt(Engine.TensorAdd(Engine.TensorSquare(real), Engine.TensorSquare(imag)));
            var gain = Engine.TensorDivide(Engine.TensorPower(compressedMagnitude, NumOps.FromDouble(inverseExponent)), compressedMagnitude);

            // A zero-magnitude bin is left as is (its gain would be 0/0).
            var silent = Engine.TensorEquals(compressedMagnitude, NumOps.Zero);
            real = Engine.TensorWhere(silent, real, Engine.TensorMultiply(real, gain));
            imag = Engine.TensorWhere(silent, imag, Engine.TensorMultiply(imag, gain));
        }

        return _stft.Inverse(Engine.TensorComplex(real, imag), originalLength);
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

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
