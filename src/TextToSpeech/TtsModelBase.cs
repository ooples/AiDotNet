using System.Collections.Generic;
// AiDotNet.Attributes is REQUIRED for [TensorLayout] to bind to the right type: two other Tensors
// namespaces declare a TensorLayout, and without this using the attribute silently resolves to one
// of those and the contract is never seen.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Validation;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// Base class for text-to-speech neural networks that can operate in both ONNX inference and native training modes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class extends <see cref="NeuralNetworkBase{T}"/> to provide TTS-specific functionality
/// while maintaining full integration with the AiDotNet neural network infrastructure.
/// </para>
/// <para>
/// <b>For Beginners:</b> Text-to-speech models convert written text into spoken audio. This base class provides:
///
/// - Support for pre-trained ONNX models (fast inference with existing models)
/// - Full training capability from scratch (like other neural networks)
/// - Audio preprocessing utilities (mel-spectrogram computation, normalization)
/// - Text encoding utilities (phoneme/token conversion)
///
/// You can use this class in two ways:
/// 1. Load a pre-trained ONNX model for quick inference
/// 2. Build and train a new model from scratch
/// </para>
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "The encoded text or conditioning the layer stack consumes.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output,
    Note = "One frame per input position: the input's second axis is carried through as TIME and the "
         + "width is appended. Models whose Predict ends somewhere else state their own width through "
         + "OutputFeatureWidth, or decline by leaving it at 0.")]
public abstract partial class TtsModelBase<T> : NeuralNetworkBase<T>, IShapeContract
{
    /// <summary>
    /// The width of this model's <c>Predict</c> output, or 0 for "not stated".
    /// </summary>
    /// <remarks>
    /// <para>
    /// The shape here is the LAYER STACK'S, not the duration-predicted synthesis path that
    /// <c>TextToMel</c> drives: every <c>PredictCore</c> in this family is a plain fold over
    /// <c>Layers</c> - Tacotron2, FastSpeech2 and GlowTTS are all literally
    /// <c>foreach (var l in Layers) c = l.Forward(c)</c>.
    /// </para>
    /// <para>
    /// DEFAULTS TO 0 - "not stated" - rather than to <see cref="MelChannels"/>, and that was measured
    /// rather than chosen. Defaulting to MelChannels gave 18 agreed and 80 DISAGREED: the family
    /// splits into acoustic models that really do end at a mel width, and codec models that end at a
    /// token vocabulary (192, 626, 4096, 8192, 12288, 65536). A default right for 18 and wrong for 80
    /// is worse than no default, because the 80 then carry a confident false claim instead of an
    /// honest silence - and the sweep would report the family as broken rather than as unfinished.
    /// </para>
    /// <para>
    /// VIRTUAL AND DEFAULTED, not abstract: adding an abstract member to a public base breaks every
    /// external subclass, and a 0 lets a model that ends somewhere else opt out honestly instead of
    /// carrying a wrong width. The vocoders do exactly that - <c>VocoderBase</c> overrides
    /// <c>OutputAxesFor</c> outright, because a waveform is not a mel frame.
    /// </para>
    /// </remarks>
    protected virtual int OutputFeatureWidth => 0;

    /// <summary>
    /// The TTS family's output law: <c>[Batch, Time, OutputFeatureWidth]</c>, where Time is the
    /// input's second axis carried through.
    /// </summary>
    /// <remarks>
    /// MEASURED, and the first version of this was wrong in exactly the way a rank assumption usually
    /// is. It declared <c>[Batch, Width]</c> - rank 2 in, rank 2 out - and the sweep returned 86
    /// DISAGREEMENTS, every one of the form "in [1,64] contract says [1,80] but Predict returned
    /// [1,64,80]". The width was right; the RANK was not. These models emit one frame per input
    /// position, so the input axis survives as TIME and the width is appended to it. Nothing about
    /// "rank 2 in" implies "rank 2 out", and assuming so is what made the audio family's six
    /// rank-mismatched models look like width errors too.
    /// </remarks>
    public virtual IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        int width = OutputFeatureWidth;
        if (inputRank != 2 || width <= 0) return null;
        return
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Features)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(width)),
        ];
    }

    /// <summary>
    /// Gets the audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; protected set; } = 22050;

    /// <summary>
    /// Gets the number of mel-spectrogram frequency channels.
    /// </summary>
    public int MelChannels { get; protected set; } = 80;

    /// <summary>
    /// Gets the hop size in audio samples for mel-spectrogram computation.
    /// </summary>
    public int HopSize { get; protected set; } = 256;

    /// <summary>
    /// Gets the model's hidden dimension.
    /// </summary>
    public int HiddenDim { get; protected set; } = 256;

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    public bool IsOnnxMode =>
        OnnxEncoder is not null || OnnxDecoder is not null || OnnxModel is not null;

    /// <summary>
    /// Gets or sets the ONNX encoder model (for two-stage architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxEncoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX decoder model (for two-stage architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxDecoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX model (for single-model architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxModel { get; set; }

    /// <summary>
    /// Initializes a new instance of the TtsModelBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, a default MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected TtsModelBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0
    )
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm) { }

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => !IsOnnxMode;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _vocoderBaseOptimizer;

    /// <summary>
    /// Vocoder generators (those implementing <see cref="AiDotNet.TextToSpeech.Interfaces.IVocoder{T}"/>)
    /// train with AMSGrad rather than plain Adam. Their MRF / dilated-conv loss
    /// surfaces are bumpy enough that plain Adam's effective step can grow as the
    /// second-moment estimate shrinks near convergence, letting long training
    /// drift back up off the minimum. AMSGrad (Reddi et al. 2018; equivalent to
    /// <c>torch.optim.Adam(amsgrad=True)</c>) keeps a non-decreasing second-moment
    /// denominator, which bounds that drift. It is a strict convergence-stability
    /// improvement and does not affect inference. Non-vocoder TTS models (acoustic
    /// / end-to-end) keep the default base optimizer.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
    {
        if (
            this is AiDotNet.TextToSpeech.Interfaces.IVocoder<T>
            || this is AiDotNet.TextToSpeech.Interfaces.IEndToEndTts<T>
        )
        {
            return _vocoderBaseOptimizer ??= new AiDotNet.Optimizers.AdamOptimizer<
                T,
                Tensor<T>,
                Tensor<T>
            >(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { UseAMSGrad = true });
        }
        return base.GetOrCreateBaseOptimizer();
    }

    /// <summary>
    /// Trainable sub-networks a model runs outside its sequential <see cref="NeuralNetworkBase{T}.Layers"/> stack:
    /// a duration predictor reading the raw text (AlignTTS), an alignment network used only in training, and the like.
    /// </summary>
    /// <remarks>Layers added here take part in parameter counting, training, serialization and cloning exactly as
    /// stack layers do; a model declares them rather than wiring each of those surfaces itself.</remarks>
    protected readonly List<LayerBase<T>> ComponentLayers = new();

    /// <summary>Surfaces <see cref="ComponentLayers"/> to the parameter walk.</summary>
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
    {
        foreach (var layer in base.GetExtraTrainableLayers())
            yield return layer;
        foreach (var layer in ComponentLayers)
            yield return layer;
    }

    private int _encoderLayerCount;

    /// <summary>
    /// The number of leading <see cref="NeuralNetworkBase{T}.Layers"/> that form the text encoder.
    /// </summary>
    /// <remarks>
    /// Synthesis runs the encoder, then the model's own step (variance adaptor, length regulator, alignment), then
    /// the decoder. The split is recorded by <see cref="AddEncoderDecoderLayers"/> from the layers actually built.
    /// Thirteen acoustic models used to recompute it from a per-block layer count; the factory had since changed to
    /// one layer per block, so the computed boundary (25 for a 10-layer FastSpeech 2) ran past the end of the stack
    /// and every call to <c>Synthesize</c> threw.
    /// </remarks>
    protected int EncoderLayerCount => _encoderLayerCount;

    /// <summary>
    /// Adds an encoder and a decoder to the layer stack and records where the encoder ends.
    /// </summary>
    protected void AddEncoderDecoderLayers(IEnumerable<ILayer<T>> encoderLayers, IEnumerable<ILayer<T>> decoderLayers)
    {
        Guard.NotNull(encoderLayers);
        Guard.NotNull(decoderLayers);
        Layers.AddRange(encoderLayers);
        _encoderLayerCount = Layers.Count;
        Layers.AddRange(decoderLayers);
    }

    /// <summary>
    /// Adds caller-supplied layers, which carry no declared encoder/decoder split.
    /// </summary>
    /// <remarks>The first half is taken as the encoder, as before; supply the split through
    /// <see cref="AddEncoderDecoderLayers"/> wherever it is known.</remarks>
    protected void AddUnsplitLayers(IEnumerable<ILayer<T>> layers)
    {
        Guard.NotNull(layers);
        Layers.AddRange(layers);
        _encoderLayerCount = Layers.Count / 2;
    }

    /// <summary>Runs the encoder layers (<c>Layers[0 .. EncoderLayerCount)</c>).</summary>
    protected Tensor<T> RunEncoder(Tensor<T> input)
    {
        var x = input;
        for (int i = 0; i < _encoderLayerCount; i++)
            x = Layers[i].Forward(x);
        return x;
    }

    /// <summary>Runs the decoder layers (<c>Layers[EncoderLayerCount ..]</c>).</summary>
    protected Tensor<T> RunDecoder(Tensor<T> hidden)
    {
        var x = hidden;
        for (int i = _encoderLayerCount; i < Layers.Count; i++)
            x = Layers[i].Forward(x);
        return x;
    }

    /// <summary>Runs <c>Layers[start .. end)</c> in order.</summary>
    protected Tensor<T> RunLayers(Tensor<T> input, int start, int end)
    {
        if (start < 0 || end > Layers.Count || start > end)
            throw new ArgumentOutOfRangeException(nameof(start), $"Layer range [{start}, {end}) is outside 0..{Layers.Count}.");
        var x = input;
        for (int i = start; i < end; i++)
            x = Layers[i].Forward(x);
        return x;
    }

    /// <summary>
    /// Converts text to the token tensor the model's first layer reads, one token per character.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The input domain the network declares decides the encoding. A model whose first layer is an embedding
    /// declares integer token indices in <c>[min, max)</c>; each character becomes <c>min + code % (max - min)</c>,
    /// so every token is a valid row of that embedding (character input, which FastSpeech 2 notes its method
    /// applies to directly, Ren et al. 2021 §2.2 fn. 2). A model with a continuous front end gets character codes
    /// scaled by 1/128, the encoding these models used before tokenization moved here.
    /// </para>
    /// <para>Text longer than the model's maximum text length is truncated.</para>
    /// </remarks>
    protected virtual Tensor<T> PreprocessText(string text)
    {
        Guard.NotNull(text);
        int length = Math.Min(text.Length, MaxTextTokens);
        if (length <= 0)
            throw new ArgumentException("Text must contain at least one character.", nameof(text));

        var tokens = new Tensor<T>(new[] { length });
        var domain = GetInputDomain(new[] { length });
        if (domain.IsResolved && domain.Kind == LayerInputDomainKind.IntegerIndices)
        {
            int range = Math.Max(1, domain.MaxExclusive - domain.MinInclusive);
            for (int i = 0; i < length; i++)
                tokens[i] = NumOps.FromDouble(domain.MinInclusive + text[i] % range);
        }
        else
        {
            for (int i = 0; i < length; i++)
                tokens[i] = NumOps.FromDouble(text[i] / 128.0);
        }
        return tokens;
    }

    /// <summary>Longest text, in tokens, the model reads.</summary>
    protected virtual int MaxTextTokens =>
        this is AiDotNet.TextToSpeech.Interfaces.ITtsModel<T> tts && tts.MaxTextLength > 0 ? tts.MaxTextLength : int.MaxValue;

    /// <summary>
    /// Synthesizes speech (or the model's acoustic output) from text: tokenize, run the network, postprocess.
    /// </summary>
    /// <remarks>
    /// The network forward is <see cref="NeuralNetworkBase{T}.Predict"/>, the same graph training updates, so what a
    /// user hears is what was trained. ONNX models run their loaded graph instead.
    /// </remarks>
    public virtual Tensor<T> Synthesize(string text)
    {
        var tokens = PreprocessText(text);
        var output = IsOnnxMode && OnnxModel is not null ? OnnxModel.Run(tokens) : Predict(tokens);
        return PostprocessAudio(output);
    }

    /// <summary>
    /// Supervision this model's paper trains on that a token/mel pair does not carry and that cannot be derived from
    /// the recording. A plain token/mel <c>Train</c> call refuses to train such a model.
    /// </summary>
    protected virtual TtsSupervision RequiredSupervision => TtsSupervision.None;

    /// <summary>
    /// Supervision outside the recording that this model trains on; anything other than
    /// <see cref="TtsSupervision.None"/> means training goes through <see cref="Train(TtsTrainingSample{T})"/>.
    /// </summary>
    public TtsSupervision TrainingSupervision => RequiredSupervision;

    /// <summary>Trains on one utterance with the supervision its paper uses.</summary>
    /// <returns>The training loss of the step.</returns>
    public T Train(TtsTrainingSample<T> sample)
    {
        Guard.NotNull(sample);
        if ((RequiredSupervision & TtsSupervision.Durations) != 0 && sample.Durations is null)
            throw new ArgumentException(
                $"{GetType().Name} trains on per-token durations from a forced alignment; set {nameof(sample.Durations)}.",
                nameof(sample));
        if ((RequiredSupervision & TtsSupervision.SpeakerId) != 0 && sample.SpeakerId is null)
            throw new ArgumentException(
                $"{GetType().Name} trains a speaker table; set {nameof(sample.SpeakerId)}.", nameof(sample));
        if ((RequiredSupervision & TtsSupervision.Recording) != 0 && sample.Audio is null && sample.LinearSpectrogram is null)
            throw new ArgumentException(
                $"{GetType().Name} trains on the recording's linear spectrogram; set {nameof(sample.Audio)} or {nameof(sample.LinearSpectrogram)}.",
                nameof(sample));
        if ((RequiredSupervision & TtsSupervision.SpeakerReference) != 0 && sample.SpeakerReference is null)
            throw new ArgumentException(
                $"{GetType().Name} trains on a reference recording of the speaker; set {nameof(sample.SpeakerReference)}.",
                nameof(sample));
        if ((RequiredSupervision & TtsSupervision.LanguageId) != 0 && sample.LanguageId is null)
            throw new ArgumentException(
                $"{GetType().Name} trains a language table; set {nameof(sample.LanguageId)}.", nameof(sample));
        return TrainOnSample(sample);
    }

    /// <summary>
    /// The voice synthesis and prediction use, for models whose paper defines inference relative to a speaker or a
    /// reference recording (see <see cref="SynthesisVoiceRequirement"/>).
    /// </summary>
    public TtsVoice<T>? Voice { get; set; }

    /// <summary>
    /// What <see cref="Voice"/> must carry for this model to synthesize: <see cref="TtsSupervision.SpeakerId"/>,
    /// <see cref="TtsSupervision.SpeakerReference"/>, or <see cref="TtsSupervision.None"/> for a model that
    /// synthesizes from text alone.
    /// </summary>
    protected virtual TtsSupervision RequiredVoice => TtsSupervision.None;

    /// <summary>What a voice must carry for this model to synthesize; <see cref="TtsSupervision.None"/> when text
    /// alone determines the output.</summary>
    public TtsSupervision SynthesisVoiceRequirement => RequiredVoice;

    /// <summary>
    /// Returns <see cref="Voice"/>, or throws naming what is missing when the model's paper needs a voice to
    /// synthesize and none (or an incomplete one) is set.
    /// </summary>
    protected TtsVoice<T> RequireVoice()
    {
        var voice = Voice ?? throw new InvalidOperationException(
            $"{GetType().Name} synthesizes in a given voice ({RequiredVoice}); set {nameof(Voice)} first.");
        if ((RequiredVoice & TtsSupervision.SpeakerReference) != 0 && voice.Reference is null)
            throw new InvalidOperationException(
                $"{GetType().Name} reads a reference recording of the speaker; set {nameof(TtsVoice<T>.Reference)} on {nameof(Voice)}.");
        if ((RequiredVoice & TtsSupervision.ReferenceRecording) != 0 && voice.ReferenceAudio is null)
            throw new InvalidOperationException(
                $"{GetType().Name} reads a recording of the speaker; set {nameof(TtsVoice<T>.ReferenceAudio)} on {nameof(Voice)}.");
        if ((RequiredVoice & TtsSupervision.LanguageId) != 0 && voice.LanguageId is null)
            throw new InvalidOperationException(
                $"{GetType().Name} speaks one of several languages; set {nameof(TtsVoice<T>.LanguageId)} on {nameof(Voice)}.");
        return voice;
    }

    /// <summary>
    /// Evaluates the training objective on <paramref name="sample"/> without updating the model, for models whose
    /// training goes through <see cref="Train(TtsTrainingSample{T})"/>.
    /// </summary>
    public virtual T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
        => throw new NotSupportedException($"{GetType().Name} does not define a sample-level training objective.");

    /// <summary>Trains on several utterances, one step each.</summary>
    public void Train(IEnumerable<TtsTrainingSample<T>> samples)
    {
        Guard.NotNull(samples);
        foreach (var sample in samples) Train(sample);
    }

    /// <summary>
    /// One training step on <paramref name="sample"/>. The default trains the token/mel pair through the ordinary
    /// <c>Train</c>, deriving the mel from the recording when it is absent; models with richer objectives override it.
    /// </summary>
    protected virtual T TrainOnSample(TtsTrainingSample<T> sample)
    {
        var targets = DeriveAcousticTargets(sample);
        Train(sample.Tokens, targets.Mel);
        return LastLoss ?? NumOps.Zero;
    }

    /// <summary>
    /// Throws when this model needs supervision that a token/mel pair cannot provide. Models call it at the top of
    /// their token/mel <c>Train</c>.
    /// </summary>
    protected void ThrowIfTokenMelTrainingUnsupported()
    {
        if (RequiredSupervision != TtsSupervision.None)
            throw new NotSupportedException(
                $"{GetType().Name} trains on {RequiredSupervision} that a token/mel pair does not carry; " +
                "use Train(TtsTrainingSample) instead.");
    }

    /// <summary>
    /// Completes an utterance's acoustic targets from its recording, as the papers derive them: the Tacotron 2 mel
    /// spectrogram, WORLD F0 at the mel frame period, and the STFT frame energy. Supplied values are kept. Pitch and
    /// energy are aligned to the mel frame count.
    /// </summary>
    protected AcousticTargets<T> DeriveAcousticTargets(TtsTrainingSample<T> sample)
    {
        Guard.NotNull(sample);
        var spectrogram = new TacotronSpectrogram(SampleRate, TargetFftSize, HopSize, TargetWindowSize, MelChannels);
        double[]? audio = sample.Audio is null ? null : ToDoubles(sample.Audio);
        double[,]? magnitude = audio is null ? null : spectrogram.Magnitude(PreEmphasize(audio));

        Tensor<T> mel;
        if (sample.Mel is not null)
        {
            mel = sample.Mel;
        }
        else
        {
            if (magnitude is null)
                throw new ArgumentException("Supply the recording or its mel spectrogram.", nameof(sample));
            var logMel = spectrogram.LogMel(magnitude);
            mel = new Tensor<T>(new[] { logMel.GetLength(0), logMel.GetLength(1) });
            for (int f = 0; f < logMel.GetLength(0); f++)
                for (int m = 0; m < logMel.GetLength(1); m++)
                    mel[f, m] = NumOps.FromDouble(logMel[f, m]);
        }
        int frames = mel.Shape[0];

        double[]? pitch = sample.Pitch;
        if (pitch is null && audio is not null)
            pitch = new AiDotNet.Audio.Pitch.WorldPitchDetector<T>(SampleRate).EstimateF0(audio, spectrogram.FramePeriodMs).F0;
        double[]? energy = sample.Energy;
        if (energy is null && magnitude is not null)
            energy = spectrogram.Energy(magnitude);

        Tensor<T>? linear = sample.LinearSpectrogram;
        if (linear is null && magnitude is not null)
        {
            int rows = magnitude.GetLength(0), bins = magnitude.GetLength(1);
            linear = new Tensor<T>(new[] { rows, bins });
            for (int f = 0; f < rows; f++)
                for (int k = 0; k < bins; k++)
                    linear[f, k] = NumOps.FromDouble(Math.Log(Math.Max(spectrogram.ClipValue, magnitude[f, k])));
        }

        return new AcousticTargets<T>(mel, frames,
            pitch is null ? null : AlignToFrames(pitch, frames),
            energy is null ? null : AlignToFrames(energy, frames),
            linear);
    }

    /// <summary>Analysis window length, in samples, of the spectrograms the targets are computed with.</summary>
    protected virtual int TargetWindowSize => TargetFftSize;

    /// <summary>Pre-emphasis coefficient applied to the recording before its spectrograms (Tacotron: 0.97); 0 for none.
    /// Pitch is estimated from the recording as recorded.</summary>
    protected virtual double PreEmphasis => 0.0;

    /// <summary>Bins of the linear spectrogram target, <c>fftSize / 2 + 1</c>.</summary>
    public int LinearSpectrogramBins => TargetFftSize / 2 + 1;

    private double[] PreEmphasize(double[] audio)
    {
        double k = PreEmphasis;
        if (k == 0.0)
            return audio;
        var y = new double[audio.Length];
        if (audio.Length > 0) y[0] = audio[0];
        for (int n = 1; n < audio.Length; n++) y[n] = audio[n] - k * audio[n - 1];
        return y;
    }

    /// <summary>FFT and window size used to derive targets from a recording (Tacotron 2: 1024).</summary>
    protected virtual int TargetFftSize => 1024;

    private static double[] AlignToFrames(double[] values, int frames)
    {
        if (values.Length == frames) return values;
        // The pitch tracker and the STFT can disagree by a frame at the end; repeat or drop the last value.
        var aligned = new double[frames];
        for (int i = 0; i < frames; i++) aligned[i] = values[Math.Min(i, values.Length - 1)];
        return aligned;
    }

    private double[] ToDoubles(Tensor<T> tensor)
    {
        var span = tensor.AsSpan();
        var values = new double[span.Length];
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.ToDouble(span[i]);
        return values;
    }


    /// <summary>
    /// Postprocesses model output into the final audio format.
    /// </summary>
    /// <param name="modelOutput">Raw output from the model.</param>
    /// <returns>Postprocessed audio tensor.</returns>
    protected abstract Tensor<T> PostprocessAudio(Tensor<T> modelOutput);

    /// <summary>
    /// Normalizes a mel-spectrogram tensor.
    /// </summary>
    /// <param name="mel">Mel-spectrogram tensor.</param>
    /// <param name="minLevel">Minimum amplitude level in dB (default: -100).</param>
    /// <param name="refLevel">Reference amplitude level in dB (default: 20).</param>
    /// <returns>Normalized mel-spectrogram tensor.</returns>
    protected Tensor<T> NormalizeMel(
        Tensor<T> mel,
        double minLevel = -100.0,
        double refLevel = 20.0
    )
    {
        var result = new Tensor<T>(mel._shape);
        double range = refLevel - minLevel;
        if (Math.Abs(range) < 1e-10)
            range = 1.0;

        for (int i = 0; i < mel.Length; i++)
        {
            double val = NumOps.ToDouble(mel[i]);
            double normalized = (val - minLevel) / range;
            normalized = Math.Max(0.0, Math.Min(1.0, normalized));
            result[i] = NumOps.FromDouble(normalized);
        }

        return result;
    }

    /// <summary>
    /// Applies GELU activation function element-wise.
    /// </summary>
    /// <param name="x">Input value.</param>
    /// <returns>GELU-activated value.</returns>
    protected static double Gelu(double x)
    {
        return x * 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
    }

    /// <summary>
    /// Applies softmax to convert logits to probabilities.
    /// </summary>
    /// <param name="logits">Raw scores.</param>
    /// <returns>Probabilities that sum to 1.</returns>
    protected Tensor<T> Softmax(Tensor<T> logits)
    {
        double maxVal = double.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            double v = NumOps.ToDouble(logits[i]);
            if (v > maxVal)
                maxVal = v;
        }

        var result = new Tensor<T>(logits._shape);
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double v = Math.Exp(NumOps.ToDouble(logits[i]) - maxVal);
            result[i] = NumOps.FromDouble(v);
            sum += v;
        }

        if (sum > 1e-8)
        {
            for (int i = 0; i < result.Length; i++)
                result[i] = NumOps.FromDouble(NumOps.ToDouble(result[i]) / sum);
        }

        return result;
    }

    /// <summary>
    /// L2-normalizes a tensor.
    /// </summary>
    /// <param name="tensor">Tensor to normalize.</param>
    /// <returns>Unit-normalized tensor.</returns>
    protected Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        double norm = 0;
        for (int i = 0; i < tensor.Length; i++)
        {
            double v = NumOps.ToDouble(tensor[i]);
            norm += v * v;
        }

        norm = Math.Sqrt(norm);
        if (norm < 1e-8)
            return tensor;

        var result = new Tensor<T>(tensor._shape);
        for (int i = 0; i < tensor.Length; i++)
            result[i] = NumOps.FromDouble(NumOps.ToDouble(tensor[i]) / norm);

        return result;
    }

    /// <summary>
    /// Gets the default loss function for this model.
    /// </summary>
    public override ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Disposes of resources used by this model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxEncoder?.Dispose();
            OnnxDecoder?.Dispose();
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
