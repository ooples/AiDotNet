using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

using System.Linq;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// CLAP (Contrastive Language-Audio Pretraining) — a dual-encoder neural network
/// that learns to align audio and text representations in a shared embedding
/// space via a contrastive objective.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLAP (Wu et al. 2023) trains an audio encoder and a text encoder so that
/// matching (audio, caption) pairs produce nearby embeddings and unrelated
/// pairs produce distant embeddings. The network is LAION-CLAP's (reference <c>LAION-AI/CLAP</c>):
/// </para>
/// <list type="bullet">
/// <item><b>Audio</b>: HTS-AT (Chen et al. 2022) — BatchNorm over the mel bins, the spectrogram folded into a
/// square image, a patch embedding, four hierarchical Swin stages (Liu et al. 2021) joined by patch merging, and
/// the mean over the final tokens. With <see cref="CLAPModelOptions.EnableFeatureFusion"/>, clips longer than
/// 10 s are fused from a global view and three local crops (Wu et al. 2023 §3.3).</item>
/// <item><b>Text</b>: RoBERTa (Liu et al. 2019) and its pooled <c>&lt;s&gt;</c> output.</item>
/// <item><b>Projections</b>: <c>Linear → ReLU → Linear</c> into the joint space, then L2 normalization.</item>
/// </list>
/// <para>
/// A learnable temperature τ scales the cosine-similarity logits during contrastive training (CLIP / CLAP
/// convention). Published LAION checkpoints load through <see cref="LoadLaionWeights"/>.
/// </para>
/// <para>
/// <b>Capabilities</b>:
/// <list type="bullet">
/// <item><description>Zero-shot audio classification with text prompts</description></item>
/// <item><description>Audio-to-text and text-to-audio retrieval</description></item>
/// <item><description>Semantic audio fingerprinting via the projection head</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Reference:</b> Wu, Y. et al. (2023), "Large-Scale Contrastive Language-Audio
/// Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation", ICASSP.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.EmbeddingModel)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation",
    "https://doi.org/10.1109/ICASSP49357.2023.10095969",
    Year = 2023,
    Authors = "Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov")]
// The inherited [Batch, Features] input is CORRECT - CLAP takes raw audio, per EncodeAudio's own
// contract of "[samples] or [batch, samples]". What the declared-input check saw is the log-mel front
// end: EncodeAudio fits the clip and computes the log-mel views BEFORE folding the stack, so Layers[0] - HTS-AT's
// bn0 over the mel bins, ahead of the image fold, the patch embedding and the Swin stages - receives
// [batch, views, frames, mels] and never the waveform. Both declarations describe different tensors, so the exemption is the honest fix here; an
// override claiming a spectrogram input would advertise a shape Predict does not accept.
[PreprocessesInput("PreprocessAudio converts the waveform to a log-mel spectrogram image before the "
    + "audio stack runs, so Layers[0] never sees the raw samples - see EncodeAudio.")]
[StackInputLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true)]
public partial class CLAPModel<T> : AudioNeuralNetworkBase<T>, IAudioFingerprinter<T>, ITrainingObjectiveProvider<T>
{

    /// <inheritdoc />
    /// <remarks>
    /// Measured: <c>PredictCore</c> delegates to <c>EncodeAudio</c>, which folds the AUDIO stack
    /// (<c>Layers</c>) and L2-normalizes - a normalization that rescales, never reshapes.
    /// <c>CreateDefaultCLAPAudioEncoderLayers</c> ends with the <c>Linear → ReLU → Linear</c> audio projection,
    /// whose output width is <c>_options.ProjectionDim</c> (512 in the paper). Not the HTS-AT width
    /// (<c>AudioEmbedDim · 2^(stages−1)</c>, 768), which the projection reads, and not
    /// <c>VocabSize</c>, which belongs to the TEXT stack in <c>TextEncoderLayers</c> - Predict never
    /// touches that stack.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.ProjectionDim;

    private readonly CLAPModelOptions _options;
    private readonly bool _useNativeMode;
    // Captured ONNX-mode constructor inputs. Preserved so CreateNewInstance
    // can rebuild an ONNX-mode clone with the same encoder weights instead
    // of silently downgrading to native (random-init) mode.
    private readonly string? _audioEncoderPath;
    private readonly string? _textEncoderPath;

    // Trainable temperature parameter (stored in log space). Gradients flow
    // through the tape via Engine ops; the optimizer updates this alongside
    // the rest of the network.
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _logTemperature = null!;

    // The waveform -> log-mel pipeline (clip fitting, feature-fusion views, filterbank).
    private readonly ClapAudioFrontEnd<T> _frontEnd;

    // Random crops of long clips. Training draws from one advancing stream; inference reseeds per call so the
    // same audio always gives the same embedding.
    private readonly Random _trainingRandom;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <inheritdoc/>
    public string Name => _useNativeMode ? "CLAP-Native" : "CLAP-ONNX";

    /// <inheritdoc/>
    public int FingerprintLength => _options.ProjectionDim;

    #region Constructors

    /// <summary>
    /// Initializes a new instance of <see cref="CLAPModel{T}"/> in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">Architecture descriptor (used for I/O shape metadata).</param>
    /// <param name="audioEncoderPath">Path to the ONNX audio encoder model.</param>
    /// <param name="textEncoderPath">Optional path to the ONNX text encoder model.</param>
    /// <param name="options">Optional CLAP configuration (defaults match the published paper).</param>
    public CLAPModel(
        NeuralNetworkArchitecture<T> architecture,
        string audioEncoderPath,
        CLAPModelOptions? options = null)
        : base(architecture: architecture)
    {
        _options = options ?? new CLAPModelOptions();
        if (string.IsNullOrWhiteSpace(audioEncoderPath))
            throw new ArgumentException("Audio encoder path is required for ONNX mode.", nameof(audioEncoderPath));
        if (!File.Exists(audioEncoderPath))
            throw new FileNotFoundException($"Audio encoder ONNX model not found: {audioEncoderPath}", audioEncoderPath);

        // Validated after the audio-encoder path check so a missing model file reports itself
        // as FileNotFoundException rather than being pre-empted by the options. It must still
        // precede the text-encoder check below, which reads _options.TextEncoderPath and so
        // needs the options to be valid first.
        _options.Validate();
        _frontEnd = new ClapAudioFrontEnd<T>(_options);
        _trainingRandom = RandomHelper.CreateSeededRandom(_options.Seed ?? 0);

        SampleRate = _options.SampleRate;
        _useNativeMode = false;
        _audioEncoderPath = audioEncoderPath;
        _textEncoderPath = _options.TextEncoderPath;
        OnnxEncoder = new OnnxModel<T>(audioEncoderPath);
        // Fail fast when a text-encoder path is provided but invalid — the
        // earlier "try-and-skip" path silently dropped a mistyped path and
        // only surfaced as a NotSupportedException deep inside EncodeText().
        // Pattern-matches into a non-null local for net471 nullable-flow.
        if (_options.TextEncoderPath is { Length: > 0 } pathLocal
            && !string.IsNullOrWhiteSpace(pathLocal))
        {
            if (!File.Exists(pathLocal))
                throw new FileNotFoundException(
                    $"Text encoder ONNX model not found: {pathLocal}",
                    pathLocal);
            OnnxDecoder = new OnnxModel<T>(pathLocal);
        }

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of <see cref="CLAPModel{T}"/> in native training / inference mode.
    /// </summary>
    /// <param name="architecture">Architecture descriptor. If <c>Architecture.Layers</c> is populated,
    /// those layers replace the default audio + text encoder stacks.</param>
    /// <param name="options">Optional CLAP configuration (defaults match the published paper).</param>
    public CLAPModel(
        NeuralNetworkArchitecture<T> architecture,
        CLAPModelOptions? options = null)
        : base(architecture: architecture)
    {
        _options = options ?? new CLAPModelOptions();
        _options.Validate();
        _frontEnd = new ClapAudioFrontEnd<T>(_options);
        _trainingRandom = RandomHelper.CreateSeededRandom(_options.Seed ?? 0);
        SampleRate = _options.SampleRate;
        _useNativeMode = true;

        InitializeLayers();
    }

    #endregion

    #region Layer Construction (Golden-Standard Pattern)

    /// <summary>
    /// Initializes the neural network layers following the codebase's golden
    /// dual-stream pattern. Prefers an <see cref="AudioTextDualStreamArchitecture{T}"/>
    /// (modality-specific custom layers per encoder), then falls back to the
    /// paper-faithful <see cref="LayerHelper{T}"/> factories.
    /// </summary>
    /// <remarks>
    /// Audio encoder layers live in the inherited <see cref="NeuralNetworkBase{T}.Layers"/>
    /// list so the standard tape-training and parameter-iteration infrastructure
    /// walks them as the primary stream. Text encoder layers live in
    /// <see cref="AudioNeuralNetworkBase{T}.TextEncoderLayers"/>; subclasses
    /// (here) walk that collection explicitly inside <see cref="EncodeText(Tensor{T})"/>.
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture is AudioTextDualStreamArchitecture<T> dual)
        {
            // Production path: caller provided custom layer stacks for both
            // encoders via the proper dual-stream architecture descriptor.
            Layers.AddRange(dual.AudioLayers);
            TextEncoderLayers.AddRange(dual.TextLayers);
        }
        else if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // Legacy / single-stack path: user supplied only one Layers list.
            // We treat that as the audio encoder and use the LayerHelper
            // default for the text side, so the model is still functional.
            // Callers wanting full control of BOTH stacks should switch to
            // AudioTextDualStreamArchitecture.
            Layers.AddRange(Architecture.Layers);
            TextEncoderLayers.AddRange(TextTower(_options));
        }
        else
        {
            // Default both stacks from the paper-faithful LayerHelper factories.
            Layers.AddRange(AudioTower(_options));
            TextEncoderLayers.AddRange(TextTower(_options));
        }

        // Learnable temperature τ in log space (CLIP / CLAP convention).
        // Initialised so exp(_logTemperature) = 1/InitialTemperature; the
        // contrastive loss uses logits·exp(_logTemperature).
        _logTemperature = new Tensor<T>([1]);
        _logTemperature[0] = NumOps.FromDouble(Math.Log(1.0 / _options.InitialTemperature));
    }

    /// <summary>The audio tower: HTS-AT (bn0 over the mel bins, the image fold, patch embedding, four Swin stages,
    /// the token mean) and the Linear-ReLU-Linear projection into the joint space.</summary>
    /// <remarks>Shared with the classification adapter <see cref="AiDotNet.Audio.Classification.CLAP{T}"/>, so CLAP
    /// has one network.</remarks>
    internal static IEnumerable<ILayer<T>> AudioTower(CLAPModelOptions o) =>
        LayerHelper<T>.CreateDefaultCLAPAudioEncoderLayers(
            numMelBands: o.NumMelBands,
            specSize: o.SpecSize,
            patchSize: o.AudioPatchSize,
            embedDim: o.AudioEmbedDim,
            stageDepths: o.AudioStageDepths,
            stageHeads: o.AudioStageHeads,
            windowSize: o.SwinWindowSize,
            mlpRatio: o.AudioMlpRatio,
            dropPathRate: o.AudioDropPathRate,
            featureFusion: o.EnableFeatureFusion,
            projectionDim: o.ProjectionDim,
            specAugmentTimeDropWidth: o.SpecAugmentTimeDropWidth,
            specAugmentTimeStripes: o.SpecAugmentTimeStripes,
            specAugmentFreqDropWidth: o.SpecAugmentFreqDropWidth,
            specAugmentFreqStripes: o.SpecAugmentFreqStripes);

    /// <summary>The text tower: RoBERTa and its pooled &lt;s&gt; output through the projection.</summary>
    internal static IEnumerable<ILayer<T>> TextTower(CLAPModelOptions o) =>
        LayerHelper<T>.CreateDefaultCLAPTextEncoderLayers(
            vocabSize: o.VocabSize,
            maxPositions: o.TextMaxPositions,
            textHiddenDim: o.TextHiddenDim,
            textEncoderLayers: o.TextEncoderLayers,
            textEncoderHeads: o.TextEncoderHeads,
            intermediateDim: o.TextIntermediateDim,
            paddingTokenId: o.TextPaddingTokenId,
            dropoutRate: o.TextDropoutRate,
            layerNormEpsilon: o.TextLayerNormEpsilon,
            projectionDim: o.ProjectionDim);

    #endregion

    #region Preprocessing — LAION-CLAP audio pipeline

    /// <summary>
    /// Fits raw audio to the 10-second clip and converts it to the log-mel input of the audio tower (see
    /// <see cref="ClapAudioFrontEnd{T}"/>): <c>[batch, 1, frames, mels]</c>, or four views
    /// <c>[batch, 4, frames, mels]</c> with feature fusion.
    /// </summary>
    /// <param name="rawAudio">Audio samples — shape <c>[samples]</c> or <c>[batch, samples]</c>.</param>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => PrepareAudio(rawAudio).Features;

    private (Tensor<T> Features, Tensor<T> Longer) PrepareAudio(Tensor<T> rawAudio)
    {
        Guard.NotNull(rawAudio);
        var random = IsTrainingMode ? _trainingRandom : RandomHelper.CreateSeededRandom(_options.Seed ?? 0);
        using (new NoGradScope<T>())
            return _frontEnd.Prepare(Engine, rawAudio, random);
    }

    /// <summary>
    /// The log-mel spectrogram of each row of <paramref name="waveforms"/>, <c>[batch, frames, mels]</c>, computed
    /// with the filterbank of this model variant and no clip fitting.
    /// </summary>
    /// <param name="waveforms">Audio <c>[batch, samples]</c>.</param>
    public Tensor<T> ComputeLogMel(Tensor<T> waveforms)
    {
        Guard.NotNull(waveforms);
        return _frontEnd.LogMel(Engine, waveforms.Rank == 1 ? Engine.Reshape(waveforms, new[] { 1, waveforms.Shape[0] }) : waveforms);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => L2Normalize(modelOutput);

    #endregion

    #region Public API

    /// <summary>
    /// Encodes audio into a CLAP embedding vector in the shared text-audio space.
    /// </summary>
    /// <param name="audio">Raw audio tensor [samples] or [batch, samples] at <see cref="CLAPModelOptions.SampleRate"/>.</param>
    /// <returns>Audio embedding [batch, projectionDim], L2-normalised.</returns>
    public Tensor<T> EncodeAudio(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return InEvaluationMode(() => EmbedAudio(audio));
    }

    // The audio embedding in the current mode: the training step's forward, and EncodeAudio's body.
    private Tensor<T> EmbedAudio(Tensor<T> audio)
    {
        var (features, longer) = PrepareAudio(audio);
        if (!_useNativeMode && OnnxEncoder is not null)
        {
            return PostprocessOutput(RunOnnxAudioEncoder(OnnxEncoder, features, longer));
        }
        return L2Normalize(RunAudioTower(features, longer));
    }

    /// <summary>
    /// Runs an exported audio encoder. The features go to the graph's feature input; a graph that also declares an
    /// input whose name contains <c>longer</c> (LAION-CLAP's fusion mask, <c>[batch]</c>, 1 where a sample's local
    /// views are fused) receives the mask there, so a fusion export behaves like the native tower.
    /// </summary>
    /// <remarks>
    /// The ONNX contract: one feature input <c>[batch, views, frames, mels]</c> (one view, or the four fusion views)
    /// and at most one more input, the mask, named with <c>longer</c>. The embedding is the graph's only output, or
    /// else the one output whose name contains <c>embed</c>. Any other layout is ambiguous and is rejected by name
    /// rather than fed or read by position.
    /// </remarks>
    private static Tensor<T> RunOnnxAudioEncoder(OnnxModel<T> encoder, Tensor<T> features, Tensor<T> longer)
    {
        var inputs = encoder.Metadata.Inputs;
        var maskInputs = inputs
            .Where(input => input.Name.IndexOf("longer", System.StringComparison.OrdinalIgnoreCase) >= 0)
            .ToList();
        var featureInputs = inputs.Where(input => !maskInputs.Contains(input)).ToList();
        if (featureInputs.Count != 1 || maskInputs.Count > 1)
        {
            throw new InvalidOperationException(
                "The CLAP audio ONNX graph must declare one feature input and at most one 'longer' mask input; it "
                + $"declares {string.Join(", ", inputs.Select(input => input.Name))}.");
        }

        var feeds = new System.Collections.Generic.Dictionary<string, Tensor<T>> { [featureInputs[0].Name] = features };
        if (maskInputs.Count == 1)
            feeds[maskInputs[0].Name] = longer;

        var outputs = encoder.Run(feeds);
        if (outputs.Count == 1)
            return outputs.Values.First();
        var embeddings = outputs
            .Where(output => output.Key.IndexOf("embed", System.StringComparison.OrdinalIgnoreCase) >= 0)
            .ToList();
        if (embeddings.Count == 1)
            return embeddings[0].Value;
        throw new InvalidOperationException(
            "The CLAP audio ONNX graph returns several outputs and not exactly one named with 'embed'; it returns "
            + $"{string.Join(", ", outputs.Keys)}.");
    }

    /// <summary>
    /// Encodes log-mel input directly, bypassing clip fitting: <paramref name="mel"/> is
    /// <c>[batch, views, frames, mels]</c> (one view, or the four fusion views) and <paramref name="longer"/> the
    /// optional <c>[batch]</c> 0/1 mask of samples whose local views are fused.
    /// </summary>
    /// <returns>Audio embedding [batch, projectionDim], L2-normalised.</returns>
    public Tensor<T> EncodeLogMel(Tensor<T> mel, Tensor<T>? longer = null)
    {
        ThrowIfDisposed();
        Guard.NotNull(mel);
        if (!_useNativeMode)
            throw new NotSupportedException("EncodeLogMel runs the native audio tower; this instance is in ONNX mode.");
        return L2Normalize(RunAudioTower(mel, longer));
    }

    // The audio tower is a plain sequence except for the fusion patch embedding, which also reads the mask.
    private Tensor<T> RunAudioTower(Tensor<T> features, Tensor<T>? longer)
        => RunAudioTower(Layers, Layers.Count, features, longer);

    /// <summary>
    /// Runs <paramref name="inference"/> with the towers in evaluation mode and restores the previous mode, as PredictCore
    /// does. Networks and layers start in training mode, so an embedding taken without this ran the text tower's dropout
    /// and the same caption encoded differently on every call.
    /// </summary>
    private TResult InEvaluationMode<TResult>(Func<TResult> inference)
    {
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            return inference();
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    /// <summary>
    /// Runs the first <paramref name="count"/> audio-tower layers on the front end's features. The fusion patch
    /// embedding also reads LAION-CLAP's per-sample fusion mask. Shared with the classification adapter.
    /// </summary>
    internal static Tensor<T> RunAudioTower(IReadOnlyList<ILayer<T>> layers, int count, Tensor<T> features, Tensor<T>? longer)
    {
        var hidden = features;
        for (int i = 0; i < count; i++)
        {
            var layer = layers[i];
            hidden = layer is HtsatFusionPatchEmbeddingLayer<T> fusion && longer is not null
                ? fusion.Forward(hidden, longer)
                : layer.Forward(hidden);
        }
        return hidden;
    }

    /// <summary>
    /// Encodes a tokenised text caption into a CLAP embedding vector.
    /// </summary>
    /// <param name="tokens">RoBERTa token IDs [batch, seqLen] or [seqLen], padded with
    /// <see cref="CLAPModelOptions.TextPaddingTokenId"/>; padding positions are masked out.</param>
    /// <returns>Text embedding [batch, projectionDim], L2-normalised.</returns>
    public Tensor<T> EncodeText(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        return InEvaluationMode(() => EmbedText(tokens));
    }

    // The text embedding in the current mode: the training step's text side, and EncodeText's body.
    private Tensor<T> EmbedText(Tensor<T> tokens)
    {
        if (!_useNativeMode)
        {
            if (OnnxDecoder is null)
                throw new NotSupportedException(
                    "CLAPModel.EncodeText() in ONNX mode requires a text encoder. " +
                    "Construct CLAPModel with both audioEncoderPath and textEncoderPath, " +
                    "or use a native-mode instance. Falling through to the native layer " +
                    "stack would walk an empty TextEncoderLayers list and L2-normalise " +
                    "the raw token IDs — silent garbage output.");
            return PostprocessOutput(OnnxDecoder.Run(tokens));
        }

        if (tokens.Shape.Length == 1)
            tokens = Engine.Reshape(tokens, [1, tokens.Shape[0]]);

        // Native path: walk the text encoder layers (held on the audio base
        // class so any future audio-text dual encoder shares the same slot).
        var hidden = tokens;
        foreach (var layer in TextEncoderLayers) hidden = layer.Forward(hidden);
        return L2Normalize(hidden);
    }

    /// <summary>Convenience overload: encodes one caption's token ids.</summary>
    public Tensor<T> EncodeText(int[] tokenIds)
    {
        Guard.NotNull(tokenIds);
        var tokenTensor = new Tensor<T>([1, tokenIds.Length]);
        for (int i = 0; i < tokenIds.Length; i++)
            tokenTensor[0, i] = NumOps.FromDouble(tokenIds[i]);
        return EncodeText(tokenTensor);
    }

    /// <summary>
    /// Tokenizes <paramref name="captions"/> as LAION-CLAP does — special tokens added, truncated and padded to
    /// <see cref="CLAPModelOptions.MaxTextLength"/> — and encodes them. Pass the RoBERTa tokenizer, for example
    /// <c>HuggingFaceTokenizerLoader.LoadFromHub("roberta-base")</c>.
    /// </summary>
    /// <returns>Text embeddings [captions, projectionDim], L2-normalised.</returns>
    public Tensor<T> EncodeText(IReadOnlyList<string> captions, ITokenizer tokenizer)
    {
        return EncodeText(Tokenize(captions, tokenizer));
    }

    /// <summary>
    /// Token ids <c>[captions, MaxTextLength]</c> for <paramref name="captions"/>: <c>tokenizer(text,
    /// padding="max_length", truncation=True, max_length=77)</c> in LAION-CLAP's <c>hook.py</c>.
    /// </summary>
    public Tensor<T> Tokenize(IReadOnlyList<string> captions, ITokenizer tokenizer)
    {
        Guard.NotNull(captions);
        Guard.NotNull(tokenizer);
        int length = _options.MaxTextLength;
        var ids = new Tensor<T>(new[] { captions.Count, length });
        var encoding = new AiDotNet.Tokenization.Models.EncodingOptions
        {
            AddSpecialTokens = true,
            MaxLength = length,
            Truncation = true,
            Padding = true,
        };
        for (int c = 0; c < captions.Count; c++)
        {
            var tokens = tokenizer.Encode(captions[c] ?? string.Empty, encoding).TokenIds;
            for (int i = 0; i < length; i++)
                ids[c, i] = NumOps.FromDouble(i < tokens.Count ? tokens[i] : _options.TextPaddingTokenId);
        }
        return ids;
    }

    /// <summary>
    /// Zero-shot audio classification with the RoBERTa tokenizer: ranks <paramref name="labels"/> by cosine
    /// similarity between their caption embeddings and the clip's audio embedding.
    /// </summary>
    public Dictionary<string, double> ZeroShotClassify(Tensor<T> audio, string[] labels, ITokenizer tokenizer)
    {
        Guard.NotNull(tokenizer);
        return ZeroShotClassify(audio, labels, label => TokenIdsOf(label, tokenizer));
    }

    private int[] TokenIdsOf(string label, ITokenizer tokenizer)
    {
        var ids = Tokenize(new[] { label }, tokenizer);
        var result = new int[ids.Shape[1]];
        for (int i = 0; i < result.Length; i++) result[i] = (int)Math.Round(NumOps.ToDouble(ids[0, i]));
        return result;
    }

    /// <summary>
    /// Performs zero-shot audio classification: rank each text prompt by its
    /// CLAP-similarity to the supplied audio clip.
    /// </summary>
    public Dictionary<string, double> ZeroShotClassify(Tensor<T> audio, string[] labels, Func<string, int[]> tokenize)
    {
        if (labels is null || labels.Length == 0) throw new ArgumentException("Labels required.", nameof(labels));
        if (tokenize is null) throw new ArgumentNullException(nameof(tokenize));
        // Reject batched inputs explicitly — this API returns a single score
        // map. EncodeAudio accepts [batch, samples], and silently flattening
        // a batch into one embedding would produce one merged score per
        // label across the whole batch.
        if (audio.Shape.Length > 1 && audio.Shape[0] != 1)
            throw new ArgumentException(
                "ZeroShotClassify expects a single audio clip. Pass clips one at a time " +
                "or call EncodeAudio directly for the batched path.",
                nameof(audio));

        var audioEmb = EncodeAudio(audio);
        var scores = new Dictionary<string, double>(labels.Length);
        foreach (var label in labels)
        {
            var textEmb = EncodeText(tokenize(label));
            scores[label] = ComputeCosineSimilarity(audioEmb, textEmb);
        }
        return scores.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        return EncodeAudio(input);
    }

    /// <inheritdoc/>
    public AudioFingerprint<T> Fingerprint(Tensor<T> audio)
    {
        // Single-clip API — reject batched inputs (same rationale as
        // ZeroShotClassify: flattening a batch into one fingerprint and one
        // Duration is silently misleading).
        if (audio.Shape.Length > 1 && audio.Shape[0] != 1)
            throw new ArgumentException(
                "Fingerprint expects a single audio clip. Pass clips one at a time " +
                "or call EncodeAudio directly for the batched path.",
                nameof(audio));

        var embedding = EncodeAudio(audio);
        var flat = embedding.ToVector();
        var data = new T[flat.Length];
        for (int i = 0; i < flat.Length; i++) data[i] = flat[i];
        return new AudioFingerprint<T>
        {
            Data = data,
            SampleRate = SampleRate,
            // Per-clip duration: use the last-axis sample count so a [1, N]
            // batched tensor still reports the correct N / SampleRate.
            Duration = audio.Shape[audio.Shape.Length - 1] / (double)SampleRate
        };
    }

    /// <inheritdoc/>
    public AudioFingerprint<T> Fingerprint(Vector<T> audio) =>
        Fingerprint(Tensor<T>.FromVector(audio));

    /// <inheritdoc/>
    public double ComputeSimilarity(AudioFingerprint<T> fp1, AudioFingerprint<T> fp2)
    {
        if (fp1.Data.Length != fp2.Data.Length)
            throw new ArgumentException("Fingerprint dimensions do not match.");
        double dot = 0.0, n1 = 0.0, n2 = 0.0;
        for (int i = 0; i < fp1.Data.Length; i++)
        {
            double a = Convert.ToDouble(fp1.Data[i]);
            double b = Convert.ToDouble(fp2.Data[i]);
            dot += a * b; n1 += a * a; n2 += b * b;
        }
        return dot / (Math.Sqrt(n1) * Math.Sqrt(n2) + 1e-12);
    }

    /// <inheritdoc/>
    public IReadOnlyList<FingerprintMatch> FindMatches(
        AudioFingerprint<T> query, AudioFingerprint<T> reference, int minMatchLength = 10)
    {
        // CLAP fingerprints are global embeddings — there's no temporal
        // segmentation to localise inside. We return a single whole-clip match
        // when the overall similarity exceeds the standard zero-shot threshold;
        // callers needing time-localised matches should use a segment-level
        // fingerprinter like ConformerFP or NeuralFP.
        double similarity = ComputeSimilarity(query, reference);
        if (similarity < 0.5) return Array.Empty<FingerprintMatch>();
        return new[]
        {
            new FingerprintMatch
            {
                QueryStartTime = 0.0,
                ReferenceStartTime = 0.0,
                Duration = Math.Max(query.Duration, reference.Duration),
                Confidence = similarity,
                MatchCount = Math.Min(query.Data.Length, reference.Data.Length)
            }
        };
    }

    #endregion

    #region Published weights

    /// <summary>
    /// Loads a LAION-CLAP checkpoint (an HTSAT audio branch with a RoBERTa text branch, e.g.
    /// <c>630k-audioset-fusion-best.pt</c> for the default options, or <c>630k-audioset-best.pt</c> with
    /// <see cref="CLAPModelOptions.EnableFeatureFusion"/> off) by its <c>state_dict</c> names.
    /// </summary>
    /// <param name="read">Returns the row-major values of the named tensor, which must have the given PyTorch
    /// shape. Names are those of the checkpoint's <c>state_dict</c> with any <c>module.</c> prefix removed.</param>
    /// <returns>The state-dict names that were read.</returns>
    /// <remarks>
    /// <para>Read: <c>audio_branch.bn0</c>, <c>audio_branch.patch_embed.*</c> (with <c>mel_conv2d</c> and
    /// <c>fusion_model</c> when fused), <c>audio_branch.layers.{s}.blocks.{i}.*</c>,
    /// <c>audio_branch.layers.{s}.downsample.*</c>, <c>audio_branch.norm</c>, <c>text_branch.*</c> (Hugging Face
    /// <c>RobertaModel</c>), <c>audio_projection.{0,2}</c>, <c>text_projection.{0,2}</c> and
    /// <c>logit_scale_a</c>.</para>
    /// <para>Not read, because the CLAP embedding never uses them: HTS-AT's AudioSet head (<c>tscam_conv</c>,
    /// <c>head</c>), the spectrogram extractor buffers, <c>audio_transform</c> / <c>text_transform</c> (the optional
    /// MLP-loss heads) and <c>logit_scale_t</c> (the text-side scale of that loss).</para>
    /// <para>A <c>.pt</c> checkpoint is a PyTorch pickle; export it once, e.g. with
    /// <c>safetensors.torch.save_file(torch.load(path)["state_dict"], out)</c>, and serve the reader from the
    /// safetensors file.</para>
    /// </remarks>
    public IReadOnlyCollection<string> LoadLaionWeights(Func<string, int[], double[]> read)
    {
        Guard.NotNull(read);
        if (!_useNativeMode)
            throw new NotSupportedException("LAION weights load into the native network; this instance is in ONNX mode.");
        MaterializeForLoading();
        var loader = new TorchStateDictLoader<T>(read, Engine);

        int index = 0;
        ILayer<T> Next() => index < Layers.Count
            ? Layers[index++]
            : throw new InvalidOperationException(
                $"LoadLaionWeights expects the default CLAP audio tower; it ended after {Layers.Count} layers, before the checkpoint's layers were all placed.");
        TLayer Expect<TLayer>(ILayer<T> layer) where TLayer : class =>
            layer as TLayer ?? throw new InvalidOperationException(
                $"LoadLaionWeights expects the default CLAP audio tower; found {layer.GetType().Name} where {typeof(TLayer).Name} belongs.");

        loader.BatchNorm(Expect<BatchNormalizationLayer<T>>(Next()), "audio_branch.bn0");
        Expect<SpectrogramStripeDropLayer<T>>(Next());
        Expect<HtsatSpectrogramImageLayer<T>>(Next());
        var embedding = Next();
        if (embedding is HtsatFusionPatchEmbeddingLayer<T> fused)
        {
            loader.Conv2d(fused.Projection, "audio_branch.patch_embed.proj");
            loader.LayerNorm(fused.Norm, "audio_branch.patch_embed.norm");
            loader.PatchConvAsLinear(fused.LocalProjection, "audio_branch.patch_embed.mel_conv2d", 1,
                _options.AudioPatchSize, 3 * _options.AudioPatchSize);
            const string aff = "audio_branch.patch_embed.fusion_model";
            var (lr, lrn, le, len) = fused.Fusion.LocalBranch;
            loader.Conv2d(lr, aff + ".local_att.0");
            loader.BatchNorm(lrn, aff + ".local_att.1");
            loader.Conv2d(le, aff + ".local_att.3");
            loader.BatchNorm(len, aff + ".local_att.4");
            var (gr, grn, ge, gen) = fused.Fusion.GlobalBranch;
            loader.Conv2d(gr, aff + ".global_att.1");
            loader.BatchNorm(grn, aff + ".global_att.2");
            loader.Conv2d(ge, aff + ".global_att.4");
            loader.BatchNorm(gen, aff + ".global_att.5");
        }
        else
        {
            var plain = Expect<SwinPatchEmbeddingLayer<T>>(embedding);
            loader.Conv2d(plain.Projection, "audio_branch.patch_embed.proj");
            loader.LayerNorm(plain.Norm, "audio_branch.patch_embed.norm");
        }

        for (int stage = 0; stage < _options.AudioStageCount; stage++)
        {
            for (int b = 0; b < _options.AudioStageDepths[stage]; b++)
            {
                var block = Expect<SwinTransformerBlockLayer<T>>(Next()).ReferenceModules;
                string p = $"audio_branch.layers.{stage}.blocks.{b}";
                loader.LayerNorm(block.Norm1, p + ".norm1");
                loader.Linear(block.Qkv, p + ".attn.qkv");
                loader.Linear(block.Projection, p + ".attn.proj");
                loader.Tensor(block.RelativePositionBiasTable, p + ".attn.relative_position_bias_table");
                loader.LayerNorm(block.Norm2, p + ".norm2");
                loader.Linear(block.Fc1, p + ".mlp.fc1");
                loader.Linear(block.Fc2, p + ".mlp.fc2");
            }
            if (stage < _options.AudioStageCount - 1)
            {
                var merge = Expect<SwinPatchMergingLayer<T>>(Next());
                loader.LayerNorm(merge.Norm, $"audio_branch.layers.{stage}.downsample.norm");
                loader.Linear(merge.Reduction, $"audio_branch.layers.{stage}.downsample.reduction");
            }
        }
        loader.LayerNorm(Expect<LayerNormalizationLayer<T>>(Next()), "audio_branch.norm");
        Expect<GlobalPoolingLayer<T>>(Next());
        loader.Linear(Expect<DenseLayer<T>>(Next()), "audio_projection.0");
        loader.Linear(Expect<DenseLayer<T>>(Next()), "audio_projection.2");
        if (index != Layers.Count)
            throw new InvalidOperationException(
                $"LoadLaionWeights expects the default CLAP audio tower; {Layers.Count - index} layer(s) after the projection have no checkpoint weights.");

        if (TextEncoderLayers.Count != 3 || TextEncoderLayers[0] is not RobertaEncoderLayer<T> roberta)
            throw new InvalidOperationException("LoadLaionWeights expects the default CLAP text tower (RoBERTa and its projection).");
        var embeddings = roberta.Embeddings;
        loader.Embedding(embeddings.WordEmbedding, "text_branch.embeddings.word_embeddings");
        loader.Embedding(embeddings.PositionEmbedding, "text_branch.embeddings.position_embeddings");
        loader.Embedding(embeddings.TokenTypeEmbedding, "text_branch.embeddings.token_type_embeddings");
        loader.LayerNorm(embeddings.Normalization, "text_branch.embeddings.LayerNorm");
        for (int i = 0; i < roberta.NumLayers; i++)
        {
            var layer = roberta.EncoderLayer(i);
            string p = $"text_branch.encoder.layer.{i}";
            loader.Linear(layer.Query, p + ".attention.self.query");
            loader.Linear(layer.Key, p + ".attention.self.key");
            loader.Linear(layer.Value, p + ".attention.self.value");
            loader.Linear(layer.AttentionOutput, p + ".attention.output.dense");
            loader.LayerNorm(layer.AttentionNorm, p + ".attention.output.LayerNorm");
            loader.Linear(layer.Intermediate, p + ".intermediate.dense");
            loader.Linear(layer.Output, p + ".output.dense");
            loader.LayerNorm(layer.OutputNorm, p + ".output.LayerNorm");
        }
        loader.Linear(roberta.Pooler, "text_branch.pooler.dense");
        loader.Linear(Expect<DenseLayer<T>>(TextEncoderLayers[1]), "text_projection.0");
        loader.Linear(Expect<DenseLayer<T>>(TextEncoderLayers[2]), "text_projection.2");

        loader.Scalar(_logTemperature, "logit_scale_a");
        return loader.Loaded;
    }

    // Sub-layers allocate their tensors on first use, so one inference pass of each tower (with the fusion branch
    // engaged) gives every parameter its storage before values are copied in. Nothing is recorded or trained.
    private void MaterializeForLoading()
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using (new NoGradScope<T>())
            {
                int views = _options.EnableFeatureFusion ? 4 : 1;
                var mel = new Tensor<T>(new[] { 1, views, _frontEnd.ClipFrames, _options.NumMelBands });
                var longer = new Tensor<T>(new[] { 1 });
                longer[0] = NumOps.One;
                RunAudioTower(mel, _options.EnableFeatureFusion ? longer : null);
                var ids = new Tensor<T>(new[] { 1, 2 });
                ids[0, 1] = NumOps.FromDouble(2);
                foreach (var layer in TextEncoderLayers) ids = layer.Forward(ids);
            }
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>The text tower (RoBERTa, then the text projection), for structural tests.</summary>
    internal IReadOnlyList<ILayer<T>> TextEncoderLayersForTests => TextEncoderLayers;

    /// <summary>The learned logit scale <c>exp(logit_scale_a) = 1 / τ</c>.</summary>
    public double LogitScale => Math.Exp(NumOps.ToDouble(_logTemperature[0]));

    #endregion

    #region Training (tape-based)

    /// <inheritdoc/>
    /// <remarks>
    /// CLAP is trained with a symmetric contrastive objective (Radford 2021
    /// Eq. 1; Wu 2023 §3.2): for a minibatch of (audio, caption) pairs, the
    /// loss is the average of audio→text and text→audio cross-entropies over
    /// the temperature-scaled cosine-similarity logit matrix. Implemented via
    /// the standard tape-based training path so all gradients (including the
    /// learnable temperature) flow correctly through every layer.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (!_useNativeMode)
            throw new NotSupportedException(
                "CLAP training requires native mode. Construct with the (architecture, options) ctor.");

        // CLAP contrastive objective (Radford 2021 Eq. 1; Wu 2023 §3.2): the
        // loss is the average of audio→text and text→audio softmax cross-
        // entropies over the temperature-scaled cosine-similarity logit
        // matrix where positive pairs are aligned on the diagonal.
        //
        // Engine-side implementation: TrainWithCustomLoss runs the audio
        // encoder forward inside its own GradientTape, then we run the text
        // encoder forward inside the same tape, build the symmetric loss,
        // and let TrainWithCustomLoss handle the gradient computation +
        // optimizer step. Parameters collected from Layers
        // (audio side); the text-encoder parameters and _logTemperature
        // participate in the loss only when they're referenced in the tape
        // graph. The optimizer.Step in TrainWithCustomLoss won't directly
        // update text-encoder or temperature weights, so we manually apply
        // SGD with the same LR on those after the audio step using their
        // tape-computed gradients (collected via a parallel
        // CollectContrastiveParameters walk).
        //
        // input:    [batch, samples] audio batch.
        // expected: [batch, seqLen]  tokenised caption batch (positive pair
        //                            in the same row).
        SetTrainingMode(true);
        try
        {
            var audioParams = Training.TapeTrainingStep<T>.CollectParameters(Layers);
            var textParams = Training.TapeTrainingStep<T>.CollectParameters(TextEncoderLayers);
            var allParams = new List<Tensor<T>>(audioParams.Count + textParams.Count + 1);
            allParams.AddRange(audioParams);
            allParams.AddRange(textParams);
            allParams.Add(_logTemperature);

            var optimizer = GetOrCreateBaseOptimizer();

            // GPU-RESIDENT fast path — audio + text encoders + the learned
            // temperature scalar. _logTemperature is NOT an ITrainableLayer, so
            // it goes through extraTensors (Phase 4A). The fused optimizer tracks
            // it via the same moment-buffer path as layer-carried params.
            var trainableLayers = new List<ITrainableLayer<T>>();
            foreach (var l in Layers) if (l is ITrainableLayer<T> t) trainableLayers.Add(t);
            foreach (var l in TextEncoderLayers) if (l is ITrainableLayer<T> t) trainableLayers.Add(t);
            var extras = new List<Tensor<T>> { _logTemperature };
            if (trainableLayers.Count > 0 || extras.Count > 0)
            {
                // The contrastive loss depends on BOTH encoders + temperature, so
                // the forward closure runs the full symmetric-alignment computation
                // and the loss closure just reduces its output. We pass the audio
                // input as the primary and the text batch as target so the closure
                // signature matches TryStep's contract.
                Tensor<T> FwdCLAP(Tensor<T> audioIn) => EmbedAudio(audioIn); // audio embedding — the loss closure re-runs the text side.
                // Re-run the text encoder on this replay's text batch.
                Tensor<T> LossCLAP(Tensor<T> audioEmb, Tensor<T> textBatch) => ContrastiveLoss(audioEmb, EmbedText(textBatch));
                if (AiDotNet.Training.GpuResidentFusedStep<T>.TryStep(
                        trainableLayers, input, expected,
                        forward: FwdCLAP, computeLoss: LossCLAP,
                        optimizer: optimizer,
                        out T fusedLoss, owner: this,
                        extraTensors: extras,
                        onGradients: gradients => PublishParameterGradients(gradients)))
                {
                    LastLoss = fusedLoss;
                    return;
                }
            }

            using var tape = new GradientTape<T>();
            // Forward both encoders inside the same tape so gradients flow
            // through every parameter. EncodeAudio / EncodeText already
            // L2-normalise along the last axis so the dot product below is
            // cosine similarity.
            var lossSum = ContrastiveLoss(EmbedAudio(input), EmbedText(expected));

            // Manual gradient + optimizer step over the combined params.
            var grads = ComputeAndPublishParameterGradients(tape, lossSum, allParams);

            T lossValue = lossSum.Length > 0 ? lossSum[0] : NumOps.Zero;
            LastLoss = lossValue;

            Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => EmbedAudio(inp);
            Tensor<T> RecomputeLoss(Tensor<T> _, Tensor<T> __) => lossSum;

            var context = new AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<T>(
                allParams, grads, lossValue,
                input, input, ComputeForward, RecomputeLoss);
            optimizer.Step(context);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// The symmetric contrastive loss of paired, L2-normalised embeddings (Radford 2021 Eq. 1; Wu 2023 §3.2):
    /// <c>0.5 · (CE(s · A Tᵀ, diag) + CE(s · T Aᵀ, diag))</c> with <c>s = exp(logit_scale)</c>, on the tape.
    /// </summary>
    private Tensor<T> ContrastiveLoss(Tensor<T> audioEmb, Tensor<T> textEmb)
    {
        int batchSize = audioEmb.Shape[0];
        int projDim = audioEmb.Shape[audioEmb.Shape.Length - 1];
        if (textEmb.Shape[0] != batchSize)
            throw new ArgumentException(
                $"CLAP pairs audio and captions row by row; got {batchSize} clips and {textEmb.Shape[0]} captions.", nameof(textEmb));
        var audioEmb2D = audioEmb.Shape.Length == 2 ? audioEmb : Engine.Reshape(audioEmb, new[] { batchSize, projDim });
        var textEmb2D = textEmb.Shape.Length == 2 ? textEmb : Engine.Reshape(textEmb, new[] { batchSize, projDim });
        var sim = Engine.TensorMatMul<T>(audioEmb2D, Engine.TensorTranspose<T>(textEmb2D));
        // Broadcast scalar exp(logTemp) across the [batch, batch] grid.
        var tau = Engine.TensorExp<T>(_logTemperature);
        var tauBroadcast = Engine.TensorTile(Engine.Reshape(tau, new[] { 1, 1 }), new[] { batchSize, batchSize });
        var logitsA2T = Engine.TensorMultiply<T>(sim, tauBroadcast);
        // Both directions explicitly (rather than relying on transpose-eq) so the autodiff graph is unambiguous.
        var logitsT2A = Engine.TensorTranspose<T>(logitsA2T);
        return Engine.TensorAdd<T>(SymmetricRowCrossEntropy(logitsA2T, batchSize), SymmetricRowCrossEntropy(logitsT2A, batchSize));
    }

    /// <inheritdoc />
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.ContrastiveAlignment;

    /// <inheritdoc />
    /// <remarks>The target of a CLAP step is the caption batch paired with the audio batch. A proposed target is
    /// read as token ids: rounded and clamped into the vocabulary.</remarks>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget)
    {
        Guard.NotNull(proposedTarget);
        var ids = new Tensor<T>(proposedTarget._shape);
        for (int i = 0; i < ids.Length; i++)
        {
            double id = Math.Round(Math.Abs(NumOps.ToDouble(proposedTarget.GetFlat(i))));
            ids.SetFlat(i, NumOps.FromDouble(Math.Min(id, _options.VocabSize - 1)));
        }
        return ids;
    }

    /// <inheritdoc />
    /// <remarks>The contrastive loss <see cref="Train"/> minimizes, for audio <paramref name="input"/> and caption
    /// token ids <paramref name="target"/>, without recording gradients.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        Guard.NotNull(input);
        Guard.NotNull(target);
        using (new NoGradScope<T>())
        {
            var loss = ContrastiveLoss(EncodeAudio(input), EncodeText(target));
            return loss[0];
        }
    }

    /// <summary>
    /// Computes 0.5 * mean cross-entropy of softmax(logits) against the
    /// diagonal target (positive pair at the same row index), tape-tracked
    /// via engine ops. Numerically stable: subtracts the row-wise max
    /// before exponentiating. Returns a [1] scalar tensor so the caller
    /// can sum the two directions into the final symmetric loss.
    /// </summary>
    private Tensor<T> SymmetricRowCrossEntropy(Tensor<T> logits, int batchSize)
    {
        // log_softmax = logits - log(sum(exp(logits − rowMax))) − rowMax
        // We need the diagonal entry of log_softmax. Easiest: compute
        // log_softmax row by row using engine ops, then sum along the
        // diagonal via a one-hot-style multiply.
        var rowMaxes = new Tensor<T>(new[] { batchSize, 1 });
        for (int i = 0; i < batchSize; i++)
        {
            T m = logits[i, 0];
            for (int j = 1; j < batchSize; j++)
                if (NumOps.GreaterThan(logits[i, j], m)) m = logits[i, j];
            rowMaxes[i, 0] = m;
        }
        var rowMaxTiled = Engine.TensorTile(rowMaxes, new[] { 1, batchSize });
        var shifted = Engine.TensorSubtract<T>(logits, rowMaxTiled);
        var expShifted = Engine.TensorExp<T>(shifted);
        var rowSum = Engine.ReduceSum(expShifted, new[] { 1 }, keepDims: true);
        var logRowSum = Engine.TensorLog<T>(rowSum);
        var logRowSumTiled = Engine.TensorTile(logRowSum, new[] { 1, batchSize });
        var logSoftmax = Engine.TensorSubtract<T>(shifted, logRowSumTiled);

        // Diagonal mask: 1 at (i, i), 0 elsewhere.
        var diagMask = new Tensor<T>(new[] { batchSize, batchSize });
        for (int i = 0; i < batchSize; i++) diagMask[i, i] = NumOps.One;
        var picked = Engine.TensorMultiply<T>(logSoftmax, diagMask);
        // Sum all entries; only the diagonal survives.
        var sumAxes = new[] { 0, 1 };
        var totalLog = Engine.ReduceSum(picked, sumAxes, keepDims: false);

        // halfLoss = -0.5 * totalLog / batchSize. Keep the reduction result on the
        // engine graph: reading totalLog[0] into T and wrapping it in a new Tensor
        // disconnects the loss from every encoder parameter, yielding all-zero grads.
        return Engine.TensorMultiplyScalar(
            totalLog, NumOps.FromDouble(-0.5 / batchSize));
    }

    #endregion

    #region Helpers

    /// <summary>
    /// L2-normalises along the last (feature) axis so cosine similarity
    /// reduces to a plain dot product downstream. Engine-routed to stay
    /// generic in <typeparamref name="T"/> and BLAS-accelerated.
    /// </summary>
    private Tensor<T> L2Normalize(Tensor<T> x) => L2Normalize(Engine, x);

    /// <summary>Scales each row of <paramref name="x"/> to unit length along its last axis.</summary>
    internal static Tensor<T> L2Normalize(IEngine engine, Tensor<T> x)
    {
        // ||x||² along the last axis, then 1/√(·+ε) via Sqrt + Reciprocal.
        var sq = engine.TensorMultiply(x, x);
        int lastAxis = x.Shape.Length - 1;
        var sumSq = engine.ReduceSum(sq, [lastAxis], keepDims: true);
        var eps = MathHelper.GetNumericOperations<T>().FromDouble(1e-12);
        var sumSqPlusEps = engine.TensorAddScalar(sumSq, eps);
        var norm = engine.TensorSqrt(sumSqPlusEps);
        var invNorm = engine.TensorReciprocal(norm);
        return engine.TensorMultiply(x, invNorm);
    }

    /// <summary>Cosine similarity between two row-aligned embeddings (already L2-normalised).</summary>
    private double ComputeCosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        int len = Math.Min(a.Length, b.Length);
        double dot = 0.0;
        for (int i = 0; i < len; i++)
            dot += Convert.ToDouble(a[i]) * Convert.ToDouble(b[i]);
        return dot;
    }

    private void ThrowIfDisposed()
    {
        // AudioNeuralNetworkBase manages disposal; no extra state to gate.
    }

 // _logTemperature (learnable τ scalar)

    // UpdateParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    /// <inheritdoc/>


    /// <inheritdoc/>
    /// <remarks>
    /// Validates that every persisted option matches the constructor option
    /// supplied for the live instance. CLAPModelOptions is init-only, so we
    /// can't rebind those — but accepting a mismatched checkpoint would silently
    /// produce wrong-shape weight loads. Throwing with the offending field
    /// surfaces the issue immediately.
    /// </remarks>


    private static void VerifyEqual<TValue>(TValue persisted, TValue current, string name)
        where TValue : IEquatable<TValue>
    {
        if (!persisted.Equals(current))
            throw new InvalidOperationException(
                $"Persisted CLAPModelOptions.{name} = {persisted} does not match constructor option {current}. " +
                "Reconstruct CLAPModel with matching CLAPModelOptions before loading this checkpoint.");
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = Name,
            Description = "CLAP — Contrastive Language-Audio Pre-training (Wu et al. 2023).",
            Complexity = _options.AudioStageDepths.Sum() + _options.TextEncoderLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["AudioEncoder"] = "HTS-AT",
                ["AudioEmbedDim"] = _options.AudioEmbedDim,
                ["AudioStageDepths"] = string.Join(",", _options.AudioStageDepths),
                ["AudioFeatureDim"] = _options.AudioFeatureDim,
                ["TextEncoder"] = "RoBERTa",
                ["TextHiddenDim"] = _options.TextHiddenDim,
                ["TextEncoderLayers"] = _options.TextEncoderLayers,
                ["ProjectionDim"] = _options.ProjectionDim,
                ["SampleRate"] = _options.SampleRate,
                ["NumMelBands"] = _options.NumMelBands,
                ["FeatureFusion"] = _options.EnableFeatureFusion,
            }
        };
    }

    #endregion
}
