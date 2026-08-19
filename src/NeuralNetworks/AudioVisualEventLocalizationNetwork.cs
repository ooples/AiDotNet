using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Neural network for audio-visual event localization - identifying WHEN and WHERE events occur
/// in video by jointly analyzing audio and visual streams with precise temporal boundaries.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// This network jointly analyzes audio and visual streams to identify when and where events
/// occur in video, producing precise temporal boundaries for detected events.
/// </para>
/// <para><b>For Beginners:</b> This model watches and listens to video simultaneously to find
/// specific events. For example, in a concert video it can identify:
/// - WHEN the guitar solo starts and ends (temporal localization)
/// - WHERE on screen the guitar player is (spatial localization)
///
/// It works by processing audio and video frames in parallel, then using cross-modal
/// attention to find moments where what's heard matches what's seen. This is useful for
/// video surveillance, sports analysis, and content moderation.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an audio-visual event localization network
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Detection,
///     inputSize: 512,
///     outputSize: 128);
///
/// var model = new AudioVisualEventLocalizationNetwork&lt;float&gt;(architecture);
///
/// // Detect and localize events in audio-visual input
/// Tensor&lt;float&gt; detections = model.Predict(inputTensor);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Audio-Visual Event Localization in Unconstrained Videos", "https://arxiv.org/abs/1803.08842")]
public partial class AudioVisualEventLocalizationNetwork<T> : MultimodalModelLayoutBase<T>, IAudioVisualEventLocalizationModel<T>
{
    private readonly AudioVisualEventLocalizationOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Constants

    private const int DEFAULT_EMBEDDING_DIM = 512;
    private const double DEFAULT_TEMPORAL_RESOLUTION = 0.1; // 100ms resolution
    private const int DEFAULT_NUM_CATEGORIES = 100;

    /// <summary>
    /// Number of patches each video frame is split into before the visual encoder sees it.
    /// </summary>
    /// <remarks>
    /// The visual encoder is an attention stack, and attention needs a sequence to attend over. A
    /// single flattened frame is one vector with nothing to compare against; splitting it into
    /// patches gives the encoder the tokens it needs, exactly as in Dosovitskiy et al. (2021),
    /// "An Image Is Worth 16x16 Words". Sixteen patches of 48 values preserves the 768-value budget
    /// the visual input projection was already built for.
    /// </remarks>
    private const int VISUAL_PATCH_COUNT = 16;

    /// <summary>
    /// Number of values in each visual patch. <see cref="VISUAL_PATCH_COUNT"/> times this is 768.
    /// </summary>
    private const int VISUAL_PATCH_DIMENSION = 48;

    #endregion

    #region Fields

    private readonly INumericOperations<T> _numOps;
    private readonly int _embeddingDimension;
    private readonly double _temporalResolution;
    private readonly int _numEncoderLayers;
    private readonly IReadOnlyList<string> _supportedCategories;
    private readonly Random _random;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    // Audio front-end. The mel transform holds no trainable parameters (its filterbank is fixed),
    // so it is NOT part of Layers; the VGGish embedding IS trainable and is registered there.
    private readonly AiDotNet.Diffusion.Audio.MelSpectrogram<T> _audioFrontEnd;
    private readonly int _audioEmbeddingFullyConnectedWidth;
    private readonly int _audioEmbeddingSize;
    private VGGishAudioEmbedding<T> _audioEmbedding;

    // Audio encoder
    private DenseLayer<T> _audioInputProjection;
    private MultiHeadAttentionLayer<T>[] _audioEncoderLayers;
    private DenseLayer<T> _audioOutputProjection;

    // Visual encoder
    private DenseLayer<T> _visualInputProjection;
    private MultiHeadAttentionLayer<T>[] _visualEncoderLayers;
    private DenseLayer<T> _visualOutputProjection;

    // Temporal modeling
    private MultiHeadAttentionLayer<T>[] _temporalAttentionLayers;
    private DenseLayer<T> _temporalProposalHead;

    // Cross-modal fusion for event detection
    private MultiHeadAttentionLayer<T>[] _crossModalAttentionLayers;

    // Task-specific heads
    private DenseLayer<T> _eventClassificationHead;
    private DenseLayer<T> _temporalBoundaryHead;
    private DenseLayer<T> _spatialLocalizationHead;
    private DenseLayer<T> _anomalyDetectionHead;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public double TemporalResolution => _temporalResolution;

    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedEventCategories => _supportedCategories;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public AudioVisualEventLocalizationNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.BinaryClassification,
            inputSize: 512,
            outputSize: 1))
    {
    }

    /// <summary>
    /// Initializes a new instance of the AudioVisualEventLocalizationNetwork.
    /// </summary>
    public AudioVisualEventLocalizationNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int embeddingDimension = DEFAULT_EMBEDDING_DIM,
        double temporalResolution = DEFAULT_TEMPORAL_RESOLUTION,
        int numEncoderLayers = 6,
        IEnumerable<string>? eventCategories = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null,
        AudioVisualEventLocalizationOptions? options = null,
        int audioEmbeddingFullyConnectedWidth = VGGishAudioEmbedding<T>.PaperFullyConnectedWidth,
        int audioEmbeddingSize = VGGishAudioEmbedding<T>.PaperEmbeddingSize)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new AudioVisualEventLocalizationOptions();
        Options = _options;

        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDimension = embeddingDimension;
        _temporalResolution = temporalResolution;
        _numEncoderLayers = numEncoderLayers;
        // Published VGGish widths by default. Exposed because the published network is ~67M
        // parameters, which is right for fidelity and wrong for a fixture; a caller can shrink it
        // without a second implementation existing.
        _audioEmbeddingFullyConnectedWidth = audioEmbeddingFullyConnectedWidth;
        _audioEmbeddingSize = audioEmbeddingSize;
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSeededRandom(42);
        _lossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();
        _optimizer = optimizer ?? new Optimizers.AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Default event categories
        _supportedCategories = (eventCategories?.ToList() ?? GetDefaultEventCategories()).AsReadOnly();

        // Real log-mel at the geometry VGGish was defined against. Previously this model computed
        // its own "spectrogram" -- the waveform chopped into 128 chunks with each chunk's RMS energy
        // -- which had no FFT, no mel filterbank, no log, and, fatally, no time axis: it returned a
        // single rank-1 vector, so the attention encoder below had no sequence to attend over and
        // MultiHeadAttentionLayer rejected it outright.
        _audioFrontEnd = new AiDotNet.Diffusion.Audio.MelSpectrogram<T>(
            sampleRate: VGGishMelSpectrogramDefaults.SampleRate,
            nMels: VGGishMelSpectrogramDefaults.MelBins,
            nFft: VGGishMelSpectrogramDefaults.WindowLengthSamples,
            hopLength: VGGishMelSpectrogramDefaults.HopLengthSamples,
            fMin: VGGishMelSpectrogramDefaults.MinFrequencyHz,
            fMax: VGGishMelSpectrogramDefaults.MaxFrequencyHz,
            logOffset: VGGishMelSpectrogramDefaults.LogOffset);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    [System.Diagnostics.CodeAnalysis.MemberNotNull(
        nameof(_audioEmbedding),
        nameof(_audioInputProjection), nameof(_audioEncoderLayers), nameof(_audioOutputProjection),
        nameof(_visualInputProjection), nameof(_visualEncoderLayers), nameof(_visualOutputProjection),
        nameof(_temporalAttentionLayers), nameof(_temporalProposalHead),
        nameof(_crossModalAttentionLayers),
        nameof(_eventClassificationHead), nameof(_temporalBoundaryHead),
        nameof(_spatialLocalizationHead), nameof(_anomalyDetectionHead))]
    protected override void InitializeLayers()
    {
        Layers.Clear();

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateAudioVisualEventLocalizationLayers(
                inputSize: Architecture.InputSize,
                embeddingDimension: _embeddingDimension,
                numEncoderLayers: _numEncoderLayers,
                numCategories: _supportedCategories.Count,
                audioEmbeddingFullyConnectedWidth: _audioEmbeddingFullyConnectedWidth,
                audioEmbeddingSize: _audioEmbeddingSize));
        }

        // Distribute layers to internal fields
        int idx = 0;

        // Audio encoder: input projection + attention × numEncoderLayers + output projection
        _audioInputProjection = (DenseLayer<T>)Layers[idx++];
        _audioEncoderLayers = new MultiHeadAttentionLayer<T>[_numEncoderLayers];
        for (int i = 0; i < _numEncoderLayers; i++)
            _audioEncoderLayers[i] = (MultiHeadAttentionLayer<T>)Layers[idx++];
        _audioOutputProjection = (DenseLayer<T>)Layers[idx++];

        // Visual encoder: input projection + attention × numEncoderLayers + output projection
        _visualInputProjection = (DenseLayer<T>)Layers[idx++];
        _visualEncoderLayers = new MultiHeadAttentionLayer<T>[_numEncoderLayers];
        for (int i = 0; i < _numEncoderLayers; i++)
            _visualEncoderLayers[i] = (MultiHeadAttentionLayer<T>)Layers[idx++];
        _visualOutputProjection = (DenseLayer<T>)Layers[idx++];

        // Temporal modeling: 4 attention layers + proposal head
        _temporalAttentionLayers = new MultiHeadAttentionLayer<T>[4];
        for (int i = 0; i < 4; i++)
            _temporalAttentionLayers[i] = (MultiHeadAttentionLayer<T>)Layers[idx++];
        _temporalProposalHead = (DenseLayer<T>)Layers[idx++];

        // Cross-modal fusion: 4 attention layers
        _crossModalAttentionLayers = new MultiHeadAttentionLayer<T>[4];
        for (int i = 0; i < 4; i++)
            _crossModalAttentionLayers[i] = (MultiHeadAttentionLayer<T>)Layers[idx++];

        // Task-specific heads
        _eventClassificationHead = (DenseLayer<T>)Layers[idx++];
        _temporalBoundaryHead = (DenseLayer<T>)Layers[idx++];
        _spatialLocalizationHead = (DenseLayer<T>)Layers[idx++];
        _anomalyDetectionHead = (DenseLayer<T>)Layers[idx++];

        // Appended LAST on purpose: every index above keeps the position it had before the audio
        // front-end existed, so this addition cannot shift the existing [idx++] contract.
        // This claim reads idx without advancing it, since nothing below consumes another layer.
        // Anything appended after this one must restore the increment here first.
        _audioEmbedding = (VGGishAudioEmbedding<T>)Layers[idx];
    }

    private static List<string> GetDefaultEventCategories()
    {
        return new List<string>
        {
            "speech", "music", "applause", "laughter", "crying", "shouting",
            "dog_bark", "cat_meow", "bird_chirp", "car_horn", "siren",
            "door_slam", "glass_break", "explosion", "gunshot", "thunder",
            "rain", "wind", "footsteps", "typing", "phone_ring",
            "cooking", "eating", "drinking", "coughing", "sneezing",
            "engine_start", "engine_idle", "tire_screech", "crash",
            "sports_crowd", "whistle", "ball_bounce", "ball_kick",
            "unknown"
        };
    }

    #endregion

    #region Audio/Visual Encoding

    /// <summary>
    /// Runs the audio branch and returns the pooled embedding as a TENSOR, keeping every step on the
    /// autodiff tape.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This exists because <see cref="EncodeAudio"/> returns <c>Vector&lt;T&gt;</c>, and a
    /// Tensor -> Vector -> Tensor round-trip severs the tape: the conversion copies values out of the
    /// graph, so no gradient reaches the audio encoder, the VGGish embedding, or anything before
    /// them. Training then reports "no parameters changed" while the forward pass keeps producing
    /// plausible numbers -- a silent failure, which is why the training path uses this method and
    /// <see cref="EncodeAudio"/> is left for the inference callers that genuinely want a vector.
    /// </para>
    /// </remarks>
    private Tensor<T> EncodeAudioTensor(Tensor<T> audioWaveform)
    {
        var segmentEmbeddings = ComputeAudioSegmentEmbeddings(audioWaveform);
        var projected = _audioInputProjection.Forward(segmentEmbeddings);

        var encoded = projected;
        foreach (var layer in _audioEncoderLayers)
        {
            encoded = layer.Forward(encoded);
        }

        var output = _audioOutputProjection.Forward(encoded);
        return MeanOverSegments(output);
    }

    /// <summary>
    /// Averages a <c>[segments, features]</c> sequence over its segment axis using traced ops.
    /// </summary>
    /// <remarks>
    /// Slice-add-scale rather than a manual loop, so the mean stays part of the graph. The previous
    /// pooling flattened the tensor and averaged arbitrary contiguous CHUNKS of it, which is not a
    /// mean over time and also left the tape behind.
    /// </remarks>
    private Tensor<T> MeanOverSegments(Tensor<T> sequence)
    {
        if (sequence.Shape.Length < 2) return sequence;

        int segments = sequence.Shape[0];
        if (segments <= 1) return Engine.TensorSliceAxis(sequence, 0, 0);

        var accumulated = Engine.TensorSliceAxis(sequence, 0, 0);
        for (int i = 1; i < segments; i++)
        {
            accumulated = Engine.TensorAdd(accumulated, Engine.TensorSliceAxis(sequence, 0, i));
        }

        return Engine.TensorMultiplyScalar(accumulated, _numOps.FromDouble(1.0 / segments));
    }

    private Vector<T> EncodeAudio(Tensor<T> audioWaveform)
    {
        // [segments, embeddingSize] -- a real sequence, which is what the attention stack below
        // requires and what the previous rank-1 energy vector could never provide.
        var segmentEmbeddings = ComputeAudioSegmentEmbeddings(audioWaveform);

        // Project to embedding dimension
        var projected = _audioInputProjection.Forward(segmentEmbeddings);

        // Apply transformer encoder layers
        var encoded = projected;
        foreach (var layer in _audioEncoderLayers)
        {
            encoded = layer.Forward(encoded);
        }

        // Final projection and pooling
        var output = _audioOutputProjection.Forward(encoded);
        return GlobalAveragePool(output);
    }

    private Vector<T> EncodeVisual(IEnumerable<Tensor<T>> frames)
    {
        var frameList = frames.ToList();
        if (frameList.Count == 0)
        {
            return new Vector<T>(_embeddingDimension);
        }

        var frameEmbeddings = new List<Vector<T>>();

        foreach (var frame in frameList)
        {
            // Split the frame into patches. FromVector would hand the encoder a rank-1 tensor, and
            // the attention layers below it need a sequence axis to attend over.
            var frameTensor = SplitFrameIntoPatches(frame);

            // Project to embedding dimension
            var projected = _visualInputProjection.Forward(frameTensor);

            // Apply transformer encoder layers
            var encoded = projected;
            foreach (var layer in _visualEncoderLayers)
            {
                encoded = layer.Forward(encoded);
            }

            var output = _visualOutputProjection.Forward(encoded);
            frameEmbeddings.Add(GlobalAveragePool(output));
        }

        // Average across frames
        return AverageVectors(frameEmbeddings);
    }

    /// <summary>
    /// Turns a waveform into the sequence of per-segment embeddings the audio encoder consumes.
    /// </summary>
    /// <param name="waveform">Raw audio samples.</param>
    /// <returns>A <c>[segments, embeddingSize]</c> tensor.</returns>
    /// <remarks>
    /// <para>
    /// The published pipeline (Tian et al. 2018) takes its audio features from VGGish: a stabilised
    /// log-mel spectrogram at 16 kHz, 25 ms window, 10 ms hop, 64 mel bins over 125-7500 Hz, cut
    /// into 96-frame (0.96 s) patches, each patch reduced to one embedding. That yields a genuine
    /// time axis, which is the whole point -- an event localiser has to say WHEN something happened,
    /// and attention cannot localise across a sequence of length one.
    /// </para>
    /// <para>
    /// Short clips are edge-padded to fill a patch rather than rejected. Repeating the final frame
    /// keeps the spectral content of the clip's end, where zero-padding would inject a silence edge
    /// that the convolution reads as a real onset.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeAudioSegmentEmbeddings(Tensor<T> waveform)
    {
        var mel = _audioFrontEnd.Forward(waveform);
        if (mel.Shape.Length < 2)
        {
            throw new InvalidOperationException(
                $"The mel front-end returned rank {mel.Shape.Length}; a [frames, mels] tensor is required.");
        }

        int frames = mel.Shape[mel.Shape.Length - 2];
        int mels = mel.Shape[mel.Shape.Length - 1];
        int patchFrames = VGGishAudioEmbedding<T>.PaperPatchFrames;
        int segments = Math.Max(1, frames / patchFrames);

        var melSpan = mel.Data.Span;
        var segmentEmbeddings = new Tensor<T>[segments];

        for (int segment = 0; segment < segments; segment++)
        {
            // The PATCH may be assembled by copy: it is cut from the mel spectrogram, and nothing
            // upstream of the mel transform carries trainable parameters (its filterbank is fixed),
            // so there is no gradient to lose here.
            var patch = new Tensor<T>(new[] { patchFrames, mels });
            var patchSpan = patch.Data.Span;
            for (int f = 0; f < patchFrames; f++)
            {
                // Clamp rather than wrap: past the end of a short clip we repeat the last frame.
                int sourceFrame = Math.Min(frames - 1, (segment * patchFrames) + f);
                for (int m = 0; m < mels; m++)
                {
                    patchSpan[(f * mels) + m] = melSpan[(sourceFrame * mels) + m];
                }
            }

            // The OUTPUT may not. Writing the embedding's values into a fresh tensor element by
            // element severs the autodiff tape: the copy is not a traced operation, so no gradient
            // reaches VGGish or anything before it and the whole audio branch trains to a standstill
            // -- silently, since the forward pass still produces plausible numbers. Keep each
            // segment's embedding as the tensor the layer returned and join them with a traced
            // concatenation instead.
            var embedded = _audioEmbedding.Forward(patch);
            segmentEmbeddings[segment] = embedded.Shape.Length == 1
                ? embedded.Reshape(new[] { 1, embedded.Shape[0] })
                : embedded;
        }

        return segments == 1
            ? segmentEmbeddings[0]
            : Engine.TensorConcatenate(segmentEmbeddings, axis: 0);
    }

    private Vector<T> FlattenFrame(Tensor<T> frame, int targetSize)
    {
        var result = new Vector<T>(targetSize);
        var frameData = frame.ToVector();

        if (frameData.Length >= targetSize)
        {
            for (int i = 0; i < targetSize; i++)
            {
                result[i] = frameData[i];
            }
        }
        else
        {
            for (int i = 0; i < frameData.Length; i++)
            {
                result[i] = frameData[i];
            }
            for (int i = frameData.Length; i < targetSize; i++)
            {
                result[i] = _numOps.Zero;
            }
        }

        return result;
    }

    /// <summary>
    /// Splits a video frame into a sequence of patches the visual encoder can attend over.
    /// </summary>
    /// <param name="frame">One video frame, of any shape.</param>
    /// <returns>A [<see cref="VISUAL_PATCH_COUNT"/>, <see cref="VISUAL_PATCH_DIMENSION"/>] tensor.</returns>
    /// <remarks>
    /// <para>
    /// The frame's values are read in order, truncated or zero-padded to the fixed patch budget, and
    /// laid out one patch per row. Fixing both dimensions matters: the visual input projection sizes
    /// itself from the first tensor it sees, so a patch width that varied with the frame's resolution
    /// would lock in one width and then reject every differently-sized frame afterwards.
    /// </para>
    /// </remarks>
    private Tensor<T> SplitFrameIntoPatches(Tensor<T> frame)
    {
        var flattened = FlattenFrame(frame, VISUAL_PATCH_COUNT * VISUAL_PATCH_DIMENSION);
        var patches = new Tensor<T>(new[] { VISUAL_PATCH_COUNT, VISUAL_PATCH_DIMENSION });

        for (int patch = 0; patch < VISUAL_PATCH_COUNT; patch++)
        {
            for (int i = 0; i < VISUAL_PATCH_DIMENSION; i++)
            {
                patches[patch, i] = flattened[patch * VISUAL_PATCH_DIMENSION + i];
            }
        }

        return patches;
    }

    /// <summary>
    /// Averages an encoder's output over its sequence axis, giving one embedding per stream.
    /// </summary>
    /// <param name="tensor">The encoder output, [steps, features] or a single [features] vector.</param>
    /// <returns>A vector of at most <see cref="_embeddingDimension"/> features.</returns>
    /// <remarks>
    /// <para>
    /// "Global average pooling" means averaging each feature across the sequence: feature <c>i</c> of
    /// the result is the mean of feature <c>i</c> over every step. The previous implementation
    /// flattened the tensor first and then averaged CONSECUTIVE runs of the flat buffer, which in
    /// row-major order groups together several different features of the SAME step rather than one
    /// feature across steps. That was invisible while the encoders produced rank-1 output, because a
    /// single step makes the two readings coincide; it becomes wrong the moment there is a real
    /// sequence to pool, which is now.
    /// </para>
    /// </remarks>
    private Vector<T> GlobalAveragePool(Tensor<T> tensor)
    {
        if (tensor.Shape.Length < 2)
        {
            var flat = tensor.ToVector();
            int width = Math.Min(_embeddingDimension, Math.Max(flat.Length, 1));
            var single = new Vector<T>(width);

            for (int i = 0; i < width && i < flat.Length; i++)
            {
                single[i] = flat[i];
            }

            return single;
        }

        int features = tensor.Shape[tensor.Shape.Length - 1];
        int steps = tensor.Length / Math.Max(features, 1);
        int outputSize = Math.Min(_embeddingDimension, features);

        var data = tensor.ToVector();
        var result = new Vector<T>(outputSize);

        for (int feature = 0; feature < outputSize; feature++)
        {
            T sum = _numOps.Zero;
            for (int step = 0; step < steps; step++)
            {
                sum = _numOps.Add(sum, data[step * features + feature]);
            }

            result[feature] = _numOps.Divide(sum, _numOps.FromDouble(steps));
        }

        return result;
    }

    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0)
        {
            return new Vector<T>(_embeddingDimension);
        }

        var result = new Vector<T>(vectors[0].Length);
        foreach (var vec in vectors)
        {
            for (int i = 0; i < result.Length && i < vec.Length; i++)
            {
                result[i] = _numOps.Add(result[i], vec[i]);
            }
        }

        var divisor = _numOps.FromDouble(vectors.Count);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = _numOps.Divide(result[i], divisor);
        }

        return result;
    }

    #endregion

    #region IAudioVisualEventLocalizationModel Implementation

    /// <inheritdoc/>
    public IEnumerable<AudioVisualEvent> DetectEvents(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var frameList = frames.ToList();
        if (frameList.Count == 0)
        {
            return Enumerable.Empty<AudioVisualEvent>();
        }

        var events = new List<AudioVisualEvent>();
        var proposals = GenerateProposals(audioWaveform, frameList, frameRate).ToList();

        foreach (var proposal in proposals)
        {
            var eventnessScore = _numOps.ToDouble(proposal.EventnessScore);
            if (eventnessScore < 0.5)
            {
                continue;
            }

            // Extract segment
            var (audioSegment, frameSegment) = ExtractSegment(
                audioWaveform, frameList, proposal.StartTime, proposal.EndTime, frameRate);

            // Classify the event
            var classification = ClassifyEvent(audioSegment, frameSegment, _supportedCategories);
            var bestLabel = classification.OrderByDescending(kvp => _numOps.ToDouble(kvp.Value)).First();

            // Get spatial localization
            var bbox = LocalizeSpatially(audioSegment, frameSegment);

            events.Add(new AudioVisualEvent
            {
                StartTime = proposal.StartTime,
                EndTime = proposal.EndTime,
                Label = bestLabel.Key,
                Confidence = _numOps.ToDouble(bestLabel.Value),
                Modality = DetermineModality(audioSegment, frameSegment),
                BoundingBox = bbox
            });
        }

        return events.OrderBy(e => e.StartTime);
    }

    /// <inheritdoc/>
    public IEnumerable<AudioVisualEvent> DetectSpecificEvents(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> targetCategories,
        double frameRate)
    {
        var allEvents = DetectEvents(audioWaveform, frames, frameRate);
        var targetSet = new HashSet<string>(targetCategories, StringComparer.OrdinalIgnoreCase);

        return allEvents.Where(e => targetSet.Contains(e.Label));
    }

    /// <inheritdoc/>
    public IEnumerable<(double StartTime, double EndTime, T Confidence)> LocalizeEventByDescription(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        string eventDescription,
        double frameRate)
    {
        var frameList = frames.ToList();
        var results = new List<(double, double, T)>();

        // Use temporal proposals and check similarity to description
        var proposals = GenerateProposals(audioWaveform, frameList, frameRate).ToList();

        foreach (var proposal in proposals)
        {
            var (audioSegment, frameSegment) = ExtractSegment(
                audioWaveform, frameList, proposal.StartTime, proposal.EndTime, frameRate);

            // Encode segment features
            var audioFeatures = EncodeAudio(audioSegment);
            var visualFeatures = EncodeVisual(frameSegment);
            var fusedFeatures = FuseFeatures(audioFeatures, visualFeatures);

            // Compute text-to-feature similarity
            var descriptionEmbedding = EncodeTextDescription(eventDescription);
            var similarity = _numOps.FromDouble(VectorHelper.CosineSimilarity(fusedFeatures, descriptionEmbedding));

            if (_numOps.ToDouble(similarity) > 0.3)
            {
                results.Add((proposal.StartTime, proposal.EndTime, similarity));
            }
        }

        return results.OrderByDescending(r => _numOps.ToDouble(r.Item3));
    }

    /// <inheritdoc/>
    public IEnumerable<(double StartTime, double EndTime, T EventnessScore)> GenerateProposals(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var frameList = frames.ToList();
        var proposals = new List<(double, double, T)>();

        if (frameList.Count == 0)
        {
            return proposals;
        }

        double videoDuration = frameList.Count / frameRate;
        int numSegments = (int)(videoDuration / _temporalResolution);

        // Compute per-segment features
        var segmentScores = new List<T>();

        for (int i = 0; i < numSegments; i++)
        {
            double segmentStart = i * _temporalResolution;
            double segmentEnd = (i + 1) * _temporalResolution;

            var (audioSegment, frameSegment) = ExtractSegment(
                audioWaveform, frameList, segmentStart, segmentEnd, frameRate);

            var audioFeatures = EncodeAudio(audioSegment);
            var visualFeatures = EncodeVisual(frameSegment);
            var fusedFeatures = FuseFeatures(audioFeatures, visualFeatures);

            // Apply temporal proposal head
            var proposalTensor = Tensor<T>.FromVector(fusedFeatures);
            var proposalOutput = _temporalProposalHead.Forward(proposalTensor);
            var proposalData = proposalOutput.ToVector();

            // Eventness score
            var score = proposalData.Length > 0 ? _numOps.Abs(proposalData[0]) : _numOps.Zero;
            segmentScores.Add(score);
        }

        // Find contiguous high-scoring regions
        int regionStart = -1;
        T threshold = _numOps.FromDouble(0.3);

        for (int i = 0; i < segmentScores.Count; i++)
        {
            bool isEvent = _numOps.Compare(segmentScores[i], threshold) > 0;

            if (isEvent && regionStart < 0)
            {
                regionStart = i;
            }
            else if (!isEvent && regionStart >= 0)
            {
                // End of region
                T avgScore = ComputeAverageScore(segmentScores, regionStart, i);
                proposals.Add((
                    regionStart * _temporalResolution,
                    i * _temporalResolution,
                    avgScore));
                regionStart = -1;
            }
        }

        // Handle region that extends to end
        if (regionStart >= 0)
        {
            T avgScore = ComputeAverageScore(segmentScores, regionStart, segmentScores.Count);
            proposals.Add((
                regionStart * _temporalResolution,
                segmentScores.Count * _temporalResolution,
                avgScore));
        }

        return proposals.OrderByDescending(p => _numOps.ToDouble(p.Item3));
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ClassifyEvent(
        Tensor<T> audioSegment,
        IEnumerable<Tensor<T>> frameSegment,
        IEnumerable<string> candidateLabels)
    {
        var labelList = candidateLabels.ToList();
        var result = new Dictionary<string, T>();

        var audioFeatures = EncodeAudio(audioSegment);
        var visualFeatures = EncodeVisual(frameSegment);
        var fusedFeatures = FuseFeatures(audioFeatures, visualFeatures);

        // Apply classification head
        var fusedTensor = Tensor<T>.FromVector(fusedFeatures);
        var logits = _eventClassificationHead.Forward(fusedTensor);
        var logitsData = logits.ToVector();

        // Apply softmax
        var probs = Softmax(logitsData);

        // Map to labels
        for (int i = 0; i < labelList.Count && i < probs.Length; i++)
        {
            result[labelList[i]] = probs[i];
        }

        // Handle remaining labels
        for (int i = probs.Length; i < labelList.Count; i++)
        {
            result[labelList[i]] = _numOps.Zero;
        }

        return result;
    }

    /// <inheritdoc/>
    public IEnumerable<AudioVisualEvent> TrackEvent(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        AudioVisualEvent initialEvent,
        double frameRate)
    {
        var frameList = frames.ToList();
        var trajectory = new List<AudioVisualEvent> { initialEvent };

        double currentTime = initialEvent.EndTime;
        double videoDuration = frameList.Count / frameRate;
        double windowSize = initialEvent.EndTime - initialEvent.StartTime;

        // Get reference features from initial event
        var (initAudio, initFrames) = ExtractSegment(
            audioWaveform, frameList, initialEvent.StartTime, initialEvent.EndTime, frameRate);
        var referenceFeatures = FuseFeatures(EncodeAudio(initAudio), EncodeVisual(initFrames));

        // Track forward in time
        while (currentTime < videoDuration)
        {
            double searchStart = currentTime;
            double searchEnd = Math.Min(currentTime + windowSize * 2, videoDuration);

            var (searchAudio, searchFrames) = ExtractSegment(
                audioWaveform, frameList, searchStart, searchEnd, frameRate);

            var searchFeatures = FuseFeatures(EncodeAudio(searchAudio), EncodeVisual(searchFrames));
            var similarity = _numOps.FromDouble(VectorHelper.CosineSimilarity(referenceFeatures, searchFeatures));

            if (_numOps.ToDouble(similarity) < 0.5)
            {
                break; // Event ended
            }

            // Refine boundaries
            var bbox = LocalizeSpatially(searchAudio, searchFrames);

            trajectory.Add(new AudioVisualEvent
            {
                StartTime = searchStart,
                EndTime = searchEnd,
                Label = initialEvent.Label,
                Confidence = _numOps.ToDouble(similarity),
                Modality = initialEvent.Modality,
                BoundingBox = bbox
            });

            currentTime = searchEnd;
        }

        return trajectory;
    }

    /// <inheritdoc/>
    public IEnumerable<(double StartTime, double EndTime, T SyncQuality, string Description)> DetectSyncEvents(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var frameList = frames.ToList();
        var syncEvents = new List<(double, double, T, string)>();

        if (frameList.Count == 0)
        {
            return syncEvents;
        }

        double videoDuration = frameList.Count / frameRate;
        double windowSize = 1.0; // 1 second windows

        for (double t = 0; t < videoDuration - windowSize; t += windowSize / 2)
        {
            var (audioSeg, frameSeg) = ExtractSegment(
                audioWaveform, frameList, t, t + windowSize, frameRate);

            var audioFeatures = EncodeAudio(audioSeg);
            var visualFeatures = EncodeVisual(frameSeg);

            // Measure sync quality via cross-correlation of feature magnitudes
            var syncQuality = ComputeSyncQuality(audioFeatures, visualFeatures);

            if (_numOps.ToDouble(syncQuality) > 0.6)
            {
                string description = DescribeSyncEvent(audioSeg, frameSeg);
                syncEvents.Add((t, t + windowSize, syncQuality, description));
            }
        }

        return MergeOverlappingSyncEvents(syncEvents);
    }

    /// <inheritdoc/>
    public IEnumerable<(double StartTime, double EndTime, string SceneDescription)> SegmentScenes(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var frameList = frames.ToList();
        var scenes = new List<(double, double, string)>();

        if (frameList.Count == 0)
        {
            return scenes;
        }

        double videoDuration = frameList.Count / frameRate;
        var segmentFeatures = new List<(double Time, Vector<T> Features)>();

        // Compute features at regular intervals
        double interval = 0.5; // 500ms intervals
        for (double t = 0; t < videoDuration; t += interval)
        {
            var (audioSeg, frameSeg) = ExtractSegment(
                audioWaveform, frameList, t, Math.Min(t + interval, videoDuration), frameRate);

            var features = FuseFeatures(EncodeAudio(audioSeg), EncodeVisual(frameSeg));
            segmentFeatures.Add((t, features));
        }

        // Find scene boundaries via feature discontinuities
        var boundaries = new List<int> { 0 };
        T threshold = _numOps.FromDouble(0.4);

        for (int i = 1; i < segmentFeatures.Count; i++)
        {
            var similarity = _numOps.FromDouble(VectorHelper.CosineSimilarity(
                segmentFeatures[i - 1].Features,
                segmentFeatures[i].Features));

            if (_numOps.Compare(_numOps.Subtract(_numOps.One, similarity), threshold) > 0)
            {
                boundaries.Add(i);
            }
        }
        boundaries.Add(segmentFeatures.Count);

        // Create scene segments
        for (int i = 0; i < boundaries.Count - 1; i++)
        {
            int startIdx = boundaries[i];
            int endIdx = boundaries[i + 1];
            double startTime = segmentFeatures[startIdx].Time;
            double endTime = segmentFeatures[Math.Min(endIdx, segmentFeatures.Count - 1)].Time + interval;

            var (audioSeg, frameSeg) = ExtractSegment(audioWaveform, frameList, startTime, endTime, frameRate);
            string description = GenerateSceneDescription(audioSeg, frameSeg);

            scenes.Add((startTime, endTime, description));
        }

        return scenes;
    }

    /// <inheritdoc/>
    public IEnumerable<(double Time, string Caption)> GenerateDenseCaptions(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var events = DetectEvents(audioWaveform, frames, frameRate).ToList();
        var captions = new List<(double, string)>();

        foreach (var evt in events)
        {
            double midTime = (evt.StartTime + evt.EndTime) / 2;
            string caption = $"{evt.Label} detected ({evt.Confidence:F2} confidence)";

            if (evt.BoundingBox.HasValue)
            {
                var box = evt.BoundingBox.Value;
                caption += $" at position ({box.X}, {box.Y})";
            }

            captions.Add((midTime, caption));
        }

        return captions.OrderBy(c => c.Item1);
    }

    /// <inheritdoc/>
    public (string Answer, IEnumerable<(double StartTime, double EndTime)> Evidence) AnswerEventQuestion(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        string question,
        double frameRate)
    {
        var events = DetectEvents(audioWaveform, frames, frameRate).ToList();
        var evidence = new List<(double, double)>();
        string answer;

        // Parse question type
        var questionLower = question.ToLowerInvariant();

        if (questionLower.Contains("when") || questionLower.Contains("what time"))
        {
            // Temporal question
            var relevantEvents = FindRelevantEvents(events, question);
            if (relevantEvents.Any())
            {
                var evt = relevantEvents.First();
                answer = $"The event occurs from {evt.StartTime:F1}s to {evt.EndTime:F1}s";
                evidence.AddRange(relevantEvents.Select(e => (e.StartTime, e.EndTime)));
            }
            else
            {
                answer = "No relevant event found in the video.";
            }
        }
        else if (questionLower.Contains("where"))
        {
            // Spatial question
            var relevantEvents = FindRelevantEvents(events, question).Where(e => e.BoundingBox.HasValue);
            if (relevantEvents.Any())
            {
                var evt = relevantEvents.First();
                var box = evt.BoundingBox ?? throw new InvalidOperationException("BoundingBox should have a value after filtering.");
                answer = $"The event is located at position ({box.X}, {box.Y}) with size ({box.Width}x{box.Height})";
                evidence.AddRange(relevantEvents.Select(e => (e.StartTime, e.EndTime)));
            }
            else
            {
                answer = "Could not determine spatial location.";
            }
        }
        else if (questionLower.Contains("how many") || questionLower.Contains("count"))
        {
            // Counting question
            var relevantEvents = FindRelevantEvents(events, question);
            answer = $"Found {relevantEvents.Count()} occurrences";
            evidence.AddRange(relevantEvents.Select(e => (e.StartTime, e.EndTime)));
        }
        else
        {
            // General question
            var relevantEvents = FindRelevantEvents(events, question);
            if (relevantEvents.Any())
            {
                var labels = relevantEvents.Select(e => e.Label).Distinct();
                answer = $"Detected: {string.Join(", ", labels)}";
                evidence.AddRange(relevantEvents.Select(e => (e.StartTime, e.EndTime)));
            }
            else
            {
                answer = "No relevant events found.";
            }
        }

        return (answer, evidence);
    }

    /// <inheritdoc/>
    public IEnumerable<(double StartTime, double EndTime, T AnomalyScore, string Description)> DetectAnomalies(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var frameList = frames.ToList();
        var anomalies = new List<(double, double, T, string)>();

        if (frameList.Count == 0)
        {
            return anomalies;
        }

        double videoDuration = frameList.Count / frameRate;
        var allFeatures = new List<Vector<T>>();

        // Compute features for all segments
        for (double t = 0; t < videoDuration; t += _temporalResolution)
        {
            var (audioSeg, frameSeg) = ExtractSegment(
                audioWaveform, frameList, t, t + _temporalResolution, frameRate);
            allFeatures.Add(FuseFeatures(EncodeAudio(audioSeg), EncodeVisual(frameSeg)));
        }

        // Compute mean and variance
        var meanFeatures = ComputeMeanVector(allFeatures);

        // Find anomalies based on distance from mean
        for (int i = 0; i < allFeatures.Count; i++)
        {
            var distance = VectorHelper.EuclideanDistance(allFeatures[i], meanFeatures);

            // Apply anomaly detection head
            var featureTensor = Tensor<T>.FromVector(allFeatures[i]);
            var anomalyOutput = _anomalyDetectionHead.Forward(featureTensor);
            var anomalyData = anomalyOutput.ToVector();

            var anomalyScore = anomalyData.Length > 0
                ? MathHelper.Sigmoid(anomalyData[0])
                : distance;

            if (_numOps.ToDouble(anomalyScore) > 0.7)
            {
                double startTime = i * _temporalResolution;
                double endTime = (i + 1) * _temporalResolution;
                string description = DescribeAnomaly(allFeatures[i], meanFeatures);

                anomalies.Add((startTime, endTime, anomalyScore, description));
            }
        }

        return MergeContiguousAnomalies(anomalies);
    }

    /// <inheritdoc/>
    public (Tensor<T> AudioToVisualAttention, Tensor<T> VisualToAudioAttention) ComputeEventAttention(
        Tensor<T> audioSegment,
        IEnumerable<Tensor<T>> frameSegment)
    {
        var audioFeatures = EncodeAudio(audioSegment);
        var visualFeatures = EncodeVisual(frameSegment);

        // Compute attention matrices
        int audioLen = audioFeatures.Length;
        int visualLen = visualFeatures.Length;

        var audioToVisual = new T[audioLen * visualLen];
        var visualToAudio = new T[visualLen * audioLen];

        // Compute attention weights
        for (int i = 0; i < audioLen; i++)
        {
            for (int j = 0; j < visualLen; j++)
            {
                var product = _numOps.Multiply(audioFeatures[i], visualFeatures[j]);
                audioToVisual[i * visualLen + j] = product;
                visualToAudio[j * audioLen + i] = product;
            }
        }

        // Apply softmax normalization - convert arrays to vectors
        var a2vVector = new Vector<T>(audioToVisual.Length);
        for (int i = 0; i < audioToVisual.Length; i++) a2vVector[i] = audioToVisual[i];
        var a2vTensor = new Tensor<T>(new[] { audioLen, visualLen }, a2vVector);

        var v2aVector = new Vector<T>(visualToAudio.Length);
        for (int i = 0; i < visualToAudio.Length; i++) v2aVector[i] = visualToAudio[i];
        var v2aTensor = new Tensor<T>(new[] { visualLen, audioLen }, v2aVector);

        return (SoftmaxTensor(a2vTensor, axis: 1), SoftmaxTensor(v2aTensor, axis: 1));
    }

    #endregion

    #region Helper Methods

    private (Tensor<T> Audio, List<Tensor<T>> Frames) ExtractSegment(
        Tensor<T> audioWaveform,
        List<Tensor<T>> frames,
        double startTime,
        double endTime,
        double frameRate)
    {
        // Extract audio segment
        var audioData = audioWaveform.ToVector();
        int audioSampleRate = 16000; // Assume 16kHz
        int startSample = Math.Max(0, (int)(startTime * audioSampleRate));
        int endSample = Math.Min(audioData.Length, (int)(endTime * audioSampleRate));

        int audioLength = Math.Max(1, endSample - startSample);
        var segmentAudioVector = new Vector<T>(audioLength);
        for (int i = 0; i < audioLength && (startSample + i) < audioData.Length; i++)
        {
            segmentAudioVector[i] = audioData[startSample + i];
        }

        // Extract frame segment
        int startFrame = Math.Max(0, (int)(startTime * frameRate));
        int endFrame = Math.Min(frames.Count, (int)(endTime * frameRate));

        var segmentFrames = new List<Tensor<T>>();
        for (int i = startFrame; i < endFrame; i++)
        {
            segmentFrames.Add(frames[i]);
        }

        if (segmentFrames.Count == 0 && frames.Count > 0)
        {
            segmentFrames.Add(frames[Math.Min(startFrame, frames.Count - 1)]);
        }

        return (Tensor<T>.FromVector(segmentAudioVector), segmentFrames);
    }

    private Vector<T> FuseFeatures(Vector<T> audioFeatures, Vector<T> visualFeatures)
    {
        // Concatenate features
        int totalLength = audioFeatures.Length + visualFeatures.Length;
        var fused = new Vector<T>(totalLength);

        for (int i = 0; i < audioFeatures.Length; i++)
        {
            fused[i] = audioFeatures[i];
        }
        for (int i = 0; i < visualFeatures.Length; i++)
        {
            fused[audioFeatures.Length + i] = visualFeatures[i];
        }

        // Apply cross-modal attention
        var fusedTensor = Tensor<T>.FromVector(fused);
        var attended = fusedTensor;
        foreach (var layer in _crossModalAttentionLayers)
        {
            attended = layer.Forward(attended);
        }

        return attended.ToVector();
    }

    private (int X, int Y, int Width, int Height)? LocalizeSpatially(
        Tensor<T> audioSegment,
        IEnumerable<Tensor<T>> frameSegment)
    {
        var audioFeatures = EncodeAudio(audioSegment);
        var visualFeatures = EncodeVisual(frameSegment);
        var fusedFeatures = FuseFeatures(audioFeatures, visualFeatures);

        // Apply spatial localization head
        var fusedTensor = Tensor<T>.FromVector(fusedFeatures);
        var spatialOutput = _spatialLocalizationHead.Forward(fusedTensor);
        var bbox = spatialOutput.ToVector();

        if (bbox.Length < 4)
        {
            return null;
        }

        // Convert normalized coordinates to pixel coordinates (assuming 224x224)
        int imgSize = 224;
        int x = (int)(_numOps.ToDouble(MathHelper.Sigmoid(bbox[0])) * imgSize);
        int y = (int)(_numOps.ToDouble(MathHelper.Sigmoid(bbox[1])) * imgSize);
        int w = (int)(_numOps.ToDouble(MathHelper.Sigmoid(bbox[2])) * imgSize);
        int h = (int)(_numOps.ToDouble(MathHelper.Sigmoid(bbox[3])) * imgSize);

        return (x, y, Math.Max(1, w), Math.Max(1, h));
    }

    private string DetermineModality(Tensor<T> audioSegment, IEnumerable<Tensor<T>> frameSegment)
    {
        var audioEnergy = ComputeEnergy(audioSegment.ToVector());
        var visualActivity = ComputeVisualActivity(frameSegment.ToList());

        double audioVal = _numOps.ToDouble(audioEnergy);
        double visualVal = _numOps.ToDouble(visualActivity);

        if (audioVal > 0.7 && visualVal > 0.7) return "both";
        if (audioVal > visualVal) return "audio";
        return "visual";
    }

    private T ComputeEnergy(Vector<T> signal)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < signal.Length; i++)
        {
            sum = _numOps.Add(sum, _numOps.Multiply(signal[i], signal[i]));
        }
        return _numOps.Sqrt(_numOps.Divide(sum, _numOps.FromDouble(Math.Max(1, signal.Length))));
    }

    private T ComputeVisualActivity(List<Tensor<T>> frames)
    {
        if (frames.Count < 2)
        {
            return _numOps.Zero;
        }

        T totalDiff = _numOps.Zero;
        for (int i = 1; i < frames.Count; i++)
        {
            var curr = frames[i].ToVector();
            var prev = frames[i - 1].ToVector();

            int minLen = Math.Min(curr.Length, prev.Length);
            for (int j = 0; j < minLen; j++)
            {
                var diff = _numOps.Abs(_numOps.Subtract(curr[j], prev[j]));
                totalDiff = _numOps.Add(totalDiff, diff);
            }
        }

        return _numOps.Divide(totalDiff, _numOps.FromDouble(frames.Count * frames[0].ToVector().Length));
    }

    private Vector<T> EncodeTextDescription(string description)
    {
        // Simple bag-of-words encoding
        var result = new Vector<T>(_embeddingDimension);
        var words = description.ToLowerInvariant().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var word in words)
        {
            // Use deterministic FNV-1a hash instead of string.GetHashCode()
            // which is not deterministic across .NET versions or processes
            int hash = Math.Abs(GetDeterministicHashCode(word)) % _embeddingDimension;
            result[hash] = _numOps.Add(result[hash], _numOps.One);
        }

        // L2 normalize
        return VectorHelper.Normalize(result);
    }

    /// <summary>
    /// Computes a deterministic hash code using the FNV-1a algorithm.
    /// Unlike string.GetHashCode(), this produces consistent results across runs.
    /// </summary>
    private static int GetDeterministicHashCode(string str)
    {
        unchecked
        {
            const int FNV_OFFSET_BASIS = unchecked((int)2166136261);
            const int FNV_PRIME = 16777619;

            int hash = FNV_OFFSET_BASIS;
            foreach (char c in str)
            {
                hash ^= c;
                hash *= FNV_PRIME;
            }
            return hash;
        }
    }

    private T ComputeAverageScore(List<T> scores, int start, int end)
    {
        T sum = _numOps.Zero;
        for (int i = start; i < end; i++)
        {
            sum = _numOps.Add(sum, scores[i]);
        }
        return _numOps.Divide(sum, _numOps.FromDouble(end - start));
    }

    private Vector<T> Softmax(Vector<T> logits)
    {
        var result = new Vector<T>(logits.Length);
        T maxVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (_numOps.Compare(logits[i], maxVal) > 0)
            {
                maxVal = logits[i];
            }
        }

        T sum = _numOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = _numOps.Exp(_numOps.Subtract(logits[i], maxVal));
            sum = _numOps.Add(sum, result[i]);
        }

        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = _numOps.Divide(result[i], sum);
        }

        return result;
    }

    private Tensor<T> SoftmaxTensor(Tensor<T> tensor, int axis)
    {
        // Use the Engine's vectorized Softmax implementation which handles
        // arbitrary tensor ranks and axis parameters efficiently
        return Engine.Softmax(tensor, axis);
    }

    private T ComputeSyncQuality(Vector<T> audioFeatures, Vector<T> visualFeatures)
    {
        return _numOps.FromDouble(VectorHelper.CosineSimilarity(audioFeatures, visualFeatures));
    }

    private string DescribeSyncEvent(Tensor<T> audioSeg, IEnumerable<Tensor<T>> frameSeg)
    {
        return "Audio-visual synchronization detected";
    }

    private IEnumerable<(double, double, T, string)> MergeOverlappingSyncEvents(
        List<(double StartTime, double EndTime, T SyncQuality, string Description)> events)
    {
        if (events.Count == 0)
        {
            return events;
        }

        var sorted = events.OrderBy(e => e.StartTime).ToList();
        var merged = new List<(double, double, T, string)>();
        var current = sorted[0];

        for (int i = 1; i < sorted.Count; i++)
        {
            if (sorted[i].StartTime <= current.EndTime)
            {
                // Merge
                var maxQuality = _numOps.Compare(sorted[i].SyncQuality, current.SyncQuality) > 0
                    ? sorted[i].SyncQuality
                    : current.SyncQuality;
                current = (current.StartTime, Math.Max(current.EndTime, sorted[i].EndTime), maxQuality, current.Description);
            }
            else
            {
                merged.Add(current);
                current = sorted[i];
            }
        }
        merged.Add(current);

        return merged;
    }

    private string GenerateSceneDescription(Tensor<T> audioSeg, IEnumerable<Tensor<T>> frameSeg)
    {
        var classification = ClassifyEvent(audioSeg, frameSeg, _supportedCategories);
        var topLabel = classification.OrderByDescending(kvp => _numOps.ToDouble(kvp.Value)).First();
        return $"Scene with {topLabel.Key}";
    }

    private IEnumerable<AudioVisualEvent> FindRelevantEvents(List<AudioVisualEvent> events, string query)
    {
        var queryWords = query.ToLowerInvariant()
            .Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)
            .ToHashSet();

        return events.Where(e =>
        {
            var labelWords = e.Label.ToLowerInvariant().Replace('_', ' ').Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            return labelWords.Any(w => queryWords.Contains(w));
        });
    }

    private Vector<T> ComputeMeanVector(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0)
        {
            return new Vector<T>(_embeddingDimension * 2);
        }

        var result = new Vector<T>(vectors[0].Length);
        foreach (var vec in vectors)
        {
            for (int i = 0; i < result.Length && i < vec.Length; i++)
            {
                result[i] = _numOps.Add(result[i], vec[i]);
            }
        }

        var divisor = _numOps.FromDouble(vectors.Count);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = _numOps.Divide(result[i], divisor);
        }

        return result;
    }


    private string DescribeAnomaly(Vector<T> features, Vector<T> mean)
    {
        return "Anomalous audio-visual pattern detected";
    }

    private IEnumerable<(double, double, T, string)> MergeContiguousAnomalies(
        List<(double StartTime, double EndTime, T AnomalyScore, string Description)> anomalies)
    {
        if (anomalies.Count == 0)
        {
            return anomalies;
        }

        var sorted = anomalies.OrderBy(a => a.StartTime).ToList();
        var merged = new List<(double, double, T, string)>();
        var current = sorted[0];

        for (int i = 1; i < sorted.Count; i++)
        {
            if (sorted[i].StartTime <= current.EndTime + _temporalResolution)
            {
                var maxScore = _numOps.Compare(sorted[i].AnomalyScore, current.AnomalyScore) > 0
                    ? sorted[i].AnomalyScore
                    : current.AnomalyScore;
                current = (current.StartTime, sorted[i].EndTime, maxScore, current.Description);
            }
            else
            {
                merged.Add(current);
                current = sorted[i];
            }
        }
        merged.Add(current);

        return merged;
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Mirrors <see cref="PredictCore"/> instead of inheriting the base's sequential walk over
    /// <c>Layers</c>. That walk is not this model's topology: <c>Layers</c> holds two independent
    /// encoder stacks (audio and visual), the temporal and cross-modal attention blocks and four
    /// task heads, and chaining them end to end feeds each stage an activation the next was never
    /// built to receive. It also bypasses the audio front-end entirely, handing the raw rank-1
    /// waveform straight to the first layer.
    /// </para>
    /// <para>
    /// Training therefore runs the same audio path inference does -- log-mel, VGGish per segment,
    /// then the audio encoder -- so the gradients the optimizer sees correspond to the computation
    /// the model actually performs. A single training tensor is the audio waveform, which is the
    /// interpretation <see cref="PredictCore"/> already fixes for this model.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => EncodeAudioTensor(input);

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        return Accelerate(input, () =>
        {
            var audioFeatures = EncodeAudio(input);
            return Tensor<T>.FromVector(audioFeatures);
        });
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    // UpdateParameters applied a GRADIENT STEP, but its one-argument form is the value setter and every caller passes values -- the override corrupted the model. Removed under AIDN082.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "AudioVisualEventLocalizationNetwork",
            FeatureCount = _embeddingDimension,
            Complexity = _numEncoderLayers * 2 + 4,
            Description = "Audio-visual event localization network for temporal and spatial event detection"
        };
        metadata.AdditionalInfo["EmbeddingDimension"] = _embeddingDimension;
        metadata.AdditionalInfo["TemporalResolution"] = _temporalResolution;
        metadata.AdditionalInfo["NumEncoderLayers"] = _numEncoderLayers;
        metadata.AdditionalInfo["NumEventCategories"] = _supportedCategories.Count;
        metadata.AdditionalInfo["ParameterCount"] = ParameterCount;
        return metadata;
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embeddingDimension);
        writer.Write(_temporalResolution);
        writer.Write(_numEncoderLayers);
        // The audio-embedding widths are part of the ARCHITECTURE, so they have to persist. Without
        // them a restored model rebuilds VGGish at its paper defaults whatever the source used, and
        // the parameter vector is then applied to a differently shaped network -- the restored model
        // predicts differently from the one that was saved.
        writer.Write(_audioEmbeddingFullyConnectedWidth);
        writer.Write(_audioEmbeddingSize);
        writer.Write(_supportedCategories.Count);
        foreach (var category in _supportedCategories)
        {
            writer.Write(category);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read serialized values
        int embDim = reader.ReadInt32();
        double tempRes = reader.ReadDouble();
        int numLayers = reader.ReadInt32();
        int audioEmbedWidth = reader.ReadInt32();
        int audioEmbedSize = reader.ReadInt32();
        int categoryCount = reader.ReadInt32();
        var categories = new List<string>();
        for (int i = 0; i < categoryCount; i++)
        {
            categories.Add(reader.ReadString());
        }

        // Validate that loaded values match current instance configuration
        if (audioEmbedWidth != _audioEmbeddingFullyConnectedWidth || audioEmbedSize != _audioEmbeddingSize)
        {
            throw new InvalidOperationException(
                $"Loaded audio-embedding shape ({audioEmbedWidth}x{audioEmbedSize}) doesn't match current " +
                $"({_audioEmbeddingFullyConnectedWidth}x{_audioEmbeddingSize}). Restoring parameters into a " +
                "differently shaped VGGish embedding would silently produce a model that predicts " +
                "differently from the one that was saved.");
        }

        if (embDim != _embeddingDimension)
        {
            throw new InvalidOperationException(
                $"Loaded embedding dimension ({embDim}) doesn't match current ({_embeddingDimension}).");
        }

        if (Math.Abs(tempRes - _temporalResolution) > 0.0001)
        {
            throw new InvalidOperationException(
                $"Loaded temporal resolution ({tempRes}) doesn't match current ({_temporalResolution}).");
        }

        if (numLayers != _numEncoderLayers)
        {
            throw new InvalidOperationException(
                $"Loaded encoder layers ({numLayers}) doesn't match current ({_numEncoderLayers}).");
        }

        if (categoryCount != _supportedCategories.Count)
        {
            throw new InvalidOperationException(
                $"Loaded category count ({categoryCount}) doesn't match current ({_supportedCategories.Count}).");
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // The audio-embedding widths MUST ride along. Omitting them rebuilds VGGish at its paper
        // defaults (FC 4096, embedding 128) while the source may hold a smaller configured variant,
        // so the clone is a structurally different model: its predictions diverge from the original
        // and, at paper scale, materialising it can exhaust the test host.
        return new AudioVisualEventLocalizationNetwork<T>(
            Architecture,
            _embeddingDimension,
            _temporalResolution,
            _numEncoderLayers,
            _supportedCategories,
            audioEmbeddingFullyConnectedWidth: _audioEmbeddingFullyConnectedWidth,
            audioEmbeddingSize: _audioEmbeddingSize);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var copy = new AudioVisualEventLocalizationNetwork<T>(
            Architecture,
            _embeddingDimension,
            _temporalResolution,
            _numEncoderLayers,
            _supportedCategories,
            _optimizer,
            _lossFunction,
            audioEmbeddingFullyConnectedWidth: _audioEmbeddingFullyConnectedWidth,
            audioEmbeddingSize: _audioEmbeddingSize);

        // Copy trained weights PER LAYER from the (materialized) source rather than via the
        // model-level SetParameters(GetParameters()). The freshly-constructed copy's layers are lazy
        // (ParameterCount == 0 until first forward), and the model-level SetParameters slices the flat
        // vector by each target layer's ParameterCount — which is 0 for a lazy layer, so every slice was
        // empty, `offset` never advanced, and NO weights were applied (the clone re-randomized on its
        // first forward). Each source layer is materialized, so copying its parameter vector directly
        // into the matching copy layer lets that layer self-materialize (DenseLayer/MultiHeadAttention
        // resolve their shape from the vector length). copy.Layers[i] is the same object the cached
        // field references (_audioInputProjection, etc.) point at, so they are materialized in place.
        for (int i = 0; i < Layers.Count; i++)
        {
            var src = Layers[i];
            if (src.ParameterCount > 0)
                copy.Layers[i].SetParameters(src.GetParameters());
        }
        return copy;
    }

    #endregion
}
