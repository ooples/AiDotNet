using AiDotNet.LearningRateSchedulers;
using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Audio-visual correspondence learning network for cross-modal understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This network learns correspondences between audio and visual modalities,
/// enabling sound source localization, audio-visual retrieval, and scene understanding.
/// </para>
/// <para><b>For Beginners:</b> This model learns to connect what it "hears" with what it "sees."
///
/// For example, if you show it a video of someone playing guitar, it learns that the
/// guitar sound corresponds to the guitar in the image. This enables:
/// - Sound source localization: "Where in the image is the sound coming from?"
/// - Audio-visual retrieval: "Find images that match this sound"
/// - Scene understanding: "What objects are making sounds in this scene?"
///
/// Separate audio and visual input projections feed one shared Dense encoder. The
/// paired training objective aligns their normalized embeddings; the feature-input
/// Predict path returns two correspondence logits through the separate fusion stack.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputSize: 512,
///     outputSize: 2);
///
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 512);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 128);
///
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new AudioVisualCorrespondenceNetwork&lt;float&gt;(architecture))
///     .Build(trainX, trainY);
///
/// var inputTensor = Tensor&lt;float&gt;.CreateRandom(1, 512);
/// Tensor&lt;float&gt; embedding = result.Predict(inputTensor);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Look, Listen and Learn", "https://arxiv.org/abs/1705.08168")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, WeightDecay = 1e-5,
                Provenance = RecipeProvenance.Searched,
                Source = "Arandjelovic and Zisserman 2017, Sec. 3: the Adam optimizer with a weight "
                        + "decay of 1e-5 and a grid search over the learning rate, of which the paper "
                        + "says 1e-4 usually works well. The provenance records that the rate came from "
                        + "a search rather than being stated outright.")]
public partial class AudioVisualCorrespondenceNetwork<T> : MultimodalModelLayoutBase<T>, IAudioVisualCorrespondenceModel<T>
{
    private readonly AudioVisualCorrespondenceOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Constants

    private const int DEFAULT_EMBEDDING_DIM = 512;
    private const int DEFAULT_SAMPLE_RATE = 16000;
    private const double DEFAULT_FRAME_RATE = 25.0;
    private const int SPECTROGRAM_BINS = 128;
    private const int SPECTROGRAM_HOP = 512;
    private const int NUM_ATTENTION_HEADS = 8;

    #endregion

    #region Fields

    private readonly int _embeddingDimension;
    private readonly int _audioSampleRate;
    private readonly int _visualChannels;

    /// <summary>
    /// Real log-mel front-end, built for <see cref="_audioSampleRate"/>. Holds no trainable
    /// parameters (the filterbank is fixed), so it is not part of the layer list.
    /// </summary>
    private readonly AiDotNet.Diffusion.Audio.MelSpectrogram<T> _audioFrontEnd;
    private readonly double _videoFrameRate;
    private readonly int _numEncoderLayers;
    // Modality-specific adapters feed one registered Dense encoder. The fusion path is
    // separate from the encoder; neither path is inferred by subtracting a magic layer count.
    private List<ILayer<T>>? _sharedEncoderLayers;
    // The fusion trunk every pair head reads. In the default layout the correspondence head that follows it is the
    // last registered layer; the trunk list itself never contains a head.
    private List<ILayer<T>>? _fusionLayers;
    private ILayer<T>? _audioInputProjection;
    private ILayer<T>? _pairInputProjection;

    // Task heads on the fusion trunk. Only the correspondence head is part of Look, Listen and Learn; these serve this
    // model's synchronization and separation APIs and are trained by LearnSynchronization and LearnSeparation. Like
    // the adapters they are discovered by the model generator (parameters, clone, serialization) without joining the
    // sequential Predict path.
    private ILayer<T>? _synchronizationHead;
    private ILayer<T>? _separationHead;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _audioPositionalEmbedding;

    // Visual encoder components
    private ILayer<T>? _visualInputProjection;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _visualPositionalEmbedding;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    // Per-label mean joint embeddings learned by LearnScene and scored by ClassifyScene; persisted by RegisterState.
    private ScenePrototypeState _scenePrototypes = new();

    #endregion

    #region Non-Null Accessors

    private List<ILayer<T>> GetSharedEncoderLayers() => _sharedEncoderLayers
        ?? throw new InvalidOperationException("The shared correspondence encoder is not initialized.");
    private List<ILayer<T>> GetFusionLayers() => _fusionLayers
        ?? throw new InvalidOperationException("The correspondence fusion trunk is not initialized.");
    private ILayer<T> GetSynchronizationHead() => _synchronizationHead ?? throw new InvalidOperationException("Synchronization head not initialized.");
    private ILayer<T> GetSeparationHead() => _separationHead ?? throw new InvalidOperationException("Separation head not initialized.");
    private ILayer<T> GetAudioInputProjection() => _audioInputProjection ?? throw new InvalidOperationException("Audio input projection not initialized.");
    private ILayer<T> GetVisualInputProjection() => _visualInputProjection ?? throw new InvalidOperationException("Visual input projection not initialized.");
    private ILayer<T> GetPairInputProjection() => _pairInputProjection ?? throw new InvalidOperationException("Pair input projection not initialized.");

    #endregion

    #region IAudioVisualCorrespondenceModel Properties

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc/>
    public int AudioSampleRate => _audioSampleRate;

    /// <inheritdoc/>
    public double VideoFrameRate => _videoFrameRate;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public AudioVisualCorrespondenceNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.BinaryClassification,
            inputSize: 512,
            // The correspondence head's width follows OutputSize; the paper classifies correspondence with a 2-way softmax.
            outputSize: 2))
    {
    }

    /// <summary>
    /// Creates a new audio-visual correspondence network.
    /// </summary>
    /// <param name="architecture">Network architecture configuration.</param>
    /// <param name="options">Shared embedding width, registered encoder-block count, audio sample rate, and nominal video frame rate.</param>
    /// <param name="optimizer">Gradient-based optimizer for training.</param>
    /// <param name="lossFunction">Loss function for training.</param>
    public AudioVisualCorrespondenceNetwork(
        NeuralNetworkArchitecture<T> architecture,
        AudioVisualCorrespondenceOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _options = options ?? new AudioVisualCorrespondenceOptions();
        _options.Validate();
        Options = _options;

        _embeddingDimension = _options.EmbeddingDimension;
        _audioSampleRate = _options.AudioSampleRate;
        _visualChannels = _options.Channels;
        _audioFrontEnd = CreateAudioFrontEnd(_options.AudioSampleRate);
        _videoFrameRate = _options.VideoFrameRate;
        _numEncoderLayers = _options.NumEncoderLayers;
        // Preserve the implementation's existing Adam learning rate. The native encoder is
        // a configurable Dense/LayerNorm/Tanh stack, not the paper's convolutional network.
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 5e-5,
            });
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        InitializeLayers();
        InitializePositionalEmbeddings();
    }

    #endregion

    #region Initialization

    private void InitializePositionalEmbeddings()
    {
        if (_sharedEncoderLayers is null)
        {
            _audioPositionalEmbedding = null;
            _visualPositionalEmbedding = null;
            return;
        }

        const int maxAudioLength = 500;
        const int maxVisualLength = 256;

        _audioPositionalEmbedding = new Tensor<T>([maxAudioLength, _embeddingDimension]);
        _visualPositionalEmbedding = new Tensor<T>([maxVisualLength, _embeddingDimension]);

        InitializeSinusoidalEmbedding(_audioPositionalEmbedding, maxAudioLength);
        InitializeSinusoidalEmbedding(_visualPositionalEmbedding, maxVisualLength);
    }

    private void InitializeSinusoidalEmbedding(Tensor<T> embedding, int maxLength)
    {
        for (int pos = 0; pos < maxLength; pos++)
        {
            for (int i = 0; i < _embeddingDimension; i++)
            {
                var divTerm = Math.Exp(i * -Math.Log(10000.0) / _embeddingDimension);
                var angle = pos * divTerm;

                if (i % 2 == 0)
                {
                    embedding[pos, i] = NumOps.FromDouble(Math.Sin(angle));
                }
                else
                {
                    embedding[pos, i] = NumOps.FromDouble(Math.Cos(angle));
                }
            }
        }
    }

    #endregion

    #region IAudioVisualCorrespondenceModel Implementation

    /// <inheritdoc/>
    public Vector<T> GetAudioEmbedding(Tensor<T> audioWaveform, int sampleRate)
    {
        var spectrogram = ComputeSpectrogram(audioWaveform, sampleRate);
        var projected = ApplyAudioEncoder(spectrogram);

        return new Vector<T>(NormalizeEmbedding(PoolFeatures(projected)).ToArray());
    }

    /// <inheritdoc/>
    public Vector<T> GetVisualEmbedding(IEnumerable<Tensor<T>> frames)
    {
        var frameList = frames.ToList();
        if (frameList.Count == 0)
        {
            return new Vector<T>(_embeddingDimension);
        }

        Tensor<T>? sum = null;
        foreach (var frame in frameList)
        {
            var pooled = PoolFeatures(ApplyVisualEncoder(frame));
            sum = sum is null ? pooled : Engine.TensorAdd(sum, pooled);
        }
        if (sum is null) throw new InvalidOperationException("A nonempty frame list produced no embeddings.");
        return new Vector<T>(NormalizeEmbedding(
            Engine.TensorDivideScalar(sum, NumOps.FromDouble(frameList.Count))).ToArray());
    }

    /// <inheritdoc/>
    public T ComputeCorrespondence(Tensor<T> audioWaveform, IEnumerable<Tensor<T>> frames)
    {
        var audioEmb = GetAudioEmbedding(audioWaveform, _audioSampleRate);
        var visualEmb = GetVisualEmbedding(frames);

        return NumOps.FromDouble(VectorHelper.CosineSimilarity(audioEmb, visualEmb));
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor<T>> LocalizeSoundSource(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames)
    {
        var audioEmbedding = GetAudioEmbedding(audioWaveform, _audioSampleRate);
        var results = new List<Tensor<T>>();

        foreach (var frame in frames)
        {
            var spatialFeatures = ExtractSpatialFeatures(frame);
            var attentionMap = ComputeSoundSourceAttention(audioEmbedding, spatialFeatures, frame._shape);
            results.Add(attentionMap);
        }

        return results;
    }

    /// <inheritdoc/>
    public (double OffsetSeconds, T Confidence) CheckSynchronization(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames)
    {
        var audioEmb = GetAudioEmbedding(audioWaveform, _audioSampleRate);
        var frameList = frames.ToList();
        var visualEmb = GetVisualEmbedding(frameList);

        // Concatenate embeddings for sync prediction
        var combined = ConcatenateVectors(audioEmb, visualEmb);
        var combinedTensor = Tensor<T>.FromVector(combined);

        // The synchronization head (trained by LearnSynchronization) reads the shared fusion trunk.
        var syncOutput = GetSynchronizationHead().Forward(ApplyPairTrunk(combinedTensor));
        var offsetValue = NumOps.ToDouble(syncOutput.Data.Span[0]);

        // Compute confidence from correspondence score
        var correspondence = ComputeCorrespondence(audioWaveform, frameList);
        var confidence = NumOps.Abs(correspondence);

        return (offsetValue, confidence);
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveVisualsFromAudio(
        Tensor<T> audioWaveform,
        IEnumerable<Vector<T>> visualDatabase,
        int topK = 10)
    {
        var audioEmb = GetAudioEmbedding(audioWaveform, _audioSampleRate);
        var scores = new List<(int Index, T Score)>();
        int index = 0;

        foreach (var visualEmb in visualDatabase)
        {
            var score = NumOps.FromDouble(VectorHelper.CosineSimilarity(audioEmb, visualEmb));
            scores.Add((index, score));
            index++;
        }

        return scores
            .OrderByDescending(x => NumOps.ToDouble(x.Score))
            .Take(topK);
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveAudioFromVisuals(
        IEnumerable<Tensor<T>> frames,
        IEnumerable<Vector<T>> audioDatabase,
        int topK = 10)
    {
        var visualEmb = GetVisualEmbedding(frames);
        var scores = new List<(int Index, T Score)>();
        int index = 0;

        foreach (var audioEmb in audioDatabase)
        {
            var score = NumOps.FromDouble(VectorHelper.CosineSimilarity(visualEmb, audioEmb));
            scores.Add((index, score));
            index++;
        }

        return scores
            .OrderByDescending(x => NumOps.ToDouble(x.Score))
            .Take(topK);
    }

    /// <inheritdoc/>
    public Tensor<T> SeparateAudioByVisual(
        Tensor<T> mixedAudio,
        Tensor<T> targetVisual)
    {
        var spectrogram = ComputeSpectrogram(mixedAudio, _audioSampleRate);
        var visualEmb = ApplyVisualEncoder(targetVisual);
        var visualPooled = GlobalAveragePool(visualEmb);

        var audioFeatures = ApplyAudioEncoder(spectrogram);
        var audioPooled = GlobalAveragePool(audioFeatures);

        // Predict separation mask
        var combined = ConcatenateVectors(audioPooled, visualPooled);
        var combinedTensor = Tensor<T>.FromVector(combined);
        // One logit per mel bin from the separation head (trained by LearnSeparation), squashed to a [0, 1] mask.
        var maskOutput = GetSeparationHead().Forward(ApplyPairTrunk(combinedTensor));
        var mask = ApplySigmoid(maskOutput);

        // Apply mask to spectrogram and reconstruct
        var maskedSpec = ApplyMask(spectrogram, mask);
        return ReconstructAudio(maskedSpec);
    }

    /// <inheritdoc/>
    public string DescribeExpectedAudio(IEnumerable<Tensor<T>> frames)
    {
        var frameList = frames.ToList();
        if (frameList.Count == 0)
        {
            return "No visual content to analyze.";
        }

        var visualEmb = GetVisualEmbedding(frameList);

        // Simple scene-based audio description
        var descriptions = new List<string>
        {
            "ambient sounds",
            "environmental audio",
            "background noise"
        };

        var embMagnitude = VectorHelper.L2Norm(visualEmb);
        var magValue = NumOps.ToDouble(embMagnitude);

        if (magValue > 0.5)
        {
            descriptions.Add("active scene sounds");
        }

        return $"Expected audio: {string.Join(", ", descriptions)}";
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// The model has no text encoder, so a label is only meaningful once <see cref="LearnScene"/> has given it a
    /// prototype: the mean joint audio-visual embedding of its examples. As in prototypical networks (Snell et al.
    /// 2017), the probability of each requested label is a softmax over the negative squared Euclidean distance
    /// between this clip's joint embedding and that label's prototype. A label without a prototype is rejected rather
    /// than given an invented score.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">A requested label has no learned prototype.</exception>
    public Dictionary<string, T> ClassifyScene(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> sceneLabels)
    {
        if (audioWaveform is null) throw new ArgumentNullException(nameof(audioWaveform));
        if (frames is null) throw new ArgumentNullException(nameof(frames));
        if (sceneLabels is null) throw new ArgumentNullException(nameof(sceneLabels));

        var labelList = sceneLabels.Distinct(StringComparer.Ordinal).ToList();
        if (labelList.Count == 0) throw new ArgumentException("At least one scene label is required.", nameof(sceneLabels));
        var missing = labelList.Where(label => !_scenePrototypes.Labels.Contains(label)).ToList();
        if (missing.Count > 0)
        {
            throw new InvalidOperationException(
                $"No scene prototype has been learned for: {string.Join(", ", missing)}. " +
                "Call LearnScene with examples of each label before classifying.");
        }

        var frameList = frames.ToList();
        if (frameList.Count == 0) throw new ArgumentException("At least one visual frame is required.", nameof(frames));
        var joint = JointSceneEmbedding(audioWaveform, frameList);

        var negativeDistances = new double[labelList.Count];
        for (int index = 0; index < labelList.Count; index++)
        {
            var prototype = _scenePrototypes.Means[_scenePrototypes.Labels.IndexOf(labelList[index])];
            if (prototype.Length != joint.Length)
                throw new InvalidOperationException($"The prototype for '{labelList[index]}' has width {prototype.Length}, not {joint.Length}.");
            double squaredDistance = 0;
            for (int column = 0; column < joint.Length; column++)
            {
                double difference = joint[column] - prototype[column];
                squaredDistance += difference * difference;
            }
            negativeDistances[index] = -squaredDistance;
        }

        // Stable softmax: shift by the largest logit before exponentiating.
        double maxLogit = negativeDistances.Max();
        var weights = negativeDistances.Select(logit => Math.Exp(logit - maxLogit)).ToArray();
        double total = weights.Sum();
        var results = new Dictionary<string, T>(StringComparer.Ordinal);
        for (int index = 0; index < labelList.Count; index++)
            results[labelList[index]] = NumOps.FromDouble(weights[index] / total);
        return results;
    }

    /// <summary>
    /// Adds one labeled audio-visual example to that label's scene prototype, the running mean of its joint embeddings.
    /// </summary>
    /// <remarks>
    /// <para>Prototypes are computed with the encoders as they are when each example is added, so relearn them after
    /// further training changes the embeddings.</para>
    /// <para><b>For Beginners:</b> Show the model a few clips of each kind of scene ("music", "speech", ...) together
    /// with the label. It remembers the average of each label's clips and later picks the label whose average is
    /// closest to a new clip.</para>
    /// </remarks>
    public void LearnScene(Tensor<T> audioWaveform, IEnumerable<Tensor<T>> frames, string label)
    {
        if (audioWaveform is null) throw new ArgumentNullException(nameof(audioWaveform));
        if (frames is null) throw new ArgumentNullException(nameof(frames));
        if (string.IsNullOrWhiteSpace(label)) throw new ArgumentException("A scene label is required.", nameof(label));
        var frameList = frames.ToList();
        if (frameList.Count == 0) throw new ArgumentException("At least one visual frame is required.", nameof(frames));

        var joint = JointSceneEmbedding(audioWaveform, frameList);
        int index = _scenePrototypes.Labels.IndexOf(label);
        if (index < 0)
        {
            _scenePrototypes.Labels.Add(label);
            _scenePrototypes.Means.Add(joint);
            _scenePrototypes.Counts.Add(1);
            return;
        }

        var mean = _scenePrototypes.Means[index];
        if (mean.Length != joint.Length)
            throw new InvalidOperationException($"The prototype for '{label}' has width {mean.Length}, not {joint.Length}.");
        int count = _scenePrototypes.Counts[index] + 1;
        for (int column = 0; column < mean.Length; column++)
            mean[column] += (joint[column] - mean[column]) / count;
        _scenePrototypes.Counts[index] = count;
    }

    /// <inheritdoc/>
    public void LearnCorrespondence(
        IEnumerable<Tensor<T>> audioSamples,
        IEnumerable<IEnumerable<Tensor<T>>> visualSamples,
        int epochs = 10)
    {
        var audioList = audioSamples.ToList();
        var visualList = visualSamples.Select(v => v.ToList()).ToList();

        if (audioList.Count != visualList.Count)
        {
            throw new ArgumentException(
                $"Number of audio samples ({audioList.Count}) must match number of visual sample groups ({visualList.Count}).",
                nameof(visualSamples));
        }

        if (audioList.Count == 0)
        {
            throw new ArgumentException("At least one sample pair is required for training.", nameof(audioSamples));
        }

        if (_lossFunction is not LossFunctionBase<T> tapeLoss)
            throw new NotSupportedException("Correspondence training requires a tape-compatible loss function.");

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            T epochLoss = NumOps.Zero;
            int sampleCount = 0;
            for (int pair = 0; pair < audioList.Count; pair++)
            {
                var frames = visualList[pair];
                if (frames.Count == 0) continue;

                var (visualInput, framePatchCounts) = PackFramePatches(frames);
                T loss = TrainWithCustomObjective(audioList[pair], visualInput, (waveform, packedPatches) =>
                {
                    var (audioEmbedding, visualEmbedding) = PairedTrainingEmbeddings(waveform, packedPatches, framePatchCounts);
                    var similarity = Engine.ReduceSum(Engine.TensorMultiply(audioEmbedding, visualEmbedding),
                        new[] { 0 }, keepDims: true);
                    var target = new Tensor<T>(new[] { 1 });
                    target[0] = NumOps.One;
                    return tapeLoss.ComputeTapeLoss(similarity, target);
                }, _optimizer);
                epochLoss = NumOps.Add(epochLoss, loss);
                sampleCount++;
            }
            if (sampleCount > 0) LastLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(sampleCount));
        }
    }

    /// <summary>
    /// Trains the synchronization head, together with the encoders, adapters and fusion trunk beneath it, to predict
    /// each pair's audio-to-visual offset in seconds.
    /// </summary>
    /// <param name="audioSamples">One waveform per pair.</param>
    /// <param name="visualSamples">The frames of each pair.</param>
    /// <param name="offsetsSeconds">The true offset of each pair, in seconds.</param>
    /// <param name="epochs">Passes over the pairs.</param>
    /// <remarks>The objective is the squared error between <see cref="CheckSynchronization"/>'s offset and the target.</remarks>
    public void LearnSynchronization(
        IEnumerable<Tensor<T>> audioSamples,
        IEnumerable<IEnumerable<Tensor<T>>> visualSamples,
        IEnumerable<double> offsetsSeconds,
        int epochs = 10)
    {
        if (audioSamples is null) throw new ArgumentNullException(nameof(audioSamples));
        if (visualSamples is null) throw new ArgumentNullException(nameof(visualSamples));
        if (offsetsSeconds is null) throw new ArgumentNullException(nameof(offsetsSeconds));
        if (epochs < 0) throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs cannot be negative.");

        var audioList = audioSamples.ToList();
        var visualList = visualSamples.Select(frames => frames.ToList()).ToList();
        var offsets = offsetsSeconds.ToList();
        if (audioList.Count == 0) throw new ArgumentException("At least one sample pair is required for training.", nameof(audioSamples));
        if (visualList.Count != audioList.Count || offsets.Count != audioList.Count)
        {
            throw new ArgumentException(
                $"Audio samples ({audioList.Count}), visual sample groups ({visualList.Count}) and offsets ({offsets.Count}) must match.",
                nameof(offsetsSeconds));
        }
        if (visualList.Any(frames => frames.Count == 0))
            throw new ArgumentException("Every pair needs at least one visual frame.", nameof(visualSamples));
        if (offsets.Any(offset => double.IsNaN(offset) || double.IsInfinity(offset)))
            throw new ArgumentException("Offsets must be finite.", nameof(offsetsSeconds));

        var synchronizationLoss = new MeanSquaredErrorLoss<T>();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            T epochLoss = NumOps.Zero;
            for (int pair = 0; pair < audioList.Count; pair++)
            {
                var (visualInput, framePatchCounts) = PackFramePatches(visualList[pair]);
                var target = new Tensor<T>(new[] { 1 });
                target[0] = NumOps.FromDouble(offsets[pair]);
                T loss = TrainWithCustomObjective(audioList[pair], visualInput, (waveform, packedPatches) =>
                {
                    var (audioEmbedding, visualEmbedding) = PairedTrainingEmbeddings(waveform, packedPatches, framePatchCounts);
                    var pairFeatures = Engine.TensorConcatenate(new[] { audioEmbedding, visualEmbedding }, axis: 0);
                    var offset = GetSynchronizationHead().Forward(ApplyPairTrunk(pairFeatures));
                    return synchronizationLoss.ComputeTapeLoss(offset, target);
                }, _optimizer);
                epochLoss = NumOps.Add(epochLoss, loss);
            }
            LastLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(audioList.Count));
        }
    }

    /// <summary>
    /// Trains the separation head, together with the encoders, adapters and fusion trunk beneath it, to predict each
    /// example's per-mel-bin mask for the sound belonging to its visual.
    /// </summary>
    /// <param name="mixedAudio">One mixed waveform per example.</param>
    /// <param name="targetVisuals">The frame showing the source to keep, per example.</param>
    /// <param name="targetMasks">The ideal mask per example: <c>[128]</c> values in [0, 1], one per mel bin.</param>
    /// <param name="epochs">Passes over the examples.</param>
    /// <remarks>The objective is binary cross-entropy between <see cref="SeparateAudioByVisual"/>'s mask and the target.</remarks>
    public void LearnSeparation(
        IEnumerable<Tensor<T>> mixedAudio,
        IEnumerable<Tensor<T>> targetVisuals,
        IEnumerable<Tensor<T>> targetMasks,
        int epochs = 10)
    {
        if (mixedAudio is null) throw new ArgumentNullException(nameof(mixedAudio));
        if (targetVisuals is null) throw new ArgumentNullException(nameof(targetVisuals));
        if (targetMasks is null) throw new ArgumentNullException(nameof(targetMasks));
        if (epochs < 0) throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs cannot be negative.");

        var audioList = mixedAudio.ToList();
        var visualList = targetVisuals.ToList();
        var maskList = targetMasks.ToList();
        if (audioList.Count == 0) throw new ArgumentException("At least one example is required for training.", nameof(mixedAudio));
        if (visualList.Count != audioList.Count || maskList.Count != audioList.Count)
        {
            throw new ArgumentException(
                $"Mixed audio ({audioList.Count}), target visuals ({visualList.Count}) and masks ({maskList.Count}) must match.",
                nameof(targetMasks));
        }
        foreach (var mask in maskList)
        {
            if (mask is null || mask.Shape.Length != 1 || mask.Shape[0] != SPECTROGRAM_BINS)
                throw new ArgumentException($"Each target mask must have shape [{SPECTROGRAM_BINS}], one value per mel bin.", nameof(targetMasks));
            for (int bin = 0; bin < SPECTROGRAM_BINS; bin++)
            {
                double value = NumOps.ToDouble(mask[bin]);
                if (double.IsNaN(value) || value < 0.0 || value > 1.0)
                    throw new ArgumentException("Target mask values must lie in [0, 1].", nameof(targetMasks));
            }
        }

        var separationLoss = new BinaryCrossEntropyLoss<T>();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            T epochLoss = NumOps.Zero;
            for (int sample = 0; sample < audioList.Count; sample++)
            {
                var mask = maskList[sample];
                T loss = TrainWithCustomObjective(audioList[sample], FlattenToPatches(visualList[sample]), (waveform, framePatches) =>
                {
                    // Mirrors SeparateAudioByVisual: pooled (unnormalized) encoder features into the fusion trunk.
                    var audioPooled = PoolFeatures(ApplyAudioEncoder(ComputeSpectrogram(waveform, _audioSampleRate)));
                    var visualPooled = PoolFeatures(ApplyVisualPatches(framePatches));
                    var pairFeatures = Engine.TensorConcatenate(new[] { audioPooled, visualPooled }, axis: 0);
                    var probabilities = Engine.Sigmoid(GetSeparationHead().Forward(ApplyPairTrunk(pairFeatures)));
                    return separationLoss.ComputeTapeLoss(probabilities, mask);
                }, _optimizer);
                epochLoss = NumOps.Add(epochLoss, loss);
            }
            LastLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(audioList.Count));
        }
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Packs frames into one patch tensor for a training objective, with the per-frame patch counts needed to unpack it.
    /// </summary>
    /// <remarks>
    /// Packing only raw frame patches preserves real optimizer inputs without capturing precomputed embeddings or
    /// severing their parameter gradients. Boundaries preserve equal per-frame averaging for unequal image sizes.
    /// </remarks>
    private (Tensor<T> Patches, int[] FramePatchCounts) PackFramePatches(IReadOnlyList<Tensor<T>> frames)
    {
        var patches = frames.Select(FlattenToPatches).ToArray();
        return (Engine.TensorConcatenate(patches, axis: 0), patches.Select(frame => frame.Shape[0]).ToArray());
    }

    /// <summary>The normalized audio and visual embeddings, recomputed on the tape from the raw training inputs.</summary>
    private (Tensor<T> Audio, Tensor<T> Visual) PairedTrainingEmbeddings(
        Tensor<T> waveform, Tensor<T> packedPatches, int[] framePatchCounts)
    {
        var audioEmbedding = NormalizeEmbedding(PoolFeatures(
            ApplyAudioEncoder(ComputeSpectrogram(waveform, _audioSampleRate))));
        Tensor<T>? visualSum = null;
        int start = 0;
        foreach (int count in framePatchCounts)
        {
            var framePatches = Engine.TensorSlice(packedPatches,
                new[] { start, 0 }, new[] { count, packedPatches.Shape[1] });
            var pooled = PoolFeatures(ApplyVisualPatches(framePatches));
            visualSum = visualSum is null ? pooled : Engine.TensorAdd(visualSum, pooled);
            start += count;
        }
        if (visualSum is null) throw new InvalidOperationException("A paired objective requires visual frames.");
        var visualEmbedding = NormalizeEmbedding(Engine.TensorDivideScalar(
            visualSum, NumOps.FromDouble(framePatchCounts.Length)));
        return (audioEmbedding, visualEmbedding);
    }

    /// <summary>The unit-length joint embedding scene prototypes are built from and compared against.</summary>
    private double[] JointSceneEmbedding(Tensor<T> audioWaveform, IReadOnlyList<Tensor<T>> frames)
    {
        var audio = GetAudioEmbedding(audioWaveform, _audioSampleRate);
        var visual = GetVisualEmbedding(frames);
        // Both embeddings are unit length, so scaling their concatenation by 1/sqrt(2) keeps the joint one unit length.
        double scale = 1.0 / Math.Sqrt(2.0);
        var joint = new double[audio.Length + visual.Length];
        for (int column = 0; column < audio.Length; column++) joint[column] = NumOps.ToDouble(audio[column]) * scale;
        for (int column = 0; column < visual.Length; column++) joint[audio.Length + column] = NumOps.ToDouble(visual[column]) * scale;
        return joint;
    }

    /// <summary>
    /// Builds the log-mel front-end for a given sample rate.
    /// </summary>
    /// <remarks>
    /// <see cref="SPECTROGRAM_BINS"/> mel bins and a <see cref="SPECTROGRAM_HOP"/>-sample hop keep
    /// this model's own frequency resolution and frame rate; only the arithmetic producing them
    /// changes. The FFT length follows librosa's convention of four times the hop.
    /// </remarks>
    private static AiDotNet.Diffusion.Audio.MelSpectrogram<T> CreateAudioFrontEnd(int sampleRate)
        => new AiDotNet.Diffusion.Audio.MelSpectrogram<T>(
            sampleRate: sampleRate,
            nMels: SPECTROGRAM_BINS,
            nFft: SPECTROGRAM_HOP * 4,
            hopLength: SPECTROGRAM_HOP);

    /// <summary>
    /// Computes a real log-mel spectrogram, <c>[frames, SPECTROGRAM_BINS]</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This previously built its own array and called it a spectrogram: for each frame it read raw
    /// waveform samples at <c>startSample + bin</c>, scaled them by a linear ramp
    /// <c>(bin + 1) / SPECTROGRAM_BINS</c>, and took the log of the magnitude. There was no Fourier
    /// transform, no mel filterbank and no window function, so the second axis indexed a sample
    /// OFFSET rather than a frequency — the values carried no spectral meaning for the audio encoder
    /// to correspond against the visual stream.
    /// </para>
    /// <para>
    /// The transform is cached for the model's configured rate and rebuilt only when a caller asks
    /// for a different one, since the mel filterbank depends on the sample rate.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeSpectrogram(Tensor<T> waveform, int sampleRate)
    {
        var frontEnd = sampleRate == _audioSampleRate
            ? _audioFrontEnd
            : CreateAudioFrontEnd(sampleRate);

        return frontEnd.Forward(waveform);
    }

    private Tensor<T> ApplyAudioEncoder(Tensor<T> spectrogram)
    {
        _ = GetSharedEncoderLayers();
        EnsureLayerRandomSeedsWired();
        if (spectrogram.Shape.Length != 2 || spectrogram.Shape[1] != SPECTROGRAM_BINS)
            throw new ArgumentException("The audio encoder requires [frames, mel bins] features.", nameof(spectrogram));
        return AddPositionalEmbedding(
            ApplySharedEncoder(GetAudioInputProjection().Forward(spectrogram)), _audioPositionalEmbedding);
    }

    private Tensor<T> ApplyVisualEncoder(Tensor<T> frame)
    {
        return ApplyVisualPatches(FlattenToPatches(frame));
    }

    private Tensor<T> ApplyVisualPatches(Tensor<T> patches)
    {
        _ = GetSharedEncoderLayers();
        EnsureLayerRandomSeedsWired();
        if (patches.Shape.Length != 2 || patches.Shape[1] != checked(_visualChannels * 16 * 16))
            throw new ArgumentException("Visual patch features do not match the configured channel count.", nameof(patches));
        return AddPositionalEmbedding(
            ApplySharedEncoder(GetVisualInputProjection().Forward(patches)), _visualPositionalEmbedding);
    }

    private Tensor<T> ApplySharedEncoder(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in GetSharedEncoderLayers()) current = layer.Forward(current);
        return current;
    }

    /// <summary>The fusion trunk every pair head reads: the pair projection followed by the fusion layers.</summary>
    private Tensor<T> ApplyPairTrunk(Tensor<T> combined)
    {
        var current = GetPairInputProjection().Forward(combined);
        foreach (var layer in GetFusionLayers()) current = layer.Forward(current);
        return current;
    }

    private Tensor<T> AddPositionalEmbedding(Tensor<T> features, Tensor<T>? positions)
    {
        if (positions is null) return features;
        int count = Math.Min(features.Shape[0], positions.Shape[0]);
        var prefix = Engine.TensorSlice(positions, new[] { 0, 0 }, new[] { count, _embeddingDimension });
        var aligned = count == features.Shape[0] ? prefix : Engine.TensorConcatenate(
            new[] { prefix, new Tensor<T>(new[] { features.Shape[0] - count, _embeddingDimension }) }, axis: 0);
        return Engine.TensorAdd(features, aligned);
    }

    private Tensor<T> FlattenToPatches(Tensor<T> frame)
    {
        if (frame is null) throw new ArgumentNullException(nameof(frame));
        _options.ValidateVisualChannels(_visualChannels);
        if (frame.Shape.Length != 3)
            throw new ArgumentException("Each visual frame must have shape [channels, height, width].", nameof(frame));

        var channels = frame.Shape[^3];
        var height = frame.Shape[^2];
        var width = frame.Shape[^1];

        if (channels != _visualChannels || height <= 0 || width <= 0)
            throw new ArgumentException("Frame dimensions must be positive and channels must match the configured options.", nameof(frame));

        const int patchSize = 16;

        // Validate dimensions are divisible by patch size
        if (height % patchSize != 0 || width % patchSize != 0)
        {
            throw new ArgumentException(
                $"Image dimensions ({height}x{width}) must be divisible by patch size ({patchSize}). " +
                $"Remainder pixels would be truncated: height remainder={height % patchSize}, " +
                $"width remainder={width % patchSize}. Consider resizing the input image.",
                nameof(frame));
        }

        var numPatchesH = height / patchSize;
        var numPatchesW = width / patchSize;
        var numPatches = numPatchesH * numPatchesW;
        var patchDim = checked(channels * patchSize * patchSize);

        // Use computed patchDim instead of hardcoded 768
        var patches = new Tensor<T>([numPatches, patchDim]);

        for (int ph = 0; ph < numPatchesH; ph++)
        {
            for (int pw = 0; pw < numPatchesW; pw++)
            {
                var patchIdx = ph * numPatchesW + pw;
                var flatIdx = 0;

                for (int c = 0; c < channels && flatIdx < patchDim; c++)
                {
                    for (int py = 0; py < patchSize && flatIdx < patchDim; py++)
                    {
                        for (int px = 0; px < patchSize && flatIdx < patchDim; px++)
                        {
                            var y = ph * patchSize + py;
                            var x = pw * patchSize + px;
                            var srcIdx = c * height * width + y * width + x;

                            if (srcIdx < frame.Length)
                            {
                                patches.Data.Span[patchIdx * patchDim + flatIdx] = frame.Data.Span[srcIdx];
                            }
                            flatIdx++;
                        }
                    }
                }
            }
        }

        return patches;
    }

    private Tensor<T> ExtractSpatialFeatures(Tensor<T> frame)
    {
        return ApplyVisualEncoder(frame);
    }

    private Tensor<T> ComputeSoundSourceAttention(Vector<T> audioEmbedding, Tensor<T> spatialFeatures, int[] originalShape)
    {
        var numPositions = spatialFeatures.Shape[0];
        var attentionScores = new Tensor<T>([numPositions]);

        // Compute attention between audio and each spatial position
        for (int pos = 0; pos < numPositions; pos++)
        {
            var spatialVec = new Vector<T>(_embeddingDimension);
            for (int j = 0; j < _embeddingDimension; j++)
            {
                var idx = pos * _embeddingDimension + j;
                if (idx < spatialFeatures.Length)
                {
                    spatialVec[j] = spatialFeatures.Data.Span[idx];
                }
            }

            var score = NumOps.FromDouble(VectorHelper.CosineSimilarity(audioEmbedding, spatialVec));
            attentionScores.Data.Span[pos] = score;
        }

        // Softmax normalization
        var maxScore = NumOps.Zero;
        for (int i = 0; i < numPositions; i++)
        {
            if (NumOps.GreaterThan(attentionScores.Data.Span[i], maxScore))
            {
                maxScore = attentionScores.Data.Span[i];
            }
        }

        var sumExp = NumOps.Zero;
        for (int i = 0; i < numPositions; i++)
        {
            var shifted = NumOps.Subtract(attentionScores.Data.Span[i], maxScore);
            attentionScores.Data.Span[i] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(shifted)));
            sumExp = NumOps.Add(sumExp, attentionScores.Data.Span[i]);
        }

        for (int i = 0; i < numPositions; i++)
        {
            attentionScores.Data.Span[i] = NumOps.Divide(attentionScores.Data.Span[i], sumExp);
        }

        // Reshape to spatial dimensions
        var patchH = originalShape[^2] / 16;
        var patchW = originalShape[^1] / 16;
        if (checked(patchH * patchW) != numPositions)
            throw new InvalidOperationException("Spatial features do not match the input frame's patch grid.");

        var attentionMap = new Tensor<T>([patchH, patchW]);
        for (int i = 0; i < Math.Min(numPositions, patchH * patchW); i++)
        {
            attentionMap.Data.Span[i] = attentionScores.Data.Span[i];
        }

        return attentionMap;
    }

    private Vector<T> GlobalAveragePool(Tensor<T> features)
    {
        return new Vector<T>(PoolFeatures(features).ToArray());
    }

    private Tensor<T> PoolFeatures(Tensor<T> features)
    {
        if (features.Shape.Length != 2 || features.Shape[0] <= 0 || features.Shape[1] != _embeddingDimension)
            throw new ArgumentException("Encoder features must have nonempty shape [positions, embedding width].", nameof(features));
        return Engine.ReduceMean(features, new[] { 0 }, keepDims: false);
    }

    private Tensor<T> NormalizeEmbedding(Tensor<T> embedding)
    {
        var squaredNorm = Engine.ReduceSum(Engine.TensorMultiply(embedding, embedding), new[] { 0 }, keepDims: true);
        var norm = Engine.TensorSqrt(Engine.TensorAddScalar(squaredNorm, NumOps.FromDouble(1e-12)));
        return Engine.TensorDivide(embedding, norm);
    }

    private Vector<T> ConcatenateVectors(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(a.Length + b.Length);

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = a[i];
        }

        for (int i = 0; i < b.Length; i++)
        {
            result[a.Length + i] = b[i];
        }

        return result;
    }

    private Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        // Use IEngine vectorized Sigmoid activation
        return Engine.Sigmoid(input);
    }

    private Tensor<T> ApplyMask(Tensor<T> spectrogram, Tensor<T> mask)
    {
        var numFrames = spectrogram.Shape[0];
        var numBins = spectrogram.Shape[1];
        // One mask value per bin. Clamping a short mask to its last element gave every bin past it the same weight.
        if (mask.Length != numBins)
            throw new InvalidOperationException($"The separation mask has {mask.Length} values for {numBins} mel bins.");

        // Broadcast mask across all frames for vectorized multiplication
        var broadcastedMask = new Tensor<T>([numFrames, numBins]);
        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int bin = 0; bin < numBins; bin++)
            {
                broadcastedMask.Data.Span[frame * numBins + bin] = mask.Data.Span[bin];
            }
        }

        // Use IEngine vectorized tensor multiplication
        return Engine.TensorMultiply(spectrogram, broadcastedMask);
    }

    private Tensor<T> ReconstructAudio(Tensor<T> maskedSpectrogram)
    {
        var numFrames = maskedSpectrogram.Shape[0];
        var outputLength = numFrames * SPECTROGRAM_HOP;
        var output = TensorAllocator.Rent<T>([outputLength]);

        // Griffin-Lim style reconstruction approximation
        for (int frame = 0; frame < numFrames; frame++)
        {
            var startSample = frame * SPECTROGRAM_HOP;
            for (int i = 0; i < SPECTROGRAM_HOP && startSample + i < outputLength; i++)
            {
                var binIdx = (i * SPECTROGRAM_BINS) / SPECTROGRAM_HOP;
                binIdx = Math.Min(binIdx, SPECTROGRAM_BINS - 1);
                var specIdx = frame * SPECTROGRAM_BINS + binIdx;

                if (specIdx < maskedSpectrogram.Length)
                {
                    var logMag = NumOps.ToDouble(maskedSpectrogram.Data.Span[specIdx]);
                    var mag = Math.Exp(logMag) - 1e-8;
                    var phase = Math.Sin(2.0 * Math.PI * i / SPECTROGRAM_HOP);
                    output.Data.Span[startSample + i] = NumOps.FromDouble(mag * phase);
                }
            }
        }

        return output;
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        Layers.Clear();
        if (Architecture.OutputSize <= 0)
        {
            throw new ArgumentException(
                "AudioVisualCorrespondenceNetwork requires a positive Architecture.OutputSize for its correspondence head.",
                "architecture");
        }

        // The fusion path ends in the correspondence head, sized to the declared output (the paper's 2-way softmax by
        // default), so Predict returns exactly the width the architecture declares.
        var layout = LayerHelper<T>.CreateAudioVisualCorrespondenceLayout(
            _embeddingDimension, _numEncoderLayers, Architecture.OutputSize);
        var correspondenceHead = layout.Fusion[layout.Fusion.Count - 1];
        _sharedEncoderLayers = layout.Encoder;
        _fusionLayers = layout.Fusion.Take(layout.Fusion.Count - 1).ToList();

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // A custom layer list replaces only the feature-input Predict/Train stack. The modality adapters, shared
            // encoder, fusion trunk and task heads are still built and registered through their fields, so every
            // IAudioVisualCorrespondenceModel member keeps working instead of throwing at its first call. The default
            // correspondence head has no role here and is released.
            Layers.AddRange(Architecture.Layers);
            if (correspondenceHead is IDisposable disposable) disposable.Dispose();
        }
        else
        {
            Layers.AddRange(layout.Encoder);
            Layers.AddRange(layout.Fusion);
        }

        // The public feature-input Predict path and both modality adapters enter the same
        // encoder width. These distinct layers are discovered by the shared model generator,
        // so they belong to parameter update, clone, and serialization rather than hidden state.
        int encoderInputWidth = Architecture.CalculatedInputSize;
        _audioInputProjection = new DenseLayer<T>(encoderInputWidth, (IActivationFunction<T>?)null);
        _visualInputProjection = new DenseLayer<T>(encoderInputWidth, (IActivationFunction<T>?)null);
        _pairInputProjection = new DenseLayer<T>(_embeddingDimension, (IActivationFunction<T>?)null);
        _synchronizationHead = new DenseLayer<T>(1, (IActivationFunction<T>?)null);
        _separationHead = new DenseLayer<T>(SPECTROGRAM_BINS, (IActivationFunction<T>?)null);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        return Accelerate(input, () =>
        {
            // VGG-style Dense encoder: simple sequential forward through all layers
            // per Arandjelovic & Zisserman 2017
            Tensor<T> current = input;
            foreach (var layer in Layers)
                current = layer.Forward(current);
            return current;
        });
    }

    // Issue #1670: do NOT override ForwardForTraining to call Predict. With the
    // inference arena enabled by default, Predict copies its result out via
    // DetachFromArena (new Tensor(output.ToArray())), which SEVERS the gradient
    // tape — the loss then has no path back to any parameter and training is a
    // silent no-op. This model's forward is a plain sequential layer stack
    // (see PredictCore), which is exactly what the tape-preserving base
    // NeuralNetworkBase.ForwardForTraining already does, so the override is
    // removed and the base (correct) implementation is used.

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
    /// <inheritdoc/>
    protected override void RegisterState(ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        // Scene prototypes are learned data, not parameters: without this a restored model forgot every label.
        state.DeclareObject<ScenePrototypeState>(
            nameof(AudioVisualCorrespondenceNetwork<T>) + "." + nameof(ScenePrototypeState),
            () => _scenePrototypes,
            restored => _scenePrototypes = restored ?? new ScenePrototypeState());
    }

    /// <summary>Scene labels with the running mean joint embedding and example count of each.</summary>
    private sealed class ScenePrototypeState
    {
        public List<string> Labels { get; set; } = new();
        public List<double[]> Means { get; set; } = new();
        public List<int> Counts { get; set; } = new();
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "AudioVisualCorrespondenceNetwork",
            FeatureCount = _embeddingDimension,
            Complexity = _numEncoderLayers * 2,
            Description = "Audio-visual correspondence learning network for cross-modal understanding",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["embedding_dimension"] = _embeddingDimension,
                ["audio_sample_rate"] = _audioSampleRate,
                ["video_frame_rate"] = _videoFrameRate,
                ["num_encoder_layers"] = _numEncoderLayers
            }
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion
}
