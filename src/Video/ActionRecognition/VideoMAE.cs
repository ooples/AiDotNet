using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.ActionRecognition;

/// <summary>
/// Video Masked Autoencoder (VideoMAE) for video understanding and action recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> VideoMAE is a self-supervised learning model for video understanding.
/// It learns powerful video representations by masking random patches in video frames
/// and training the model to reconstruct the missing content. This learned representation
/// can then be used for various tasks:
/// - Action recognition (identifying what's happening in a video)
/// - Video classification
/// - Temporal reasoning
/// - Video captioning
///
/// The key insight is that learning to reconstruct masked video teaches the model
/// about motion, appearance, and temporal patterns in videos.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Vision Transformer (ViT) architecture with temporal extension
/// - Tube masking strategy for spatiotemporal masking
/// - High masking ratio (75-90%) for efficient training
/// - Joint space-time attention mechanism
/// </para>
/// <para>
/// <b>Reference:</b> Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
/// NeurIPS 2022.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a VideoMAE model for action recognition on Kinetics-400
/// var videoMAE = new VideoMAE&lt;double&gt;();
///
/// // Or configure with a custom architecture and parameters
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3,
///     outputSize: 400);
/// var model = new VideoMAE&lt;double&gt;(architecture, numClasses: 400, numFrames: 16);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training",
    "https://arxiv.org/abs/2203.12602",
    Year = 2022,
    Authors = "Zhan Tong, Yibing Song, Jue Wang, Limin Wang")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Classes,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class VideoMAE<T> : NeuralNetworkBase<T>
{
    private readonly VideoMAEOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private int _height;
    private int _width;
    private int _channels;
    private int _numFrames;
    private int _numClasses;
    private int _numFeatures;
    private readonly int _patchSize;
    private readonly int _tubeletSize;
    private double _maskRatio;
    private bool _videoMaeShapesResolved;
    private bool _useNativeMode;
    private string? _onnxModelPath;
    private InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    private readonly Random _random = RandomHelper.CreateSecureRandom();
    private bool _disposed;

    /// <summary>
    /// Default clip length (frames) when neither the caller nor the architecture declares one — the
    /// 16-frame clips of the VideoMAE paper.
    /// </summary>
    private const int DefaultNumFrames = 16;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether training is supported.
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the input height for frames.
    /// </summary>
    internal int InputHeight => _height;

    /// <summary>
    /// Gets the input width for frames.
    /// </summary>
    internal int InputWidth => _width;

    /// <summary>
    /// Gets the number of frames processed.
    /// </summary>
    internal int NumFrames => _numFrames;

    /// <summary>
    /// Gets the number of action classes.
    /// </summary>
    internal int NumClasses => _numClasses;

    /// <summary>
    /// Gets the masking ratio for pretraining.
    /// </summary>
    internal double MaskRatio => _maskRatio;

    /// <summary>
    /// Gets whether using native mode (trainable) or ONNX mode (inference only).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of VideoMAE with default architecture (224x224, 400 classes).
    /// </summary>
    public VideoMAE()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 224, inputWidth: 224, inputDepth: 3,
            outputSize: 400)) { }

    /// <summary>
    /// Initializes a new instance of the VideoMAE class in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">The number of action classes for classification.</param>
    /// <param name="numFrames">The number of video frames per clip. When the architecture declares a
    /// frame count (<see cref="NeuralNetworkArchitecture{T}.InputFrames"/> &gt; 0) that count is used; passing a
    /// different non-default value throws, since the two would describe different clips.</param>
    /// <param name="numFeatures">The embedding dimension.</param>
    /// <param name="maskRatio">The masking ratio for pretraining (default: 0.9).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a trainable VideoMAE model.
    /// Use this when you want to train or fine-tune the model on your own video data.
    /// </para>
    /// </remarks>
    public VideoMAE(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 400,
        int numFrames = DefaultNumFrames,
        int numFeatures = 768,
        double maskRatio = 0.9,
        VideoMAEOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new VideoMAEOptions();
        Options = _options;

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 224;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _numFrames = ResolveNumFrames(architecture, numFrames);
        _numFeatures = numFeatures;
        _patchSize = 16;
        _tubeletSize = 2;
        _maskRatio = maskRatio;
        _useNativeMode = true;
        _onnxModelPath = null;
        _optimizer = optimizer;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the VideoMAE class in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="numClasses">The number of action classes for classification.</param>
    /// <param name="numFrames">The number of video frames per clip. When the architecture declares a
    /// frame count (<see cref="NeuralNetworkArchitecture{T}.InputFrames"/> &gt; 0) that count is used; passing a
    /// different non-default value throws, since the two would describe different clips.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor loads a pre-trained VideoMAE model from ONNX format.
    /// Use this for fast inference when you don't need to train the model.
    /// </para>
    /// </remarks>
    public VideoMAE(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 400,
        int numFrames = DefaultNumFrames,
        VideoMAEOptions? options = null)
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new VideoMAEOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"VideoMAE ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 224;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _numFrames = ResolveNumFrames(architecture, numFrames);
        _numFeatures = 768;
        _patchSize = 16;
        _tubeletSize = 2;
        _maskRatio = 0.9;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    /// <summary>
    /// Resolves the clip length from the architecture and the <c>numFrames</c> constructor argument.
    /// </summary>
    /// <remarks>
    /// The architecture is the model's declared input contract (<c>GetInputShape()</c> is
    /// <c>[InputFrames, C, H, W]</c> for a temporal-video architecture), so a frame count it declares
    /// wins — the same rule BSVD applies to <c>Architecture.InputFrames</c>. Previously the constructors
    /// ignored it and always used <c>numFrames</c> (default 16), so an architecture declaring 8-frame
    /// clips produced a model reporting and masking for 16. An explicit <c>numFrames</c> that disagrees
    /// with the declared count is rejected rather than silently overridden. The parameter's default
    /// value cannot be told apart from an explicit 16, so it defers to the architecture. The rule lives
    /// in <see cref="VideoClipFrameCount"/>, shared with TimeSformer.
    /// </remarks>
    private static int ResolveNumFrames(NeuralNetworkArchitecture<T> architecture, int numFrames)
        => VideoClipFrameCount.Resolve(architecture, numFrames, DefaultNumFrames);

    #endregion

    #region Public Methods

    /// <summary>
    /// Classifies actions in a video clip.
    /// </summary>
    /// <param name="video">Video tensor [T, C, H, W] or [B, T, C, H, W].</param>
    /// <returns>Action class probabilities [NumClasses] or [B, NumClasses].</returns>
    public Tensor<T> ClassifyAction(Tensor<T> video)
    {
        bool hasBatch = video.Rank == 5;
        if (!hasBatch)
        {
            video = AddBatchDimension5D(video);
        }

        var features = EncodeVideo(video);
        var logits = ClassificationForward(features);
        var probs = ApplySoftmax(logits);

        if (!hasBatch)
        {
            probs = RemoveBatchDimension(probs);
        }

        return probs;
    }

    /// <summary>
    /// Gets the top-k predicted actions for a video.
    /// </summary>
    /// <param name="video">Video tensor.</param>
    /// <param name="k">Number of top predictions to return.</param>
    /// <returns>List of (classIndex, probability) tuples.</returns>
    public List<(int ClassIndex, double Probability)> GetTopKPredictions(Tensor<T> video, int k = 5)
    {
        if (video.Rank == 5)
            throw new ArgumentException("GetTopKPredictions only supports single video input [T,C,H,W]. Remove batch dimension.", nameof(video));

        var probs = ClassifyAction(video);
        var results = new List<(int ClassIndex, double Probability)>();

        for (int i = 0; i < probs.Data.Length; i++)
        {
            results.Add((i, Convert.ToDouble(probs.Data.Span[i])));
        }

        return results.OrderByDescending(x => x.Probability).Take(k).ToList();
    }

    /// <summary>
    /// Performs masked autoencoder pretraining on a video.
    /// </summary>
    /// <param name="video">Video tensor [T, C, H, W] or [B, T, C, H, W].</param>
    /// <returns>Reconstruction loss.</returns>
    public T PretrainMAE(Tensor<T> video)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException("Pretraining is not supported in ONNX mode.");
        }

        bool hasBatch = video.Rank == 5;
        if (!hasBatch)
        {
            video = AddBatchDimension5D(video);
        }

        // Create the tube mask on this clip's own patch grid (one spatial mask per clip, shared by
        // every tubelet).
        var mask = CreateTubeMask(video.Shape[0], video.Shape[3] / _patchSize, video.Shape[4] / _patchSize);

        // Encode visible patches
        var visibleFeatures = EncodeVisiblePatches(video, mask);

        // Decode to reconstruct full video
        var reconstruction = DecodeForReconstruction(visibleFeatures);

        // Compute reconstruction loss on masked patches
        T loss = ComputeReconstructionLoss(reconstruction, video, mask);

        return loss;
    }

    /// <summary>
    /// Extracts video features for downstream tasks.
    /// </summary>
    /// <param name="video">Video tensor.</param>
    /// <returns>Feature tensor.</returns>
    public Tensor<T> ExtractFeatures(Tensor<T> video)
    {
        bool hasBatch = video.Rank == 5;
        if (!hasBatch)
        {
            video = AddBatchDimension5D(video);
        }

        var features = EncodeVideo(video);

        if (!hasBatch)
        {
            features = RemoveBatchDimension(features);
        }

        return features;
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return ClassifyAction(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// VideoMAE's forward is the tubelet patch-embedding + transformer encoder
    /// in <see cref="EncodeVideo"/>/<see cref="ClassifyAction"/>, not a
    /// sequential pass over the flat <c>Layers</c> list (Layers[0] is the
    /// tubelet conv that consumes channels*tubeletSize input; the remaining
    /// layers operate on pooled features and the classification head). The
    /// base <see cref="NeuralNetworkBase{T}.ForwardForTraining"/> runs the
    /// layers in order and feeds the raw 5-D video straight into a transformer
    /// block. Route the training forward through the real graph so the tape
    /// records the actual operations.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        // EncodeVideo/PatchEmbed operate on batched clips [B, T, C, H, W] and
        // index the frame/channel/spatial axes positionally. Every inference
        // path (ClassifyAction, ExtractFeatures, PretrainMAE) normalizes a
        // single unbatched clip [T, C, H, W] to rank 5 before encoding; the
        // training forward must do the same or PatchEmbed reads a missing axis
        // (IndexOutOfRange) on the rank-4 input the harness feeds. The batch
        // dim is prepended on the raw leaf input, so no gradient flows back
        // through the copy and the tape is unaffected. The output is left
        // batched ([1, NumClasses]); TrainWithTape reshapes the target to a
        // matching leading-1 batch, so no tape-severing RemoveBatchDimension
        // is needed on the logits.
        if (input.Rank != 5)
        {
            input = AddBatchDimension5D(input);
        }

        // Train on raw logits, not the post-softmax probabilities ClassifyAction
        // returns: this model is wired with CrossEntropyWithLogitsLoss<T>, which
        // applies softmax internally. Feeding probabilities into the loss would
        // double-normalize the head and produce wrong gradients on every step.
        // Inference (ClassifyAction) keeps the softmax; training drops it.
        var features = EncodeVideo(input);
        return ClassificationForward(features);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// VideoMAE's <c>Layers</c> are NOT a plain sequential chain over the raw
    /// <c>[C, H, W]</c> architecture input: <see cref="PatchEmbed"/> folds
    /// <c>tubeletSize</c> consecutive frames into <c>Layers[0]</c>'s channel
    /// axis, so the tubelet convolution consumes <c>channels * tubeletSize</c>
    /// input, not <c>channels</c>. The base <see cref="NeuralNetworkBase{T}.ResolveLazyLayerShapes"/>
    /// walk feeds the raw (channels-wide) architecture input and resolves the
    /// tubelet conv to <c>InputDepth = channels</c>; the first real forward then
    /// feeds <c>channels * tubeletSize</c> and throws
    /// "Expected input depth {channels}, but got {channels*tubeletSize}". Resolve
    /// the lazy layers through the model's OWN forward on a dummy single-tubelet
    /// clip so every layer materializes at the shape it actually sees on the
    /// classification forward path (the reconstruction/decoder tail is exercised
    /// only by pretraining and resolves lazily there).
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_videoMaeShapesResolved)
        {
            return;
        }
        _videoMaeShapesResolved = true;

        if (!_useNativeMode || Layers is null || Layers.Count == 0)
        {
            return;
        }

        // [1, tubeletSize, channels, H, W]: PatchEmbed folds the tubelet axis
        // into channels*tubeletSize, resolving Layers[0]; the encoder loop and
        // classification head then resolve on the pooled features.
        var probe = new Tensor<T>([1, _tubeletSize, _channels, _height, _width]);
        var features = EncodeVideo(probe);
        ClassificationForward(features);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use native mode constructor for training.");
        }

        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    #endregion

    #region Private Methods

    private Tensor<T> EncodeVideo(Tensor<T> video)
    {
        if (!_useNativeMode)
        {
            return RunOnnxInference(video);
        }

        int batchSize = video.Shape[0];
        int numTubelets = video.Shape[1] / _tubeletSize;

        // Reshape video to batch of frame pairs. PatchEmbed folds the tubelet
        // axis into the leading dim, producing [batchSize * numTubelets, ...].
        var patchEmbedded = PatchEmbed(video);

        // Apply encoder blocks. Each Layers[i] is a shape-preserving
        // Conv(numFeatures, 3, 1, 1) whose built-in ReLU records on the autodiff
        // tape, so a straight sequential dispatch keeps the gradient path intact.
        // The previous code applied a SECOND, Transform-based GELU on top of the
        // layer's own activation, which had no GradFn and severed the tape, so
        // gradients never reached any encoder or embedding weight (frozen
        // training). A residual skip was NOT added here on purpose: with
        // ReLU-activated blocks (output >= 0) an additive skip makes the feature
        // magnitude grow monotonically across the 12-block stack, blowing up the
        // classification logits and saturating the softmax so the output stops
        // responding to input changes.
        var features = patchEmbedded;
        int encoderLayerCount = Math.Min(VideoMAELayerLayout.EncoderEndIndex, Layers.Count);
        for (int i = VideoMAELayerLayout.FirstEncoderBlockIndex; i < encoderLayerCount; i++)
        {
            features = Layers[i].Forward(features);
        }

        // Spatial global average pool: [B * numTubelets, C, 1, 1].
        var pooled = GlobalAveragePool(features);

        // Temporal pool: collapse the tubelet axis that PatchEmbed folded into
        // the batch dimension back down so the result is grouped per input
        // video ([batchSize, C, 1, 1]). Without this, downstream
        // RemoveBatchDimension assumes a leading dim of 1 and throws
        // "Destination is too short" whenever numFrames > tubeletSize (i.e.
        // numTubelets > 1), and the classification head produced one logit row
        // per tubelet instead of one per video.
        return PoolTubelets(pooled, batchSize, numTubelets);
    }

    /// <summary>
    /// Averages the per-tubelet feature rows that <see cref="PatchEmbed"/>
    /// folded into the batch dimension back down to one row per input video.
    /// </summary>
    /// <param name="pooled">Spatially pooled features of shape
    /// [batchSize * numTubelets, channels, 1, 1].</param>
    /// <param name="batchSize">Number of input videos.</param>
    /// <param name="numTubelets">Tubelets per video.</param>
    /// <returns>Temporally pooled features [batchSize, channels, 1, 1].</returns>
    private Tensor<T> PoolTubelets(Tensor<T> pooled, int batchSize, int numTubelets)
    {
        if (numTubelets <= 1)
        {
            return pooled;
        }

        // Express the temporal pooling as engine tensor ops so the operation
        // stays on the autodiff tape. The previous scalar-read/write loop
        // (sum-then-write per element) severed the gradient path: the classifier
        // loss stopped at PoolTubelets and never reached the encoder weights,
        // leaving the video encoder frozen across training. The reshape +
        // reduce-along-axis form below records as tape-tracked ops on the
        // forward pass and produces the same numeric values.
        int channels = pooled.Shape[1];

        // Reshape [B*T, C, 1, 1] -> [B, T, C] so we can mean over axis 1 (T) via the
        // engine's ReduceMean (tape-aware). The Reshape -> ReduceMean -> Reshape chain
        // is all engine ops, so the gradient flows from the classifier head back
        // through the pooled features and into the tubelet encoder.
        var reshaped = Engine.Reshape(pooled, new[] { batchSize, numTubelets, channels });
        var meanBC = Engine.ReduceMean(reshaped, new[] { 1 }, keepDims: false);  // [B, C]
        // Restore the [B, C, 1, 1] output shape the caller expects.
        return Engine.Reshape(meanBC, new[] { batchSize, channels, 1, 1 });
    }

    private Tensor<T> PatchEmbed(Tensor<T> video)
    {
        int batchSize = video.Shape[0];
        int numFrames = video.Shape[1];
        int channels = video.Shape[2];
        int height = video.Shape[3];
        int width = video.Shape[4];

        // Combine frames into tubelet pairs
        int numTubelets = numFrames / _tubeletSize;
        var tubeletInput = new Tensor<T>([batchSize * numTubelets, channels * _tubeletSize, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < numTubelets; t++)
            {
                for (int ts = 0; ts < _tubeletSize; ts++)
                {
                    int frameIdx = t * _tubeletSize + ts;
                    for (int c = 0; c < channels; c++)
                    {
                        for (int h = 0; h < height; h++)
                        {
                            for (int w = 0; w < width; w++)
                            {
                                tubeletInput[b * numTubelets + t, ts * channels + c, h, w] = video[b, frameIdx, c, h, w];
                            }
                        }
                    }
                }
            }
        }

        // Apply patch embedding layer
        if (Layers.Count > 0)
        {
            return Layers[0].Forward(tubeletInput);
        }

        return tubeletInput;
    }

    private Tensor<T> ClassificationForward(Tensor<T> features)
    {
        // Classification head: feature-reduce conv, (global pool), classifier Dense — see
        // VideoMAELayerLayout for the indices.
        if (_useNativeMode && Layers.Count > VideoMAELayerLayout.ClassifierIndex)
        {
            // features: [B, C, 1, 1] pooled per-video embedding from EncodeVideo.
            // Apply the 1x1 feature-reduce conv (Layers[13], Conv+ReLU on the
            // autodiff tape), flatten to [B, C], and project to numClasses raw
            // logits with the final Dense (Layers[15]). The previous code returned
            // the SECOND global-pooling layer (Layers[14]) output, which reduced
            // the whole embedding to a single scalar — softmax over one element is
            // 1.0 for EVERY input, so the classifier could neither distinguish
            // inputs nor train (constant output, zero gradient). Layers[15] emits
            // logits (identity activation); ClassifyAction softmaxes for inference
            // and training uses CrossEntropyWithLogitsLoss.
            var reduced = Layers[VideoMAELayerLayout.FeatureReduceIndex].Forward(features);
            int batch = reduced.Shape[0];
            int channels = reduced.Shape[1];
            var flat = Engine.Reshape(reduced, new[] { batch, channels });
            return Layers[VideoMAELayerLayout.ClassifierIndex].Forward(flat);
        }

        // In ONNX mode, this is handled by RunOnnxInference
        int batchSize = features.Shape[0];
        return new Tensor<T>([batchSize, _numClasses]);
    }

    private Tensor<T> RunOnnxInference(Tensor<T> input)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(VideoMAE<T>));
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_onnxSession.InputMetadata.Keys.First(), onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
    }

    /// <summary>
    /// Creates a tube mask on the model's configured patch grid.
    /// </summary>
    internal bool[,,] CreateTubeMask(int batchSize)
        => CreateTubeMask(batchSize, _height / _patchSize, _width / _patchSize);

    /// <summary>
    /// Creates a tube mask: one random spatial mask per clip, shared by every tubelet of that clip.
    /// </summary>
    /// <remarks>
    /// Tube masking (Tong et al. 2022, Sec. 3.3; <c>TubeMaskingGenerator</c> in the reference code) masks
    /// <c>int(maskRatio * patchesPerFrame)</c> spatial positions and extends each through time, so the
    /// masked fraction of the whole clip equals <c>maskRatio</c> regardless of its length. The mask is
    /// indexed <c>[clip, patchRow, patchCol]</c> and applied to every tubelet, so the masked count must be
    /// computed from the SPATIAL patch count. It used to be computed from
    /// <c>numTubelets * patchesPerFrame</c> while sampling from only <c>patchesPerFrame</c> indices, so
    /// for any clip with two or more tubelets <c>Take(numMasked)</c> took every index and the encoder saw
    /// nothing at all.
    /// </remarks>
    internal bool[,,] CreateTubeMask(int batchSize, int patchesH, int patchesW)
    {
        int spatialPatches = patchesH * patchesW;
        int numMasked = (int)(spatialPatches * _maskRatio);

        var mask = new bool[batchSize, patchesH, patchesW];

        for (int b = 0; b < batchSize; b++)
        {
            var indices = Enumerable.Range(0, spatialPatches).OrderBy(_ => _random.Next()).Take(numMasked).ToHashSet();

            for (int h = 0; h < patchesH; h++)
            {
                for (int w = 0; w < patchesW; w++)
                {
                    mask[b, h, w] = indices.Contains(h * patchesW + w);
                }
            }
        }

        return mask;
    }

    private Tensor<T> EncodeVisiblePatches(Tensor<T> video, bool[,,] mask)
    {
        var patchEmbedded = PatchEmbed(video);

        // Apply mask (zero out masked patches)
        // Note: patchEmbedded has shape [B * numTubelets, C, H, W] while mask has shape [B, patchesH, patchesW]
        int batchSize = video.Shape[0];
        int numTubelets = video.Shape[1] / _tubeletSize;
        int channels = patchEmbedded.Shape[1];
        int height = patchEmbedded.Shape[2];
        int width = patchEmbedded.Shape[3];

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < numTubelets; t++)
            {
                int tubeletIdx = b * numTubelets + t;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        if (mask[b, h % mask.GetLength(1), w % mask.GetLength(2)])
                        {
                            for (int c = 0; c < channels; c++)
                            {
                                patchEmbedded[tubeletIdx, c, h, w] = NumOps.Zero;
                            }
                        }
                    }
                }
            }
        }

        // Apply encoder blocks. Each Layers[i] is a shape-preserving
        // Conv(numFeatures, 3, 1, 1) whose built-in ReLU records on the autodiff
        // tape, so a straight sequential dispatch keeps the gradient path intact.
        // The previous code applied a SECOND, Transform-based GELU on top of the
        // layer's own activation, which had no GradFn and severed the tape, so
        // gradients never reached any encoder or embedding weight (frozen
        // training). A residual skip was NOT added here on purpose: with
        // ReLU-activated blocks (output >= 0) an additive skip makes the feature
        // magnitude grow monotonically across the 12-block stack, blowing up the
        // classification logits and saturating the softmax so the output stops
        // responding to input changes.
        var features = patchEmbedded;
        int encoderLayerCount = Math.Min(VideoMAELayerLayout.EncoderEndIndex, Layers.Count);
        for (int i = VideoMAELayerLayout.FirstEncoderBlockIndex; i < encoderLayerCount; i++)
        {
            features = Layers[i].Forward(features);
        }

        return features;
    }

    /// <summary>
    /// Runs the reconstruction decoder blocks and the reconstruction head over encoder features.
    /// </summary>
    /// <param name="features">Encoder features [B * numTubelets, numFeatures, patchesH, patchesW].</param>
    /// <returns>Per-patch pixel predictions [B * numTubelets, channels * tubeletSize * P * P, patchesH, patchesW].</returns>
    /// <remarks>
    /// The decoder slice is addressed through <see cref="VideoMAELayerLayout"/>. It used to start at a
    /// hard-coded index 15 — the classifier's <c>DenseLayer</c> — so pretraining pushed the encoder maps
    /// through the class head, ran three of the four decoder blocks, used the fourth as the "head", and
    /// never reached the real reconstruction head (whose output is the only one sized to a tubelet patch).
    /// </remarks>
    internal Tensor<T> DecodeForReconstruction(Tensor<T> features)
    {
        var decoded = features;

        if (_useNativeMode && Layers.Count > VideoMAELayerLayout.ReconstructionHeadIndex)
        {
            for (int i = VideoMAELayerLayout.FirstDecoderBlockIndex; i < VideoMAELayerLayout.ReconstructionHeadIndex; i++)
            {
                decoded = Layers[i].Forward(decoded);
                decoded = ApplyGELU(decoded);
            }

            decoded = Layers[VideoMAELayerLayout.ReconstructionHeadIndex].Forward(decoded);
        }

        return decoded;
    }

    /// <summary>
    /// Mean squared error between the predicted and the true pixels of the MASKED tubelet patches.
    /// </summary>
    /// <param name="reconstructed">Reconstruction-head output
    /// [B * numTubelets, channels * tubeletSize * P * P, patchesH, patchesW].</param>
    /// <param name="original">The clip [B, T, C, H, W].</param>
    /// <param name="mask">Tube mask [B, patchesH, patchesW] (true = masked), shared by all tubelets.</param>
    /// <remarks>
    /// <para>
    /// VideoMAE (Tong et al. 2022, Sec. 3.3) reconstructs each masked tubelet patch — the
    /// <c>tubeletSize x P x P</c> pixels of every channel at that position — and averages the squared
    /// error over masked patches only. Prediction channel <c>((ts * C + c) * P + y) * P + x</c> at patch
    /// <c>(ph, pw)</c> of tubelet <c>t</c> is compared with pixel
    /// <c>original[b, t * tubeletSize + ts, c, ph * P + y, pw * P + x]</c>; the <c>(ts, c)</c> order matches
    /// how <see cref="PatchEmbed"/> folds a tubelet into the embedding conv's channels.
    /// </para>
    /// <para>
    /// The previous loss treated the head's leading axis (<c>B * numTubelets</c>) as the clip batch and
    /// compared channel <c>c</c> at patch <c>(h, w)</c> with the frame-average of pixel <c>(h, w)</c> of
    /// channel <c>c % C</c> — not the patch's pixels at all — and read past the end of the clip as soon
    /// as a clip had two or more tubelets (IndexOutOfRange from <see cref="PretrainMAE"/>).
    /// </para>
    /// <para>
    /// The target is raw pixels. The reference implementation's optional per-patch target normalisation
    /// (<c>normlize_target</c>) is not applied.
    /// </para>
    /// </remarks>
    internal T ComputeReconstructionLoss(Tensor<T> reconstructed, Tensor<T> original, bool[,,] mask)
    {
        if (original.Rank != 5)
        {
            throw new ArgumentException("The original clip must be [B, T, C, H, W].", nameof(original));
        }

        int batchSize = original.Shape[0];
        int numFrames = original.Shape[1];
        int channels = original.Shape[2];
        int numTubelets = numFrames / _tubeletSize;
        int patch = _patchSize;
        int patchDim = channels * _tubeletSize * patch * patch;
        int patchesH = mask.GetLength(1);
        int patchesW = mask.GetLength(2);

        if (reconstructed.Rank != 4
            || reconstructed.Shape[0] != batchSize * numTubelets
            || reconstructed.Shape[1] != patchDim
            || reconstructed.Shape[2] != patchesH
            || reconstructed.Shape[3] != patchesW)
        {
            throw new InvalidOperationException(
                $"Reconstruction shape [{string.Join(", ", reconstructed.Shape.ToArray())}] does not match the " +
                $"expected per-patch prediction [{batchSize * numTubelets}, {patchDim}, {patchesH}, {patchesW}] " +
                "for this clip and mask.");
        }

        if (mask.GetLength(0) != batchSize
            || patchesH * patch > original.Shape[3]
            || patchesW * patch > original.Shape[4])
        {
            throw new ArgumentException("The mask does not match the clip's batch size and patch grid.", nameof(mask));
        }

        T loss = NumOps.Zero;
        int count = 0;

        for (int b = 0; b < batchSize; b++)
        {
            for (int ph = 0; ph < patchesH; ph++)
            {
                for (int pw = 0; pw < patchesW; pw++)
                {
                    if (!mask[b, ph, pw])
                    {
                        continue;
                    }

                    for (int t = 0; t < numTubelets; t++)
                    {
                        int row = b * numTubelets + t;
                        for (int ts = 0; ts < _tubeletSize; ts++)
                        {
                            int frame = t * _tubeletSize + ts;
                            for (int c = 0; c < channels; c++)
                            {
                                for (int y = 0; y < patch; y++)
                                {
                                    for (int x = 0; x < patch; x++)
                                    {
                                        int channel = (((ts * channels) + c) * patch + y) * patch + x;
                                        T diff = NumOps.Subtract(
                                            reconstructed[row, channel, ph, pw],
                                            original[b, frame, c, ph * patch + y, pw * patch + x]);
                                        loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return count > 0 ? NumOps.Divide(loss, NumOps.FromDouble(count)) : NumOps.Zero;
    }

    private Tensor<T> GlobalAveragePool(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        // Tape-safe spatial mean expressed as engine ops (mirrors PoolTubelets):
        // reshape [B, C, H, W] -> [B, C, H*W], reduce-mean over the spatial axis,
        // restore [B, C, 1, 1]. The previous scalar sum/write loop built a fresh
        // tensor element by element, which has no GradFn and severed the autodiff
        // tape between the classifier head and the encoder/tubelet-embedding
        // weights — the whole video encoder stayed frozen across training.
        var reshaped = Engine.Reshape(input, new[] { batchSize, channels, height * width });
        var meanBC = Engine.ReduceMean(reshaped, new[] { 2 }, keepDims: false);  // [B, C]
        return Engine.Reshape(meanBC, new[] { batchSize, channels, 1, 1 });
    }

    private Tensor<T> ApplyGELU(Tensor<T> input)
    {
        return input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            double c = Math.Sqrt(2.0 / Math.PI);
            double gelu = 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
            return NumOps.FromDouble(gelu);
        });
    }

    private Tensor<T> ApplySoftmax(Tensor<T> input)
    {
        return Engine.Softmax(input);
    }

    private Tensor<T> AddBatchDimension5D(Tensor<T> tensor)
    {
        int t = tensor.Shape[0];
        int c = tensor.Shape[1];
        int h = tensor.Shape[2];
        int w = tensor.Shape[3];

        var result = new Tensor<T>([1, t, c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++)
        {
            newShape[i] = tensor.Shape[i + 1];
        }

        var result = new Tensor<T>(newShape);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    #endregion

    #region Abstract Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            ClearLayers();
            return;
        }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoMAELayers(
                _channels,
                _height,
                _width,
                _numFeatures,
                _numClasses,
                _tubeletSize));
        }
    }

    // UpdateParameters redistributed the vector across Layers, which the base already folds -- and
    // did it less safely: the `offset + count <= parameters.Length` guard silently left the
    // remaining layers untouched on a short vector instead of failing. Removed under AIDN082.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "VideoMAE" },
            { "Description", "Video Masked Autoencoder for Action Recognition" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "NumFrames", _numFrames },
            { "NumClasses", _numClasses },
            { "NumFeatures", _numFeatures },
            { "MaskRatio", _maskRatio },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count }
        };

        return new ModelMetadata<T>
        {
            AdditionalInfo = additionalInfo,
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion

    #region IDisposable

    /// <summary>
    /// Releases the unmanaged resources and optionally releases managed resources.
    /// </summary>
    /// <param name="disposing">True to release both managed and unmanaged resources; false to release only unmanaged resources.</param>
    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _onnxSession?.Dispose();
                _onnxSession = null;
            }
            _disposed = true;
        }
        base.Dispose(disposing);
    }

    #endregion
}
