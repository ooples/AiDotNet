using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// SAM-HQ: Segment Anything in High Quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM-HQ extends Meta's Segment Anything Model with a High-Quality output
/// token that produces significantly sharper and more accurate mask boundaries. While SAM sometimes
/// produces coarse masks (especially on thin structures like bicycle spokes or fences), SAM-HQ adds
/// learnable components that refine boundaries to pixel-level precision.
///
/// Common use cases:
/// - High-precision object segmentation where boundary quality matters
/// - Thin structure segmentation (wires, fences, poles)
/// - Medical imaging requiring precise boundaries
/// - Any SAM use case where mask quality needs improvement
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Adds an HQ output token alongside SAM's original output tokens
/// - Global-local feature fusion: combines early ViT features (local) with final features (global)
/// - Trained on only 44K fine-grained masks from HQSeg-44K dataset
/// - +17.6 mBIoU improvement on DIS-val5K (thin/complex structures)
/// - Backbone: ViT-B/L/H (same as SAM)
/// </para>
/// <para>
/// <b>Reference:</b> Ke et al., "Segment Anything in High Quality", NeurIPS 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SAM-HQ model for high-quality boundary segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new SAMHQ&lt;double&gt;(architecture);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new SAMHQ&lt;double&gt;(architecture,
///     options: new SAMHQOptions { NumClasses = 1, DropRate = 0.1, ModelSize = SAMHQModelSize.ViTBase });
///
/// // Or load a pre-trained ONNX model for precise mask boundaries
/// var onnxModel = new SAMHQ&lt;double&gt;(architecture, "sam_hq_vit_h.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Segment Anything in High Quality", "https://arxiv.org/abs/2306.01567", Year = 2023, Authors = "Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu")]
public partial class SAMHQ<T> : Common.PromptableSegmentationBase<T>
{
    /// <inheritdoc />
    /// <remarks>Downsamples by 16, not the family's 32 - measured: [1,3,64,64] returns [1,C,4,4].</remarks>
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => SpatialStrideContract(inputRank, 16);

    private readonly SAMHQOptions _options;

    #region Fields

    // Only SAM-HQ's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed, _encoderLayerEnd and the
    // promptable state (_imageEmbedding, _imageSet) all come from PromptableSegmentationBase ->
    // SegmentationModelBase.
    private readonly SAMHQModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;

    #endregion

    #region Properties

    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal SAMHQModelSize ModelSize => _modelSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes SAM-HQ in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW, as used in the SAM-HQ paper).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss; the paper uses focal loss
    /// + dice loss + IoU loss).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable SAM-HQ model. The HQ output token learns to
    /// produce sharper boundaries than the original SAM. Training uses only 44K fine-grained masks
    /// from the HQSeg-44K dataset, making it efficient to fine-tune.
    /// </para>
    /// </remarks>
    public SAMHQ(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SAMHQOptions? options = null)
        // The base resolves numClasses/native-mode, and defaults `optimizer` LAZILY via
        // CreateDefaultOptimizer() - which is why null is passed straight through instead of
        // `optimizer ?? new AdamWOptimizer<...>(this)`, an expression that cannot appear in a
        // constructor initializer.
        : base(architecture, optimizer, lossFunction, (options ??= new SAMHQOptions()).NumClasses)
    {
        _options = options;
        Options = _options;
        // SAM-HQ's own 1024x1024 input default, which differs from the base's 512x512.
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _modelSize = _options.ModelSize;
        _dropRate = _options.DropRate;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes SAM-HQ in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained SAM-HQ model from an ONNX file for fast inference.
    /// ONNX mode does not support training.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public SAMHQ(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SAMHQOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry and opens the InferenceSession - the same twenty lines this used to repeat.
        : base(architecture, onnxModelPath, (options ??= new SAMHQOptions()).NumClasses)
    {
        _options = options;
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _modelSize = _options.ModelSize;
        _dropRate = _options.DropRate;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through SAM-HQ to produce high-quality segmentation masks.
    /// </summary>
    /// <param name="input">The input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel mask logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass an image and get back high-quality segmentation masks.
    /// The HQ output token ensures boundary precision is significantly better than standard SAM.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : PredictOnnx(input);
    }

    #endregion

    #region Private Methods

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(SAMHQModelSize modelSize)
    {
        return modelSize switch
        {
            SAMHQModelSize.ViTBase => ([768, 768, 768, 768], [12, 0, 0, 0], 256),
            SAMHQModelSize.ViTLarge => ([1024, 1024, 1024, 1024], [24, 0, 0, 0], 256),
            SAMHQModelSize.ViTHuge => ([1280, 1280, 1280, 1280], [32, 0, 0, 0], 256),
            _ => ([768, 768, 768, 768], [12, 0, 0, 0], 256)
        };
    }

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            features = Layers[i].Forward(features);

        if (!hasBatch) features = RemoveBatchDimension(features);
        return features;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var onnxInput = CvTensorOps<T>.ToOnnxFloat(input);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "pixel_values";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var result = CvTensorOps<T>.FromOnnxFloat(outputTensor);
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
    }

    // AddBatchDimension and RemoveBatchDimension come from SegmentationModelBase.

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the encoder and decoder layers for SAM-HQ.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the ViT encoder and HQ mask decoder layers.
    /// In ONNX mode, no layers are created since the runtime handles everything.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Architecture.Layers.Count / 2;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateSAMHQEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int featureH = _height / 16;
            int featureW = _width / 16;
            int encoderOutputChannels = _channelDims[^1];
            var decoderLayers = LayerHelper<T>.CreateSAMHQDecoderLayers(
                encoderOutputChannels, _decoderDim, _numClasses, featureH, featureW);
            Layers.AddRange(decoderLayers);
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Collects metadata describing this SAM-HQ model's configuration.
    /// </summary>
    /// <returns>A <see cref="ModelMetadata{T}"/> with model type, architecture, and serialized data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary of the model's configuration for saving or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "SAMHQ" }, { "InputHeight", _height }, { "InputWidth", _width },
                { "InputChannels", _channels }, { "NumClasses", _numClasses },
                { "ModelSize", _modelSize.ToString() }, { "DecoderDim", _decoderDim },
                { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and sets _disposed,
    // and SAM-HQ owns no further unmanaged resources.

    #endregion

    #region IPromptableSegmentation Implementation

    // SAM-HQ's own promptable state. _imageEmbedding and _imageSet live on PromptableSegmentationBase.
    [Scratch]
    private Tensor<T>? _imageProbabilities;

    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase;
    // SupportsPointPrompts/BoxPrompts/MaskPrompts/TextPrompts come from PromptableSegmentationBase
    // with exactly these values (true, true, true, false).

    /// <summary>
    /// Encodes an image into the embedding the prompt methods score against. SAM-HQ's encoder is the
    /// full forward pass, so this is simply <see cref="NeuralNetworkBase{T}.Predict"/>.
    /// </summary>
    protected override Tensor<T> EncodeImage(Tensor<T> image) => Predict(image);

    /// <inheritdoc/>
    public override void SetImage(Tensor<T> image)
    {
        // Same two steps as before - encode once, then cache the softmax of that embedding for
        // SegmentEverything - plus the base's _imageSet flag that EnsureImageSet() reads.
        var embedding = EncodeImage(image);
        _imageEmbedding = embedding;
        _imageSet = true;
        _imageProbabilities = Common.SegmentationTensorOps.SoftmaxAlongClassDim(embedding);
    }

    /// <inheritdoc/>
    public override PromptedSegmentationResult<T> SegmentFromPoints(Tensor<T> points, Tensor<T> labels)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var scoreMap = Common.SegmentationTensorOps.PointPromptScoreMap(features, points, labels);
        return BuildPromptMaskResult(scoreMap, features.Shape[1], features.Shape[2]);
    }

    /// <inheritdoc/>
    public override PromptedSegmentationResult<T> SegmentFromBox(Tensor<T> box)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var scoreMap = Common.SegmentationTensorOps.BoxPromptScoreMap(features, box);
        return BuildPromptMaskResult(scoreMap, features.Shape[1], features.Shape[2]);
    }

    /// <inheritdoc/>
    public override PromptedSegmentationResult<T> SegmentFromMask(Tensor<T> mask)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var scoreMap = Common.SegmentationTensorOps.MaskPromptScoreMap(features, mask);
        return BuildPromptMaskResult(scoreMap, features.Shape[1], features.Shape[2]);
    }

    /// <inheritdoc/>
    public override List<PromptedSegmentationResult<T>> SegmentEverything()
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var probs = _imageProbabilities ?? Common.SegmentationTensorOps.SoftmaxAlongClassDim(features);
        return Common.SegmentationTensorOps.ClassMaskCandidates(features, probs);
    }

    private PromptedSegmentationResult<T> BuildPromptMaskResult(Tensor<T> scoreMap, int h, int w)
        => Common.SegmentationTensorOps.BuildPromptMaskResult(scoreMap, h, w);

    #endregion
}
