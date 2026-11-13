using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;

namespace AiDotNet.Document.OCR.TextDetection;

/// <summary>
/// PSENet (Progressive Scale Expansion Network) for text detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PSENet uses a novel progressive scale expansion algorithm to accurately detect
/// text instances of various shapes and sizes, especially useful for closely spaced text.
/// </para>
/// <para>
/// <b>For Beginners:</b> PSENet handles difficult text detection scenarios:
/// 1. Detects text at multiple scales (kernels)
/// 2. Progressively expands from smallest to largest
/// 3. Separates closely spaced text instances
/// 4. Handles arbitrary-shaped text
///
/// Key features:
/// - Multi-scale kernel prediction
/// - Progressive scale expansion algorithm
/// - Handles closely adjacent text
/// - Accurate boundary detection
///
/// Example usage:
/// <code>
/// var model = new PSENet&lt;float&gt;(architecture);
/// var result = model.DetectText(documentImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Shape Robust Text Detection with Progressive Scale Expansion Network" (CVPR 2019)
/// https://arxiv.org/abs/1903.12473
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Shape Robust Text Detection with Progressive Scale Expansion Network", "https://doi.org/10.48550/arXiv.1903.12473", Year = 2019, Authors = "Wenhai Wang, Enze Xie, Xiang Li, Wenbo Hou, Tong Lu, Gang Yu, Shuai Shao")]
[PaperOptimizer(OptimizerKind.SgdMomentum, LearningRate = 0.001, WeightDecay = 5e-4,
                Momentum = 0.99, UseNesterov = true, ReferenceBatchSize = 16,
                Phase = TrainingPhase.PreTraining,
                Source = "Wang et al. 2019, Sec. 4: training for 300 epochs at a batch size of 16 with "
                        + "an initial learning rate of 1e-3, a weight decay of 5e-4 and a Nesterov "
                        + "momentum of 0.99 without dampening. Recorded verify-only because SGD is a "
                        + "weaker optimizer than this model's Adam default.")]
[PaperOptimizer(OptimizerKind.SgdMomentum, LearningRate = 1e-4, WeightDecay = 5e-4,
                Momentum = 0.99, UseNesterov = true, ReferenceBatchSize = 16,
                Phase = TrainingPhase.FineTuning,
                Source = "Wang et al. 2019, Sec. 4: fine-tuning runs 400 epochs at a batch size of 16 "
                        + "with an initial learning rate of 1e-4, otherwise as the training stage.")]
public partial class PSENet<T> : DocumentNeuralNetworkBase<T>, ITextDetector<T>
{
    private readonly PSENetOptions _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _backboneChannels;
    private readonly int _featureChannels;
    private readonly int _numKernels;

    // Native mode layers
    private readonly List<ILayer<T>> _backboneLayers = [];
    private readonly List<ILayer<T>> _fpnLayers = [];
    private readonly List<ILayer<T>> _segmentationLayers = [];

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => false;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    public bool SupportsRotatedText => true;

    /// <inheritdoc/>
    public int MinTextHeight => 8;

    /// <inheritdoc/>
    public bool SupportsPolygonOutput => true;

    /// <summary>
    /// Gets the number of scale kernels.
    /// </summary>
    public int NumKernels => _numKernels;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a PSENet model with default configuration for native training.
    /// </summary>
    public PSENet()
        : this(new NeuralNetworkArchitecture<T>(
            // PSENet (CVPR 2019) is an RGB scene-text detector: a ResNet/FPN backbone
            // whose first convolution consumes 3-channel color images. Declare the input
            // as ThreeDimensional with inputDepth:3 so GetInputShape() reports [3, H, W]
            // and the first backbone conv resolves its InputDepth to 3. A TwoDimensional
            // declaration forces InputDepth=1 (grayscale), which mismatches the 3-channel
            // RGB tensors fed at inference/training ("Expected input depth 1, but got 3").
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputDepth: 3,
            inputHeight: 640, inputWidth: 640,
            outputSize: 7))
    {
    }

    /// <summary>
    /// Creates a PSENet model using a pre-trained ONNX model for inference.
    /// </summary>
    public PSENet(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        PSENetOptions? options = null)
        // PSENet trains with Dice, not cross-entropy (Wang et al. 2019, Eq. 5): 0.7 x OHEM Dice on the complete
        // map plus 0.3 x Dice on the shrunk kernels inside the predicted text region. PSENetLoss applies the
        // sigmoid itself, so PredictCore keeps returning raw logits. The previous default (BCE with logits),
        // documented here as the paper's objective, was not.
        : base(architecture, lossFunction ?? new PSENetLoss<T>(), 1.0)
    {
        _options = options ?? new PSENetOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _backboneChannels = _options.BackboneChannels;
        _featureChannels = _options.FeatureChannels;
        _numKernels = _options.NumKernels;
        // The rate the options publish. It defaults to 1e-4 rather than Adam's 1e-3 because the
        // detector's multi-million-parameter ResNet/FPN stack needs a bounded fine-tuning step;
        // the generic first step overshoots. (That was measured under the former BCE objective;
        // the Dice objective (PSENetLoss) is bounded in [0, 1] per term, and the native training
        // tests confirm it still converges at 1e-4.) The rate lives on PSENetOptions.LearningRate,
        // where a caller can see and change it.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });

        ImageSize = _options.ImageSize;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a PSENet model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (PSENet from CVPR 2019):</b>
    /// - Backbone: ResNet-50/152
    /// - FPN: Feature Pyramid Network
    /// - Output: Multi-scale kernels (default 7)
    /// - Post-processing: Progressive scale expansion
    /// </para>
    /// </remarks>
    public PSENet(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        PSENetOptions? options = null)
        // PSENet trains with Dice, not cross-entropy (Wang et al. 2019, Eq. 5): 0.7 x OHEM Dice on the complete
        // map plus 0.3 x Dice on the shrunk kernels inside the predicted text region. PSENetLoss applies the
        // sigmoid itself, so PredictCore keeps returning raw logits. The previous default (BCE with logits),
        // documented here as the paper's objective, was not.
        : base(architecture, lossFunction ?? new PSENetLoss<T>(), 1.0)
    {
        _options = options ?? new PSENetOptions();
        Options = _options;

        _useNativeMode = true;
        _backboneChannels = _options.BackboneChannels;
        _featureChannels = _options.FeatureChannels;
        _numKernels = _options.NumKernels;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });

        ImageSize = _options.ImageSize;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            return;
        }

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        Layers.AddRange(LayerHelper<T>.CreateDefaultPSENetLayers(
            imageSize: ImageSize,
            backboneChannels: _backboneChannels,
            featureChannels: _featureChannels,
            numKernels: _numKernels));
    }

    #endregion

    #region ITextDetector Implementation

    /// <inheritdoc/>
    public TextDetectionResult<T> DetectText(Tensor<T> documentImage)
    {
        return DetectText(documentImage, 0.5);
    }

    /// <inheritdoc/>
    public TextDetectionResult<T> DetectText(Tensor<T> documentImage, double confidenceThreshold)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var regions = ParsePSENetOutput(output, confidenceThreshold);

        return new TextDetectionResult<T>
        {
            TextRegions = regions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public IEnumerable<TextDetectionResult<T>> DetectTextBatch(IEnumerable<Tensor<T>> documentImages)
    {
        foreach (var image in documentImages)
            yield return DetectText(image);
    }

    /// <inheritdoc/>
    public Tensor<T> GetHeatmap()
    {
        return Tensor<T>.CreateDefault([ImageSize, ImageSize], NumOps.Zero);
    }

    /// <inheritdoc/>
    public Tensor<T> GetProbabilityMap(Tensor<T> image)
    {
        ValidateImageShape(image);
        var preprocessed = PreprocessDocument(image);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        // The sigmoid of the largest kernel (the complete text map) of the first image: the network emits logits.
        var (maps, outH, outW) = KernelMaps(output);
        return Engine.Reshape(Engine.Sigmoid(
            Engine.TensorSlice(maps, new[] { _numKernels - 1, 0 }, new[] { 1, outH * outW })), new[] { outH, outW });
    }

    /// <summary>The first image's kernel logits as <c>[kernels, H * W]</c>, with the map height and width.</summary>
    private (Tensor<T> Maps, int Height, int Width) KernelMaps(Tensor<T> output)
    {
        int outH = output.Shape.Length > 2 ? output.Shape[2] : ImageSize;
        int outW = output.Shape.Length > 3 ? output.Shape[3] : ImageSize;
        var first = Engine.TensorSlice(Engine.Reshape(output, new[] { output.Length / (_numKernels * outH * outW), _numKernels, outH * outW }),
            new[] { 0, 0, 0 }, new[] { 1, _numKernels, outH * outW });
        return (Engine.Reshape(first, new[] { _numKernels, outH * outW }), outH, outW);
    }

    /// <summary>
    /// PSENet's progressive scale expansion (Wang et al. 2019, Algorithm 1; the reference <c>pse</c>):
    /// kernels are the sigmoid maps above 0.5, each shrunk kernel restricted to the complete text map. The
    /// smallest kernel's 4-connected components seed the instances; each larger kernel in turn grows them by
    /// breadth-first search from every labelled pixel, first come first served, so instances meeting at a
    /// boundary split it. An instance is kept when it has at least <see cref="MinRegionArea"/> pixels and its
    /// mean complete-map probability reaches <paramref name="threshold"/>.
    /// </summary>
    internal List<TextRegion<T>> ParsePSENetOutput(Tensor<T> output, double threshold)
    {
        var regions = new List<TextRegion<T>>();

        // PSENet output: [batch, numKernels, H, W], kernels ordered from smallest (0) to largest (numKernels-1).
        var (maps, outH, outW) = KernelMaps(output);
        int plane = outH * outW;
        var probabilities = Engine.Sigmoid(maps);
        var text = Engine.TensorSlice(probabilities, new[] { _numKernels - 1, 0 }, new[] { 1, plane });
        var inside = Engine.TensorBroadcastTo(Engine.TensorGreaterThan(text, NumOps.FromDouble(0.5)), new[] { _numKernels, plane });
        int[] kernels = Engine.TensorMultiply(Engine.TensorGreaterThan(probabilities, NumOps.FromDouble(0.5)), inside)
            .Cast<int>().ToArray();

        // Step 1: 4-connected components of the smallest kernel seed the text instances. As in the reference
        // pse, a component smaller than MinKernelArea is noise: it is dropped (its pixels stay unlabelled, free
        // for the expansion to claim) instead of seeding an instance of its own.
        const int Dropped = -1;
        var labels = new int[plane];
        var queue = new Queue<int>();
        var component = new List<int>();
        int count = 0;
        for (int p = 0; p < plane; p++)
        {
            if (kernels[p] == 0 || labels[p] != 0) continue;
            labels[p] = ++count;
            queue.Enqueue(p);
            component.Clear();
            while (queue.Count > 0)
            {
                int q = queue.Dequeue();
                component.Add(q);
                foreach (int n in Neighbours(q, outH, outW))
                {
                    if (kernels[n] != 0 && labels[n] == 0)
                    {
                        labels[n] = count;
                        queue.Enqueue(n);
                    }
                }
            }

            if (component.Count < MinKernelArea)
            {
                foreach (int q in component) labels[q] = Dropped;
                count--;
            }
        }
        for (int p = 0; p < plane; p++)
            if (labels[p] == Dropped) labels[p] = 0;

        // Step 2: progressive scale expansion through the larger kernels, seeded in raster order.
        var frontier = new Queue<int>(Enumerable.Range(0, plane).Where(p => labels[p] != 0));
        for (int k = 1; k < _numKernels; k++)
        {
            int offset = k * plane;
            var edge = new Queue<int>();
            while (frontier.Count > 0)
            {
                int q = frontier.Dequeue();
                bool isEdge = true;
                foreach (int n in Neighbours(q, outH, outW))
                {
                    if (kernels[offset + n] == 0 || labels[n] != 0) continue;
                    labels[n] = labels[q];
                    frontier.Enqueue(n);
                    isEdge = false;
                }
                if (isEdge) edge.Enqueue(q);
            }
            frontier = edge;
        }

        // Step 3: area filter, mean complete-map score, bounding box and polygon of each instance.
        var pixels = Enumerable.Range(0, count + 1).Select(_ => new List<(int y, int x)>()).ToArray();
        for (int p = 0; p < plane; p++)
            if (labels[p] != 0) pixels[labels[p]].Add((p / outW, p % outW));

        // Each instance's score is the mean complete-map probability of its pixels, gathered at their int
        // positions (labels are never compared in T, which holds integers exactly only up to 2048 in Half) and
        // averaged in double (a Half running sum stalls at 2048 pixels).
        var textScores = Engine.Reshape(text, new[] { plane });
        var sized = Enumerable.Range(1, count).Where(label => pixels[label].Count >= MinRegionArea).ToArray();
        if (sized.Length == 0)
            return regions;
        T[] means = sized.Select(label => DocumentTensorOps.Mean(Engine,
            DocumentTensorOps.Pick(Engine, textScores, pixels[label].Select(c => (c.y * outW) + c.x).ToArray()))).ToArray();
        // One engine comparison for every instance (NaN never reaches the threshold).
        int[] passed = DocumentTensorOps.Hits(Engine, DocumentTensorOps.AtLeast(Engine,
            new Tensor<T>(means, new[] { means.Length }), NumOps.FromDouble(threshold)));
        int regionId = 0;
        foreach (int k in passed)
        {
            var instance = pixels[sized[k]];
            T mean = means[k];

            var polygon = ExtractConvexHull(instance);
            T[] box = DocumentTensorOps.ToT<T>(new[] { instance.Min(p => p.x), instance.Min(p => p.y), instance.Max(p => p.x), instance.Max(p => p.y) });
            T[] corners = polygon.SelectMany(c => new[] { c.x, c.y }).Select(NumOps.FromDouble).ToArray();
            regions.Add(new TextRegion<T>
            {
                Confidence = mean,
                BoundingBox = new Vector<T>(box),
                PolygonPoints = Enumerable.Range(0, polygon.Count).Select(i => new Vector<T>(new[] { corners[2 * i], corners[(2 * i) + 1] })).ToList(),
                Index = regionId++
            });
        }

        return regions;
    }

    /// <summary>Instances with fewer pixels than this are dropped as noise.</summary>
    private const int MinRegionArea = 10;

    /// <summary>
    /// Smallest-kernel components with fewer pixels than this do not seed an instance (the reference pse's
    /// <c>min_kernel_area</c>, 5 at scale 1).
    /// </summary>
    private const int MinKernelArea = 5;

    /// <summary>The in-bounds 4-connected neighbours of row-major pixel <paramref name="p"/>, in the reference order.</summary>
    private static IEnumerable<int> Neighbours(int p, int height, int width)
    {
        int y = p / width, x = p % width;
        if (y > 0) yield return p - width;
        if (y < height - 1) yield return p + width;
        if (x > 0) yield return p - 1;
        if (x < width - 1) yield return p + 1;
    }

    /// <summary>
    /// Extracts convex hull from pixel coordinates (simplified algorithm).
    /// </summary>
    private static List<(double x, double y)> ExtractConvexHull(List<(int y, int x)> pixels)
    {
        if (pixels.Count < 3) return pixels.Select(p => ((double)p.x, (double)p.y)).ToList();

        // Find extreme points
        var topLeft = pixels.OrderBy(p => p.y).ThenBy(p => p.x).First();
        var topRight = pixels.OrderBy(p => p.y).ThenByDescending(p => p.x).First();
        var bottomRight = pixels.OrderByDescending(p => p.y).ThenByDescending(p => p.x).First();
        var bottomLeft = pixels.OrderByDescending(p => p.y).ThenBy(p => p.x).First();

        // Return quadrilateral approximation of hull
        return
        [
            (topLeft.x, topLeft.y),
            (topRight.x, topRight.y),
            (bottomRight.x, bottomRight.y),
            (bottomLeft.x, bottomLeft.y)
        ];
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
        var preprocessed = PreprocessDocument(documentImage);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("PSENet Model Summary");
        sb.AppendLine("====================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: ResNet + FPN + Multi-scale Kernels");
        sb.AppendLine($"Backbone Channels: {_backboneChannels}");
        sb.AppendLine($"Feature Channels: {_featureChannels}");
        sb.AppendLine($"Number of Kernels: {_numKernels}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Progressive Expansion: Yes");
        sb.AppendLine($"Arbitrary Shapes: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies PSENet's industry-standard preprocessing: ImageNet normalization with scale.
    /// </summary>
    /// <remarks>
    /// PSENet (Progressive Scale Expansion Network) uses ImageNet normalization with
    /// mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225], with /255 scaling.
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
        => DocumentTensorOps.NormalizeChannels(Engine, EnsureBatchDimension(rawImage),
            DocumentTensorOps.ImageNetMean, DocumentTensorOps.ImageNetStd);

    /// <summary>
    /// Applies PSENet's industry-standard postprocessing: pass-through (kernel maps are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "PSENet",
            Description = "PSENet for progressive scale expansion text detection (CVPR 2019)",
            FeatureCount = _featureChannels,
            Complexity = _numKernels,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "backbone_channels", _backboneChannels },
                { "feature_channels", _featureChannels },
                { "num_kernels", _numKernels },
                { "image_size", ImageSize },
                { "use_native_mode", _useNativeMode }
            },
            ModelDataProvider = () => SafeSerialize()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <summary>
    /// Trains on the SAME preprocessed tensor <see cref="PredictCore"/> evaluates, rather than on
    /// the raw input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Without this override the base implementation walks <c>Layers</c> on the raw tensor while
    /// prediction runs on the preprocessed one, so the model was trained on one input distribution
    /// and measured on another. Preprocessing here maps a U[0,1) fixture (mean 0.5, std 0.29) to
    /// roughly mean 0.07, std 1.26, and additionally promotes [3, H, W] to [1, 3, H, W] -- so the
    /// two paths differed in rank as well as scale.
    /// </para>
    /// <para>
    /// That is what made the loss climb MONOTONICALLY with training rather than merely fail to
    /// improve. BatchNormalization takes its batch-statistics branch during training and updates
    /// its running mean/variance, so those running stats drifted toward the RAW activation
    /// distribution; eval-mode prediction then normalized with them and got progressively more
    /// wrong the longer training ran. Starting values are (0, 1) -- a pure identity affine -- which
    /// is exactly why the untrained baseline looked healthy at 0.876 and the trained model reached
    /// 1.418.
    /// </para>
    /// <para>
    /// Same fix as the siblings that already do this: DocBank and MATCHA. DBNet, EAST and CRAFT
    /// share the identical asymmetry and are not fixed here.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        EnsureLayerRandomSeedsWired();
        return base.ForwardForTraining(PreprocessDocument(input));
    }

    // UpdateParameters applied a GRADIENT STEP, but its one-argument form is the value setter and every caller passes values -- the override corrupted the model. Removed under AIDN082.


    /// <summary>
    /// Parameters cannot be written while the model is backed by a loaded ONNX graph: the weights
    /// belong to that graph, not to this instance.
    /// </summary>
    /// <remarks>
    /// Replaces a hand-written throw that used to sit inside UpdateParameters. The base checks this
    /// on every mutating entry point rather than the one member the throw happened to guard, and
    /// reading -- ParameterCount and GetParameters -- stays available either way.
    /// </remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    private Vector<T> CollectGradients()
    {
        var grads = new List<T>();
        foreach (var layer in Layers)
            grads.AddRange(layer.GetParameterGradients());
        return new Vector<T>([.. grads]);
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _onnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
