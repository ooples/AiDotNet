using AiDotNet.LearningRateSchedulers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.ComputerVision;
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
using AiDotNet.Models.Options;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;

namespace AiDotNet.Document.GraphBased;

/// <summary>
/// PICK (Processing Key Information extraction) for document key information extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PICK treats each OCR text segment as a graph node. It encodes each segment and learns how the
/// segments relate, and a CRF then tags every token (Yu et al., ICPR 2020; reference
/// wenwenyu/PICK-pytorch):
/// <list type="bullet">
/// <item>Encoder: word embeddings plus a sinusoidal position embedding, plus the segment's image embedding
/// (the ResNet page features ROI-aligned 7x7 inside the segment's box, then a 7x7 conv, BatchNorm and ReLU).
/// A transformer runs over each segment, and each segment is mean-pooled into its node.</item>
/// <item>Graph: <see cref="PickGraphLayer{T}"/> learns a soft adjacency from the nodes (with its
/// graph-learning loss) and convolves over node-edge-node triplets seeded by six box-relation features.</item>
/// <item>Decoder: every token's features plus its segment's graph embedding, joined into one document
/// sequence, then a BiLSTM, a linear layer and a CRF.</item>
/// <item>Training minimizes the CRF negative log-likelihood plus 0.01 x the graph-learning loss.</item>
/// </list>
/// </para>
/// <para>
/// <b>Input.</b> <see cref="NeuralNetworkBase{T}.Predict"/> takes a packed segment tensor <c>[N, 4 + T]</c>. Each row is
/// one text segment: its box <c>(x0, y0, x1, y1)</c> in page pixels, then up to <c>T</c> token ids, padded with
/// 0. The output is the per-slot CRF emissions <c>[N * T, NumEntityTypes]</c> (zeros for padding), as the reference forward
/// returns its logits. <see cref="PredictDocument"/> also accepts the page image, which drives the image branch (without
/// one, the image embedding is zero), and returns the CRF's Viterbi tags one-hot per real token.
/// </para>
/// <para>
/// <b>For Beginners:</b> Give PICK the text boxes an OCR engine found on an invoice or receipt, optionally
/// with the page image. It labels every word (for example "TOTAL" or "DATE") using both what the words say
/// and where they sit relative to each other. Pass an <see cref="IOCRModel{T}"/> to the constructor to run
/// straight from page images.
/// </para>
/// <para>
/// <b>Reference:</b> "PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks" (ICPR 2020)
/// https://arxiv.org/abs/2004.07464
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks", "https://doi.org/10.1109/ICPR48806.2021.9412927", Year = 2020, Authors = "Wenwen Yu, Ning Lu, Xianbiao Qi, Ping Gong, Rong Xiao")]
[PaperOptimizer(OptimizerKind.Adam, ReferenceBatchSize = 16,
                Source = "Yu et al. 2020, Sec. 4: trained from scratch using Adam to minimize the CRF "
                        + "and graph learning losses jointly, at a batch size of 16. The paper states no "
                        + "learning rate in its training description, so none is declared.")]
public partial class PICK<T> : DocumentNeuralNetworkBase<T>, IFormUnderstanding<T>
{
    private readonly PICKOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly ITokenizer _tokenizer;
    private readonly IOCRModel<T>? _ocr;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _hiddenDim;
    private readonly int _numGcnLayers;
    private readonly int _numHeads;
    private readonly int _vocabSize;
    private readonly int _numEntityTypes;

    // Positions of each component in Layers, set by InitializeLayers for the default PICK stack.
    private bool _defaultStack;
    private int _embeddingIndex;
    private int _encoderStart;
    private int _encoderNormIndex;
    private int _imageStart;
    private int _imageEnd;
    private int _roiConvIndex;
    private int _roiNormIndex;
    private int _graphIndex;
    private int _lstmStart;
    private int _emissionIndex;
    private int _crfIndex;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.Form;

    /// <inheritdoc/>
    public override bool RequiresOCR => true;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <summary>
    /// Gets the supported entity types for extraction.
    /// </summary>
    public IReadOnlyList<string> SupportedEntityTypes { get; } =
    [
        "SELLER", "ADDRESS", "DATE", "TOTAL", "TAX", "ITEM", "QUANTITY", "PRICE",
        "INVOICE_NUMBER", "BUYER", "PAYMENT_METHOD", "DUE_DATE", "CURRENCY", "OTHER"
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a PICK model backed by an ONNX export.
    /// </summary>
    public PICK(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        PICKOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        IOCRModel<T>? ocr = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new PICKOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        Guard.NotNull(tokenizer);
        _tokenizer = tokenizer;
        _ocr = ocr;
        _useNativeMode = false;
        // Validated after the path checks so a missing model file reports itself
        // as FileNotFoundException rather than being pre-empted by the options.
        _options.Validate();

        _numEntityTypes = _options.NumEntityTypes;
        _hiddenDim = _options.HiddenDim;
        _numGcnLayers = _options.NumGcnLayers;
        _numHeads = _options.NumHeads;
        _vocabSize = _options.VocabSize;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = 1e-4 });

        ImageSize = _options.ImageSize;
        MaxSequenceLength = _options.MaxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a trainable native PICK model.
    /// </summary>
    /// <param name="architecture">The network architecture.</param>
    /// <param name="options">Hyperparameters; defaults to the reference configuration.</param>
    /// <param name="tokenizer">Tokenizer for OCR text; defaults to the OPT tokenizer.</param>
    /// <param name="optimizer">Optimizer; defaults to Adam at the reference learning rate 1e-4.</param>
    /// <param name="lossFunction">Base loss (unused by the CRF objective; kept for the base contract).</param>
    /// <param name="ocr">Optional OCR model that turns page images into text segments for the image-based APIs.</param>
    public PICK(
        NeuralNetworkArchitecture<T> architecture,
        PICKOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        IOCRModel<T>? ocr = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new PICKOptions();
        _options.Validate();
        Options = _options;

        _useNativeMode = true;
        _numEntityTypes = _options.NumEntityTypes;
        _hiddenDim = _options.HiddenDim;
        _numGcnLayers = _options.NumGcnLayers;
        _numHeads = _options.NumHeads;
        _vocabSize = _options.VocabSize;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = 1e-4 });

        ImageSize = _options.ImageSize;
        MaxSequenceLength = _options.MaxSequenceLength;

        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
        _ocr = ocr;

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
            _defaultStack = false;
            return;
        }

        IActivationFunction<T> identity = new IdentityActivation<T>();
        int d = _hiddenDim;

        _embeddingIndex = Layers.Count;
        Layers.Add(new EmbeddingLayer<T>(_vocabSize, d));

        _encoderStart = Layers.Count;
        for (int i = 0; i < _options.NumEncoderLayers; i++)
            Layers.Add(new TransformerEncoderLayer<T>(_numHeads, _options.FeedForwardDim, d));
        _encoderNormIndex = Layers.Count;
        Layers.Add(new LayerNormalizationLayer<T>());

        // Image branch: a ResNet over the page, projected to ImageFeatureDim (the reference resnet's
        // output_channels), then a 7x7 conv over each 7x7 ROI-aligned box with BatchNorm (ReLU applied in forward).
        _imageStart = Layers.Count;
        Layers.AddRange(LayerHelper<T>.CreateResNetBottleneckEncoderLayers(
            3, ImageSize, ImageSize, new[] { 256, 512, 1024, 2048 }, _options.ImageEncoderDepths));
        Layers.Add(new ConvolutionalLayer<T>(_options.ImageFeatureDim, 1, 1, 0, identity));
        _imageEnd = Layers.Count;
        _roiConvIndex = Layers.Count;
        Layers.Add(new ConvolutionalLayer<T>(d, 7, 1, 0, identity));
        _roiNormIndex = Layers.Count;
        Layers.Add(new BatchNormalizationLayer<T>());

        _graphIndex = Layers.Count;
        Layers.Add(new PickGraphLayer<T>(d, _options.GraphLearningDim, _numGcnLayers, _options.GraphEta, _options.GraphGamma));

        _lstmStart = Layers.Count;
        IActivationFunction<T> tanh = new TanhActivation<T>();
        IActivationFunction<T> sigmoid = new SigmoidActivation<T>();
        for (int i = 0; i < _options.LstmLayers; i++)
        {
            var lstm = new LSTMLayer<T>(hiddenSize: _options.LstmHiddenDim, activation: tanh, recurrentActivation: sigmoid);
            Layers.Add(new BidirectionalLayer<T>(lstm, mergeMode: false, activationFunction: identity));
        }

        // Reference MLPLayer with no hidden dims: a single Linear(2 * lstm_hidden -> tags).
        _emissionIndex = Layers.Count;
        Layers.Add(new DenseLayer<T>(_numEntityTypes, identity));
        _crfIndex = Layers.Count;
        Layers.Add(new ConditionalRandomFieldLayer<T>(numClasses: _numEntityTypes, scalarActivation: identity));
        _defaultStack = true;
    }

    #endregion

    #region Input contract

    /// <inheritdoc/>
    /// <remarks>
    /// The packed segment tensor holds box coordinates in page pixels and token ids, both non-negative
    /// integers. One bound covers both (the larger of the vocabulary and the page size); token ids above the vocabulary are clamped when parsed.
    /// </remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape)
        => _defaultStack ? LayerInputDomain.Indices(Math.Max(_vocabSize, ImageSize + 1)) : base.GetInputDomain(inputShape);

    /// <summary>A parsed document: one box and one token run per segment.</summary>
    private sealed class Segments
    {
        public Segments(double[][] boxes, int[][] tokens, int slots)
        {
            Boxes = boxes;
            Tokens = tokens;
            Slots = slots;
        }

        public double[][] Boxes { get; }
        public int[][] Tokens { get; }
        public int Slots { get; }
        public int Count => Boxes.Length;
    }

    /// <summary>
    /// Parses <c>[N, 4 + T]</c>. A segment's tokens run up to its last non-zero id; an all-padding segment is one
    /// padding token, so every segment is a graph node.
    /// </summary>
    private Segments ParseSegments(Tensor<T> packed)
    {
        if (packed.Rank != 2 || packed.Shape[1] < 5)
            throw new ArgumentException(
                "PICK expects packed segments [N, 4 + T]: each row a box (x0, y0, x1, y1) followed by T token ids.", nameof(packed));
        int n = packed.Shape[0], slots = packed.Shape[1] - 4;
        var boxes = new double[n][];
        var tokens = new int[n][];
        for (int i = 0; i < n; i++)
        {
            double a = NumOps.ToDouble(packed[i, 0]), b = NumOps.ToDouble(packed[i, 1]);
            double c = NumOps.ToDouble(packed[i, 2]), e = NumOps.ToDouble(packed[i, 3]);
            boxes[i] = new[] { Math.Min(a, c), Math.Min(b, e), Math.Max(a, c), Math.Max(b, e) };
            int last = 0;
            for (int t = 0; t < slots; t++)
                if (NumOps.ToDouble(packed[i, 4 + t]) != 0) last = t + 1;
            int length = Math.Max(1, last);
            tokens[i] = new int[length];
            for (int t = 0; t < length; t++)
                tokens[i][t] = Math.Min(Math.Max((int)Math.Round(NumOps.ToDouble(packed[i, 4 + t])), 0), _vocabSize - 1);
        }
        return new Segments(boxes, tokens, slots);
    }

    /// <summary>
    /// The reference's initial relation features for every node pair:
    /// <c>[|cx_i - cx_j| / W, |cy_i - cy_j| / H, w_i / h_i, h_j / h_i, w_j / h_i, len_j / len_i]</c>, with -1 where
    /// h_i is 0. Features 2-5 are min-max normalized. The reference writes <c>x - min / (max - min)</c>, whose operator
    /// precedence subtracts only <c>min / (max - min)</c>; this implements the min-max scaling that line intends.
    /// </summary>
    private Tensor<T> RelationFeatures(Segments segments, double pageWidth, double pageHeight)
    {
        int n = segments.Count;
        var features = new double[n, n, PickGraphLayer<T>.RelationFeatures];
        for (int i = 0; i < n; i++)
        {
            var bi = segments.Boxes[i];
            double wi = bi[2] - bi[0], hi = bi[3] - bi[1], cxi = (bi[0] + bi[2]) / 2, cyi = (bi[1] + bi[3]) / 2;
            for (int j = 0; j < n; j++)
            {
                var bj = segments.Boxes[j];
                double wj = bj[2] - bj[0], hj = bj[3] - bj[1];
                features[i, j, 0] = Math.Abs(cxi - ((bj[0] + bj[2]) / 2)) / pageWidth;
                features[i, j, 1] = Math.Abs(cyi - ((bj[1] + bj[3]) / 2)) / pageHeight;
                features[i, j, 2] = hi != 0 ? wi / hi : -1;
                features[i, j, 3] = hi != 0 ? hj / hi : -1;
                features[i, j, 4] = hi != 0 ? wj / hi : -1;
                features[i, j, 5] = (double)segments.Tokens[j].Length / segments.Tokens[i].Length;
            }
        }
        for (int k = 2; k < PickGraphLayer<T>.RelationFeatures; k++)
        {
            double min = double.PositiveInfinity, max = double.NegativeInfinity;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) { min = Math.Min(min, features[i, j, k]); max = Math.Max(max, features[i, j, k]); }
            if (max != min)
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) features[i, j, k] = (features[i, j, k] - min) / (max - min);
        }
        var tensor = new Tensor<T>(new[] { n, n, PickGraphLayer<T>.RelationFeatures });
        int index = 0;
        foreach (var value in features) tensor[index++] = NumOps.FromDouble(value);
        return tensor;
    }

    #endregion

    #region Forward

    /// <summary>Everything one PICK forward produces.</summary>
    private sealed class PickPass
    {
        public PickPass(Tensor<T> emissions, Tensor<T> graphLoss, Segments segments)
        {
            Emissions = emissions;
            GraphLoss = graphLoss;
            Segments = segments;
        }

        /// <summary>CRF emissions over the document's real tokens, segment by segment: <c>[L, tags]</c>.</summary>
        public Tensor<T> Emissions { get; }
        public Tensor<T> GraphLoss { get; }
        public Segments Segments { get; }
    }

    private PickPass RunPick(Tensor<T> packed, Tensor<T>? pageImage)
    {
        var segments = ParseSegments(packed);
        int n = segments.Count, d = _hiddenDim;
        double pageWidth = ImageSize, pageHeight = ImageSize;
        var imageEmbeddings = pageImage is null ? null : ImageEmbeddings(pageImage, segments, out pageWidth, out pageHeight);

        // Encoder, one segment at a time: exact for the reference's key-padding mask, whose padded positions
        // contribute nothing to a real token and are discarded afterwards.
        var tokenFeatures = new Tensor<T>[n];
        var nodes = new Tensor<T>[n];
        for (int i = 0; i < n; i++)
        {
            int length = segments.Tokens[i].Length;
            var ids = new Tensor<T>(new[] { length });
            for (int t = 0; t < length; t++) ids[t] = NumOps.FromDouble(segments.Tokens[i][t]);
            var x = Engine.TensorAdd(Layers[_embeddingIndex].Forward(ids), SinusoidalPositions(length, d));
            if (imageEmbeddings is not null)
                x = Engine.TensorAdd(x, Engine.TensorBroadcastTo(
                    Engine.TensorSlice(imageEmbeddings, new[] { i, 0 }, new[] { 1, d }), new[] { length, d }));
            for (int l = _encoderStart; l < _encoderNormIndex; l++) x = Layers[l].Forward(x);
            x = Layers[_encoderNormIndex].Forward(x);
            tokenFeatures[i] = x;
            nodes[i] = Engine.TensorMultiplyScalar(Engine.ReduceSum(x, new[] { 0 }, keepDims: true), NumOps.FromDouble(1.0 / length));
        }

        var graph = (PickGraphLayer<T>)Layers[_graphIndex];
        var (graphNodes, _, graphLoss) = graph.Forward(Engine.TensorConcatenate(nodes, 0), RelationFeatures(segments, pageWidth, pageHeight));

        // Decoder: each token plus its segment's graph embedding, as one document sequence.
        var document = new Tensor<T>[n];
        for (int i = 0; i < n; i++)
        {
            int length = segments.Tokens[i].Length;
            document[i] = Engine.TensorAdd(tokenFeatures[i], Engine.TensorBroadcastTo(
                Engine.TensorSlice(graphNodes, new[] { i, 0 }, new[] { 1, d }), new[] { length, d }));
        }
        var sequence = Engine.TensorConcatenate(document, 0);
        int total = sequence.Shape[0];
        for (int l = _lstmStart; l < _emissionIndex; l++)
        {
            var directions = Layers[l].Forward(sequence);                              // [2, L, H]
            int hidden = directions.Shape[2];
            sequence = Engine.Reshape(Engine.TensorPermute(directions, new[] { 1, 0, 2 }), new[] { total, 2 * hidden });
        }
        var emissions = Layers[_emissionIndex].Forward(sequence);
        return new PickPass(emissions, graphLoss, segments);
    }

    /// <summary>
    /// The encoder's image embedding of every segment: ReLU(BN(Conv7x7(RoIAlign_7x7(ResNet(page))))),
    /// with boxes in page pixels (reference Encoder).
    /// </summary>
    private Tensor<T> ImageEmbeddings(Tensor<T> pageImage, Segments segments, out double pageWidth, out double pageHeight)
    {
        var image = pageImage.Rank == 3
            ? Engine.Reshape(pageImage, new[] { 1, pageImage.Shape[0], pageImage.Shape[1], pageImage.Shape[2] })
            : pageImage;
        if (image.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3)
            throw new ArgumentException("PICK's page image must be [3, H, W] or [1, 3, H, W].", nameof(pageImage));
        pageHeight = image.Shape[2];
        pageWidth = image.Shape[3];
        var features = image;
        for (int l = _imageStart; l < _imageEnd; l++) features = Layers[l].Forward(features);
        double scale = features.Shape[3] / pageWidth;
        var boxes = segments.Boxes.SelectMany(b => b).ToArray();
        var pooled = CvTensorOps<T>.RoIAlign(features, boxes, new int[segments.Count], scale, 7, 2);   // [N, C, 7, 7]
        var embedded = Engine.ReLU(Layers[_roiNormIndex].Forward(Layers[_roiConvIndex].Forward(pooled)));  // [N, d, 1, 1]
        return Engine.Reshape(embedded, new[] { segments.Count, _hiddenDim });
    }

    private Tensor<T> SinusoidalPositions(int length, int dim)
    {
        var pe = new Tensor<T>(new[] { length, dim });
        for (int p = 0; p < length; p++)
            for (int i = 0; i < dim; i += 2)
            {
                double angle = p / Math.Pow(10000, (double)i / dim);
                pe[p, i] = NumOps.FromDouble(Math.Sin(angle));
                if (i + 1 < dim) pe[p, i + 1] = NumOps.FromDouble(Math.Cos(angle));
            }
        return pe;
    }

    /// <summary>
    /// Tags a document: packed segments <c>[N, 4 + T]</c> and an optional page image <c>[3, H, W]</c>.
    /// Returns <c>[N * T, NumEntityTypes]</c>, the CRF's Viterbi tags one-hot for real tokens and zeros for padding.
    /// </summary>
    public Tensor<T> PredictDocument(Tensor<T> segments, Tensor<T>? pageImage = null)
    {
        if (segments is null) throw new ArgumentNullException(nameof(segments));
        if (!_useNativeMode) return RunOnnxInference(segments);
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return ToGrid(RunPick(segments, pageImage), decode: true);
    }

    /// <summary>The CRF emissions over real tokens (<c>[L, tags]</c>), in inference mode, for tests that inspect the tagger's input.</summary>
    internal Tensor<T> EmissionsForTest(Tensor<T> segments, Tensor<T>? pageImage)
    {
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return RunPick(segments, pageImage).Emissions;
    }

    /// <summary>Scatters per-token rows back onto the <c>[N * T, tags]</c> slot grid.</summary>
    private Tensor<T> ToGrid(PickPass pass, bool decode)
    {
        var rows = decode ? Layers[_crfIndex].Forward(pass.Emissions) : pass.Emissions;
        int n = pass.Segments.Count, slots = pass.Segments.Slots;
        var positions = new List<int>();
        for (int i = 0; i < n; i++)
            for (int t = 0; t < pass.Segments.Tokens[i].Length; t++) positions.Add((i * slots) + t);
        // A gather in reverse: select every real token's row into its slot, leaving padding slots zero.
        var slotOfRow = new int[n * slots];
        var mask = new T[n * slots * _numEntityTypes];
        for (int s = 0; s < slotOfRow.Length; s++) slotOfRow[s] = 0;
        for (int r = 0; r < positions.Count; r++)
        {
            slotOfRow[positions[r]] = r;
            for (int c = 0; c < _numEntityTypes; c++) mask[(positions[r] * _numEntityTypes) + c] = NumOps.One;
        }
        var gathered = CvTensorOps<T>.Select(rows, slotOfRow, 0);
        return Engine.TensorMultiply(gathered, new Tensor<T>(mask, new[] { n * slots, _numEntityTypes }));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Like the reference <c>PICKModel.forward</c>, which returns the tagger's logits and leaves Viterbi decoding to
    /// the caller: the per-slot CRF emissions <c>[N * T, tags]</c> (zeros for padding). <see cref="PredictDocument"/>
    /// and the extraction APIs decode them with the CRF.
    /// </remarks>
    protected override Tensor<T> Forward(Tensor<T> input)
        => _defaultStack ? ToGrid(RunPick(input, null), decode: false) : base.Forward(input);

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _defaultStack ? ToGrid(RunPick(input, null), decode: false) : base.ForwardForTraining(input);

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <paramref name="expectedOutput"/> gives each slot's tag, either as indices <c>[N * T]</c> or as scores
    /// <c>[N * T, NumEntityTypes]</c> (argmax), in the same slot grid as the output; padding slots are ignored.
    /// The objective is the reference's: CRF negative log-likelihood + GraphLossWeight x graph-learning loss.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput) => TrainDocument(input, null, expectedOutput);

    /// <summary>Trains on packed segments with an optional page image and per-slot tags (see <see cref="Train"/>).</summary>
    public void TrainDocument(Tensor<T> segments, Tensor<T>? pageImage, Tensor<T> tags)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        if (segments is null) throw new ArgumentNullException(nameof(segments));
        if (tags is null) throw new ArgumentNullException(nameof(tags));
        if (!_defaultStack)
        {
            SetTrainingMode(true);
            try { TrainWithTape(segments, tags, _optimizer); }
            finally { SetTrainingMode(false); }
            return;
        }

        var crf = (ConditionalRandomFieldLayer<T>)Layers[_crfIndex];
        TrainWithCustomObjective(segments, tags, (input, target) =>
        {
            var pass = RunPick(input, pageImage);
            var labels = TokenLabels(pass.Segments, target);
            var nll = crf.ComputeNegativeLogLikelihood(pass.Emissions, labels);
            return Engine.TensorAdd(nll, Engine.TensorMultiplyScalar(pass.GraphLoss, NumOps.FromDouble(_options.GraphLossWeight)));
        }, _optimizer);
    }

    /// <summary>Reads each real token's tag out of the slot grid.</summary>
    private Tensor<T> TokenLabels(Segments segments, Tensor<T> target)
    {
        int n = segments.Count, slots = segments.Slots;
        bool indices = target.Rank == 1 || (target.Rank == 2 && target.Shape[1] == 1);
        if (target.Shape[0] != n * slots)
            throw new ArgumentException($"PICK tags must cover the {n} x {slots} slot grid ({n * slots} rows).", nameof(target));
        var labels = new List<T>();
        for (int i = 0; i < n; i++)
            for (int t = 0; t < segments.Tokens[i].Length; t++)
            {
                int slot = (i * slots) + t, tag;
                if (indices)
                {
                    tag = (int)Math.Round(NumOps.ToDouble(target.Rank == 1 ? target[slot] : target[slot, 0]));
                }
                else
                {
                    tag = 0;
                    for (int c = 1; c < target.Shape[1]; c++)
                        if (NumOps.ToDouble(target[slot, c]) > NumOps.ToDouble(target[slot, tag])) tag = c;
                }
                labels.Add(NumOps.FromDouble(Math.Min(Math.Max(tag, 0), _numEntityTypes - 1)));
            }
        return new Tensor<T>(labels.ToArray(), new[] { labels.Count });
    }

    #endregion

    #region IFormUnderstanding Implementation

    /// <inheritdoc/>
    public FormFieldResult<T> ExtractFormFields(Tensor<T> documentImage)
    {
        return ExtractFormFields(documentImage, 0.5);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// PICK reads OCR text segments, not pixels alone: the image is run through the <see cref="IOCRModel{T}"/>
    /// passed to the constructor, its lines become the segments, and the page image drives the image branch.
    /// </remarks>
    public FormFieldResult<T> ExtractFormFields(Tensor<T> documentImage, double confidenceThreshold)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;
        var (packed, lines) = SegmentsFromImage(documentImage);
        var output = _useNativeMode ? PredictDocument(packed, documentImage) : RunOnnxInference(packed);
        return new FormFieldResult<T>
        {
            Fields = ParseFieldOutput(output, lines, packed.Shape[1] - 4, confidenceThreshold),
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <summary>OCR lines as packed segments: each line's box, then its token ids, truncated to MaxSequenceLength.</summary>
    private (Tensor<T> Packed, IReadOnlyList<OCRLine<T>> Lines) SegmentsFromImage(Tensor<T> documentImage)
    {
        var ocr = _ocr ?? throw new InvalidOperationException(
            "PICK tags OCR text segments. Pass an IOCRModel<T> to the constructor to run from page images, " +
            "or call PredictDocument with packed segments [N, 4 + T].");
        var lines = ocr.RecognizeText(documentImage).Lines.Where(line => line.BoundingBox.Length >= 4).ToList();
        if (lines.Count == 0)
            throw new InvalidOperationException("The OCR model found no text lines with boxes on this page.");
        var encoded = lines.Select(line => _tokenizer.Encode(line.Text).TokenIds.Take(MaxSequenceLength).ToList()).ToList();
        int slots = Math.Max(1, encoded.Max(ids => ids.Count));
        var packed = new Tensor<T>(new[] { lines.Count, 4 + slots });
        for (int i = 0; i < lines.Count; i++)
        {
            for (int c = 0; c < 4; c++) packed[i, c] = lines[i].BoundingBox[c];
            for (int t = 0; t < encoded[i].Count; t++) packed[i, 4 + t] = NumOps.FromDouble(encoded[i][t]);
        }
        return (packed, lines);
    }

    /// <inheritdoc/>
    public Dictionary<string, string> ExtractKeyValuePairs(Tensor<T> documentImage)
    {
        var result = ExtractFormFields(documentImage);
        var pairs = new Dictionary<string, string>();

        foreach (var field in result.Fields)
        {
            if (!string.IsNullOrEmpty(field.FieldName) && !string.IsNullOrEmpty(field.FieldValue))
            {
                pairs[field.FieldName] = field.FieldValue;
            }
        }

        return pairs;
    }

    /// <inheritdoc/>
    public IEnumerable<CheckboxResult<T>> DetectCheckboxes(Tensor<T> documentImage)
    {
        // PICK is designed for text extraction, not checkbox detection
        yield break;
    }

    /// <inheritdoc/>
    public IEnumerable<SignatureResult<T>> DetectSignatures(Tensor<T> documentImage)
    {
        // PICK is designed for text extraction, not signature detection
        yield break;
    }

    /// <summary>
    /// One field per segment whose tokens were tagged: its most frequent non-zero tag, with the fraction of its
    /// tokens carrying that tag as the confidence.
    /// </summary>
    private List<FormField<T>> ParseFieldOutput(Tensor<T> output, IReadOnlyList<OCRLine<T>> lines, int slots, double threshold)
    {
        var fields = new List<FormField<T>>();
        int classes = output.Shape[1];
        for (int i = 0; i < lines.Count; i++)
        {
            var counts = new int[classes];
            int tokens = 0;
            for (int t = 0; t < slots; t++)
            {
                int row = (i * slots) + t, best = -1;
                for (int c = 0; c < classes; c++)
                    if (NumOps.ToDouble(output[row, c]) > 0.5) best = c;
                if (best < 0) continue;
                counts[best]++;
                tokens++;
            }
            if (tokens == 0) continue;
            int tag = Enumerable.Range(1, Math.Max(0, classes - 1)).OrderByDescending(c => counts[c]).FirstOrDefault();
            double confidence = (double)counts[tag] / tokens;
            if (tag <= 0 || counts[tag] == 0 || confidence < threshold) continue;
            string entityType = tag < SupportedEntityTypes.Count ? SupportedEntityTypes[tag] : "UNKNOWN";
            fields.Add(new FormField<T>
            {
                FieldName = entityType,
                FieldValue = lines[i].Text,
                FieldType = entityType,
                Confidence = NumOps.FromDouble(confidence),
                ConfidenceValue = confidence,
                BoundingBox = lines[i].BoundingBox
            });
        }
        return fields;
    }

    /// <summary>
    /// Extracts key information entities from a page image (requires an <see cref="IOCRModel{T}"/>).
    /// </summary>
    public KeyInfoExtractionResult<T> ExtractKeyInfo(Tensor<T> documentImage)
    {
        var formResult = ExtractFormFields(documentImage);

        var entities = formResult.Fields.Select(f => new ExtractedEntity<T>
        {
            Label = f.FieldName,
            Text = f.FieldValue,
            EntityType = f.FieldType,
            Confidence = f.Confidence,
            ConfidenceValue = f.ConfidenceValue,
            BoundingBox = f.BoundingBox
        }).ToList();

        return new KeyInfoExtractionResult<T>
        {
            Entities = entities,
            ProcessingTimeMs = formResult.ProcessingTimeMs
        };
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    /// <remarks>PICK encodes packed text segments <c>[N, 4 + T]</c>; see the class remarks.</remarks>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateInputShape(documentImage);
        return _useNativeMode ? Forward(documentImage) : RunOnnxInference(documentImage);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        if (documentImage is null) throw new ArgumentNullException(nameof(documentImage));
        if (documentImage.Rank != 2 || documentImage.Shape[1] < 5)
            throw new ArgumentException("PICK expects packed segments [N, 4 + T].", nameof(documentImage));
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("PICK Model Summary");
        sb.AppendLine("==================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine("Architecture: Transformer + ResNet encoder, graph learning-convolution, BiLSTM-CRF");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Encoder Layers: {_options.NumEncoderLayers} x {_numHeads} heads");
        sb.AppendLine($"GCN Layers: {_numGcnLayers}");
        sb.AppendLine($"BiLSTM: {_options.LstmLayers} x {_options.LstmHiddenDim}");
        sb.AppendLine($"Number of Entity Types: {_numEntityTypes}");
        sb.AppendLine($"Supported Entity Types: {string.Join(", ", SupportedEntityTypes.Take(5))}...");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// PICK takes packed text segments, not pixels, so there is nothing to normalize.
    /// </summary>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage) => rawImage;

    /// <summary>
    /// The CRF output is already final.
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "PICK",
            Description = "PICK for key information extraction (ICPR 2020)",
            FeatureCount = _hiddenDim,
            Complexity = _numGcnLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_gcn_layers", _numGcnLayers },
                { "num_heads", _numHeads },
                { "vocab_size", _vocabSize },
                { "num_entity_types", _numEntityTypes },
                { "use_native_mode", _useNativeMode }
            },
            ModelDataProvider = () => SafeSerialize()
        };
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <summary>
    /// Parameters cannot be written while the model is backed by a loaded ONNX graph: the weights
    /// belong to that graph, not to this instance.
    /// </summary>
    protected override bool SupportsParameterMutation => _useNativeMode;

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

/// <summary>
/// Result of key information extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class KeyInfoExtractionResult<T>
{
    /// <summary>
    /// Gets or sets the extracted entities.
    /// </summary>
    public IList<ExtractedEntity<T>> Entities { get; set; } = [];

    /// <summary>
    /// Gets or sets the processing time in milliseconds.
    /// </summary>
    public double ProcessingTimeMs { get; set; }
}

/// <summary>
/// An extracted entity from a document.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ExtractedEntity<T>
{
    /// <summary>
    /// Gets or sets the entity label.
    /// </summary>
    public string Label { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the extracted text.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the entity type.
    /// </summary>
    public string EntityType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score.
    /// </summary>
    public required T Confidence { get; set; }

    /// <summary>
    /// Gets or sets the confidence as a double.
    /// </summary>
    public double ConfidenceValue { get; set; }

    /// <summary>
    /// Gets or sets the bounding box.
    /// </summary>
    public Vector<T> BoundingBox { get; set; } = Vector<T>.Empty();
}
