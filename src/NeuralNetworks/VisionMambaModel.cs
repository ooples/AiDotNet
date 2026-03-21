using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Defines the scan pattern used by the Vision Mamba model to convert 2D patch grids into 1D sequences.
/// </summary>
public enum VisionScanPattern
{
    /// <summary>
    /// Bidirectional scan: forward + reverse, used by the original Vision Mamba (Vim) paper.
    /// </summary>
    Bidirectional,

    /// <summary>
    /// Cross-scan: four directional scans (L→R, R→L, T→B, B→T), used by VMamba.
    /// </summary>
    CrossScan,

    /// <summary>
    /// Continuous/zigzag scan preserving spatial locality, used by PlainMamba.
    /// </summary>
    Continuous
}

/// <summary>
/// Implements the Vision Mamba (Vim) model: PatchEmbed + scan pattern + bidirectional Mamba + classifier.
/// </summary>
/// <remarks>
/// <para>
/// Vision Mamba adapts the Mamba SSM architecture for image classification by:
/// 1. Dividing the image into non-overlapping patches (like ViT)
/// 2. Projecting patches to an embedding dimension
/// 3. Adding learnable positional embeddings
/// 4. Scanning patches through Mamba blocks using configurable spatial scan patterns
/// 5. Pooling the output and projecting to class logits
/// </para>
/// <para>
/// Different scan patterns capture different spatial relationships:
/// - <see cref="VisionScanPattern.Bidirectional"/>: forward + reverse (Vim paper)
/// - <see cref="VisionScanPattern.CrossScan"/>: 4 directional scans (VMamba)
/// - <see cref="VisionScanPattern.Continuous"/>: zigzag preserving locality (PlainMamba)
/// </para>
/// <para><b>For Beginners:</b> This model classifies images using Mamba instead of attention.
/// It cuts the image into small patches, reads them in a specific order (the scan pattern),
/// processes with Mamba blocks (fast sequence processing), and outputs class probabilities.
/// Vision Mamba needs O(n) computation vs O(n^2) for Transformers, making it faster for
/// high-resolution images like medical imaging and satellite photos.</para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item><description>Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", 2024</description></item>
/// <item><description>Liu et al., "VMamba: Visual State Space Model", 2024</description></item>
/// <item><description>Yang et al., "PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition", 2024</description></item>
/// </list>
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new VisionMambaOptions { ImageSize = 224, PatchSize = 16, ModelDim = 384, NumLayers = 24 };
/// var model = new VisionMambaModel&lt;float&gt;(options);
/// var image = Tensor&lt;float&gt;.Random(new[] { 1, 3, 224, 224 });
/// var output = model.Predict(image);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", "https://arxiv.org/abs/2401.09417", Year = 2024, Authors = "Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, Xinggang Wang")]
public class VisionMambaModel<T> : NeuralNetworkBase<T>
{
    private readonly VisionMambaOptions _options;
    private readonly int _imageHeight;
    private readonly int _imageWidth;
    private readonly int _patchSize;
    private readonly int _channels;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _numClasses;
    private readonly int _stateDimension;
    private readonly int _numPatchesH;
    private readonly int _numPatchesW;
    private readonly int _numPatches;
    private readonly int _mambaInputDim;
    private readonly VisionScanPattern _scanPattern;

    // Patch embedding weights (model-level state, not in Layers)
    private Tensor<T> _patchProjectionWeights;
    private Tensor<T> _patchProjectionBias;
    private Tensor<T> _positionalEmbedding;

    // Final normalization and classification head
    private Tensor<T> _finalNormGamma;
    private Tensor<T> _classifierWeights;
    private Tensor<T> _classifierBias;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the image height.</summary>
    public int ImageHeight => _imageHeight;

    /// <summary>Gets the image width.</summary>
    public int ImageWidth => _imageWidth;

    /// <summary>Gets the patch size.</summary>
    public int PatchSize => _patchSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of Mamba layers.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Gets the number of output classes.</summary>
    public int NumClasses => _numClasses;

    /// <summary>Gets the total number of patches.</summary>
    public int NumPatches => _numPatches;

    /// <summary>Gets the scan pattern used.</summary>
    public VisionScanPattern ScanPattern => _scanPattern;

    #region Constructors

    public VisionMambaModel(
        NeuralNetworkArchitecture<T> architecture,
        int imageHeight = 224,
        int imageWidth = 224,
        int patchSize = 16,
        int channels = 3,
        int modelDimension = 192,
        int numLayers = 4,
        int numClasses = 10,
        int stateDimension = 16,
        VisionScanPattern scanPattern = VisionScanPattern.Bidirectional,
        ILossFunction<T>? lossFunction = null,
        VisionMambaOptions? options = null)
        : base(architecture,
            lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.ImageClassification))
    {
        if (imageHeight <= 0) throw new ArgumentException($"Image height ({imageHeight}) must be positive.", nameof(imageHeight));
        if (imageWidth <= 0) throw new ArgumentException($"Image width ({imageWidth}) must be positive.", nameof(imageWidth));
        if (patchSize <= 0) throw new ArgumentException($"Patch size ({patchSize}) must be positive.", nameof(patchSize));
        if (imageHeight % patchSize != 0) throw new ArgumentException($"Image height ({imageHeight}) must be divisible by patch size ({patchSize}).", nameof(imageHeight));
        if (imageWidth % patchSize != 0) throw new ArgumentException($"Image width ({imageWidth}) must be divisible by patch size ({patchSize}).", nameof(imageWidth));
        if (numClasses <= 0) throw new ArgumentException($"Number of classes ({numClasses}) must be positive.", nameof(numClasses));

        _options = options ?? new VisionMambaOptions();
        Options = _options;
        _imageHeight = imageHeight;
        _imageWidth = imageWidth;
        _patchSize = patchSize;
        _channels = channels;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _numClasses = numClasses;
        _stateDimension = stateDimension;
        _scanPattern = scanPattern;

        _numPatchesH = imageHeight / patchSize;
        _numPatchesW = imageWidth / patchSize;
        _numPatches = _numPatchesH * _numPatchesW;
        _mambaInputDim = scanPattern == VisionScanPattern.Bidirectional
            ? modelDimension * 2
            : modelDimension;

        // Initialize patch embedding weights
        int patchDim = patchSize * patchSize * channels;
        _patchProjectionWeights = new Tensor<T>(new[] { patchDim, modelDimension });
        _patchProjectionBias = new Tensor<T>(new[] { modelDimension });
        InitializeTensor(_patchProjectionWeights);
        _patchProjectionBias.Fill(NumOps.Zero);

        _positionalEmbedding = new Tensor<T>(new[] { _numPatches, modelDimension });
        InitializeTensor(_positionalEmbedding, scale: 0.02);

        // Final norm
        _finalNormGamma = new Tensor<T>(new[] { _mambaInputDim });
        _finalNormGamma.Fill(NumOps.One);

        // Classification head
        _classifierWeights = new Tensor<T>(new[] { _mambaInputDim, numClasses });
        InitializeTensor(_classifierWeights);
        _classifierBias = new Tensor<T>(new[] { numClasses });
        _classifierBias.Fill(NumOps.Zero);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateVisionMambaClassifierLayers(
                _numPatches, _mambaInputDim, _numLayers, _stateDimension));
        }
    }

    private void InitializeTensor(Tensor<T> tensor, double scale = -1)
    {
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape[1];
        T s = scale > 0
            ? NumOps.FromDouble(scale)
            : NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));

        var rand = Tensors.Helpers.RandomHelper.CreateSecureRandom();
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(
                NumOps.FromDouble(rand.NextDouble() - 0.5), s);
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);

        int rank = input.Shape.Length;
        int batchSize;
        Tensor<T> input4D;

        if (rank == 3)
        {
            batchSize = 1;
            input4D = input.Reshape(1, _channels, _imageHeight, _imageWidth);
        }
        else if (rank == 4)
        {
            batchSize = input.Shape[0];
            input4D = input;
        }
        else
        {
            throw new ArgumentException(
                $"Input must be 3D [C,H,W] or 4D [batch,C,H,W], got {rank}D.");
        }

        // Step 1: Extract and flatten patches
        var patches = ExtractPatches(input4D, batchSize);

        // Step 2: Project patches to embeddings
        int patchDim = _patchSize * _patchSize * _channels;
        var patchesFlat = patches.Reshape(batchSize * _numPatches, patchDim);
        var embeddedFlat = Engine.TensorMatMul(patchesFlat, _patchProjectionWeights);
        var bias2D = _patchProjectionBias.Reshape(1, _modelDimension);
        embeddedFlat = Engine.TensorBroadcastAdd(embeddedFlat, bias2D);
        var embedded = embeddedFlat.Reshape(batchSize, _numPatches, _modelDimension);

        // Add positional embedding
        var posEmb3D = _positionalEmbedding.Reshape(1, _numPatches, _modelDimension);
        embedded = Engine.TensorBroadcastAdd(embedded, posEmb3D);

        // Step 3: Apply scan pattern
        Tensor<T> scannedInput;
        if (_scanPattern == VisionScanPattern.Bidirectional)
        {
            scannedInput = ScanPatterns<T>.BidirectionalScan(embedded);
        }
        else if (_scanPattern == VisionScanPattern.CrossScan)
        {
            var scans = ScanPatterns<T>.CrossScan(embedded, _numPatchesH, _numPatchesW);
            scannedInput = ScanPatterns<T>.MergeScanOutputs(scans);
        }
        else
        {
            scannedInput = ScanPatterns<T>.ContinuousScan(embedded, _numPatchesH, _numPatchesW);
        }

        // Step 4: Pass through Mamba blocks (Layers) with residuals
        var current = scannedInput;
        for (int i = 0; i < Layers.Count; i++)
        {
            var blockOut = Layers[i].Forward(current);
            current = Engine.TensorAdd(current, blockOut);
        }

        // Step 5: Global average pooling over sequence dimension
        int featureDim = current.Shape[2];
        var pooled = Engine.ReduceSum(current, new int[] { 1 });
        T seqLenT = NumOps.FromDouble(_numPatches);
        for (int i = 0; i < pooled.Length; i++)
        {
            pooled[i] = NumOps.Divide(pooled[i], seqLenT);
        }

        // Step 6: Final RMS norm on pooled output
        var normedPooled = ApplyRMSNorm1D(pooled, _finalNormGamma, batchSize, featureDim);

        // Step 7: Classification head
        var logits = Engine.TensorMatMul(normedPooled, _classifierWeights);
        var clsBias = _classifierBias.Reshape(1, _numClasses);
        logits = Engine.TensorBroadcastAdd(logits, clsBias);

        if (rank == 3)
            return logits.Reshape(_numClasses);

        return logits;
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        var predictions = Predict(input);
        SetTrainingMode(true); // Re-enable after Predict set it to false
        LastLoss = LossFunction.CalculateLoss(predictions.ToVector(), expectedOutput.ToVector());
        var outputGradients = LossFunction.CalculateDerivative(predictions.ToVector(), expectedOutput.ToVector());
        Backpropagate(Tensor<T>.FromVector(outputGradients));
        var parameterGradients = GetParameterGradients();
        parameterGradients = ClipGradient(parameterGradients);
        UpdateParameters(parameterGradients);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> gradients)
    {
        if (gradients.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {ParameterCount} gradients, but got {gradients.Length}",
                nameof(gradients));
        }

        var currentParams = GetParameters();
        T learningRate = NumOps.FromDouble(0.001);
        currentParams = Engine.Subtract(currentParams, Engine.Multiply(gradients, learningRate));
        SetParameters(currentParams);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "VisionMamba" },
                { "ImageHeight", _imageHeight },
                { "ImageWidth", _imageWidth },
                { "PatchSize", _patchSize },
                { "Channels", _channels },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
                { "NumClasses", _numClasses },
                { "StateDimension", _stateDimension },
                { "ScanPattern", _scanPattern.ToString() },
                { "NumPatches", _numPatches },
                { "LayerCount", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_imageHeight);
        writer.Write(_imageWidth);
        writer.Write(_patchSize);
        writer.Write(_channels);
        writer.Write(_modelDimension);
        writer.Write(_numLayers);
        writer.Write(_numClasses);
        writer.Write(_stateDimension);
        writer.Write((int)_scanPattern);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new VisionMambaModel<T>(
            Architecture, _imageHeight, _imageWidth, _patchSize, _channels,
            _modelDimension, _numLayers, _numClasses, _stateDimension,
            _scanPattern, LossFunction, _options);
    }

    #endregion

    #region Private Helpers

    private Tensor<T> ExtractPatches(Tensor<T> input, int batchSize)
    {
        int patchDim = _patchSize * _patchSize * _channels;
        var patches = new Tensor<T>(new[] { batchSize, _numPatches, patchDim });

        for (int b = 0; b < batchSize; b++)
        {
            int patchIdx = 0;
            for (int ph = 0; ph < _numPatchesH; ph++)
            {
                for (int pw = 0; pw < _numPatchesW; pw++)
                {
                    int flatIdx = 0;
                    for (int c = 0; c < _channels; c++)
                    {
                        for (int h = 0; h < _patchSize; h++)
                        {
                            for (int w = 0; w < _patchSize; w++)
                            {
                                int srcH = ph * _patchSize + h;
                                int srcW = pw * _patchSize + w;
                                patches[new[] { b, patchIdx, flatIdx }] =
                                    input[new[] { b, c, srcH, srcW }];
                                flatIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
        }

        return patches;
    }

    private Tensor<T> ApplyRMSNorm1D(Tensor<T> input, Tensor<T> gamma, int batchSize, int dim)
    {
        var output = TensorAllocator.Rent<T>(input.Shape);
        T eps = NumOps.FromDouble(1e-6);

        for (int b = 0; b < batchSize; b++)
        {
            T sumSq = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                T val = input[new[] { b, d }];
                sumSq = NumOps.Add(sumSq, NumOps.Multiply(val, val));
            }
            T meanSq = NumOps.Divide(sumSq, NumOps.FromDouble(dim));
            T rms = NumOps.Sqrt(NumOps.Add(meanSq, eps));

            for (int d = 0; d < dim; d++)
            {
                output[new[] { b, d }] = NumOps.Multiply(
                    NumOps.Divide(input[new[] { b, d }], rms), gamma[d]);
            }
        }

        return output;
    }

    #endregion
}
