using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

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
///
/// How it works:
/// 1. Cut the image into small patches (like cutting a photo into tiles)
/// 2. Convert each patch into a vector of numbers
/// 3. Read the patches in a specific order (the scan pattern)
/// 4. Process with Mamba blocks (fast sequence processing)
/// 5. Output: probability for each class (e.g., "cat", "dog", "car")
///
/// Why use Mamba for vision?
/// - Transformers (ViT) need O(n^2) computation for n patches
/// - Vision Mamba needs O(n) computation → faster for high-resolution images
/// - Particularly useful for large images (medical imaging, satellite photos)
///
/// The scan pattern determines how the 2D grid of patches is read as a 1D sequence.
/// Bidirectional is the simplest (read forward and backward), while cross-scan reads
/// in 4 directions to capture all spatial relationships.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item><description>Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", 2024</description></item>
/// <item><description>Liu et al., "VMamba: Visual State Space Model", 2024</description></item>
/// <item><description>Yang et al., "PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition", 2024</description></item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class VisionMambaModel<T> : LayerBase<T>
{
    private readonly int _imageHeight;
    private readonly int _imageWidth;
    private readonly int _patchSize;
    private readonly int _channels;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _numClasses;
    private readonly int _numPatchesH;
    private readonly int _numPatchesW;
    private readonly int _numPatches;
    private readonly VisionScanPattern _scanPattern;

    // Patch embedding: [patchSize * patchSize * channels, modelDim]
    private Tensor<T> _patchProjectionWeights;
    private Tensor<T> _patchProjectionBias;

    // Positional embedding: [numPatches, modelDim]
    private Tensor<T> _positionalEmbedding;

    // Mamba blocks
    private readonly MambaBlock<T>[] _blocks;

    // Final normalization
    private Tensor<T> _finalNormGamma;

    // Classification head: [modelDim, numClasses]
    private Tensor<T> _classifierWeights;
    private Tensor<T> _classifierBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastPatches;
    private Tensor<T>? _lastEmbedded;
    private Tensor<T>? _lastScannedInput;
    private Tensor<T>? _lastPooled;
    private Tensor<T>? _lastNormedPooled;
    private Tensor<T>[]? _lastBlockInputs;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _patchProjectionWeightsGradient;
    private Tensor<T>? _patchProjectionBiasGradient;
    private Tensor<T>? _positionalEmbeddingGradient;
    private Tensor<T>? _finalNormGammaGradient;
    private Tensor<T>? _classifierWeightsGradient;
    private Tensor<T>? _classifierBiasGradient;

    /// <inheritdoc />
    /// <summary>
    /// Training is not yet supported. The backward pass uses a simplified RMSNorm derivative and
    /// does not invert cross-scan/continuous scan patterns. Full gradient computation is required
    /// before enabling training.
    /// </summary>
    public override bool SupportsTraining => false;

    /// <summary>Gets the image height.</summary>
    /// <remarks><para><b>For Beginners:</b> The expected height of input images in pixels. Images must have this exact height.</para></remarks>
    public int ImageHeight => _imageHeight;

    /// <summary>Gets the image width.</summary>
    /// <remarks><para><b>For Beginners:</b> The expected width of input images in pixels. Images must have this exact width.</para></remarks>
    public int ImageWidth => _imageWidth;

    /// <summary>Gets the patch size.</summary>
    /// <remarks><para><b>For Beginners:</b> The size of each square patch the image is divided into.
    /// A 224x224 image with patch size 16 produces 196 patches (14x14 grid).</para></remarks>
    public int PatchSize => _patchSize;

    /// <summary>Gets the model dimension.</summary>
    /// <remarks><para><b>For Beginners:</b> The embedding dimension each patch is projected to.
    /// Larger values capture more detail per patch but require more memory.</para></remarks>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of Mamba layers.</summary>
    /// <remarks><para><b>For Beginners:</b> How many Mamba blocks process the patch sequence.
    /// More layers = deeper understanding but more computation.</para></remarks>
    public int NumLayers => _numLayers;

    /// <summary>Gets the number of output classes.</summary>
    /// <remarks><para><b>For Beginners:</b> The number of categories the model can classify images into
    /// (e.g., 10 for CIFAR-10, 1000 for ImageNet).</para></remarks>
    public int NumClasses => _numClasses;

    /// <summary>Gets the total number of patches.</summary>
    /// <remarks><para><b>For Beginners:</b> How many patches the image is divided into.
    /// Equals (imageHeight / patchSize) * (imageWidth / patchSize).</para></remarks>
    public int NumPatches => _numPatches;

    /// <summary>Gets the scan pattern used.</summary>
    /// <remarks><para><b>For Beginners:</b> How the 2D grid of patches is read as a 1D sequence.
    /// Different patterns (bidirectional, cross-scan, continuous) capture different spatial relationships.</para></remarks>
    public VisionScanPattern ScanPattern => _scanPattern;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks><para><b>For Beginners:</b> Total count of learnable numbers including patch projection,
    /// positional embeddings, all Mamba blocks, normalization, and the classification head.</para></remarks>
    public override int ParameterCount
    {
        get
        {
            int count = _patchProjectionWeights.Length + _patchProjectionBias.Length;
            count += _positionalEmbedding.Length;
            foreach (var block in _blocks)
                count += block.ParameterCount;
            count += _finalNormGamma.Length;
            count += _classifierWeights.Length + _classifierBias.Length;
            return count;
        }
    }

    /// <summary>
    /// Creates a new Vision Mamba model.
    /// </summary>
    /// <param name="imageHeight">Height of input images. Must be divisible by patchSize.</param>
    /// <param name="imageWidth">Width of input images. Must be divisible by patchSize.</param>
    /// <param name="patchSize">Size of each square patch. Default: 16.</param>
    /// <param name="channels">Number of input channels (e.g., 3 for RGB). Default: 3.</param>
    /// <param name="modelDimension">Model embedding dimension. Default: 192.</param>
    /// <param name="numLayers">Number of Mamba blocks. Default: 4.</param>
    /// <param name="numClasses">Number of output classes. Default: 10.</param>
    /// <param name="stateDimension">SSM state dimension. Default: 16.</param>
    /// <param name="scanPattern">Scanning pattern for converting 2D patches to 1D. Default: Bidirectional.</param>
    /// <param name="activationFunction">Optional activation on final output.</param>
    public VisionMambaModel(
        int imageHeight,
        int imageWidth,
        int patchSize = 16,
        int channels = 3,
        int modelDimension = 192,
        int numLayers = 4,
        int numClasses = 10,
        int stateDimension = 16,
        VisionScanPattern scanPattern = VisionScanPattern.Bidirectional,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [channels, imageHeight, imageWidth],
            [numClasses],
            activationFunction ?? new ActivationFunctions.IdentityActivation<T>())
    {
        if (imageHeight <= 0) throw new ArgumentException($"Image height ({imageHeight}) must be positive.", nameof(imageHeight));
        if (imageWidth <= 0) throw new ArgumentException($"Image width ({imageWidth}) must be positive.", nameof(imageWidth));
        if (patchSize <= 0) throw new ArgumentException($"Patch size ({patchSize}) must be positive.", nameof(patchSize));
        if (imageHeight % patchSize != 0) throw new ArgumentException($"Image height ({imageHeight}) must be divisible by patch size ({patchSize}).", nameof(imageHeight));
        if (imageWidth % patchSize != 0) throw new ArgumentException($"Image width ({imageWidth}) must be divisible by patch size ({patchSize}).", nameof(imageWidth));
        if (channels <= 0) throw new ArgumentException($"Channels ({channels}) must be positive.", nameof(channels));
        if (modelDimension <= 0) throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numLayers <= 0) throw new ArgumentException($"Number of layers ({numLayers}) must be positive.", nameof(numLayers));
        if (numClasses <= 0) throw new ArgumentException($"Number of classes ({numClasses}) must be positive.", nameof(numClasses));

        _imageHeight = imageHeight;
        _imageWidth = imageWidth;
        _patchSize = patchSize;
        _channels = channels;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _numClasses = numClasses;
        _scanPattern = scanPattern;

        _numPatchesH = imageHeight / patchSize;
        _numPatchesW = imageWidth / patchSize;
        _numPatches = _numPatchesH * _numPatchesW;

        int patchDim = patchSize * patchSize * channels;

        // Patch projection: flatten patch -> embedding
        _patchProjectionWeights = new Tensor<T>(new[] { patchDim, modelDimension });
        _patchProjectionBias = new Tensor<T>(new[] { modelDimension });
        InitializeTensor(_patchProjectionWeights);
        _patchProjectionBias.Fill(NumOps.Zero);

        // Positional embedding
        _positionalEmbedding = new Tensor<T>(new[] { _numPatches, modelDimension });
        InitializeTensor(_positionalEmbedding, scale: 0.02);

        // Determine the effective sequence length for Mamba blocks based on scan pattern
        int effectiveSeqLen = scanPattern == VisionScanPattern.Bidirectional
            ? _numPatches  // bidirectional doubles dim, not seqLen
            : _numPatches;

        // Mamba blocks
        int mambaInputDim = scanPattern == VisionScanPattern.Bidirectional
            ? modelDimension * 2
            : modelDimension;

        _blocks = new MambaBlock<T>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            _blocks[i] = new MambaBlock<T>(
                effectiveSeqLen, mambaInputDim, stateDimension);
        }

        // Final norm
        _finalNormGamma = new Tensor<T>(new[] { mambaInputDim });
        _finalNormGamma.Fill(NumOps.One);

        // Classification head
        _classifierWeights = new Tensor<T>(new[] { mambaInputDim, numClasses });
        InitializeTensor(_classifierWeights);
        _classifierBias = new Tensor<T>(new[] { numClasses });
        _classifierBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor, double scale = -1)
    {
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape[1];
        T s = scale > 0
            ? NumOps.FromDouble(scale)
            : NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));

        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(
                NumOps.FromDouble(Random.NextDouble() - 0.5), s);
        }
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;

        int rank = input.Shape.Length;
        // Expect input: [batch, channels, height, width] or [channels, height, width]
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

        _lastInput = input4D;

        // Step 1: Extract and flatten patches
        var patches = ExtractPatches(input4D, batchSize);
        _lastPatches = patches;

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
        _lastEmbedded = embedded;

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
        else // Continuous
        {
            scannedInput = ScanPatterns<T>.ContinuousScan(embedded, _numPatchesH, _numPatchesW);
        }
        _lastScannedInput = scannedInput;

        // Step 4: Pass through Mamba blocks with residuals
        _lastBlockInputs = new Tensor<T>[_numLayers];
        var current = scannedInput;
        for (int i = 0; i < _numLayers; i++)
        {
            _lastBlockInputs[i] = current;
            var blockOut = _blocks[i].Forward(current);
            current = Engine.TensorAdd(current, blockOut);
        }

        // Step 5: Global average pooling over sequence dimension
        int featureDim = current.Shape[2];
        var pooled = Engine.ReduceSum(current, new int[] { 1 }); // [batch, featureDim]
        T seqLenT = NumOps.FromDouble(_numPatches);
        for (int i = 0; i < pooled.Length; i++)
        {
            pooled[i] = NumOps.Divide(pooled[i], seqLenT);
        }
        _lastPooled = pooled;

        // Step 6: Final norm on pooled output
        var normedPooled = ApplyRMSNorm1D(pooled, _finalNormGamma, batchSize, featureDim);
        _lastNormedPooled = normedPooled;

        // Step 7: Classification head
        var logits = Engine.TensorMatMul(normedPooled, _classifierWeights);
        var clsBias = _classifierBias.Reshape(1, _numClasses);
        logits = Engine.TensorBroadcastAdd(logits, clsBias);

        var result = ApplyActivation(logits);
        _lastOutput = result;

        if (rank == 3)
            return result.Reshape(_numClasses);

        return result;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastPatches == null ||
            _lastEmbedded == null || _lastScannedInput == null ||
            _lastPooled == null || _lastNormedPooled == null || _lastBlockInputs == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _lastInput.Shape[0];
        int featureDim = _lastScannedInput.Shape[2];

        var grad = outputGradient.Rank == 1
            ? outputGradient.Reshape(1, _numClasses)
            : outputGradient;

        grad = ApplyActivationDerivative(_lastOutput, grad);

        // Step 7 backward: Classification head
        _classifierBiasGradient = Engine.ReduceSum(grad, new int[] { 0 });
        _classifierWeightsGradient = Engine.TensorMatMul(
            _lastNormedPooled.Transpose(new[] { 1, 0 }), grad);
        var dNormed = Engine.TensorMatMul(grad, _classifierWeights.Transpose(new[] { 1, 0 }));

        // Step 6 backward: Final norm (simplified - pass through with gamma scaling)
        var gamma2D = _finalNormGamma.Reshape(1, featureDim);
        var dPooled = Engine.TensorBroadcastMultiply(dNormed, gamma2D);
        _finalNormGammaGradient = new Tensor<T>(new[] { featureDim });
        for (int d = 0; d < featureDim; d++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(dNormed[new[] { b, d }], _lastPooled[new[] { b, d }]));
            }
            _finalNormGammaGradient[d] = sum;
        }

        // Step 5 backward: Global average pooling
        T seqLenT = NumOps.FromDouble(_numPatches);
        var dCurrent = new Tensor<T>(new[] { batchSize, _numPatches, featureDim });
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < _numPatches; s++)
            {
                for (int d = 0; d < featureDim; d++)
                {
                    dCurrent[new[] { b, s, d }] = NumOps.Divide(dPooled[new[] { b, d }], seqLenT);
                }
            }
        }

        // Step 4 backward: Mamba blocks
        for (int i = _numLayers - 1; i >= 0; i--)
        {
            var blockGrad = _blocks[i].Backward(dCurrent);
            dCurrent = Engine.TensorAdd(dCurrent, blockGrad);
        }

        // Step 3 backward: Scan pattern (simplified - pass gradient through)
        // For bidirectional, the gradient needs to be projected back from dim*2 to dim
        Tensor<T> dEmbedded;
        if (_scanPattern == VisionScanPattern.Bidirectional)
        {
            // Split gradient from [batch, numPatches, dim*2] back to forward and reverse
            dEmbedded = new Tensor<T>(new[] { batchSize, _numPatches, _modelDimension });
            for (int b = 0; b < batchSize; b++)
            {
                for (int p = 0; p < _numPatches; p++)
                {
                    for (int d = 0; d < _modelDimension; d++)
                    {
                        // Forward gradient
                        T fwdGrad = dCurrent[new[] { b, p, d }];
                        // Reverse gradient (maps to reverse position)
                        int revP = _numPatches - 1 - p;
                        T revGrad = dCurrent[new[] { b, revP, d + _modelDimension }];
                        dEmbedded[new[] { b, p, d }] = NumOps.Add(fwdGrad, revGrad);
                    }
                }
            }
        }
        else
        {
            dEmbedded = dCurrent;
        }

        // Step 2 backward: Embedding projection
        _positionalEmbeddingGradient = Engine.ReduceSum(dEmbedded, new int[] { 0 });

        int patchDim = _patchSize * _patchSize * _channels;
        var dEmbFlat = dEmbedded.Reshape(batchSize * _numPatches, _modelDimension);
        _patchProjectionBiasGradient = Engine.ReduceSum(dEmbedded, new int[] { 0, 1 });

        var patchesFlat = _lastPatches.Reshape(batchSize * _numPatches, patchDim);
        _patchProjectionWeightsGradient = Engine.TensorMatMul(
            patchesFlat.Transpose(new[] { 1, 0 }), dEmbFlat);

        var dPatches = Engine.TensorMatMul(dEmbFlat, _patchProjectionWeights.Transpose(new[] { 1, 0 }));
        var dPatches3D = dPatches.Reshape(batchSize, _numPatches, patchDim);

        // Reconstruct input gradient from patches
        var dInput = ReconstructFromPatches(dPatches3D, batchSize);

        if (_originalInputShape != null && _originalInputShape.Length == 3)
            return dInput.Reshape(_originalInputShape);

        return dInput;
    }

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

    private Tensor<T> ReconstructFromPatches(Tensor<T> dPatches, int batchSize)
    {
        var dInput = new Tensor<T>(new[] { batchSize, _channels, _imageHeight, _imageWidth });

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
                                dInput[new[] { b, c, srcH, srcW }] = NumOps.Add(
                                    dInput[new[] { b, c, srcH, srcW }],
                                    dPatches[new[] { b, patchIdx, flatIdx }]);
                                flatIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
        }

        return dInput;
    }

    private Tensor<T> ApplyRMSNorm1D(Tensor<T> input, Tensor<T> gamma, int batchSize, int dim)
    {
        var output = new Tensor<T>(input.Shape);
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

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(ParameterCount);
        int index = 0;

        foreach (var tensor in new[] { _patchProjectionWeights, _patchProjectionBias, _positionalEmbedding })
        {
            for (int i = 0; i < tensor.Length; i++)
                parameters[index++] = tensor[i];
        }

        foreach (var block in _blocks)
        {
            var blockParams = block.GetParameters();
            for (int i = 0; i < blockParams.Length; i++)
                parameters[index++] = blockParams[i];
        }

        for (int i = 0; i < _finalNormGamma.Length; i++)
            parameters[index++] = _finalNormGamma[i];

        for (int i = 0; i < _classifierWeights.Length; i++)
            parameters[index++] = _classifierWeights[i];

        for (int i = 0; i < _classifierBias.Length; i++)
            parameters[index++] = _classifierBias[i];

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");

        int index = 0;

        foreach (var tensor in new[] { _patchProjectionWeights, _patchProjectionBias, _positionalEmbedding })
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
        }

        foreach (var block in _blocks)
        {
            var blockParams = new Vector<T>(block.ParameterCount);
            for (int i = 0; i < block.ParameterCount; i++)
                blockParams[i] = parameters[index++];
            block.SetParameters(blockParams);
        }

        for (int i = 0; i < _finalNormGamma.Length; i++)
            _finalNormGamma[i] = parameters[index++];

        for (int i = 0; i < _classifierWeights.Length; i++)
            _classifierWeights[i] = parameters[index++];

        for (int i = 0; i < _classifierBias.Length; i++)
            _classifierBias[i] = parameters[index++];
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_patchProjectionWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);

        _patchProjectionWeights = Engine.TensorAdd(_patchProjectionWeights,
            Engine.TensorMultiplyScalar(_patchProjectionWeightsGradient, negLR));
        _patchProjectionBias = Engine.TensorAdd(_patchProjectionBias,
            Engine.TensorMultiplyScalar(_patchProjectionBiasGradient!, negLR));
        _positionalEmbedding = Engine.TensorAdd(_positionalEmbedding,
            Engine.TensorMultiplyScalar(_positionalEmbeddingGradient!, negLR));

        foreach (var block in _blocks)
            block.UpdateParameters(learningRate);

        _finalNormGamma = Engine.TensorAdd(_finalNormGamma,
            Engine.TensorMultiplyScalar(_finalNormGammaGradient!, negLR));
        _classifierWeights = Engine.TensorAdd(_classifierWeights,
            Engine.TensorMultiplyScalar(_classifierWeightsGradient!, negLR));
        _classifierBias = Engine.TensorAdd(_classifierBias,
            Engine.TensorMultiplyScalar(_classifierBiasGradient!, negLR));
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastPatches = null;
        _lastEmbedded = null;
        _lastScannedInput = null;
        _lastPooled = null;
        _lastNormedPooled = null;
        _lastBlockInputs = null;
        _originalInputShape = null;
        _patchProjectionWeightsGradient = null;
        _patchProjectionBiasGradient = null;
        _positionalEmbeddingGradient = null;
        _finalNormGammaGradient = null;
        _classifierWeightsGradient = null;
        _classifierBiasGradient = null;

        foreach (var block in _blocks)
            block.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        int patchDim = _patchSize * _patchSize * _channels;

        // Input: [1, patchDim] (single flattened patch)
        var patchPlaceholder = new Tensor<T>(new int[] { 1, patchDim });
        var patchNode = TensorOperations<T>.Variable(patchPlaceholder, "patch_input");
        inputNodes.Add(patchNode);

        // Patch projection
        var projWeightsNode = TensorOperations<T>.Variable(_patchProjectionWeights, "W_patch");
        var projBiasNode = TensorOperations<T>.Variable(_patchProjectionBias, "b_patch");
        inputNodes.Add(projWeightsNode);
        inputNodes.Add(projBiasNode);

        var projWeightsT = TensorOperations<T>.Transpose(projWeightsNode);
        var embedded = TensorOperations<T>.MatrixMultiply(patchNode, projWeightsT);
        embedded = TensorOperations<T>.Add(embedded, projBiasNode);

        // Classification head
        var clsWeightsNode = TensorOperations<T>.Variable(_classifierWeights, "W_cls");
        var clsBiasNode = TensorOperations<T>.Variable(_classifierBias, "b_cls");
        inputNodes.Add(clsWeightsNode);
        inputNodes.Add(clsBiasNode);

        var clsWeightsT = TensorOperations<T>.Transpose(clsWeightsNode);
        var logits = TensorOperations<T>.MatrixMultiply(embedded, clsWeightsT);
        var output = TensorOperations<T>.Add(logits, clsBiasNode);

        return output;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ImageHeight"] = _imageHeight.ToString();
        metadata["ImageWidth"] = _imageWidth.ToString();
        metadata["PatchSize"] = _patchSize.ToString();
        metadata["Channels"] = _channels.ToString();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumLayers"] = _numLayers.ToString();
        metadata["NumClasses"] = _numClasses.ToString();
        metadata["NumPatches"] = _numPatches.ToString();
        metadata["ScanPattern"] = _scanPattern.ToString();
        return metadata;
    }
}
