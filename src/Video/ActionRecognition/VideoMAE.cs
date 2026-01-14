using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
public class VideoMAE<T> : NeuralNetworkBase<T>
{
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
    private bool _useNativeMode;
    private string? _onnxModelPath;
    private InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    private readonly Random _random = RandomHelper.CreateSecureRandom();
    private bool _disposed;

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
    /// Initializes a new instance of the VideoMAE class in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">The number of action classes for classification.</param>
    /// <param name="numFrames">The number of video frames to process.</param>
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
        int numFrames = 16,
        int numFeatures = 768,
        double maskRatio = 0.9)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 224;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _numFrames = numFrames;
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
    /// <param name="numFrames">The number of video frames to process.</param>
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
        int numFrames = 16)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"VideoMAE ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 224;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _numFrames = numFrames;
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

        // Create tube mask
        var mask = CreateTubeMask(video.Shape[0]);

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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return ClassifyAction(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use native mode constructor for training.");
        }

        var predicted = Predict(input);

        // Compute loss gradient using the configured loss function
        var gradientVector = LossFunction.CalculateDerivative(predicted.ToVector(), expectedOutput.ToVector());
        var lossGradient = new Tensor<T>(predicted.Shape, gradientVector);

        BackwardPass(lossGradient);

        if (_optimizer != null)
        {
            _optimizer.UpdateParameters(Layers);
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

        // Reshape video to batch of frame pairs
        var patchEmbedded = PatchEmbed(video);

        // Apply encoder blocks
        var features = patchEmbedded;
        int encoderLayerCount = Math.Min(13, Layers.Count);
        for (int i = 1; i < encoderLayerCount; i++)
        {
            features = Layers[i].Forward(features);
            features = ApplyGELU(features);
        }

        // Global average pool for features
        return GlobalAveragePool(features);
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
        // Classification head layers are at indices 13 and 14
        if (_useNativeMode && Layers.Count > 14)
        {
            var classHead = Layers[13].Forward(features);
            classHead = ApplyGELU(classHead);
            return Layers[14].Forward(classHead);
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

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_onnxSession.InputMetadata.Keys.First(), onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
    }

    private bool[,,] CreateTubeMask(int batchSize)
    {
        int numTubelets = _numFrames / _tubeletSize;
        int patchesH = _height / _patchSize;
        int patchesW = _width / _patchSize;
        int totalPatches = numTubelets * patchesH * patchesW;
        int numMasked = (int)(totalPatches * _maskRatio);

        var mask = new bool[batchSize, patchesH, patchesW];

        for (int b = 0; b < batchSize; b++)
        {
            var indices = Enumerable.Range(0, patchesH * patchesW).OrderBy(_ => _random.Next()).Take(numMasked).ToHashSet();

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

        // Apply encoder blocks
        var features = patchEmbedded;
        int encoderLayerCount = Math.Min(13, Layers.Count);
        for (int i = 1; i < encoderLayerCount; i++)
        {
            features = Layers[i].Forward(features);
            features = ApplyGELU(features);
        }

        return features;
    }

    private Tensor<T> DecodeForReconstruction(Tensor<T> features)
    {
        // Decoder blocks start at index 15 (after classification head)
        var decoded = features;
        int decoderStartIdx = 15;

        if (_useNativeMode && Layers.Count > decoderStartIdx + 4)
        {
            for (int i = 0; i < 4; i++)
            {
                decoded = Layers[decoderStartIdx + i].Forward(decoded);
                decoded = ApplyGELU(decoded);
            }

            // Reconstruction head
            if (Layers.Count > decoderStartIdx + 4)
            {
                decoded = Layers[decoderStartIdx + 4].Forward(decoded);
            }
        }

        return decoded;
    }

    private T ComputeReconstructionLoss(Tensor<T> reconstructed, Tensor<T> original, bool[,,] mask)
    {
        T loss = NumOps.Zero;
        int count = 0;

        int batchSize = reconstructed.Shape[0];
        int channels = reconstructed.Shape[1];
        int height = reconstructed.Shape[2];
        int width = reconstructed.Shape[3];

        // Handle both 4D [B,C,H,W] and 5D [B,T,C,H,W] original tensors
        bool is5D = original.Rank == 5;
        int origChannels = is5D ? original.Shape[2] : original.Shape[1];
        int origHeight = is5D ? original.Shape[3] : original.Shape[2];
        int origWidth = is5D ? original.Shape[4] : original.Shape[3];
        int numFrames = is5D ? original.Shape[1] : 1;

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int maskB = b % mask.GetLength(0);
                    int maskH = h % mask.GetLength(1);
                    int maskW = w % mask.GetLength(2);

                    if (mask[maskB, maskH, maskW])
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            // Map reconstructed position to original position
                            int origH = h % origHeight;
                            int origW = w % origWidth;
                            int origC = c % origChannels;

                            // Get original value - average over all frames if 5D
                            T origVal;
                            if (is5D)
                            {
                                // Average over all frames for proper temporal reconstruction loss
                                T sum = NumOps.Zero;
                                for (int t = 0; t < numFrames; t++)
                                {
                                    int idx = b * numFrames * origChannels * origHeight * origWidth +
                                              t * origChannels * origHeight * origWidth +
                                              origC * origHeight * origWidth +
                                              origH * origWidth + origW;
                                    sum = NumOps.Add(sum, original.Data.Span[idx]);
                                }
                                origVal = NumOps.Divide(sum, NumOps.FromDouble(numFrames));
                            }
                            else
                            {
                                origVal = original[b, origC, origH, origW];
                            }

                            T diff = NumOps.Subtract(reconstructed[b, c, h, w], origVal);
                            loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
                            count++;
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

        var output = new Tensor<T>([batchSize, channels, 1, 1]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                T sum = NumOps.Zero;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        sum = NumOps.Add(sum, input[b, c, h, w]);
                    }
                }
                output[b, c, 0, 0] = NumOps.Divide(sum, NumOps.FromDouble(height * width));
            }
        }

        return output;
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
        int lastDim = input.Shape[^1];
        var output = new Tensor<T>(input.Shape);

        int totalElements = input.Data.Length;
        int numVectors = totalElements / lastDim;

        for (int v = 0; v < numVectors; v++)
        {
            int offset = v * lastDim;

            double maxVal = double.MinValue;
            for (int i = 0; i < lastDim; i++)
            {
                double val = Convert.ToDouble(input.Data.Span[offset + i]);
                if (val > maxVal) maxVal = val;
            }

            double sum = 0;
            for (int i = 0; i < lastDim; i++)
            {
                double val = Convert.ToDouble(input.Data.Span[offset + i]);
                sum += Math.Exp(val - maxVal);
            }

            for (int i = 0; i < lastDim; i++)
            {
                double val = Convert.ToDouble(input.Data.Span[offset + i]);
                output.Data.Span[offset + i] = NumOps.FromDouble(Math.Exp(val - maxVal) / sum);
            }
        }

        return output;
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

    private void BackwardPass(Tensor<T> gradient)
    {
        if (!_useNativeMode || Layers.Count == 0)
        {
            return;
        }

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }
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

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (offset + layerParamCount <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.UpdateParameters(newParams);
                offset += layerParamCount;
            }
        }
    }

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
            ModelType = ModelType.VideoActionRecognition,
            AdditionalInfo = additionalInfo,
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numFrames);
        writer.Write(_numClasses);
        writer.Write(_numFeatures);
        writer.Write(_maskRatio);
        writer.Write(_useNativeMode);
        writer.Write(_onnxModelPath ?? string.Empty);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _height = reader.ReadInt32();
        _width = reader.ReadInt32();
        _channels = reader.ReadInt32();
        _numFrames = reader.ReadInt32();
        _numClasses = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        _maskRatio = reader.ReadDouble();
        _useNativeMode = reader.ReadBoolean();
        _onnxModelPath = reader.ReadString();
        if (string.IsNullOrEmpty(_onnxModelPath)) _onnxModelPath = null;

        // Recreate ONNX session if in ONNX mode
        if (!_useNativeMode && !string.IsNullOrEmpty(_onnxModelPath))
        {
            if (File.Exists(_onnxModelPath))
            {
                try { _onnxSession = new InferenceSession(_onnxModelPath); }
                catch (Exception ex) { throw new InvalidOperationException($"Failed to restore ONNX session: {ex.Message}", ex); }
            }
            else
            {
                throw new FileNotFoundException($"ONNX model file not found during deserialization: {_onnxModelPath}");
            }
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new VideoMAE<T>(Architecture, _optimizer, LossFunction, _numClasses, _numFrames, _numFeatures, _maskRatio);
        }
        else
        {
            return new VideoMAE<T>(Architecture, _onnxModelPath!, _numClasses, _numFrames);
        }
    }

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
