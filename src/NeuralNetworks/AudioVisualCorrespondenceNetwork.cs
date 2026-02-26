using System.IO;
using AiDotNet.ActivationFunctions;
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
/// </remarks>
public class AudioVisualCorrespondenceNetwork<T> : NeuralNetworkBase<T>, IAudioVisualCorrespondenceModel<T>
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
    private readonly double _videoFrameRate;
    private readonly int _numEncoderLayers;
    private readonly int _hiddenDim;

    // Audio encoder components
    private List<ILayer<T>>? _audioEncoderLayers;
    private ILayer<T>? _audioInputProjection;
    private ILayer<T>? _audioOutputProjection;
    private Matrix<T>? _audioPositionalEmbedding;

    // Visual encoder components
    private List<ILayer<T>>? _visualEncoderLayers;
    private ILayer<T>? _visualInputProjection;
    private ILayer<T>? _visualOutputProjection;
    private Matrix<T>? _visualPositionalEmbedding;

    // Cross-modal attention for localization
    private List<ILayer<T>>? _crossModalAttentionLayers;
    private ILayer<T>? _localizationHead;

    // Synchronization head
    private ILayer<T>? _syncHead;

    // Scene classification head
    private ILayer<T>? _sceneClassificationHead;

    // Separation network components
    private ILayer<T>? _separationMaskPredictor;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    // Scene labels for classification
    private readonly List<string> _sceneLabels;

    #endregion

    #region Non-Null Accessors

    private List<ILayer<T>> AudioEncoderLayers => _audioEncoderLayers ?? throw new InvalidOperationException("Audio encoder layers not initialized.");
    private ILayer<T> AudioInputProjection => _audioInputProjection ?? throw new InvalidOperationException("Audio input projection not initialized.");
    private ILayer<T> AudioOutputProjection => _audioOutputProjection ?? throw new InvalidOperationException("Audio output projection not initialized.");
    private List<ILayer<T>> VisualEncoderLayers => _visualEncoderLayers ?? throw new InvalidOperationException("Visual encoder layers not initialized.");
    private ILayer<T> VisualInputProjection => _visualInputProjection ?? throw new InvalidOperationException("Visual input projection not initialized.");
    private ILayer<T> VisualOutputProjection => _visualOutputProjection ?? throw new InvalidOperationException("Visual output projection not initialized.");
    private List<ILayer<T>> CrossModalAttentionLayers => _crossModalAttentionLayers ?? throw new InvalidOperationException("Cross-modal attention layers not initialized.");
    private ILayer<T> LocalizationHead => _localizationHead ?? throw new InvalidOperationException("Localization head not initialized.");
    private ILayer<T> SyncHead => _syncHead ?? throw new InvalidOperationException("Sync head not initialized.");
    private ILayer<T> SceneClassificationHead => _sceneClassificationHead ?? throw new InvalidOperationException("Scene classification head not initialized.");
    private ILayer<T> SeparationMaskPredictor => _separationMaskPredictor ?? throw new InvalidOperationException("Separation mask predictor not initialized.");

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
    /// Creates a new audio-visual correspondence network.
    /// </summary>
    /// <param name="architecture">Network architecture configuration.</param>
    /// <param name="embeddingDimension">Dimension of shared embedding space.</param>
    /// <param name="audioSampleRate">Expected audio sample rate.</param>
    /// <param name="videoFrameRate">Expected video frame rate.</param>
    /// <param name="numEncoderLayers">Number of encoder layers per modality.</param>
    /// <param name="optimizer">Gradient-based optimizer for training.</param>
    /// <param name="lossFunction">Loss function for training.</param>
    public AudioVisualCorrespondenceNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int embeddingDimension = DEFAULT_EMBEDDING_DIM,
        int audioSampleRate = DEFAULT_SAMPLE_RATE,
        double videoFrameRate = DEFAULT_FRAME_RATE,
        int numEncoderLayers = 6,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        AudioVisualCorrespondenceOptions? options = null)
        : base(architecture, lossFunction ?? new ContrastiveLoss<T>(), 1.0)
    {
        _options = options ?? new AudioVisualCorrespondenceOptions();
        Options = _options;

        _embeddingDimension = embeddingDimension;
        _audioSampleRate = audioSampleRate;
        _videoFrameRate = videoFrameRate;
        _numEncoderLayers = numEncoderLayers;
        _hiddenDim = embeddingDimension * 4;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new ContrastiveLoss<T>();

        _sceneLabels = new List<string>();

        InitializeLayers();
        InitializePositionalEmbeddings();
    }

    #endregion

    #region Initialization

    private void InitializePositionalEmbeddings()
    {
        const int maxAudioLength = 500;
        const int maxVisualLength = 256;

        _audioPositionalEmbedding = new Matrix<T>(maxAudioLength, _embeddingDimension);
        _visualPositionalEmbedding = new Matrix<T>(maxVisualLength, _embeddingDimension);

        InitializeSinusoidalEmbedding(_audioPositionalEmbedding, maxAudioLength);
        InitializeSinusoidalEmbedding(_visualPositionalEmbedding, maxVisualLength);
    }

    private void InitializeSinusoidalEmbedding(Matrix<T> embedding, int maxLength)
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

        // Global average pooling
        var embedding = GlobalAveragePool(projected);
        return NormalizeVector(embedding);
    }

    /// <inheritdoc/>
    public Vector<T> GetVisualEmbedding(IEnumerable<Tensor<T>> frames)
    {
        var frameList = frames.ToList();
        if (frameList.Count == 0)
        {
            return new Vector<T>(_embeddingDimension);
        }

        // Aggregate frame embeddings
        var aggregatedEmbedding = new Vector<T>(_embeddingDimension);

        foreach (var frame in frameList)
        {
            var frameEmbedding = ApplyVisualEncoder(frame);
            var pooled = GlobalAveragePool(frameEmbedding);

            for (int i = 0; i < _embeddingDimension; i++)
            {
                aggregatedEmbedding[i] = NumOps.Add(aggregatedEmbedding[i], pooled[i]);
            }
        }

        // Average
        var scale = NumOps.FromDouble(1.0 / frameList.Count);
        for (int i = 0; i < _embeddingDimension; i++)
        {
            aggregatedEmbedding[i] = NumOps.Multiply(aggregatedEmbedding[i], scale);
        }

        return NormalizeVector(aggregatedEmbedding);
    }

    /// <inheritdoc/>
    public T ComputeCorrespondence(Tensor<T> audioWaveform, IEnumerable<Tensor<T>> frames)
    {
        var audioEmb = GetAudioEmbedding(audioWaveform, _audioSampleRate);
        var visualEmb = GetVisualEmbedding(frames);

        return ComputeCosineSimilarity(audioEmb, visualEmb);
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
            var attentionMap = ComputeSoundSourceAttention(audioEmbedding, spatialFeatures, frame.Shape);
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

        var syncOutput = SyncHead.Forward(combinedTensor);
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
            var score = ComputeCosineSimilarity(audioEmb, visualEmb);
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
            var score = ComputeCosineSimilarity(visualEmb, audioEmb);
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
        var maskOutput = SeparationMaskPredictor.Forward(combinedTensor);

        // Apply sigmoid to get mask
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

        var embMagnitude = ComputeVectorMagnitude(visualEmb);
        var magValue = NumOps.ToDouble(embMagnitude);

        if (magValue > 0.5)
        {
            descriptions.Add("active scene sounds");
        }

        return $"Expected audio: {string.Join(", ", descriptions)}";
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ClassifyScene(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> sceneLabels)
    {
        var audioEmb = GetAudioEmbedding(audioWaveform, _audioSampleRate);
        var visualEmb = GetVisualEmbedding(frames);

        var combined = ConcatenateVectors(audioEmb, visualEmb);
        var combinedTensor = Tensor<T>.FromVector(combined);

        var classFeatures = SceneClassificationHead.Forward(combinedTensor);

        var labelList = sceneLabels.ToList();
        var results = new Dictionary<string, T>();
        var softmaxDenominator = NumOps.Zero;

        // Compute logits for each label
        var logits = new List<T>();
        for (int i = 0; i < labelList.Count; i++)
        {
            var logit = i < classFeatures.Length ? classFeatures.Data.Span[i] : NumOps.Zero;
            logits.Add(logit);
            softmaxDenominator = NumOps.Add(softmaxDenominator,
                NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logit))));
        }

        // Apply softmax
        for (int i = 0; i < labelList.Count; i++)
        {
            var expLogit = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logits[i])));
            var prob = NumOps.Divide(expLogit, softmaxDenominator);
            results[labelList[i]] = prob;
        }

        return results;
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

        SetTrainingMode(true);

        try
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                T epochLoss = NumOps.Zero;
                int sampleCount = 0;

                for (int i = 0; i < audioList.Count; i++)
                {
                    var audio = audioList[i];
                    var frames = visualList[i];

                    if (frames.Count == 0) continue;

                    // Forward pass: compute embeddings
                    var audioEmb = GetAudioEmbedding(audio, _audioSampleRate);
                    var visualEmb = GetVisualEmbedding(frames);

                    // Compute contrastive loss using vectorized cosine similarity
                    T similarity = Engine.CosineSimilarity(audioEmb, visualEmb);

                    // Target is 1.0 for matched pairs (positive pairs)
                    T target = NumOps.One;
                    T loss = _lossFunction.CalculateLoss(
                        new Vector<T>(1) { [0] = similarity },
                        new Vector<T>(1) { [0] = target });

                    // Backward pass: compute gradients for contrastive loss
                    var outputGrad = _lossFunction.CalculateDerivative(
                        new Vector<T>(1) { [0] = similarity },
                        new Vector<T>(1) { [0] = target });

                    // Propagate gradient through embedding layers
                    // For cosine similarity d(sim)/d(a) = (b - sim*a) / (||a|| * ||b||)
                    var audioNorm = ComputeVectorMagnitude(audioEmb);
                    var visualNorm = ComputeVectorMagnitude(visualEmb);
                    T normProduct = NumOps.Multiply(audioNorm, visualNorm);

                    if (NumOps.ToDouble(normProduct) > 1e-8)
                    {
                        // Gradient w.r.t. audio embedding
                        var audioGrad = new Vector<T>(_embeddingDimension);
                        var visualGrad = new Vector<T>(_embeddingDimension);
                        T gradScale = NumOps.Divide(outputGrad[0], normProduct);

                        for (int j = 0; j < _embeddingDimension; j++)
                        {
                            // d(cos)/d(a_j) = (b_j - sim * a_j) / (||a|| * ||b||)
                            audioGrad[j] = NumOps.Multiply(gradScale,
                                NumOps.Subtract(visualEmb[j], NumOps.Multiply(similarity, audioEmb[j])));
                            // d(cos)/d(b_j) = (a_j - sim * b_j) / (||a|| * ||b||)
                            visualGrad[j] = NumOps.Multiply(gradScale,
                                NumOps.Subtract(audioEmb[j], NumOps.Multiply(similarity, visualEmb[j])));
                        }

                        // Backpropagate through audio encoder
                        var audioGradTensor = Tensor<T>.FromVector(audioGrad);
                        BackpropagateAudioEncoder(audioGradTensor);

                        // Backpropagate through visual encoder
                        var visualGradTensor = Tensor<T>.FromVector(visualGrad);
                        BackpropagateVisualEncoder(visualGradTensor);

                        // Update parameters using optimizer
                        _optimizer.UpdateParameters(Layers);
                    }

                    epochLoss = NumOps.Add(epochLoss, loss);
                    sampleCount++;
                }

                if (sampleCount > 0)
                {
                    LastLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(sampleCount));
                }
            }
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Backpropagates gradients through the audio encoder layers.
    /// </summary>
    private void BackpropagateAudioEncoder(Tensor<T> gradient)
    {
        // Backprop through output projection
        var current = AudioOutputProjection.Backward(gradient);

        // Backprop through encoder layers in reverse (3 layers per block)
        for (int i = AudioEncoderLayers.Count - 1; i >= 0; i -= 3)
        {
            if (i >= 2)
            {
                current = AudioEncoderLayers[i].Backward(current);     // FFN2
                current = AudioEncoderLayers[i - 1].Backward(current); // FFN1
                current = AudioEncoderLayers[i - 2].Backward(current); // Attention
            }
        }

        // Backprop through input projection
        AudioInputProjection.Backward(current);
    }

    /// <summary>
    /// Backpropagates gradients through the visual encoder layers.
    /// </summary>
    private void BackpropagateVisualEncoder(Tensor<T> gradient)
    {
        // Backprop through output projection
        var current = VisualOutputProjection.Backward(gradient);

        // Backprop through encoder layers in reverse (3 layers per block)
        for (int i = VisualEncoderLayers.Count - 1; i >= 0; i -= 3)
        {
            if (i >= 2)
            {
                current = VisualEncoderLayers[i].Backward(current);     // FFN2
                current = VisualEncoderLayers[i - 1].Backward(current); // FFN1
                current = VisualEncoderLayers[i - 2].Backward(current); // Attention
            }
        }

        // Backprop through input projection
        VisualInputProjection.Backward(current);
    }

    #endregion

    #region Helper Methods

    private Tensor<T> ComputeSpectrogram(Tensor<T> waveform, int sampleRate)
    {
        var waveLength = waveform.Length;
        var numFrames = (waveLength - SPECTROGRAM_HOP) / SPECTROGRAM_HOP + 1;
        numFrames = Math.Max(1, numFrames);

        var spectrogram = new Tensor<T>([numFrames, SPECTROGRAM_BINS]);

        // Simplified spectrogram computation
        for (int frame = 0; frame < numFrames; frame++)
        {
            var startSample = frame * SPECTROGRAM_HOP;
            for (int bin = 0; bin < SPECTROGRAM_BINS; bin++)
            {
                var freq = (bin + 1.0) / SPECTROGRAM_BINS;
                var sampleIdx = Math.Min(startSample + bin, waveLength - 1);
                var value = NumOps.ToDouble(waveform.Data.Span[sampleIdx]);
                spectrogram.Data.Span[frame * SPECTROGRAM_BINS + bin] =
                    NumOps.FromDouble(Math.Log(Math.Abs(value * freq) + 1e-8));
            }
        }

        return spectrogram;
    }

    private Tensor<T> ApplyAudioEncoder(Tensor<T> spectrogram)
    {
        // Validate layer count is a multiple of 3 (attention + 2 FFN layers per block)
        if (AudioEncoderLayers.Count % 3 != 0)
        {
            throw new InvalidOperationException(
                $"Audio encoder layer count ({AudioEncoderLayers.Count}) must be a multiple of 3 " +
                "(each block contains attention + 2 FFN layers).");
        }

        var projected = AudioInputProjection.Forward(spectrogram);

        // Add positional embeddings
        if (_audioPositionalEmbedding is not null)
        {
            var seqLen = Math.Min(projected.Shape[0], _audioPositionalEmbedding.Rows);
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < _embeddingDimension; j++)
                {
                    var idx = i * _embeddingDimension + j;
                    if (idx < projected.Length)
                    {
                        projected.Data.Span[idx] = NumOps.Add(
                            projected.Data.Span[idx],
                            _audioPositionalEmbedding[i, j]);
                    }
                }
            }
        }

        var current = projected;
        for (int i = 0; i < AudioEncoderLayers.Count; i += 3)
        {
            var attnOutput = AudioEncoderLayers[i].Forward(current);
            var ffn1 = AudioEncoderLayers[i + 1].Forward(attnOutput);
            // ffn1 layer already has GELU activation - don't apply again
            var ffn2 = AudioEncoderLayers[i + 2].Forward(ffn1);
            current = AddResidual(attnOutput, ffn2);
        }

        return AudioOutputProjection.Forward(current);
    }

    private Tensor<T> ApplyVisualEncoder(Tensor<T> frame)
    {
        // Validate layer count is a multiple of 3 (attention + 2 FFN layers per block)
        if (VisualEncoderLayers.Count % 3 != 0)
        {
            throw new InvalidOperationException(
                $"Visual encoder layer count ({VisualEncoderLayers.Count}) must be a multiple of 3 " +
                "(each block contains attention + 2 FFN layers).");
        }

        // Flatten spatial dimensions for transformer
        var flattened = FlattenToPatches(frame);
        var projected = VisualInputProjection.Forward(flattened);

        // Add positional embeddings
        if (_visualPositionalEmbedding is not null)
        {
            var seqLen = Math.Min(projected.Shape[0], _visualPositionalEmbedding.Rows);
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < _embeddingDimension; j++)
                {
                    var idx = i * _embeddingDimension + j;
                    if (idx < projected.Length)
                    {
                        projected.Data.Span[idx] = NumOps.Add(
                            projected.Data.Span[idx],
                            _visualPositionalEmbedding[i, j]);
                    }
                }
            }
        }

        var current = projected;
        for (int i = 0; i < VisualEncoderLayers.Count; i += 3)
        {
            var attnOutput = VisualEncoderLayers[i].Forward(current);
            var ffn1 = VisualEncoderLayers[i + 1].Forward(attnOutput);
            // ffn1 layer already has GELU activation - don't apply again
            var ffn2 = VisualEncoderLayers[i + 2].Forward(ffn1);
            current = AddResidual(attnOutput, ffn2);
        }

        return VisualOutputProjection.Forward(current);
    }

    private Tensor<T> FlattenToPatches(Tensor<T> frame)
    {
        if (frame.Shape.Length < 3)
        {
            return frame;
        }

        var channels = frame.Shape[^3];
        var height = frame.Shape[^2];
        var width = frame.Shape[^1];

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
        var patchDim = channels * patchSize * patchSize;

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

            var score = ComputeCosineSimilarity(audioEmbedding, spatialVec);
            attentionScores.Data.Span[pos] = score;
        }

        // Softmax normalization
        var maxScore = NumOps.Zero;
        for (int i = 0; i < numPositions; i++)
        {
            if (NumOps.ToDouble(attentionScores.Data.Span[i]) > NumOps.ToDouble(maxScore))
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
        var height = originalShape.Length >= 2 ? originalShape[^2] : 1;
        var width = originalShape.Length >= 1 ? originalShape[^1] : numPositions;
        var patchH = (int)Math.Sqrt(numPositions);
        var patchW = patchH;

        var attentionMap = new Tensor<T>([patchH, patchW]);
        for (int i = 0; i < Math.Min(numPositions, patchH * patchW); i++)
        {
            attentionMap.Data.Span[i] = attentionScores.Data.Span[i];
        }

        return attentionMap;
    }

    private Vector<T> GlobalAveragePool(Tensor<T> features)
    {
        var seqLen = features.Shape[0];
        var embedding = new Vector<T>(_embeddingDimension);

        // Build vectors for each embedding dimension and use vectorized sum
        for (int j = 0; j < _embeddingDimension; j++)
        {
            var columnVector = new Vector<T>(seqLen);
            for (int i = 0; i < seqLen; i++)
            {
                var idx = i * _embeddingDimension + j;
                columnVector[i] = idx < features.Length ? features.Data.Span[idx] : NumOps.Zero;
            }
            // Use IEngine vectorized sum
            T sum = Engine.Sum(columnVector);
            embedding[j] = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
        }

        return embedding;
    }

    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        return vector.SafeNormalize();
    }

    private T ComputeVectorMagnitude(Vector<T> vector)
    {
        // Use IEngine vectorized dot product for sum of squares
        T sumSq = Engine.DotProduct(vector, vector);
        return NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sumSq)));
    }

    private T ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            return NumOps.Zero;
        }

        // Use IEngine vectorized cosine similarity
        return Engine.CosineSimilarity(a, b);
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

    private Tensor<T> ApplyGelu(Tensor<T> input)
    {
        // Use IEngine vectorized GELU activation
        return Engine.GELU(input);
    }

    private Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        // Use IEngine vectorized Sigmoid activation
        return Engine.Sigmoid(input);
    }

    private Tensor<T> AddResidual(Tensor<T> residual, Tensor<T> output)
    {
        // Use IEngine vectorized tensor addition
        return Engine.TensorAdd(residual, output);
    }

    private Tensor<T> ApplyMask(Tensor<T> spectrogram, Tensor<T> mask)
    {
        var numFrames = spectrogram.Shape[0];
        var numBins = spectrogram.Shape[1];

        // Broadcast mask across all frames for vectorized multiplication
        var broadcastedMask = new Tensor<T>([numFrames, numBins]);
        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int bin = 0; bin < numBins; bin++)
            {
                var maskIdx = bin < mask.Length ? bin : mask.Length - 1;
                broadcastedMask.Data.Span[frame * numBins + bin] = mask.Data.Span[maskIdx];
            }
        }

        // Use IEngine vectorized tensor multiplication
        return Engine.TensorMultiply(spectrogram, broadcastedMask);
    }

    private Tensor<T> ReconstructAudio(Tensor<T> maskedSpectrogram)
    {
        var numFrames = maskedSpectrogram.Shape[0];
        var outputLength = numFrames * SPECTROGRAM_HOP;
        var output = new Tensor<T>([outputLength]);

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

    private T ComputeContrastiveLoss(T similarity, T target)
    {
        var diff = NumOps.Subtract(target, similarity);
        return NumOps.Multiply(diff, diff);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        Layers.Clear();

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateAudioVisualCorrespondenceLayers(
                _embeddingDimension, _numEncoderLayers, NUM_ATTENTION_HEADS));
        }

        // Distribute layers to internal fields
        int idx = 0;

        // Audio encoder: input projection
        _audioInputProjection = Layers[idx++];

        // Audio encoder: (attention + FFN1 + FFN2) × numEncoderLayers
        _audioEncoderLayers = new List<ILayer<T>>();
        for (int i = 0; i < _numEncoderLayers * 3; i++)
            _audioEncoderLayers.Add(Layers[idx++]);

        // Audio output projection
        _audioOutputProjection = Layers[idx++];

        // Visual encoder: input projection
        _visualInputProjection = Layers[idx++];

        // Visual encoder: (attention + FFN1 + FFN2) × numEncoderLayers
        _visualEncoderLayers = new List<ILayer<T>>();
        for (int i = 0; i < _numEncoderLayers * 3; i++)
            _visualEncoderLayers.Add(Layers[idx++]);

        // Visual output projection
        _visualOutputProjection = Layers[idx++];

        // Cross-modal attention (2 layers)
        _crossModalAttentionLayers = new List<ILayer<T>>();
        for (int i = 0; i < 2; i++)
            _crossModalAttentionLayers.Add(Layers[idx++]);

        // Task heads
        _localizationHead = Layers[idx++];
        _syncHead = Layers[idx++];
        _sceneClassificationHead = Layers[idx++];
        _separationMaskPredictor = Layers[idx++];
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        var embedding = GetAudioEmbedding(input, _audioSampleRate);
        return Tensor<T>.FromVector(embedding);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);

        try
        {
            // Forward pass: compute audio embedding from input
            var prediction = Predict(input);

            // Calculate loss
            var predictionVec = prediction.ToVector();
            var expectedVec = expectedOutput.ToVector();
            var loss = _lossFunction.CalculateLoss(predictionVec, expectedVec);
            LastLoss = loss;

            // Backward pass: compute gradients
            var outputGradient = _lossFunction.CalculateDerivative(predictionVec, expectedVec);

            // Convert gradient to tensor and backpropagate through audio encoder
            var gradientTensor = new Tensor<T>(prediction.Shape, outputGradient);
            BackpropagateAudioEncoder(gradientTensor);

            // Update parameters using the optimizer
            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (gradients.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Gradient vector length ({gradients.Length}) must match parameter count ({ParameterCount}).",
                nameof(gradients));
        }

        // Get current parameters
        var currentParams = GetParameters();

        // Apply gradient descent update: params = params - learning_rate * gradients
        T learningRate = NumOps.FromDouble(0.001); // Default learning rate
        for (int i = 0; i < currentParams.Length; i++)
        {
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        // Set the updated parameters
        SetParameters(currentParams);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "AudioVisualCorrespondenceNetwork",
            ModelType = ModelType.NeuralNetwork,
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
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embeddingDimension);
        writer.Write(_audioSampleRate);
        writer.Write(_videoFrameRate);
        writer.Write(_numEncoderLayers);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read serialized values
        var embDim = reader.ReadInt32();
        var sampleRate = reader.ReadInt32();
        var frameRate = reader.ReadDouble();
        var numLayers = reader.ReadInt32();

        // Validate that loaded values match current instance configuration
        if (embDim != _embeddingDimension)
        {
            throw new InvalidOperationException(
                $"Loaded embedding dimension ({embDim}) doesn't match current ({_embeddingDimension}).");
        }

        if (sampleRate != _audioSampleRate)
        {
            throw new InvalidOperationException(
                $"Loaded audio sample rate ({sampleRate}) doesn't match current ({_audioSampleRate}).");
        }

        if (Math.Abs(frameRate - _videoFrameRate) > 0.001)
        {
            throw new InvalidOperationException(
                $"Loaded video frame rate ({frameRate}) doesn't match current ({_videoFrameRate}).");
        }

        if (numLayers != _numEncoderLayers)
        {
            throw new InvalidOperationException(
                $"Loaded encoder layers ({numLayers}) doesn't match current ({_numEncoderLayers}).");
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new AudioVisualCorrespondenceNetwork<T>(
            Architecture,
            _embeddingDimension,
            _audioSampleRate,
            _videoFrameRate,
            _numEncoderLayers);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        void AddLayerParams(ILayer<T> layer)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++)
            {
                allParams.Add(p[i]);
            }
        }

        AddLayerParams(AudioInputProjection);
        foreach (var layer in AudioEncoderLayers) AddLayerParams(layer);
        AddLayerParams(AudioOutputProjection);

        AddLayerParams(VisualInputProjection);
        foreach (var layer in VisualEncoderLayers) AddLayerParams(layer);
        AddLayerParams(VisualOutputProjection);

        foreach (var layer in CrossModalAttentionLayers) AddLayerParams(layer);
        AddLayerParams(LocalizationHead);
        AddLayerParams(SyncHead);
        AddLayerParams(SceneClassificationHead);
        AddLayerParams(SeparationMaskPredictor);

        var result = new Vector<T>(allParams.Count);
        for (int i = 0; i < allParams.Count; i++)
        {
            result[i] = allParams[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var offset = 0;

        void SetLayerParams(ILayer<T> layer)
        {
            var count = layer.ParameterCount;
            var p = new Vector<T>(count);
            for (int i = 0; i < count; i++)
            {
                if (offset + i < parameters.Length)
                {
                    p[i] = parameters[offset + i];
                }
            }
            layer.SetParameters(p);
            offset += count;
        }

        SetLayerParams(AudioInputProjection);
        foreach (var layer in AudioEncoderLayers) SetLayerParams(layer);
        SetLayerParams(AudioOutputProjection);

        SetLayerParams(VisualInputProjection);
        foreach (var layer in VisualEncoderLayers) SetLayerParams(layer);
        SetLayerParams(VisualOutputProjection);

        foreach (var layer in CrossModalAttentionLayers) SetLayerParams(layer);
        SetLayerParams(LocalizationHead);
        SetLayerParams(SyncHead);
        SetLayerParams(SceneClassificationHead);
        SetLayerParams(SeparationMaskPredictor);
    }

    /// <inheritdoc/>
    public override int ParameterCount
    {
        get
        {
            var count = 0;

            count += AudioInputProjection.ParameterCount;
            foreach (var layer in AudioEncoderLayers) count += layer.ParameterCount;
            count += AudioOutputProjection.ParameterCount;

            count += VisualInputProjection.ParameterCount;
            foreach (var layer in VisualEncoderLayers) count += layer.ParameterCount;
            count += VisualOutputProjection.ParameterCount;

            foreach (var layer in CrossModalAttentionLayers) count += layer.ParameterCount;
            count += LocalizationHead.ParameterCount;
            count += SyncHead.ParameterCount;
            count += SceneClassificationHead.ParameterCount;
            count += SeparationMaskPredictor.ParameterCount;

            return count;
        }
    }

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        // Create a new instance without passing optimizer/loss to get fresh instances
        // Passing the same optimizer would share mutable state (momentum, etc.)
        var copy = new AudioVisualCorrespondenceNetwork<T>(
            Architecture,
            _embeddingDimension,
            _audioSampleRate,
            _videoFrameRate,
            _numEncoderLayers);

        copy.SetParameters(GetParameters());
        return copy;
    }

    #endregion
}
