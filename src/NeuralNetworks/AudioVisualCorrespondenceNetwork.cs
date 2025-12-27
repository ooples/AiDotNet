using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
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
    private readonly List<ILayer<T>> _audioEncoderLayers;
    private readonly ILayer<T> _audioInputProjection;
    private readonly ILayer<T> _audioOutputProjection;
    private Matrix<T>? _audioPositionalEmbedding;

    // Visual encoder components
    private readonly List<ILayer<T>> _visualEncoderLayers;
    private readonly ILayer<T> _visualInputProjection;
    private readonly ILayer<T> _visualOutputProjection;
    private Matrix<T>? _visualPositionalEmbedding;

    // Cross-modal attention for localization
    private readonly List<ILayer<T>> _crossModalAttentionLayers;
    private readonly ILayer<T> _localizationHead;

    // Synchronization head
    private readonly ILayer<T> _syncHead;

    // Scene classification head
    private readonly ILayer<T> _sceneClassificationHead;

    // Separation network components
    private readonly ILayer<T> _separationMaskPredictor;

    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly Random _randomGenerator;

    // Scene labels for classification
    private readonly List<string> _sceneLabels;

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
    /// <param name="optimizer">Optimizer for training.</param>
    /// <param name="lossFunction">Loss function for training.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public AudioVisualCorrespondenceNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int embeddingDimension = DEFAULT_EMBEDDING_DIM,
        int audioSampleRate = DEFAULT_SAMPLE_RATE,
        double videoFrameRate = DEFAULT_FRAME_RATE,
        int numEncoderLayers = 6,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(architecture, lossFunction ?? new ContrastiveLoss<T>(), 1.0)
    {
        _embeddingDimension = embeddingDimension;
        _audioSampleRate = audioSampleRate;
        _videoFrameRate = videoFrameRate;
        _numEncoderLayers = numEncoderLayers;
        _hiddenDim = embeddingDimension * 4;

        _randomGenerator = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSeededRandom(Environment.TickCount);

        _optimizer = optimizer ?? new Optimizers.AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new ContrastiveLoss<T>();

        _sceneLabels = new List<string>();

        IActivationFunction<T>? nullActivation = null;
        var geluActivation = new GELUActivation<T>() as IActivationFunction<T>;

        // Initialize audio encoder
        _audioEncoderLayers = new List<ILayer<T>>();
        _audioInputProjection = new DenseLayer<T>(SPECTROGRAM_BINS, _embeddingDimension, nullActivation);

        for (int i = 0; i < numEncoderLayers; i++)
        {
            _audioEncoderLayers.Add(new MultiHeadAttentionLayer<T>(
                1,
                _embeddingDimension,
                NUM_ATTENTION_HEADS,
                geluActivation));
            _audioEncoderLayers.Add(new DenseLayer<T>(_embeddingDimension, _hiddenDim, geluActivation));
            _audioEncoderLayers.Add(new DenseLayer<T>(_hiddenDim, _embeddingDimension, nullActivation));
        }

        _audioOutputProjection = new DenseLayer<T>(_embeddingDimension, _embeddingDimension, nullActivation);

        // Initialize visual encoder
        _visualEncoderLayers = new List<ILayer<T>>();
        _visualInputProjection = new DenseLayer<T>(768, _embeddingDimension, nullActivation);

        for (int i = 0; i < numEncoderLayers; i++)
        {
            _visualEncoderLayers.Add(new MultiHeadAttentionLayer<T>(
                1,
                _embeddingDimension,
                NUM_ATTENTION_HEADS,
                geluActivation));
            _visualEncoderLayers.Add(new DenseLayer<T>(_embeddingDimension, _hiddenDim, geluActivation));
            _visualEncoderLayers.Add(new DenseLayer<T>(_hiddenDim, _embeddingDimension, nullActivation));
        }

        _visualOutputProjection = new DenseLayer<T>(_embeddingDimension, _embeddingDimension, nullActivation);

        // Initialize cross-modal attention
        _crossModalAttentionLayers = new List<ILayer<T>>();
        for (int i = 0; i < 2; i++)
        {
            _crossModalAttentionLayers.Add(new MultiHeadAttentionLayer<T>(
                1,
                _embeddingDimension,
                NUM_ATTENTION_HEADS,
                geluActivation));
        }

        // Localization head outputs spatial attention map
        _localizationHead = new DenseLayer<T>(_embeddingDimension, 1, nullActivation);

        // Sync head predicts time offset
        _syncHead = new DenseLayer<T>(_embeddingDimension * 2, 1, nullActivation);

        // Scene classification head
        _sceneClassificationHead = new DenseLayer<T>(_embeddingDimension * 2, 256, nullActivation);

        // Separation mask predictor
        _separationMaskPredictor = new DenseLayer<T>(_embeddingDimension * 2, SPECTROGRAM_BINS, nullActivation);

        InitializePositionalEmbeddings();
        InitializeLayers();
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

        var syncOutput = _syncHead.Forward(combinedTensor);
        var offsetValue = NumOps.ToDouble(syncOutput.Data[0]);

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
        var maskOutput = _separationMaskPredictor.Forward(combinedTensor);

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

        var classFeatures = _sceneClassificationHead.Forward(combinedTensor);

        var labelList = sceneLabels.ToList();
        var results = new Dictionary<string, T>();
        var softmaxDenominator = NumOps.Zero;

        // Compute logits for each label
        var logits = new List<T>();
        for (int i = 0; i < labelList.Count; i++)
        {
            var logit = i < classFeatures.Length ? classFeatures.Data[i] : NumOps.Zero;
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
            throw new ArgumentException("Audio and visual samples must have matching counts.");
        }

        SetTrainingMode(true);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            var totalLoss = NumOps.Zero;

            for (int i = 0; i < audioList.Count; i++)
            {
                var audioEmb = GetAudioEmbedding(audioList[i], _audioSampleRate);
                var visualEmb = GetVisualEmbedding(visualList[i]);

                // Compute contrastive loss
                var similarity = ComputeCosineSimilarity(audioEmb, visualEmb);
                var target = NumOps.FromDouble(1.0);
                var loss = ComputeContrastiveLoss(similarity, target);

                totalLoss = NumOps.Add(totalLoss, loss);

                // Backward pass would go here in full implementation
            }
        }

        SetTrainingMode(false);
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
                var value = NumOps.ToDouble(waveform.Data[sampleIdx]);
                spectrogram.Data[frame * SPECTROGRAM_BINS + bin] =
                    NumOps.FromDouble(Math.Log(Math.Abs(value * freq) + 1e-8));
            }
        }

        return spectrogram;
    }

    private Tensor<T> ApplyAudioEncoder(Tensor<T> spectrogram)
    {
        var projected = _audioInputProjection.Forward(spectrogram);

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
                        projected.Data[idx] = NumOps.Add(
                            projected.Data[idx],
                            _audioPositionalEmbedding[i, j]);
                    }
                }
            }
        }

        var current = projected;
        for (int i = 0; i < _audioEncoderLayers.Count; i += 3)
        {
            var attnOutput = _audioEncoderLayers[i].Forward(current);
            var ffn1 = _audioEncoderLayers[i + 1].Forward(attnOutput);
            var ffn2 = _audioEncoderLayers[i + 2].Forward(ApplyGelu(ffn1));
            current = AddResidual(attnOutput, ffn2);
        }

        return _audioOutputProjection.Forward(current);
    }

    private Tensor<T> ApplyVisualEncoder(Tensor<T> frame)
    {
        // Flatten spatial dimensions for transformer
        var flattened = FlattenToPatches(frame);
        var projected = _visualInputProjection.Forward(flattened);

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
                        projected.Data[idx] = NumOps.Add(
                            projected.Data[idx],
                            _visualPositionalEmbedding[i, j]);
                    }
                }
            }
        }

        var current = projected;
        for (int i = 0; i < _visualEncoderLayers.Count; i += 3)
        {
            var attnOutput = _visualEncoderLayers[i].Forward(current);
            var ffn1 = _visualEncoderLayers[i + 1].Forward(attnOutput);
            var ffn2 = _visualEncoderLayers[i + 2].Forward(ApplyGelu(ffn1));
            current = AddResidual(attnOutput, ffn2);
        }

        return _visualOutputProjection.Forward(current);
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
        var numPatchesH = height / patchSize;
        var numPatchesW = width / patchSize;
        var numPatches = numPatchesH * numPatchesW;
        var patchDim = channels * patchSize * patchSize;

        var patches = new Tensor<T>([numPatches, 768]);

        for (int ph = 0; ph < numPatchesH; ph++)
        {
            for (int pw = 0; pw < numPatchesW; pw++)
            {
                var patchIdx = ph * numPatchesW + pw;
                var flatIdx = 0;

                for (int c = 0; c < channels && flatIdx < 768; c++)
                {
                    for (int py = 0; py < patchSize && flatIdx < 768; py++)
                    {
                        for (int px = 0; px < patchSize && flatIdx < 768; px++)
                        {
                            var y = ph * patchSize + py;
                            var x = pw * patchSize + px;
                            var srcIdx = c * height * width + y * width + x;

                            if (srcIdx < frame.Length)
                            {
                                patches.Data[patchIdx * 768 + flatIdx] = frame.Data[srcIdx];
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
                    spatialVec[j] = spatialFeatures.Data[idx];
                }
            }

            var score = ComputeCosineSimilarity(audioEmbedding, spatialVec);
            attentionScores.Data[pos] = score;
        }

        // Softmax normalization
        var maxScore = NumOps.Zero;
        for (int i = 0; i < numPositions; i++)
        {
            if (NumOps.ToDouble(attentionScores.Data[i]) > NumOps.ToDouble(maxScore))
            {
                maxScore = attentionScores.Data[i];
            }
        }

        var sumExp = NumOps.Zero;
        for (int i = 0; i < numPositions; i++)
        {
            var shifted = NumOps.Subtract(attentionScores.Data[i], maxScore);
            attentionScores.Data[i] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(shifted)));
            sumExp = NumOps.Add(sumExp, attentionScores.Data[i]);
        }

        for (int i = 0; i < numPositions; i++)
        {
            attentionScores.Data[i] = NumOps.Divide(attentionScores.Data[i], sumExp);
        }

        // Reshape to spatial dimensions
        var height = originalShape.Length >= 2 ? originalShape[^2] : 1;
        var width = originalShape.Length >= 1 ? originalShape[^1] : numPositions;
        var patchH = (int)Math.Sqrt(numPositions);
        var patchW = patchH;

        var attentionMap = new Tensor<T>([patchH, patchW]);
        for (int i = 0; i < Math.Min(numPositions, patchH * patchW); i++)
        {
            attentionMap.Data[i] = attentionScores.Data[i];
        }

        return attentionMap;
    }

    private Vector<T> GlobalAveragePool(Tensor<T> features)
    {
        var embedding = new Vector<T>(_embeddingDimension);
        var seqLen = features.Shape[0];

        for (int j = 0; j < _embeddingDimension; j++)
        {
            var sum = NumOps.Zero;
            for (int i = 0; i < seqLen; i++)
            {
                var idx = i * _embeddingDimension + j;
                if (idx < features.Length)
                {
                    sum = NumOps.Add(sum, features.Data[idx]);
                }
            }
            embedding[j] = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
        }

        return embedding;
    }

    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        var magnitude = ComputeVectorMagnitude(vector);
        var magValue = NumOps.ToDouble(magnitude);

        if (magValue < 1e-8)
        {
            return vector;
        }

        var normalized = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = NumOps.Divide(vector[i], magnitude);
        }

        return normalized;
    }

    private T ComputeVectorMagnitude(Vector<T> vector)
    {
        var sumSq = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(vector[i], vector[i]));
        }

        return NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sumSq)));
    }

    private T ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            return NumOps.Zero;
        }

        var dot = NumOps.Zero;
        var magA = NumOps.Zero;
        var magB = NumOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            dot = NumOps.Add(dot, NumOps.Multiply(a[i], b[i]));
            magA = NumOps.Add(magA, NumOps.Multiply(a[i], a[i]));
            magB = NumOps.Add(magB, NumOps.Multiply(b[i], b[i]));
        }

        var magProduct = NumOps.Multiply(
            NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(magA))),
            NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(magB))));

        if (NumOps.ToDouble(magProduct) < 1e-8)
        {
            return NumOps.Zero;
        }

        return NumOps.Divide(dot, magProduct);
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
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            var x = NumOps.ToDouble(input.Data[i]);
            var gelu = 0.5 * x * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
            output.Data[i] = NumOps.FromDouble(gelu);
        }

        return output;
    }

    private Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            var x = NumOps.ToDouble(input.Data[i]);
            output.Data[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-x)));
        }

        return output;
    }

    private Tensor<T> AddResidual(Tensor<T> residual, Tensor<T> output)
    {
        var result = new Tensor<T>(output.Shape);
        var minLength = Math.Min(residual.Length, output.Length);

        for (int i = 0; i < minLength; i++)
        {
            result.Data[i] = NumOps.Add(residual.Data[i], output.Data[i]);
        }

        return result;
    }

    private Tensor<T> ApplyMask(Tensor<T> spectrogram, Tensor<T> mask)
    {
        var result = new Tensor<T>(spectrogram.Shape);
        var numFrames = spectrogram.Shape[0];
        var numBins = spectrogram.Shape[1];

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int bin = 0; bin < numBins; bin++)
            {
                var specIdx = frame * numBins + bin;
                var maskIdx = bin < mask.Length ? bin : mask.Length - 1;
                result.Data[specIdx] = NumOps.Multiply(spectrogram.Data[specIdx], mask.Data[maskIdx]);
            }
        }

        return result;
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
                    var logMag = NumOps.ToDouble(maskedSpectrogram.Data[specIdx]);
                    var mag = Math.Exp(logMag) - 1e-8;
                    var phase = Math.Sin(2.0 * Math.PI * i / SPECTROGRAM_HOP);
                    output.Data[startSample + i] = NumOps.FromDouble(mag * phase);
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
        // Layers are already initialized in constructor
        // Add all layers to base Layers collection
        Layers.Clear();
        Layers.Add(_audioInputProjection);
        foreach (var layer in _audioEncoderLayers) Layers.Add(layer);
        Layers.Add(_audioOutputProjection);
        Layers.Add(_visualInputProjection);
        foreach (var layer in _visualEncoderLayers) Layers.Add(layer);
        Layers.Add(_visualOutputProjection);
        foreach (var layer in _crossModalAttentionLayers) Layers.Add(layer);
        Layers.Add(_localizationHead);
        Layers.Add(_syncHead);
        Layers.Add(_sceneClassificationHead);
        Layers.Add(_separationMaskPredictor);
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var embedding = GetAudioEmbedding(input, _audioSampleRate);
        return Tensor<T>.FromVector(embedding);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        var prediction = Predict(input);
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;
        SetTrainingMode(false);
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

        AddLayerParams(_audioInputProjection);
        foreach (var layer in _audioEncoderLayers) AddLayerParams(layer);
        AddLayerParams(_audioOutputProjection);

        AddLayerParams(_visualInputProjection);
        foreach (var layer in _visualEncoderLayers) AddLayerParams(layer);
        AddLayerParams(_visualOutputProjection);

        foreach (var layer in _crossModalAttentionLayers) AddLayerParams(layer);
        AddLayerParams(_localizationHead);
        AddLayerParams(_syncHead);
        AddLayerParams(_sceneClassificationHead);
        AddLayerParams(_separationMaskPredictor);

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

        SetLayerParams(_audioInputProjection);
        foreach (var layer in _audioEncoderLayers) SetLayerParams(layer);
        SetLayerParams(_audioOutputProjection);

        SetLayerParams(_visualInputProjection);
        foreach (var layer in _visualEncoderLayers) SetLayerParams(layer);
        SetLayerParams(_visualOutputProjection);

        foreach (var layer in _crossModalAttentionLayers) SetLayerParams(layer);
        SetLayerParams(_localizationHead);
        SetLayerParams(_syncHead);
        SetLayerParams(_sceneClassificationHead);
        SetLayerParams(_separationMaskPredictor);
    }

    /// <inheritdoc/>
    public override int ParameterCount
    {
        get
        {
            var count = 0;

            count += _audioInputProjection.ParameterCount;
            foreach (var layer in _audioEncoderLayers) count += layer.ParameterCount;
            count += _audioOutputProjection.ParameterCount;

            count += _visualInputProjection.ParameterCount;
            foreach (var layer in _visualEncoderLayers) count += layer.ParameterCount;
            count += _visualOutputProjection.ParameterCount;

            foreach (var layer in _crossModalAttentionLayers) count += layer.ParameterCount;
            count += _localizationHead.ParameterCount;
            count += _syncHead.ParameterCount;
            count += _sceneClassificationHead.ParameterCount;
            count += _separationMaskPredictor.ParameterCount;

            return count;
        }
    }

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var copy = new AudioVisualCorrespondenceNetwork<T>(
            Architecture,
            _embeddingDimension,
            _audioSampleRate,
            _videoFrameRate,
            _numEncoderLayers,
            _optimizer,
            _lossFunction);

        copy.SetParameters(GetParameters());
        return copy;
    }

    #endregion
}
