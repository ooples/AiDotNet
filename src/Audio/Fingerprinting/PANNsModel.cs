using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// PANNs (Pretrained Audio Neural Networks) - Large-scale pretrained CNN models for audio pattern recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PANNs are convolutional neural networks pretrained on AudioSet (2 million audio clips, 527 classes).
/// They provide state-of-the-art audio embeddings that can be used for:
/// <list type="bullet">
/// <item><description>Audio tagging (multi-label classification)</description></item>
/// <item><description>Sound event detection (localization in time)</description></item>
/// <item><description>Audio fingerprinting and retrieval</description></item>
/// <item><description>Transfer learning for custom audio tasks</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> PANNs are like ImageNet-pretrained models but for audio!
///
/// Just as image models can recognize cats, dogs, and cars after seeing millions of images,
/// PANNs can recognize 527 different sounds after hearing 2 million audio clips:
/// - Musical instruments (piano, guitar, drums)
/// - Human sounds (speech, laughter, coughing)
/// - Environmental sounds (rain, thunder, traffic)
/// - Animal sounds (dog bark, bird song)
/// - And many more!
///
/// Use cases:
/// - "What sounds are in this audio?" (audio tagging)
/// - "When does the dog bark?" (sound event detection)
/// - "Find similar sounding audio" (audio retrieval)
/// - Build custom sound classifiers with less training data
/// </para>
/// <para>
/// Reference: Kong, Q., et al. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks
/// for Audio Pattern Recognition.
/// </para>
/// </remarks>
public class PANNsModel<T> : AudioNeuralNetworkBase<T>, IAudioFingerprinter<T>
{
    private readonly INumericOperations<T> _numOps;

    // Model configuration (non-readonly for deserialization support)
    private int _numClasses;
    private int _embeddingDim;
    private int _numMelBands;
    private int _windowSize;
    private int _hopSize;

    // CNN architecture type
    private PANNsArchitecture _architectureType;

    // Convolutional layers
    private List<ConvBlock> _convBlocks;

    // Global pooling and classifier
    private T[] _fcWeight;
    private T[] _fcBias;
    private T[] _embeddingWeight;
    private T[] _embeddingBias;

    // AudioSet class labels (subset for common sounds)
    private string[] _classLabels;

    // Optimizer for training
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <inheritdoc/>
    public string Name => $"PANNs-{_architectureType}";

    /// <inheritdoc/>
    public int FingerprintLength => _embeddingDim;

    /// <summary>
    /// Gets the number of output classes (527 for AudioSet).
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => _embeddingDim;

    /// <summary>
    /// Gets the architecture type.
    /// </summary>
    public PANNsArchitecture ArchitectureType => _architectureType;

    /// <summary>
    /// Initializes a new instance of the <see cref="PANNsModel{T}"/> class for ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="sampleRate">Sample rate of input audio (default: 32000 Hz).</param>
    /// <param name="numClasses">Number of output classes (default: 527 for AudioSet).</param>
    /// <param name="embeddingDim">Embedding dimension (default: 2048).</param>
    /// <param name="onnxOptions">Optional ONNX model options.</param>
    /// <exception cref="FileNotFoundException">Thrown when the ONNX model file is not found.</exception>
    public PANNsModel(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        int sampleRate = 32000,
        int numClasses = 527,
        int embeddingDim = 2048,
        OnnxModelOptions? onnxOptions = null)
        : base(architecture)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        }

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        }

        SampleRate = sampleRate;
        _numClasses = numClasses;
        _embeddingDim = embeddingDim;
        _numMelBands = 64;
        _windowSize = 1024;
        _hopSize = 320;
        _architectureType = PANNsArchitecture.Cnn14;

        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath, onnxOptions);

        // Initialize empty arrays (not used in ONNX mode)
        _convBlocks = new List<ConvBlock>();
        _fcWeight = Array.Empty<T>();
        _fcBias = Array.Empty<T>();
        _embeddingWeight = Array.Empty<T>();
        _embeddingBias = Array.Empty<T>();
        _classLabels = GetDefaultClassLabels();

        // Initialize optimizer (not used in ONNX mode but required for readonly field)
        _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PANNsModel{T}"/> class for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="sampleRate">Sample rate of input audio (default: 32000 Hz).</param>
    /// <param name="architectureType">CNN architecture variant (default: Cnn14).</param>
    /// <param name="numClasses">Number of output classes (default: 527 for AudioSet).</param>
    /// <param name="embeddingDim">Embedding dimension (default: 2048).</param>
    /// <param name="numMelBands">Number of mel spectrogram bands (default: 64).</param>
    /// <param name="windowSize">STFT window size (default: 1024).</param>
    /// <param name="hopSize">STFT hop size (default: 320).</param>
    /// <param name="dropout">Dropout rate (default: 0.2).</param>
    /// <param name="optimizer">Optimizer for training. If null, a default Adam optimizer is used.</param>
    /// <param name="lossFunction">Loss function. If null, BCE loss is used for multi-label.</param>
    public PANNsModel(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 32000,
        PANNsArchitecture architectureType = PANNsArchitecture.Cnn14,
        int numClasses = 527,
        int embeddingDim = 2048,
        int numMelBands = 64,
        int windowSize = 1024,
        int hopSize = 320,
        double dropout = 0.2,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        SampleRate = sampleRate;
        _architectureType = architectureType;
        _numClasses = numClasses;
        _embeddingDim = embeddingDim;
        _numMelBands = numMelBands;
        _windowSize = windowSize;
        _hopSize = hopSize;

        // Initialize CNN blocks based on architecture
        _convBlocks = CreateConvBlocks(architectureType, numMelBands);

        // Get the output channels from last conv block
        int lastChannels = _convBlocks.Count > 0 ? _convBlocks[^1].OutChannels : 2048;

        // Initialize fully connected layers
        _embeddingWeight = InitializeWeights(embeddingDim * lastChannels);
        _embeddingBias = InitializeWeights(embeddingDim, 0.0);

        _fcWeight = InitializeWeights(numClasses * embeddingDim);
        _fcBias = InitializeWeights(numClasses, 0.0);

        _classLabels = GetDefaultClassLabels();

        // Initialize optimizer (Adam by default)
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates convolutional blocks based on architecture type.
    /// </summary>
    private List<ConvBlock> CreateConvBlocks(PANNsArchitecture arch, int numMelBands)
    {
        var blocks = new List<ConvBlock>();

        switch (arch)
        {
            case PANNsArchitecture.Cnn6:
                blocks.Add(new ConvBlock(_numOps, 1, 64, 5, 2, true));
                blocks.Add(new ConvBlock(_numOps, 64, 128, 5, 2, true));
                blocks.Add(new ConvBlock(_numOps, 128, 256, 5, 2, true));
                blocks.Add(new ConvBlock(_numOps, 256, 512, 5, 2, true));
                break;

            case PANNsArchitecture.Cnn10:
                blocks.Add(new ConvBlock(_numOps, 1, 64, 3, 2, false));
                blocks.Add(new ConvBlock(_numOps, 64, 64, 3, 1, true));
                blocks.Add(new ConvBlock(_numOps, 64, 128, 3, 2, false));
                blocks.Add(new ConvBlock(_numOps, 128, 128, 3, 1, true));
                blocks.Add(new ConvBlock(_numOps, 128, 256, 3, 2, false));
                blocks.Add(new ConvBlock(_numOps, 256, 256, 3, 1, true));
                blocks.Add(new ConvBlock(_numOps, 256, 512, 3, 2, false));
                blocks.Add(new ConvBlock(_numOps, 512, 512, 3, 1, true));
                break;

            case PANNsArchitecture.Cnn14:
            default:
                blocks.Add(new ConvBlock(_numOps, 1, 64, 3, 2, false));
                blocks.Add(new ConvBlock(_numOps, 64, 64, 3, 1, false));
                blocks.Add(new ConvBlock(_numOps, 64, 64, 3, 1, true));
                blocks.Add(new ConvBlock(_numOps, 64, 128, 3, 2, false));
                blocks.Add(new ConvBlock(_numOps, 128, 128, 3, 1, false));
                blocks.Add(new ConvBlock(_numOps, 128, 128, 3, 1, true));
                blocks.Add(new ConvBlock(_numOps, 128, 256, 3, 2, false));
                blocks.Add(new ConvBlock(_numOps, 256, 256, 3, 1, false));
                blocks.Add(new ConvBlock(_numOps, 256, 256, 3, 1, true));
                blocks.Add(new ConvBlock(_numOps, 256, 512, 3, 2, false));
                blocks.Add(new ConvBlock(_numOps, 512, 512, 3, 1, false));
                blocks.Add(new ConvBlock(_numOps, 512, 512, 3, 1, true));
                blocks.Add(new ConvBlock(_numOps, 512, 1024, 3, 2, false));
                blocks.Add(new ConvBlock(_numOps, 1024, 2048, 3, 1, true));
                break;
        }

        return blocks;
    }

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        // Layers are handled manually for PANNs' CNN architecture
    }

    /// <summary>
    /// Preprocesses raw audio waveform for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Convert to log mel spectrogram
        int numSamples = rawAudio.Shape[^1];
        int numFrames = Math.Max(1, (numSamples - _windowSize) / _hopSize + 1);

        var melSpec = new T[numFrames * _numMelBands];
        var frameData = new double[_windowSize];

        // Precompute mel filterbank
        var melFilterbank = ComputeMelFilterbank(_windowSize, _numMelBands, SampleRate);

        // Compute mel spectrogram
        for (int f = 0; f < numFrames; f++)
        {
            int start = f * _hopSize;

            // Extract frame and apply Hann window
            Array.Clear(frameData, 0, frameData.Length);
            for (int i = 0; i < _windowSize; i++)
            {
                int idx = start + i;
                if (idx < numSamples && idx < rawAudio.Length)
                {
                    double sample = _numOps.ToDouble(rawAudio.Data.Span[idx]);
                    // Hann window
                    double window = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (_windowSize - 1)));
                    frameData[i] = sample * window;
                }
            }

            // Compute FFT using FftSharp (proper complex FFT)
            var spectrum = FftSharp.FFT.Forward(frameData);

            // Convert to power spectrum (magnitude squared)
            var powerSpectrum = new double[_windowSize / 2 + 1];
            for (int k = 0; k < powerSpectrum.Length; k++)
            {
                powerSpectrum[k] = spectrum[k].Magnitude * spectrum[k].Magnitude;
            }

            // Apply mel filterbank
            for (int m = 0; m < _numMelBands; m++)
            {
                double melSum = 0;
                for (int k = 0; k < powerSpectrum.Length; k++)
                {
                    melSum += powerSpectrum[k] * melFilterbank[m, k];
                }

                // Log mel spectrogram
                melSpec[f * _numMelBands + m] = _numOps.FromDouble(Math.Log(Math.Max(melSum, 1e-10)));
            }
        }

        // Reshape to [batch, channels, frames, mels]
        return new Tensor<T>(melSpec, new[] { 1, 1, numFrames, _numMelBands });
    }

    /// <summary>
    /// Computes mel filterbank matrix.
    /// </summary>
    private static double[,] ComputeMelFilterbank(int windowSize, int numMelBands, int sampleRate)
    {
        int numBins = windowSize / 2 + 1;
        var filterbank = new double[numMelBands, numBins];

        double melMin = HzToMel(0);
        double melMax = HzToMel(sampleRate / 2.0);

        // Create mel-spaced center frequencies
        var melPoints = new double[numMelBands + 2];
        for (int i = 0; i < melPoints.Length; i++)
        {
            melPoints[i] = melMin + (melMax - melMin) * i / (numMelBands + 1);
        }

        // Convert mel points to Hz and then to bin indices
        var binPoints = new int[numMelBands + 2];
        for (int i = 0; i < melPoints.Length; i++)
        {
            double hz = MelToHz(melPoints[i]);
            binPoints[i] = (int)Math.Floor((windowSize + 1) * hz / sampleRate);
        }

        // Create triangular filters
        for (int m = 0; m < numMelBands; m++)
        {
            int startBin = binPoints[m];
            int centerBin = binPoints[m + 1];
            int endBin = binPoints[m + 2];

            // Rising slope
            for (int k = startBin; k < centerBin && k < numBins; k++)
            {
                if (centerBin != startBin)
                {
                    filterbank[m, k] = (double)(k - startBin) / (centerBin - startBin);
                }
            }

            // Falling slope
            for (int k = centerBin; k < endBin && k < numBins; k++)
            {
                if (endBin != centerBin)
                {
                    filterbank[m, k] = (double)(endBin - k) / (endBin - centerBin);
                }
            }
        }

        return filterbank;
    }

    /// <summary>
    /// Converts mel frequency to Hz.
    /// </summary>
    private static double MelToHz(double mel)
    {
        return 700.0 * (Math.Exp(mel / 1127.0) - 1);
    }

    /// <summary>
    /// Postprocesses model output.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return modelOutput;
    }

    /// <summary>
    /// Extracts audio embedding.
    /// </summary>
    /// <param name="audio">Audio tensor [samples] or [batch, samples].</param>
    /// <returns>Audio embedding [batch, embeddingDim].</returns>
    public Tensor<T> ExtractEmbedding(Tensor<T> audio)
    {
        var preprocessed = PreprocessAudio(audio);

        if (IsOnnxMode && OnnxModel is not null)
        {
            // ONNX model should return embedding
            return OnnxModel.Run(preprocessed);
        }

        return ExtractEmbeddingNative(preprocessed);
    }

    /// <summary>
    /// Native embedding extraction.
    /// </summary>
    private Tensor<T> ExtractEmbeddingNative(Tensor<T> melSpec)
    {
        var current = melSpec;

        // Apply convolutional blocks
        foreach (var block in _convBlocks)
        {
            current = block.Forward(current);
        }

        // Global average pooling
        int batchSize = current.Shape[0];
        int channels = current.Shape[1];
        int height = current.Shape.Length > 2 ? current.Shape[2] : 1;
        int width = current.Shape.Length > 3 ? current.Shape[3] : 1;

        var pooled = new T[batchSize * channels];
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                double sum = 0;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int idx = b * channels * height * width + c * height * width + h * width + w;
                        if (idx < current.Length)
                        {
                            sum += _numOps.ToDouble(current.Data.Span[idx]);
                        }
                    }
                }
                pooled[b * channels + c] = _numOps.FromDouble(sum / (height * width));
            }
        }

        // Project to embedding space
        var embedding = new T[batchSize * _embeddingDim];
        for (int b = 0; b < batchSize; b++)
        {
            for (int e = 0; e < _embeddingDim; e++)
            {
                T sum = e < _embeddingBias.Length ? _embeddingBias[e] : _numOps.Zero;
                for (int c = 0; c < channels; c++)
                {
                    int wIdx = e * channels + c;
                    if (wIdx < _embeddingWeight.Length)
                    {
                        sum = _numOps.Add(sum, _numOps.Multiply(
                            pooled[b * channels + c],
                            _embeddingWeight[wIdx]));
                    }
                }
                embedding[b * _embeddingDim + e] = sum;
            }
        }

        return new Tensor<T>(embedding, new[] { batchSize, _embeddingDim });
    }

    /// <summary>
    /// Classifies audio into AudioSet categories.
    /// </summary>
    /// <param name="audio">Audio tensor to classify.</param>
    /// <param name="threshold">Probability threshold for positive labels (default: 0.5).</param>
    /// <returns>Dictionary of label to probability for labels above threshold.</returns>
    public Dictionary<string, double> Classify(Tensor<T> audio, double threshold = 0.5)
    {
        var embedding = ExtractEmbedding(audio);
        var logits = ComputeLogits(embedding);

        // Apply sigmoid for multi-label classification
        var results = new Dictionary<string, double>();
        for (int i = 0; i < _numClasses && i < logits.Length; i++)
        {
            double prob = Sigmoid(_numOps.ToDouble(logits.Data.Span[i]));
            if (prob >= threshold && i < _classLabels.Length)
            {
                results[_classLabels[i]] = prob;
            }
        }

        return results.OrderByDescending(kv => kv.Value)
                      .ToDictionary(kv => kv.Key, kv => kv.Value);
    }

    /// <summary>
    /// Gets top-k predictions.
    /// </summary>
    /// <param name="audio">Audio tensor to classify.</param>
    /// <param name="k">Number of top predictions to return.</param>
    /// <returns>Top-k predictions with probabilities.</returns>
    public List<(string Label, double Probability)> GetTopK(Tensor<T> audio, int k = 5)
    {
        var embedding = ExtractEmbedding(audio);
        var logits = ComputeLogits(embedding);

        var predictions = new List<(string Label, double Probability)>();
        for (int i = 0; i < _numClasses && i < logits.Length; i++)
        {
            double prob = Sigmoid(_numOps.ToDouble(logits.Data.Span[i]));
            string label = i < _classLabels.Length ? _classLabels[i] : $"class_{i}";
            predictions.Add((label, prob));
        }

        return predictions.OrderByDescending(p => p.Probability)
                          .Take(k)
                          .ToList();
    }

    /// <summary>
    /// Computes classification logits from embedding.
    /// </summary>
    private Tensor<T> ComputeLogits(Tensor<T> embedding)
    {
        int batchSize = embedding.Shape[0];
        var logits = new T[batchSize * _numClasses];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                T sum = c < _fcBias.Length ? _fcBias[c] : _numOps.Zero;
                for (int e = 0; e < _embeddingDim; e++)
                {
                    int embIdx = b * _embeddingDim + e;
                    int wIdx = c * _embeddingDim + e;
                    if (embIdx < embedding.Length && wIdx < _fcWeight.Length)
                    {
                        sum = _numOps.Add(sum, _numOps.Multiply(
                            embedding.Data.Span[embIdx],
                            _fcWeight[wIdx]));
                    }
                }
                logits[b * _numClasses + c] = sum;
            }
        }

        return new Tensor<T>(logits, new[] { batchSize, _numClasses });
    }

    /// <summary>
    /// Predicts audio class logits.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var embedding = ExtractEmbedding(input);
        return ComputeLogits(embedding);
    }

    #region IAudioFingerprinter Implementation

    /// <inheritdoc/>
    public AudioFingerprint<T> Fingerprint(Tensor<T> audio)
    {
        var embedding = ExtractEmbedding(audio);
        double duration = (double)audio.Shape[^1] / SampleRate;

        return new AudioFingerprint<T>
        {
            Data = embedding.Data.ToArray(),
            Duration = duration,
            SampleRate = SampleRate,
            Algorithm = Name,
            FrameCount = 1,
            Metadata = new Dictionary<string, object>
            {
                { "EmbeddingDim", _embeddingDim },
                { "Architecture", _architectureType.ToString() },
                { "NumClasses", _numClasses }
            }
        };
    }

    /// <inheritdoc/>
    public AudioFingerprint<T> Fingerprint(Vector<T> audio)
    {
        var tensor = new Tensor<T>(audio.ToArray(), new[] { audio.Length });
        return Fingerprint(tensor);
    }

    /// <inheritdoc/>
    public double ComputeSimilarity(AudioFingerprint<T> fp1, AudioFingerprint<T> fp2)
    {
        // Cosine similarity
        double dot = 0, norm1 = 0, norm2 = 0;
        int len = Math.Min(fp1.Data.Length, fp2.Data.Length);

        for (int i = 0; i < len; i++)
        {
            double v1 = _numOps.ToDouble(fp1.Data[i]);
            double v2 = _numOps.ToDouble(fp2.Data[i]);
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        if (norm1 < 1e-10 || norm2 < 1e-10) return 0;
        return dot / (Math.Sqrt(norm1) * Math.Sqrt(norm2));
    }

    /// <inheritdoc/>
    public IReadOnlyList<FingerprintMatch> FindMatches(
        AudioFingerprint<T> query,
        AudioFingerprint<T> reference,
        int minMatchLength = 10)
    {
        double similarity = ComputeSimilarity(query, reference);

        if (similarity > 0.7)
        {
            return new List<FingerprintMatch>
            {
                new FingerprintMatch
                {
                    QueryStartTime = 0,
                    ReferenceStartTime = 0,
                    Duration = Math.Min(query.Duration, reference.Duration),
                    Confidence = similarity,
                    MatchCount = 1
                }
            };
        }

        return new List<FingerprintMatch>();
    }

    #endregion

    #region Training

    /// <summary>
    /// Trains the model on audio-label pairs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Note:</b> Full training from scratch is not yet implemented. PANNs models
    /// are designed to be used as pre-trained feature extractors. For best results:
    /// - Use ONNX mode with pre-trained weights for inference
    /// - Fine-tune only the final classification layers if needed
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
        {
            throw new InvalidOperationException("Cannot train in ONNX inference mode.");
        }

        // Full training from scratch is not yet implemented.
        // PANNs models require:
        // 1. Cached activations at each layer for proper backpropagation
        // 2. CNN layer gradient computation (not just FC layers)
        // 3. Batch normalization gradient handling
        // 4. Learning rate scheduling and data augmentation
        //
        // For transfer learning, use the pre-trained ONNX model and
        // fine-tune only the classification head with a separate classifier.
        throw new NotImplementedException(
            "Full PANNs training from scratch is not yet implemented. " +
            "Use ONNX mode with pre-trained weights for inference, or " +
            "implement transfer learning by fine-tuning the classification head.");
    }

    private T ComputeBCELoss(Tensor<T> predicted, Tensor<T> target)
    {
        double totalLoss = 0;
        int len = Math.Min(predicted.Length, target.Length);

        for (int i = 0; i < len; i++)
        {
            double p = Sigmoid(_numOps.ToDouble(predicted.Data.Span[i]));
            double t = _numOps.ToDouble(target.Data.Span[i]);

            // Binary cross-entropy
            p = Math.Max(1e-7, Math.Min(1 - 1e-7, p));
            totalLoss -= t * Math.Log(p) + (1 - t) * Math.Log(1 - p);
        }

        return _numOps.FromDouble(totalLoss / len);
    }

    private void UpdateWeights(Tensor<T> predicted, Tensor<T> target)
    {
        // Compute output gradients (dL/dlogits for BCE loss with sigmoid)
        int numClasses = Math.Min(predicted.Length, target.Length);
        var outputGrad = new double[numClasses];
        for (int i = 0; i < numClasses; i++)
        {
            double p = Sigmoid(_numOps.ToDouble(predicted.Data.Span[i]));
            double t = _numOps.ToDouble(target.Data.Span[i]);
            outputGrad[i] = (p - t) / numClasses; // dL/dz = sigmoid(z) - target
        }

        // Compute gradients for FC layer (output -> embedding)
        // FC: output = embedding @ fcWeight.T + fcBias
        // dL/d_fcWeight[e, c] = dL/d_output[c] * embedding[e]
        // dL/d_fcBias[c] = dL/d_output[c]
        // dL/d_embedding[e] = sum over c: dL/d_output[c] * fcWeight[e, c]
        int embeddingDim = _embeddingDim;
        var fcWeightGrad = new double[_fcWeight.Length];
        var fcBiasGrad = new double[_fcBias.Length];
        var embeddingGrad = new double[embeddingDim];

        // Get current embedding (we need to cache this during forward pass in production)
        // For now, use zero gradients for embedding layer backprop
        for (int c = 0; c < numClasses && c < _fcBias.Length; c++)
        {
            fcBiasGrad[c] = outputGrad[c];
            for (int e = 0; e < embeddingDim; e++)
            {
                int wIdx = c * embeddingDim + e;
                if (wIdx < fcWeightGrad.Length)
                {
                    // Simplified: assume unit embedding for gradient computation
                    fcWeightGrad[wIdx] = outputGrad[c];
                }
            }
        }

        // Compute gradients for embedding layer
        var embeddingWeightGrad = new double[_embeddingWeight.Length];
        var embeddingBiasGrad = new double[_embeddingBias.Length];
        for (int e = 0; e < embeddingDim && e < _embeddingBias.Length; e++)
        {
            // Propagate gradient from FC layer
            double grad = 0;
            for (int c = 0; c < numClasses && c < _numClasses; c++)
            {
                int wIdx = c * embeddingDim + e;
                if (wIdx < _fcWeight.Length)
                {
                    grad += outputGrad[c] * _numOps.ToDouble(_fcWeight[wIdx]);
                }
            }
            embeddingBiasGrad[e] = grad;
        }

        // Use optimizer to update FC and embedding layer weights
        // Collect all gradients into a single vector for optimizer
        var allGradients = new List<T>();
        foreach (var g in fcWeightGrad) allGradients.Add(_numOps.FromDouble(g));
        foreach (var g in fcBiasGrad) allGradients.Add(_numOps.FromDouble(g));
        foreach (var g in embeddingWeightGrad) allGradients.Add(_numOps.FromDouble(g));
        foreach (var g in embeddingBiasGrad) allGradients.Add(_numOps.FromDouble(g));

        var gradientVector = new Vector<T>(allGradients.ToArray());

        // Collect current parameters
        var allParams = new List<T>();
        allParams.AddRange(_fcWeight);
        allParams.AddRange(_fcBias);
        allParams.AddRange(_embeddingWeight);
        allParams.AddRange(_embeddingBias);

        var paramVector = new Vector<T>(allParams.ToArray());

        // Use optimizer to compute updated parameters
        var updatedParams = _optimizer.UpdateParameters(paramVector, gradientVector);

        // Distribute updated parameters back to weight arrays
        int idx = 0;
        for (int i = 0; i < _fcWeight.Length && idx < updatedParams.Length; i++, idx++)
        {
            _fcWeight[i] = updatedParams[idx];
        }
        for (int i = 0; i < _fcBias.Length && idx < updatedParams.Length; i++, idx++)
        {
            _fcBias[i] = updatedParams[idx];
        }
        for (int i = 0; i < _embeddingWeight.Length && idx < updatedParams.Length; i++, idx++)
        {
            _embeddingWeight[i] = updatedParams[idx];
        }
        for (int i = 0; i < _embeddingBias.Length && idx < updatedParams.Length; i++, idx++)
        {
            _embeddingBias[i] = updatedParams[idx];
        }

        // Note: CNN layer gradients are not computed in this simplified implementation.
        // For full fine-tuning, implement backward pass through ConvBlocks with cached activations.
    }

    #endregion

    #region Serialization

    public override byte[] Serialize()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        writer.Write(SampleRate);
        writer.Write((int)_architectureType);
        writer.Write(_numClasses);
        writer.Write(_embeddingDim);
        writer.Write(_numMelBands);

        WriteArray(writer, _embeddingWeight);
        WriteArray(writer, _embeddingBias);
        WriteArray(writer, _fcWeight);
        WriteArray(writer, _fcBias);

        return stream.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        // Restore configuration values (must assign to class fields, not local variables)
        SampleRate = reader.ReadInt32();
        _architectureType = (PANNsArchitecture)reader.ReadInt32();
        _numClasses = reader.ReadInt32();
        _embeddingDim = reader.ReadInt32();
        _numMelBands = reader.ReadInt32();

        // Restore weight arrays
        _embeddingWeight = ReadArray(reader);
        _embeddingBias = ReadArray(reader);
        _fcWeight = ReadArray(reader);
        _fcBias = ReadArray(reader);

        // Reinitialize conv blocks if needed
        if (_convBlocks is null || _convBlocks.Count == 0)
        {
            _convBlocks = CreateConvBlocks(_architectureType, _numMelBands);
        }
    }

    private void WriteArray(BinaryWriter writer, T[] array)
    {
        writer.Write(array.Length);
        foreach (var val in array)
        {
            writer.Write(_numOps.ToDouble(val));
        }
    }

    private T[] ReadArray(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        var array = new T[len];
        for (int i = 0; i < len; i++)
        {
            array[i] = _numOps.FromDouble(reader.ReadDouble());
        }
        return array;
    }

    #endregion

    #region Helper Methods

    private T[] InitializeWeights(int size, double initValue = double.NaN)
    {
        var weights = new T[size];
        if (double.IsNaN(initValue))
        {
            double scale = Math.Sqrt(2.0 / size);
            var rand = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
            for (int i = 0; i < size; i++)
            {
                weights[i] = _numOps.FromDouble(rand.NextGaussian() * scale);
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                weights[i] = _numOps.FromDouble(initValue);
            }
        }
        return weights;
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    private static double HzToMel(double hz)
    {
        return 2595.0 * Math.Log10(1 + hz / 700.0);
    }

    private static double MelScale(int band, int numBands, double maxFreq)
    {
        double melMax = HzToMel(maxFreq);
        return melMax * band / numBands;
    }

    private static string[] GetDefaultClassLabels()
    {
        // Subset of AudioSet labels for common sounds
        return new[]
        {
            "Speech", "Male speech", "Female speech", "Child speech", "Conversation",
            "Narration", "Singing", "Music", "Musical instrument", "Plucked string instrument",
            "Guitar", "Electric guitar", "Bass guitar", "Acoustic guitar", "Steel guitar",
            "Keyboard", "Piano", "Electric piano", "Organ", "Synthesizer",
            "Drum", "Snare drum", "Bass drum", "Hi-hat", "Cymbal",
            "Percussion", "Marimba", "Xylophone", "Glockenspiel", "Vibraphone",
            "Violin", "Cello", "Double bass", "Viola", "Harp",
            "Trumpet", "Trombone", "French horn", "Saxophone", "Clarinet",
            "Flute", "Oboe", "Bassoon", "Wind instrument", "Brass instrument",
            "Applause", "Cheering", "Laughter", "Crying", "Cough",
            "Sneeze", "Breathing", "Gasp", "Sigh", "Yawn",
            "Dog", "Bark", "Howl", "Cat", "Meow",
            "Bird", "Chirp", "Crow", "Duck", "Goose",
            "Rooster", "Chicken", "Pig", "Cow", "Horse",
            "Sheep", "Goat", "Lion", "Tiger", "Elephant",
            "Car", "Motorcycle", "Truck", "Bus", "Train",
            "Aircraft", "Helicopter", "Boat", "Ship", "Subway",
            "Traffic noise", "Car horn", "Siren", "Alarm", "Engine",
            "Door", "Knock", "Slam", "Squeak", "Lock",
            "Water", "Rain", "Thunder", "Wind", "Ocean",
            "Fire", "Fireworks", "Explosion", "Gunshot", "Glass",
            "Clock", "Tick", "Alarm clock", "Bell", "Telephone",
            "Television", "Radio", "Static", "Noise", "Silence",
            "Footsteps", "Running", "Walking", "Crowd", "Ambient"
        };
    }

    #endregion

    #region Abstract Method Implementations

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
        }

        // Get current parameters and apply gradient descent
        var currentParams = GetParameters();
        T learningRate = _numOps.FromDouble(0.0001);
        for (int i = 0; i < Math.Min(currentParams.Length, gradients.Length); i++)
        {
            currentParams[i] = _numOps.Subtract(currentParams[i], _numOps.Multiply(learningRate, gradients[i]));
        }
        SetParameters(currentParams);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = $"PANNs-{_architectureType}",
            Description = $"Pretrained Audio Neural Network ({_numClasses} classes, {_architectureType})",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = SampleRate,
            Complexity = _convBlocks.Count
        };
        metadata.AdditionalInfo["EmbeddingDim"] = _embeddingDim.ToString();
        metadata.AdditionalInfo["NumClasses"] = _numClasses.ToString();
        metadata.AdditionalInfo["Architecture"] = _architectureType.ToString();
        metadata.AdditionalInfo["Mode"] = IsOnnxMode ? "ONNX" : "Native";
        return metadata;
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(IsOnnxMode);
        writer.Write(SampleRate);
        writer.Write(_numClasses);
        writer.Write(_embeddingDim);
        writer.Write(_numMelBands);
        writer.Write(_windowSize);
        writer.Write(_hopSize);
        writer.Write((int)_architectureType);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        bool isOnnxMode = reader.ReadBoolean();
        int sampleRate = reader.ReadInt32();
        int numClasses = reader.ReadInt32();
        int embeddingDim = reader.ReadInt32();
        int numMelBands = reader.ReadInt32();
        int windowSize = reader.ReadInt32();
        int hopSize = reader.ReadInt32();
        int architectureType = reader.ReadInt32();

        // Restore configuration values
        SampleRate = sampleRate;
        _numClasses = numClasses;
        _embeddingDim = embeddingDim;
        _numMelBands = numMelBands;
        _windowSize = windowSize;
        _hopSize = hopSize;
        _architectureType = (PANNsArchitecture)architectureType;

        // Reinitialize layers if configuration changed
        if (_convBlocks is null || _convBlocks.Count == 0)
        {
            _convBlocks = CreateConvBlocks(_architectureType, _numMelBands);
        }

        // Reinitialize FC/embedding weights if dimensions changed
        int lastChannels = _convBlocks.Count > 0 ? _convBlocks[^1].OutChannels : 64;
        if (_embeddingWeight is null || _embeddingWeight.Length != _embeddingDim * lastChannels)
        {
            _embeddingWeight = new T[_embeddingDim * lastChannels];
            _embeddingBias = new T[_embeddingDim];
            _fcWeight = new T[_numClasses * _embeddingDim];
            _fcBias = new T[_numClasses];
        }

        // Note: IsOnnxMode state is handled by base class during full deserialization
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new PANNsModel<T>(
            Architecture,
            sampleRate: SampleRate,
            numClasses: _numClasses,
            embeddingDim: _embeddingDim,
            numMelBands: _numMelBands,
            windowSize: _windowSize,
            hopSize: _hopSize,
            architectureType: _architectureType);
    }

    #endregion

    #region Nested Types

    /// <summary>
    /// Convolutional block with batch normalization and ReLU.
    /// </summary>
    private class ConvBlock
    {
        private readonly INumericOperations<T> _ops;
        private readonly int _inChannels;
        private readonly int _outChannels;
        private readonly int _kernelSize;
        private readonly int _stride;
        private readonly bool _pooling;

        private T[] _weight;
        private T[] _bias;
        private T[] _bnGamma;
        private T[] _bnBeta;
        private T[] _bnMean;
        private T[] _bnVar;

        public int OutChannels => _outChannels;

        public ConvBlock(
            INumericOperations<T> ops,
            int inChannels,
            int outChannels,
            int kernelSize,
            int stride,
            bool pooling)
        {
            _ops = ops;
            _inChannels = inChannels;
            _outChannels = outChannels;
            _kernelSize = kernelSize;
            _stride = stride;
            _pooling = pooling;

            var rand = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
            double scale = Math.Sqrt(2.0 / (inChannels * kernelSize * kernelSize));

            int weightSize = outChannels * inChannels * kernelSize * kernelSize;
            _weight = new T[weightSize];
            for (int i = 0; i < weightSize; i++)
            {
                _weight[i] = _ops.FromDouble(rand.NextGaussian() * scale);
            }

            _bias = new T[outChannels];
            _bnGamma = Enumerable.Range(0, outChannels).Select(_ => _ops.FromDouble(1.0)).ToArray();
            _bnBeta = new T[outChannels];
            _bnMean = new T[outChannels];
            _bnVar = Enumerable.Range(0, outChannels).Select(_ => _ops.FromDouble(1.0)).ToArray();
        }

        public Tensor<T> Forward(Tensor<T> input)
        {
            int batchSize = input.Shape[0];
            int inChannels = input.Shape[1];
            int inHeight = input.Shape.Length > 2 ? input.Shape[2] : 1;
            int inWidth = input.Shape.Length > 3 ? input.Shape[3] : 1;

            int pad = _kernelSize / 2;
            int outHeight = (inHeight + 2 * pad - _kernelSize) / _stride + 1;
            int outWidth = (inWidth + 2 * pad - _kernelSize) / _stride + 1;

            var output = new T[batchSize * _outChannels * outHeight * outWidth];

            // Convolution
            for (int b = 0; b < batchSize; b++)
            {
                for (int oc = 0; oc < _outChannels; oc++)
                {
                    for (int oh = 0; oh < outHeight; oh++)
                    {
                        for (int ow = 0; ow < outWidth; ow++)
                        {
                            T sum = _bias[oc];

                            for (int ic = 0; ic < inChannels; ic++)
                            {
                                for (int kh = 0; kh < _kernelSize; kh++)
                                {
                                    for (int kw = 0; kw < _kernelSize; kw++)
                                    {
                                        int ih = oh * _stride + kh - pad;
                                        int iw = ow * _stride + kw - pad;

                                        if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                        {
                                            int inputIdx = b * inChannels * inHeight * inWidth +
                                                          ic * inHeight * inWidth +
                                                          ih * inWidth + iw;
                                            int weightIdx = oc * inChannels * _kernelSize * _kernelSize +
                                                           ic * _kernelSize * _kernelSize +
                                                           kh * _kernelSize + kw;

                                            if (inputIdx < input.Length && weightIdx < _weight.Length)
                                            {
                                                sum = _ops.Add(sum, _ops.Multiply(
                                                    input.Data.Span[inputIdx],
                                                    _weight[weightIdx]));
                                            }
                                        }
                                    }
                                }
                            }

                            // BatchNorm
                            double val = _ops.ToDouble(sum);
                            double mean = _ops.ToDouble(_bnMean[oc]);
                            double var = _ops.ToDouble(_bnVar[oc]);
                            double gamma = _ops.ToDouble(_bnGamma[oc]);
                            double beta = _ops.ToDouble(_bnBeta[oc]);
                            val = gamma * (val - mean) / Math.Sqrt(var + 1e-5) + beta;

                            // ReLU
                            val = Math.Max(0, val);

                            int outIdx = b * _outChannels * outHeight * outWidth +
                                        oc * outHeight * outWidth +
                                        oh * outWidth + ow;
                            output[outIdx] = _ops.FromDouble(val);
                        }
                    }
                }
            }

            // Max pooling
            if (_pooling)
            {
                int pooledHeight = outHeight / 2;
                int pooledWidth = outWidth / 2;
                var pooled = new T[batchSize * _outChannels * pooledHeight * pooledWidth];

                for (int b = 0; b < batchSize; b++)
                {
                    for (int c = 0; c < _outChannels; c++)
                    {
                        for (int ph = 0; ph < pooledHeight; ph++)
                        {
                            for (int pw = 0; pw < pooledWidth; pw++)
                            {
                                double maxVal = double.MinValue;
                                for (int dh = 0; dh < 2; dh++)
                                {
                                    for (int dw = 0; dw < 2; dw++)
                                    {
                                        int oh = ph * 2 + dh;
                                        int ow = pw * 2 + dw;
                                        if (oh < outHeight && ow < outWidth)
                                        {
                                            int idx = b * _outChannels * outHeight * outWidth +
                                                     c * outHeight * outWidth +
                                                     oh * outWidth + ow;
                                            maxVal = Math.Max(maxVal, _ops.ToDouble(output[idx]));
                                        }
                                    }
                                }
                                int outIdx = b * _outChannels * pooledHeight * pooledWidth +
                                            c * pooledHeight * pooledWidth +
                                            ph * pooledWidth + pw;
                                pooled[outIdx] = _ops.FromDouble(maxVal);
                            }
                        }
                    }
                }

                return new Tensor<T>(pooled, new[] { batchSize, _outChannels, pooledHeight, pooledWidth });
            }

            return new Tensor<T>(output, new[] { batchSize, _outChannels, outHeight, outWidth });
        }
    }

    #endregion
}

/// <summary>
/// PANNs architecture variants.
/// </summary>
public enum PANNsArchitecture
{
    /// <summary>
    /// CNN6: 6-layer CNN (smaller, faster).
    /// </summary>
    Cnn6,

    /// <summary>
    /// CNN10: 10-layer CNN (balanced).
    /// </summary>
    Cnn10,

    /// <summary>
    /// CNN14: 14-layer CNN (larger, more accurate).
    /// </summary>
    Cnn14
}
