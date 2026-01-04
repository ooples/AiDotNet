using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.OCR.Recognition;

/// <summary>
/// CRNN (Convolutional Recurrent Neural Network) for text recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> CRNN combines CNNs for visual feature extraction with
/// RNNs (specifically LSTM) for sequence modeling. It's trained with CTC loss to
/// handle variable-length text without requiring character-level alignment.</para>
///
/// <para>Key features:
/// - CNN backbone for visual features
/// - Bidirectional LSTM for sequence modeling
/// - CTC (Connectionist Temporal Classification) decoding
/// - Handles variable-length text naturally
/// </para>
///
/// <para>Reference: Shi et al., "An End-to-End Trainable Neural Network for
/// Image-based Sequence Recognition and Its Application to Scene Text Recognition", TPAMI 2017</para>
/// </remarks>
public class CRNN<T> : OCRBase<T>
{
    private readonly Conv2D<T> _conv1;
    private readonly Conv2D<T> _conv2;
    private readonly Conv2D<T> _conv3;
    private readonly Conv2D<T> _conv4;
    private readonly Conv2D<T> _conv5;
    private readonly Conv2D<T> _conv6;
    private readonly Conv2D<T> _conv7;

    // Bidirectional LSTM layers using actual LSTMLayer
    private readonly LSTMLayer<T> _lstm1Forward;
    private readonly LSTMLayer<T> _lstm1Backward;
    private readonly LSTMLayer<T> _lstm2Forward;
    private readonly LSTMLayer<T> _lstm2Backward;

    private readonly Dense<T> _outputLayer;
    private readonly int _hiddenDim;
    private readonly int _sequenceFeatureDim;

    // LSTM state tracking
    private Tensor<T>? _lstm1FwHidden;
    private Tensor<T>? _lstm1FwCell;
    private Tensor<T>? _lstm1BwHidden;
    private Tensor<T>? _lstm1BwCell;
    private Tensor<T>? _lstm2FwHidden;
    private Tensor<T>? _lstm2FwCell;
    private Tensor<T>? _lstm2BwHidden;
    private Tensor<T>? _lstm2BwCell;

    /// <inheritdoc/>
    public override string Name => "CRNN";

    /// <summary>
    /// Creates a new CRNN text recognizer.
    /// </summary>
    public CRNN(OCROptions<T> options) : base(options)
    {
        _hiddenDim = 256;

        // CNN backbone for feature extraction (VGG-style architecture)
        // Stage 1
        _conv1 = new Conv2D<T>(1, 64, kernelSize: 3, padding: 1); // 1 channel for grayscale
        _conv2 = new Conv2D<T>(64, 128, kernelSize: 3, padding: 1);

        // Stage 2
        _conv3 = new Conv2D<T>(128, 256, kernelSize: 3, padding: 1);
        _conv4 = new Conv2D<T>(256, 256, kernelSize: 3, padding: 1);

        // Stage 3
        _conv5 = new Conv2D<T>(256, 512, kernelSize: 3, padding: 1);
        _conv6 = new Conv2D<T>(512, 512, kernelSize: 3, padding: 1);

        // Stage 4
        _conv7 = new Conv2D<T>(512, 512, kernelSize: 2, padding: 0);

        // After conv layers, assuming input height 32, the feature map height becomes 1
        // Width is preserved (roughly input_width / 4 due to pooling)
        // Feature dimension = 512 channels * 1 height = 512
        _sequenceFeatureDim = 512;

        // Bidirectional LSTM Layer 1
        // Input: [batch, seqLen, 512], Output: [batch, seqLen, 256]
        int[] inputShape1 = new[] { 1, _sequenceFeatureDim }; // [batch, features] for single timestep
        IActivationFunction<T> tanhActivation = new TanhActivation<T>();
        _lstm1Forward = new LSTMLayer<T>(_sequenceFeatureDim, _hiddenDim, inputShape1, tanhActivation);
        _lstm1Backward = new LSTMLayer<T>(_sequenceFeatureDim, _hiddenDim, inputShape1, tanhActivation);

        // Bidirectional LSTM Layer 2
        // Input: [batch, seqLen, 512 (256*2)], Output: [batch, seqLen, 256]
        int[] inputShape2 = new[] { 1, _hiddenDim * 2 };
        _lstm2Forward = new LSTMLayer<T>(_hiddenDim * 2, _hiddenDim, inputShape2, tanhActivation);
        _lstm2Backward = new LSTMLayer<T>(_hiddenDim * 2, _hiddenDim, inputShape2, tanhActivation);

        // Output layer to vocabulary (512 = 256*2 from bidirectional)
        _outputLayer = new Dense<T>(_hiddenDim * 2, VocabularySize);

        // Initialize LSTM states
        ResetLSTMStates(1);
    }

    /// <summary>
    /// Resets the LSTM hidden and cell states.
    /// </summary>
    private void ResetLSTMStates(int batchSize)
    {
        _lstm1FwHidden = new Tensor<T>(new[] { batchSize, _hiddenDim });
        _lstm1FwCell = new Tensor<T>(new[] { batchSize, _hiddenDim });
        _lstm1BwHidden = new Tensor<T>(new[] { batchSize, _hiddenDim });
        _lstm1BwCell = new Tensor<T>(new[] { batchSize, _hiddenDim });
        _lstm2FwHidden = new Tensor<T>(new[] { batchSize, _hiddenDim });
        _lstm2FwCell = new Tensor<T>(new[] { batchSize, _hiddenDim });
        _lstm2BwHidden = new Tensor<T>(new[] { batchSize, _hiddenDim });
        _lstm2BwCell = new Tensor<T>(new[] { batchSize, _hiddenDim });
    }

    /// <inheritdoc/>
    public override OCRResult<T> Recognize(Tensor<T> image)
    {
        var startTime = DateTime.UtcNow;

        int imageWidth = image.Shape[3];
        int imageHeight = image.Shape[2];

        // Preprocess
        var input = PreprocessCrop(image);

        // Forward pass
        var (text, confidence) = RecognizeText(input);

        var result = new OCRResult<T>
        {
            FullText = text,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = imageWidth,
            ImageHeight = imageHeight
        };

        if (!string.IsNullOrEmpty(text))
        {
            result.TextRegions.Add(new RecognizedText<T>(text, confidence));
        }

        return result;
    }

    /// <inheritdoc/>
    public override (string text, T confidence) RecognizeText(Tensor<T> croppedImage)
    {
        int batch = croppedImage.Shape[0];

        // Reset LSTM states for new sequence
        ResetLSTMStates(batch);

        // Convert to grayscale if needed
        var grayImage = ConvertToGrayscale(croppedImage);

        // Forward pass through CNN backbone
        var x = _conv1.Forward(grayImage);
        x = ApplyReLU(x);
        x = MaxPool2D(x, 2, 2);

        x = _conv2.Forward(x);
        x = ApplyReLU(x);
        x = MaxPool2D(x, 2, 2);

        x = _conv3.Forward(x);
        x = ApplyReLU(x);

        x = _conv4.Forward(x);
        x = ApplyReLU(x);
        x = MaxPool2D(x, 2, 1); // Pool height only

        x = _conv5.Forward(x);
        x = ApplyReLU(x);
        x = ApplyBatchNorm(x);

        x = _conv6.Forward(x);
        x = ApplyReLU(x);
        x = ApplyBatchNorm(x);
        x = MaxPool2D(x, 2, 1); // Pool height only

        x = _conv7.Forward(x);
        x = ApplyReLU(x);

        // Squeeze height dimension and transpose to (batch, width, channels)
        var seqFeatures = SqueezeAndPermute(x);

        // Bidirectional LSTM processing
        var lstmOut = ApplyBidirectionalLSTM(seqFeatures, batch);

        // Output projection
        var logits = ApplyOutputLayer(lstmOut);

        // Apply softmax for probabilities
        var probs = ApplySoftmax(logits);

        // CTC decoding
        string text = DecodeCTC(probs);
        T confidence = ComputeConfidence(probs, text);

        return (text, confidence);
    }

    /// <summary>
    /// Converts RGB image to grayscale.
    /// </summary>
    private Tensor<T> ConvertToGrayscale(Tensor<T> image)
    {
        int batch = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        if (channels == 1)
        {
            return image;
        }

        var gray = new Tensor<T>(new[] { batch, 1, height, width });

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
                    double r = NumOps.ToDouble(image[b, 0, h, w]);
                    double g = channels > 1 ? NumOps.ToDouble(image[b, 1, h, w]) : r;
                    double bl = channels > 2 ? NumOps.ToDouble(image[b, 2, h, w]) : r;

                    double grayVal = 0.299 * r + 0.587 * g + 0.114 * bl;
                    gray[b, 0, h, w] = NumOps.FromDouble(grayVal);
                }
            }
        }

        return gray;
    }

    /// <summary>
    /// Applies bidirectional LSTM using proper LSTMLayer cells.
    /// </summary>
    private Tensor<T> ApplyBidirectionalLSTM(Tensor<T> x, int batch)
    {
        // x: [batch, seq_len, features]
        int seqLen = x.Shape[1];
        int features = x.Shape[2];

        // First bidirectional layer
        var fw1Outputs = new Tensor<T>(new[] { batch, seqLen, _hiddenDim });
        var bw1Outputs = new Tensor<T>(new[] { batch, seqLen, _hiddenDim });

        // Forward direction
        _lstm1Forward.ResetState();
        for (int t = 0; t < seqLen; t++)
        {
            var input = ExtractTimestep(x, t, batch, features);
            var output = _lstm1Forward.Forward(input);
            StoreTimestep(fw1Outputs, output, t, batch, _hiddenDim);
        }

        // Backward direction
        _lstm1Backward.ResetState();
        for (int t = seqLen - 1; t >= 0; t--)
        {
            var input = ExtractTimestep(x, t, batch, features);
            var output = _lstm1Backward.Forward(input);
            StoreTimestep(bw1Outputs, output, t, batch, _hiddenDim);
        }

        // Concatenate forward and backward outputs
        var concat1 = ConcatenateBidirectional(fw1Outputs, bw1Outputs, batch, seqLen, _hiddenDim);

        // Second bidirectional layer
        var fw2Outputs = new Tensor<T>(new[] { batch, seqLen, _hiddenDim });
        var bw2Outputs = new Tensor<T>(new[] { batch, seqLen, _hiddenDim });

        // Forward direction
        _lstm2Forward.ResetState();
        for (int t = 0; t < seqLen; t++)
        {
            var input = ExtractTimestep(concat1, t, batch, _hiddenDim * 2);
            var output = _lstm2Forward.Forward(input);
            StoreTimestep(fw2Outputs, output, t, batch, _hiddenDim);
        }

        // Backward direction
        _lstm2Backward.ResetState();
        for (int t = seqLen - 1; t >= 0; t--)
        {
            var input = ExtractTimestep(concat1, t, batch, _hiddenDim * 2);
            var output = _lstm2Backward.Forward(input);
            StoreTimestep(bw2Outputs, output, t, batch, _hiddenDim);
        }

        // Final concatenation
        return ConcatenateBidirectional(fw2Outputs, bw2Outputs, batch, seqLen, _hiddenDim);
    }

    /// <summary>
    /// Extracts a single timestep from the sequence tensor.
    /// </summary>
    private Tensor<T> ExtractTimestep(Tensor<T> x, int t, int batch, int features)
    {
        var timestep = new Tensor<T>(new[] { batch, features });

        for (int b = 0; b < batch; b++)
        {
            for (int f = 0; f < features; f++)
            {
                timestep[b, f] = x[b, t, f];
            }
        }

        return timestep;
    }

    /// <summary>
    /// Stores LSTM output into the sequence tensor at a specific timestep.
    /// </summary>
    private void StoreTimestep(Tensor<T> output, Tensor<T> lstmOut, int t, int batch, int hiddenDim)
    {
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < hiddenDim; h++)
            {
                output[b, t, h] = lstmOut[b, h];
            }
        }
    }

    /// <summary>
    /// Concatenates forward and backward LSTM outputs.
    /// </summary>
    private Tensor<T> ConcatenateBidirectional(Tensor<T> forward, Tensor<T> backward, int batch, int seqLen, int hiddenDim)
    {
        var concat = new Tensor<T>(new[] { batch, seqLen, hiddenDim * 2 });

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < hiddenDim; h++)
                {
                    concat[b, t, h] = forward[b, t, h];
                    concat[b, t, hiddenDim + h] = backward[b, t, h];
                }
            }
        }

        return concat;
    }

    /// <summary>
    /// Applies softmax normalization across the vocabulary dimension.
    /// </summary>
    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        int batch = logits.Shape[0];
        int seqLen = logits.Shape[1];
        int vocabSize = logits.Shape[2];

        var probs = new Tensor<T>(logits.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Find max for numerical stability
                double maxVal = double.NegativeInfinity;
                for (int v = 0; v < vocabSize; v++)
                {
                    maxVal = Math.Max(maxVal, NumOps.ToDouble(logits[b, t, v]));
                }

                // Compute exp and sum
                double sum = 0;
                var expVals = new double[vocabSize];
                for (int v = 0; v < vocabSize; v++)
                {
                    expVals[v] = Math.Exp(NumOps.ToDouble(logits[b, t, v]) - maxVal);
                    sum += expVals[v];
                }

                // Normalize
                for (int v = 0; v < vocabSize; v++)
                {
                    probs[b, t, v] = NumOps.FromDouble(expVals[v] / sum);
                }
            }
        }

        return probs;
    }

    /// <summary>
    /// Applies simple batch normalization.
    /// </summary>
    private Tensor<T> ApplyBatchNorm(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];

        var result = new Tensor<T>(x.Shape);
        double epsilon = 1e-5;

        for (int c = 0; c < channels; c++)
        {
            // Compute mean and variance for this channel
            double sum = 0;
            double sumSq = 0;
            int count = batch * height * width;

            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        double val = NumOps.ToDouble(x[b, c, h, w]);
                        sum += val;
                        sumSq += val * val;
                    }
                }
            }

            double mean = sum / count;
            double variance = (sumSq / count) - (mean * mean);
            double stdDev = Math.Sqrt(variance + epsilon);

            // Normalize
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        double val = NumOps.ToDouble(x[b, c, h, w]);
                        double normalized = (val - mean) / stdDev;
                        result[b, c, h, w] = NumOps.FromDouble(normalized);
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        return _conv1.GetParameterCount() +
               _conv2.GetParameterCount() +
               _conv3.GetParameterCount() +
               _conv4.GetParameterCount() +
               _conv5.GetParameterCount() +
               _conv6.GetParameterCount() +
               _conv7.GetParameterCount() +
               _lstm1Forward.GetParameters().Length +
               _lstm1Backward.GetParameters().Length +
               _lstm2Forward.GetParameters().Length +
               _lstm2Backward.GetParameters().Length +
               _outputLayer.GetParameterCount();
    }

    /// <inheritdoc/>
    public override async Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        string localPath = pathOrUrl;

        // Download if URL
        if (pathOrUrl.StartsWith("http://") || pathOrUrl.StartsWith("https://"))
        {
            var downloader = new WeightDownloader();
            string fileName = Path.GetFileName(new Uri(pathOrUrl).LocalPath);
            localPath = await downloader.DownloadIfNeededAsync(pathOrUrl, fileName, null, cancellationToken);
        }

        // Check if this is our native format (starts with CRNN magic number)
        if (File.Exists(localPath))
        {
            using var checkStream = File.OpenRead(localPath);
            using var checkReader = new BinaryReader(checkStream);
            if (checkStream.Length >= 4)
            {
                int magic = checkReader.ReadInt32();
                if (magic == 0x43524E4E) // "CRNN"
                {
                    checkStream.Close();
                    LoadWeightsFromFile(localPath);
                    return;
                }
            }
        }

        // Fallback to external weight format
        var loader = new WeightLoader();
        var weights = loader.LoadWeights(localPath);

        // Map weights to layers
        MapWeightsToLayers(weights);
    }

    /// <summary>
    /// Maps loaded weights to the model's layers.
    /// </summary>
    private void MapWeightsToLayers(Dictionary<string, Tensor<float>> weights)
    {
        // Map CNN weights
        MapConvWeights(weights, "cnn.conv0", _conv1);
        MapConvWeights(weights, "cnn.conv1", _conv2);
        MapConvWeights(weights, "cnn.conv2", _conv3);
        MapConvWeights(weights, "cnn.conv3", _conv4);
        MapConvWeights(weights, "cnn.conv4", _conv5);
        MapConvWeights(weights, "cnn.conv5", _conv6);
        MapConvWeights(weights, "cnn.conv6", _conv7);

        // Map LSTM weights (typical PyTorch naming: rnn.weight_ih_l0, rnn.weight_hh_l0, etc.)
        MapLSTMWeights(weights, "rnn", 0, _lstm1Forward, _lstm1Backward);
        MapLSTMWeights(weights, "rnn", 1, _lstm2Forward, _lstm2Backward);

        // Map output layer weights
        MapDenseWeights(weights, "fc", _outputLayer);
    }

    private void MapConvWeights(Dictionary<string, Tensor<float>> weights, string prefix, Conv2D<T> conv)
    {
        if (weights.TryGetValue($"{prefix}.weight", out var weight))
        {
            CopyWeights(weight, conv.Weights);
        }
        if (weights.TryGetValue($"{prefix}.bias", out var bias) && conv.Bias is not null)
        {
            CopyWeights(bias, conv.Bias);
        }
    }

    private void MapLSTMWeights(Dictionary<string, Tensor<float>> weights, string prefix, int layer,
        LSTMLayer<T> forward, LSTMLayer<T> backward)
    {
        // Forward direction weights
        MapLSTMDirection(weights, prefix, layer, string.Empty, forward);

        // Backward direction weights (PyTorch adds "_reverse" suffix)
        MapLSTMDirection(weights, prefix, layer, "_reverse", backward);
    }

    private void MapLSTMDirection(Dictionary<string, Tensor<float>> weights, string prefix, int layer,
        string suffix, LSTMLayer<T> lstm)
    {
        // PyTorch LSTM weights are concatenated for gates in order: i (input), f (forget), g (cell), o (output)
        // Each gate has hidden_size rows, so total is 4*hidden_size x input_size for weight_ih
        string weightIhKey = $"{prefix}.weight_ih_l{layer}{suffix}";
        string weightHhKey = $"{prefix}.weight_hh_l{layer}{suffix}";
        string biasIhKey = $"{prefix}.bias_ih_l{layer}{suffix}";
        string biasHhKey = $"{prefix}.bias_hh_l{layer}{suffix}";

        // Map input-hidden weights (split by gates)
        if (weights.TryGetValue(weightIhKey, out var weightIh))
        {
            int hiddenSize = weightIh.Shape[0] / 4;
            int inputSize = weightIh.Shape[1];

            // Split into 4 gates: i, f, g, o (PyTorch order)
            CopyWeightSlice(weightIh, lstm.WeightsIi, 0, hiddenSize, inputSize);
            CopyWeightSlice(weightIh, lstm.WeightsFi, hiddenSize, hiddenSize, inputSize);
            CopyWeightSlice(weightIh, lstm.WeightsCi, 2 * hiddenSize, hiddenSize, inputSize);
            CopyWeightSlice(weightIh, lstm.WeightsOi, 3 * hiddenSize, hiddenSize, inputSize);
        }

        // Map hidden-hidden weights (split by gates)
        if (weights.TryGetValue(weightHhKey, out var weightHh))
        {
            int hiddenSize = weightHh.Shape[0] / 4;

            CopyWeightSlice(weightHh, lstm.WeightsIh, 0, hiddenSize, hiddenSize);
            CopyWeightSlice(weightHh, lstm.WeightsFh, hiddenSize, hiddenSize, hiddenSize);
            CopyWeightSlice(weightHh, lstm.WeightsCh, 2 * hiddenSize, hiddenSize, hiddenSize);
            CopyWeightSlice(weightHh, lstm.WeightsOh, 3 * hiddenSize, hiddenSize, hiddenSize);
        }

        // Map biases (PyTorch has separate ih and hh biases, we combine them)
        if (weights.TryGetValue(biasIhKey, out var biasIh) &&
            weights.TryGetValue(biasHhKey, out var biasHh))
        {
            int hiddenSize = biasIh.Length / 4;

            CopyBiasSlice(biasIh, biasHh, lstm.BiasI, 0, hiddenSize);
            CopyBiasSlice(biasIh, biasHh, lstm.BiasF, hiddenSize, hiddenSize);
            CopyBiasSlice(biasIh, biasHh, lstm.BiasC, 2 * hiddenSize, hiddenSize);
            CopyBiasSlice(biasIh, biasHh, lstm.BiasO, 3 * hiddenSize, hiddenSize);
        }
    }

    private void CopyWeightSlice(Tensor<float> source, Tensor<T> dest, int startRow, int numRows, int numCols)
    {
        int destRows = dest.Shape[0];
        int destCols = dest.Shape[1];
        int rows = Math.Min(numRows, destRows);
        int cols = Math.Min(numCols, destCols);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int srcIdx = (startRow + r) * source.Shape[1] + c;
                if (srcIdx < source.Length)
                {
                    dest[r, c] = NumOps.FromDouble(source[srcIdx]);
                }
            }
        }
    }

    private void CopyBiasSlice(Tensor<float> biasIh, Tensor<float> biasHh, Tensor<T> dest, int start, int length)
    {
        int count = Math.Min(length, dest.Length);
        for (int i = 0; i < count; i++)
        {
            // Combine ih and hh biases (standard LSTM convention)
            double val = 0;
            if (start + i < biasIh.Length)
                val += biasIh[start + i];
            if (start + i < biasHh.Length)
                val += biasHh[start + i];
            dest[i] = NumOps.FromDouble(val);
        }
    }

    private void MapDenseWeights(Dictionary<string, Tensor<float>> weights, string prefix, Dense<T> dense)
    {
        if (weights.TryGetValue($"{prefix}.weight", out var weight))
        {
            CopyWeights(weight, dense.Weights);
        }
        if (weights.TryGetValue($"{prefix}.bias", out var bias))
        {
            CopyWeights(bias, dense.Bias);
        }
    }

    private void CopyWeights(Tensor<float> source, Tensor<T> dest)
    {
        int count = Math.Min(source.Length, dest.Length);
        for (int i = 0; i < count; i++)
        {
            dest[i] = NumOps.FromDouble(source[i]);
        }
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write header
        writer.Write(0x43524E4E); // "CRNN" in ASCII
        writer.Write(1); // Version 1
        writer.Write(Name);
        writer.Write(_hiddenDim);
        writer.Write(_sequenceFeatureDim);
        writer.Write(VocabularySize);

        // Write CNN weights
        _conv1.WriteParameters(writer);
        _conv2.WriteParameters(writer);
        _conv3.WriteParameters(writer);
        _conv4.WriteParameters(writer);
        _conv5.WriteParameters(writer);
        _conv6.WriteParameters(writer);
        _conv7.WriteParameters(writer);

        // Write LSTM weights using existing Serialize method
        _lstm1Forward.Serialize(writer);
        _lstm1Backward.Serialize(writer);
        _lstm2Forward.Serialize(writer);
        _lstm2Backward.Serialize(writer);

        // Write output layer
        _outputLayer.WriteParameters(writer);
    }

    /// <summary>
    /// Loads weights from a file.
    /// </summary>
    private void LoadWeightsFromFile(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        // Read and verify header
        int magic = reader.ReadInt32();
        if (magic != 0x43524E4E) // "CRNN"
        {
            throw new InvalidDataException($"Invalid CRNN model file. Expected magic 0x43524E4E, got 0x{magic:X8}");
        }

        int version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported CRNN model version: {version}");
        }

        string name = reader.ReadString();
        int hiddenDim = reader.ReadInt32();
        int seqFeatureDim = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();

        if (hiddenDim != _hiddenDim || seqFeatureDim != _sequenceFeatureDim)
        {
            throw new InvalidOperationException(
                $"CRNN configuration mismatch. Expected hiddenDim={_hiddenDim}, seqFeatureDim={_sequenceFeatureDim}, " +
                $"got hiddenDim={hiddenDim}, seqFeatureDim={seqFeatureDim}");
        }

        // Read CNN weights
        _conv1.ReadParameters(reader);
        _conv2.ReadParameters(reader);
        _conv3.ReadParameters(reader);
        _conv4.ReadParameters(reader);
        _conv5.ReadParameters(reader);
        _conv6.ReadParameters(reader);
        _conv7.ReadParameters(reader);

        // Read LSTM weights using existing Deserialize method
        _lstm1Forward.Deserialize(reader);
        _lstm1Backward.Deserialize(reader);
        _lstm2Forward.Deserialize(reader);
        _lstm2Backward.Deserialize(reader);

        // Read output layer
        _outputLayer.ReadParameters(reader);
    }

    private Tensor<T> ApplyReLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = NumOps.FromDouble(Math.Max(0, val));
        }
        return result;
    }

    private Tensor<T> MaxPool2D(Tensor<T> x, int kernelH, int kernelW)
    {
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];

        int outH = height / kernelH;
        int outW = width / kernelW;

        var result = new Tensor<T>(new[] { batch, channels, outH, outW });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < outH; h++)
                {
                    for (int w = 0; w < outW; w++)
                    {
                        double maxVal = double.NegativeInfinity;

                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int srcH = h * kernelH + kh;
                                int srcW = w * kernelW + kw;

                                if (srcH < height && srcW < width)
                                {
                                    maxVal = Math.Max(maxVal, NumOps.ToDouble(x[b, c, srcH, srcW]));
                                }
                            }
                        }

                        result[b, c, h, w] = NumOps.FromDouble(maxVal);
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> SqueezeAndPermute(Tensor<T> x)
    {
        // x: [batch, channels, height, width]
        // Output: [batch, width, channels*height]
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];

        int featureDim = channels * height;

        var result = new Tensor<T>(new[] { batch, width, featureDim });

        for (int b = 0; b < batch; b++)
        {
            for (int w = 0; w < width; w++)
            {
                int idx = 0;
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        result[b, w, idx++] = x[b, c, h, w];
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyOutputLayer(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int features = x.Shape[2];

        var result = new Tensor<T>(new[] { batch, seqLen, VocabularySize });

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                var feat = new Tensor<T>(new[] { 1, features });
                for (int f = 0; f < features; f++)
                {
                    feat[0, f] = x[b, t, f];
                }

                var output = _outputLayer.Forward(feat);
                for (int v = 0; v < VocabularySize; v++)
                {
                    result[b, t, v] = output[0, v];
                }
            }
        }

        return result;
    }
}
