using System.IO;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.OCR.Recognition;

/// <summary>
/// TrOCR (Transformer-based OCR) for text recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> TrOCR uses a Vision Transformer (ViT) as the encoder
/// to extract visual features, and a Transformer decoder to generate text autoregressively.
/// This architecture leverages the power of pre-trained language models.</para>
///
/// <para>Key features:
/// - Vision Transformer encoder for image understanding
/// - Transformer decoder with attention for text generation
/// - Autoregressive decoding with beam search
/// - Can leverage pre-trained models
/// </para>
///
/// <para>Reference: Li et al., "TrOCR: Transformer-based Optical Character Recognition
/// with Pre-trained Models", AAAI 2023</para>
/// </remarks>
public class TrOCR<T> : OCRBase<T>
{
    private readonly Conv2D<T> _patchEmbed;
    private readonly TrOCREncoderLayer<T>[] _encoderLayers;
    private readonly TrOCRDecoderLayer<T>[] _decoderLayers;
    private readonly Dense<T> _outputProjection;
    private readonly Dense<T> _tokenEmbedding;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _numLayers;
    private readonly int _patchSize;
    private readonly int _startTokenId;
    private readonly int _endTokenId;

    /// <inheritdoc/>
    public override string Name => "TrOCR";

    /// <summary>
    /// Gets the number of attention heads in the transformer.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Creates a new TrOCR text recognizer.
    /// </summary>
    public TrOCR(OCROptions<T> options) : base(options)
    {
        _hiddenDim = 512;
        _numHeads = 8;
        _numLayers = 6;
        _patchSize = 16;

        // Special tokens (add to vocabulary)
        _startTokenId = VocabularySize; // SOS token
        _endTokenId = VocabularySize + 1; // EOS token

        // Patch embedding layer
        _patchEmbed = new Conv2D<T>(3, _hiddenDim, kernelSize: _patchSize, stride: _patchSize);

        // Token embedding for decoder
        _tokenEmbedding = new Dense<T>(VocabularySize + 2, _hiddenDim);

        // Proper transformer encoder layers with multi-head self-attention
        _encoderLayers = new TrOCREncoderLayer<T>[_numLayers];
        for (int i = 0; i < _numLayers; i++)
        {
            _encoderLayers[i] = new TrOCREncoderLayer<T>(_hiddenDim, _numHeads);
        }

        // Proper transformer decoder layers with self-attention and cross-attention
        _decoderLayers = new TrOCRDecoderLayer<T>[_numLayers];
        for (int i = 0; i < _numLayers; i++)
        {
            _decoderLayers[i] = new TrOCRDecoderLayer<T>(_hiddenDim, _numHeads);
        }

        // Output projection to vocabulary + special tokens
        _outputProjection = new Dense<T>(_hiddenDim, VocabularySize + 2);
    }

    /// <inheritdoc/>
    public override OCRResult<T> Recognize(Tensor<T> image)
    {
        var startTime = DateTime.UtcNow;

        int imageWidth = image.Shape[3];
        int imageHeight = image.Shape[2];

        var input = PreprocessCrop(image);
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
        // Encode image
        var encoderOutput = EncodeImage(croppedImage);

        // Decode text autoregressively
        var (text, confidence) = DecodeText(encoderOutput);

        return (text, confidence);
    }

    private Tensor<T> EncodeImage(Tensor<T> image)
    {
        // Patch embedding
        var patches = _patchEmbed.Forward(image);

        // Flatten patches: [batch, channels, h, w] -> [batch, seq_len, hidden_dim]
        int batch = patches.Shape[0];
        int channels = patches.Shape[1];
        int h = patches.Shape[2];
        int w = patches.Shape[3];
        int seqLen = h * w;

        var x = new Tensor<T>(new[] { batch, seqLen, channels });

        for (int b = 0; b < batch; b++)
        {
            int idx = 0;
            for (int ph = 0; ph < h; ph++)
            {
                for (int pw = 0; pw < w; pw++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        x[b, idx, c] = patches[b, c, ph, pw];
                    }
                    idx++;
                }
            }
        }

        // Add positional encoding
        x = AddPositionalEncoding(x);

        // Apply encoder layers
        for (int l = 0; l < _numLayers; l++)
        {
            x = ApplyEncoderLayer(x, l);
        }

        return x;
    }

    private (string text, T confidence) DecodeText(Tensor<T> encoderOutput)
    {
        int batch = encoderOutput.Shape[0];
        int maxLen = Options.MaxSequenceLength;

        var tokens = new List<int> { _startTokenId };
        var confidences = new List<double>();

        // Autoregressive decoding
        for (int step = 0; step < maxLen - 1; step++)
        {
            // Create decoder input from current tokens
            var decoderInput = CreateDecoderInput(tokens);

            // Apply decoder
            var decoderOutput = ApplyDecoder(decoderInput, encoderOutput);

            // Get output for last position
            int lastPos = tokens.Count - 1;
            var logits = new double[VocabularySize + 2];

            for (int v = 0; v < VocabularySize + 2; v++)
            {
                logits[v] = NumOps.ToDouble(decoderOutput[0, lastPos, v]);
            }

            // Apply softmax and get best token
            double maxLogit = logits.Max();
            double sumExp = 0;
            for (int v = 0; v < logits.Length; v++)
            {
                logits[v] = Math.Exp(logits[v] - maxLogit);
                sumExp += logits[v];
            }

            int bestToken = 0;
            double bestProb = 0;
            for (int v = 0; v < logits.Length; v++)
            {
                double prob = logits[v] / sumExp;
                if (prob > bestProb)
                {
                    bestProb = prob;
                    bestToken = v;
                }
            }

            // Stop if end token
            if (bestToken == _endTokenId)
                break;

            tokens.Add(bestToken);
            confidences.Add(bestProb);
        }

        // Convert tokens to text
        var textChars = new List<char>();
        for (int i = 1; i < tokens.Count; i++) // Skip start token
        {
            int tokenId = tokens[i];
            if (tokenId > 0 && tokenId < VocabularySize && IndexToChar.TryGetValue(tokenId, out char ch))
            {
                textChars.Add(ch);
            }
        }

        string text = new string(textChars.ToArray());
        double avgConf = confidences.Count > 0 ? confidences.Average() : 0;

        return (text, NumOps.FromDouble(avgConf));
    }

    private Tensor<T> CreateDecoderInput(List<int> tokens)
    {
        int seqLen = tokens.Count;
        int vocabSize = VocabularySize + 2; // +2 for start/end tokens

        // Create one-hot representation for embedding lookup
        var oneHot = new Tensor<T>(new[] { 1, seqLen, vocabSize });
        for (int t = 0; t < seqLen; t++)
        {
            int tokenId = MathHelper.Clamp(tokens[t], 0, vocabSize - 1);
            oneHot[0, t, tokenId] = NumOps.FromDouble(1.0);
        }

        // Apply learned token embedding projection
        var embedded = new Tensor<T>(new[] { 1, seqLen, _hiddenDim });
        for (int t = 0; t < seqLen; t++)
        {
            // Extract single token one-hot vector
            var tokenOneHot = new Tensor<T>(new[] { 1, vocabSize });
            for (int v = 0; v < vocabSize; v++)
            {
                tokenOneHot[0, v] = oneHot[0, t, v];
            }

            // Apply embedding projection
            var tokenEmb = _tokenEmbedding.Forward(tokenOneHot);

            // Copy to output
            for (int h = 0; h < _hiddenDim; h++)
            {
                embedded[0, t, h] = tokenEmb[0, h];
            }
        }

        // Add positional encoding - critical for transformer to understand token positions
        // Uses sinusoidal positional encoding matching the encoder's positional encoding
        var embeddedWithPos = AddPositionalEncoding(embedded);

        return embeddedWithPos;
    }

    private Tensor<T> AddPositionalEncoding(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int hiddenDim = x.Shape[2];

        var result = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int pos = 0; pos < seqLen; pos++)
            {
                for (int i = 0; i < hiddenDim; i++)
                {
                    // Use explicit floor division to get the pair index (0,1->0, 2,3->1, etc.)
                    int pairIndex = i / 2;
                    double exponent = (2.0 * pairIndex) / hiddenDim;
                    double angle = pos / Math.Pow(10000.0, exponent);
                    double pe = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);

                    result[b, pos, i] = NumOps.FromDouble(
                        NumOps.ToDouble(x[b, pos, i]) + pe
                    );
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyEncoderLayer(Tensor<T> x, int layerIdx)
    {
        // Apply proper transformer encoder layer with multi-head self-attention
        return _encoderLayers[layerIdx].Forward(x);
    }

    private Tensor<T> ApplyDecoder(Tensor<T> decoderInput, Tensor<T> encoderOutput)
    {
        int batch = decoderInput.Shape[0];
        int seqLen = decoderInput.Shape[1];

        var x = decoderInput;

        // Apply proper transformer decoder layers with self-attention and cross-attention
        for (int l = 0; l < _numLayers; l++)
        {
            x = _decoderLayers[l].Forward(x, encoderOutput);
        }

        // Output projection
        var logits = new Tensor<T>(new[] { batch, seqLen, VocabularySize + 2 });

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                var feat = new Tensor<T>(new[] { 1, _hiddenDim });
                for (int h = 0; h < _hiddenDim; h++)
                {
                    feat[0, h] = x[b, t, h];
                }

                var output = _outputProjection.Forward(feat);
                for (int v = 0; v < VocabularySize + 2; v++)
                {
                    logits[b, t, v] = output[0, v];
                }
            }
        }

        return logits;
    }

    private static double GELU(double x)
    {
        double c = Math.Sqrt(2.0 / Math.PI);
        return 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        long count = _patchEmbed.GetParameterCount();
        count += _tokenEmbedding.GetParameterCount();

        foreach (var layer in _encoderLayers)
        {
            count += layer.GetParameterCount();
        }

        foreach (var layer in _decoderLayers)
        {
            count += layer.GetParameterCount();
        }

        count += _outputProjection.GetParameterCount();

        return count;
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

        // Check if this is our native format (starts with TROC magic number)
        if (File.Exists(localPath))
        {
            using var checkStream = File.OpenRead(localPath);
            using var checkReader = new BinaryReader(checkStream);
            if (checkStream.Length >= 4)
            {
                int magic = checkReader.ReadInt32();
                if (magic == 0x54524F43) // "TROC"
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

        // Map patch embedding weights
        MapConvWeights(weights, "encoder.embeddings.patch_embeddings.projection", _patchEmbed);

        // Map token embedding weights
        MapDenseWeights(weights, "decoder.embed_tokens", _tokenEmbedding);

        // Map encoder layer weights
        for (int i = 0; i < _numLayers; i++)
        {
            MapEncoderLayerWeights(weights, $"encoder.layers.{i}", _encoderLayers[i]);
        }

        // Map decoder layer weights
        for (int i = 0; i < _numLayers; i++)
        {
            MapDecoderLayerWeights(weights, $"decoder.layers.{i}", _decoderLayers[i]);
        }

        // Map output projection weights
        MapDenseWeights(weights, "lm_head", _outputProjection);
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

    private void MapEncoderLayerWeights(Dictionary<string, Tensor<float>> weights, string prefix, TrOCREncoderLayer<T> layer)
    {
        // Self-attention
        MapDenseWeights(weights, $"{prefix}.self_attn.q_proj", layer.QueryProj);
        MapDenseWeights(weights, $"{prefix}.self_attn.k_proj", layer.KeyProj);
        MapDenseWeights(weights, $"{prefix}.self_attn.v_proj", layer.ValueProj);
        MapDenseWeights(weights, $"{prefix}.self_attn.out_proj", layer.OutputProj);

        // FFN
        MapDenseWeights(weights, $"{prefix}.fc1", layer.FFN1);
        MapDenseWeights(weights, $"{prefix}.fc2", layer.FFN2);
    }

    private void MapDecoderLayerWeights(Dictionary<string, Tensor<float>> weights, string prefix, TrOCRDecoderLayer<T> layer)
    {
        // Self-attention
        MapDenseWeights(weights, $"{prefix}.self_attn.q_proj", layer.SelfQueryProj);
        MapDenseWeights(weights, $"{prefix}.self_attn.k_proj", layer.SelfKeyProj);
        MapDenseWeights(weights, $"{prefix}.self_attn.v_proj", layer.SelfValueProj);
        MapDenseWeights(weights, $"{prefix}.self_attn.out_proj", layer.SelfOutputProj);

        // Cross-attention
        MapDenseWeights(weights, $"{prefix}.encoder_attn.q_proj", layer.CrossQueryProj);
        MapDenseWeights(weights, $"{prefix}.encoder_attn.k_proj", layer.CrossKeyProj);
        MapDenseWeights(weights, $"{prefix}.encoder_attn.v_proj", layer.CrossValueProj);
        MapDenseWeights(weights, $"{prefix}.encoder_attn.out_proj", layer.CrossOutputProj);

        // FFN
        MapDenseWeights(weights, $"{prefix}.fc1", layer.FFN1);
        MapDenseWeights(weights, $"{prefix}.fc2", layer.FFN2);
    }

    private void CopyWeights(Tensor<float> source, Tensor<T> dest)
    {
        if (source.Length != dest.Length)
        {
            throw new ArgumentException(
                $"Weight shape mismatch: source has {source.Length} elements, " +
                $"destination has {dest.Length} elements. " +
                $"Source shape: [{string.Join(", ", source.Shape)}], " +
                $"Destination shape: [{string.Join(", ", dest.Shape)}]");
        }

        for (int i = 0; i < source.Length; i++)
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
        writer.Write(0x54524F43); // "TROC" in ASCII
        writer.Write(1); // Version 1
        writer.Write(Name);
        writer.Write(_hiddenDim);
        writer.Write(_numHeads);
        writer.Write(_numLayers);
        writer.Write(_patchSize);
        writer.Write(VocabularySize);
        writer.Write(_startTokenId);
        writer.Write(_endTokenId);

        // Write component weights
        _patchEmbed.WriteParameters(writer);
        _tokenEmbedding.WriteParameters(writer);

        foreach (var layer in _encoderLayers)
        {
            layer.WriteParameters(writer);
        }

        foreach (var layer in _decoderLayers)
        {
            layer.WriteParameters(writer);
        }

        _outputProjection.WriteParameters(writer);
    }

    /// <summary>
    /// Loads weights from a native TrOCR file format.
    /// </summary>
    private void LoadWeightsFromFile(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        // Read and verify header
        int magic = reader.ReadInt32();
        if (magic != 0x54524F43) // "TROC"
        {
            throw new InvalidDataException($"Invalid TrOCR model file. Expected magic 0x54524F43, got 0x{magic:X8}");
        }

        int version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported TrOCR model version: {version}");
        }

        string name = reader.ReadString();
        int hiddenDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int patchSize = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();
        int startTokenId = reader.ReadInt32();
        int endTokenId = reader.ReadInt32();

        if (hiddenDim != _hiddenDim || numHeads != _numHeads || numLayers != _numLayers || patchSize != _patchSize)
        {
            throw new InvalidOperationException(
                $"TrOCR configuration mismatch. Expected hiddenDim={_hiddenDim}, numHeads={_numHeads}, " +
                $"numLayers={_numLayers}, patchSize={_patchSize}, " +
                $"got hiddenDim={hiddenDim}, numHeads={numHeads}, numLayers={numLayers}, patchSize={patchSize}");
        }

        // Read component weights
        _patchEmbed.ReadParameters(reader);
        _tokenEmbedding.ReadParameters(reader);

        foreach (var layer in _encoderLayers)
        {
            layer.ReadParameters(reader);
        }

        foreach (var layer in _decoderLayers)
        {
            layer.ReadParameters(reader);
        }

        _outputProjection.ReadParameters(reader);
    }
}

/// <summary>
/// Transformer encoder layer with proper multi-head self-attention for TrOCR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class TrOCREncoderLayer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly double _scale;

    // Multi-head self-attention projections
    private readonly Dense<T> _queryProj;
    private readonly Dense<T> _keyProj;
    private readonly Dense<T> _valueProj;
    private readonly Dense<T> _outputProj;

    // Feed-forward network
    private readonly Dense<T> _ffn1;
    private readonly Dense<T> _ffn2;

    // Layer normalization with learnable affine parameters
    private readonly TrOCRLayerNorm<T> _norm1;
    private readonly TrOCRLayerNorm<T> _norm2;

    // Public properties for weight loading
    public Dense<T> QueryProj => _queryProj;
    public Dense<T> KeyProj => _keyProj;
    public Dense<T> ValueProj => _valueProj;
    public Dense<T> OutputProj => _outputProj;
    public Dense<T> FFN1 => _ffn1;
    public Dense<T> FFN2 => _ffn2;
    public TrOCRLayerNorm<T> Norm1 => _norm1;
    public TrOCRLayerNorm<T> Norm2 => _norm2;

    public TrOCREncoderLayer(int hiddenDim, int numHeads)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _headDim = hiddenDim / numHeads;
        _scale = 1.0 / Math.Sqrt(_headDim);

        _queryProj = new Dense<T>(hiddenDim, hiddenDim);
        _keyProj = new Dense<T>(hiddenDim, hiddenDim);
        _valueProj = new Dense<T>(hiddenDim, hiddenDim);
        _outputProj = new Dense<T>(hiddenDim, hiddenDim);

        _ffn1 = new Dense<T>(hiddenDim, hiddenDim * 4);
        _ffn2 = new Dense<T>(hiddenDim * 4, hiddenDim);

        // Layer normalization with learnable gamma/beta parameters
        _norm1 = new TrOCRLayerNorm<T>(hiddenDim);
        _norm2 = new TrOCRLayerNorm<T>(hiddenDim);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];

        // Self-attention with proper scaled dot-product attention
        var attnOut = ApplySelfAttention(x, batch, seqLen);

        // Add residual & LayerNorm with learnable parameters
        var residual1 = AddTensors(x, attnOut, batch, seqLen);
        var x1 = _norm1.Forward(residual1);

        // FFN
        var ffnOut = ApplyFFN(x1, batch, seqLen);

        // Add residual & LayerNorm with learnable parameters
        var residual2 = AddTensors(x1, ffnOut, batch, seqLen);
        var output = _norm2.Forward(residual2);

        return output;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b, int batch, int seqLen)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < seqLen; j++)
            {
                for (int k = 0; k < _hiddenDim; k++)
                {
                    result[i, j, k] = _numOps.Add(a[i, j, k], b[i, j, k]);
                }
            }
        }
        return result;
    }

    private Tensor<T> ApplySelfAttention(Tensor<T> x, int batch, int seqLen)
    {
        // Project Q, K, V
        var q = ProjectSequence(x, _queryProj);
        var k = ProjectSequence(x, _keyProj);
        var v = ProjectSequence(x, _valueProj);

        // Compute multi-head attention
        var attnOutput = ComputeMultiHeadAttention(q, k, v, batch, seqLen, seqLen);

        // Output projection
        return ProjectSequence(attnOutput, _outputProj);
    }

    private Tensor<T> ComputeMultiHeadAttention(Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batch, int queryLen, int keyLen)
    {
        var output = new Tensor<T>(new[] { batch, queryLen, _hiddenDim });

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                int headOffset = h * _headDim;

                // Compute attention scores: Q * K^T / sqrt(d_k)
                var scores = new double[queryLen, keyLen];
                for (int i = 0; i < queryLen; i++)
                {
                    for (int j = 0; j < keyLen; j++)
                    {
                        double score = 0;
                        for (int d = 0; d < _headDim; d++)
                        {
                            score += _numOps.ToDouble(q[b, i, headOffset + d]) *
                                     _numOps.ToDouble(k[b, j, headOffset + d]);
                        }
                        scores[i, j] = score * _scale;
                    }
                }

                // Softmax over key dimension
                for (int i = 0; i < queryLen; i++)
                {
                    double maxScore = double.NegativeInfinity;
                    for (int j = 0; j < keyLen; j++)
                    {
                        maxScore = Math.Max(maxScore, scores[i, j]);
                    }

                    double sumExp = 0;
                    for (int j = 0; j < keyLen; j++)
                    {
                        scores[i, j] = Math.Exp(scores[i, j] - maxScore);
                        sumExp += scores[i, j];
                    }

                    for (int j = 0; j < keyLen; j++)
                    {
                        scores[i, j] /= sumExp;
                    }
                }

                // Apply attention weights to values
                for (int i = 0; i < queryLen; i++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        double value = 0;
                        for (int j = 0; j < keyLen; j++)
                        {
                            value += scores[i, j] * _numOps.ToDouble(v[b, j, headOffset + d]);
                        }
                        output[b, i, headOffset + d] = _numOps.FromDouble(value);
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> ProjectSequence(Tensor<T> x, Dense<T> proj)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int dim = x.Shape[2];
        int outDim = proj.OutputSize;

        var result = new Tensor<T>(new[] { batch, seqLen, outDim });

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                var feat = new Tensor<T>(new[] { 1, dim });
                for (int d = 0; d < dim; d++)
                {
                    feat[0, d] = x[b, s, d];
                }

                var projected = proj.Forward(feat);
                for (int d = 0; d < outDim; d++)
                {
                    result[b, s, d] = projected[0, d];
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyFFN(Tensor<T> x, int batch, int seqLen)
    {
        int ffnDim = _ffn1.OutputSize;
        var result = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                var feat = new Tensor<T>(new[] { 1, _hiddenDim });
                for (int d = 0; d < _hiddenDim; d++)
                {
                    feat[0, d] = x[b, s, d];
                }

                // FFN1 with GELU
                var h = _ffn1.Forward(feat);
                for (int d = 0; d < ffnDim; d++)
                {
                    double val = _numOps.ToDouble(h[0, d]);
                    h[0, d] = _numOps.FromDouble(GELU(val));
                }

                // FFN2
                var output = _ffn2.Forward(h);
                for (int d = 0; d < _hiddenDim; d++)
                {
                    result[b, s, d] = output[0, d];
                }
            }
        }

        return result;
    }

    private static double GELU(double x)
    {
        double c = Math.Sqrt(2.0 / Math.PI);
        return 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
    }

    public long GetParameterCount()
    {
        return _queryProj.GetParameterCount() +
               _keyProj.GetParameterCount() +
               _valueProj.GetParameterCount() +
               _outputProj.GetParameterCount() +
               _ffn1.GetParameterCount() +
               _ffn2.GetParameterCount() +
               _norm1.GetParameterCount() +
               _norm2.GetParameterCount();
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numHeads);
        _queryProj.WriteParameters(writer);
        _keyProj.WriteParameters(writer);
        _valueProj.WriteParameters(writer);
        _outputProj.WriteParameters(writer);
        _ffn1.WriteParameters(writer);
        _ffn2.WriteParameters(writer);
        _norm1.WriteParameters(writer);
        _norm2.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        if (hiddenDim != _hiddenDim || numHeads != _numHeads)
        {
            throw new InvalidOperationException(
                $"TrOCREncoderLayer configuration mismatch. Expected hiddenDim={_hiddenDim}, numHeads={_numHeads}, " +
                $"got hiddenDim={hiddenDim}, numHeads={numHeads}");
        }
        _queryProj.ReadParameters(reader);
        _keyProj.ReadParameters(reader);
        _valueProj.ReadParameters(reader);
        _outputProj.ReadParameters(reader);
        _ffn1.ReadParameters(reader);
        _ffn2.ReadParameters(reader);
        _norm1.ReadParameters(reader);
        _norm2.ReadParameters(reader);
    }
}

/// <summary>
/// Transformer decoder layer with proper multi-head self-attention and cross-attention for TrOCR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class TrOCRDecoderLayer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly double _scale;

    // Self-attention projections
    private readonly Dense<T> _selfQueryProj;
    private readonly Dense<T> _selfKeyProj;
    private readonly Dense<T> _selfValueProj;
    private readonly Dense<T> _selfOutputProj;

    // Cross-attention projections
    private readonly Dense<T> _crossQueryProj;
    private readonly Dense<T> _crossKeyProj;
    private readonly Dense<T> _crossValueProj;
    private readonly Dense<T> _crossOutputProj;

    // Feed-forward network
    private readonly Dense<T> _ffn1;
    private readonly Dense<T> _ffn2;

    // Layer normalization with learnable affine parameters
    private readonly TrOCRLayerNorm<T> _norm1;
    private readonly TrOCRLayerNorm<T> _norm2;
    private readonly TrOCRLayerNorm<T> _norm3;

    // Public properties for weight loading - self-attention
    public Dense<T> SelfQueryProj => _selfQueryProj;
    public Dense<T> SelfKeyProj => _selfKeyProj;
    public Dense<T> SelfValueProj => _selfValueProj;
    public Dense<T> SelfOutputProj => _selfOutputProj;

    // Public properties for weight loading - cross-attention
    public Dense<T> CrossQueryProj => _crossQueryProj;
    public Dense<T> CrossKeyProj => _crossKeyProj;
    public Dense<T> CrossValueProj => _crossValueProj;
    public Dense<T> CrossOutputProj => _crossOutputProj;

    // Public properties for weight loading - FFN
    public Dense<T> FFN1 => _ffn1;
    public Dense<T> FFN2 => _ffn2;

    // Public properties for weight loading - LayerNorm
    public TrOCRLayerNorm<T> Norm1 => _norm1;
    public TrOCRLayerNorm<T> Norm2 => _norm2;
    public TrOCRLayerNorm<T> Norm3 => _norm3;

    public TrOCRDecoderLayer(int hiddenDim, int numHeads)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _headDim = hiddenDim / numHeads;
        _scale = 1.0 / Math.Sqrt(_headDim);

        // Self-attention
        _selfQueryProj = new Dense<T>(hiddenDim, hiddenDim);
        _selfKeyProj = new Dense<T>(hiddenDim, hiddenDim);
        _selfValueProj = new Dense<T>(hiddenDim, hiddenDim);
        _selfOutputProj = new Dense<T>(hiddenDim, hiddenDim);

        // Cross-attention
        _crossQueryProj = new Dense<T>(hiddenDim, hiddenDim);
        _crossKeyProj = new Dense<T>(hiddenDim, hiddenDim);
        _crossValueProj = new Dense<T>(hiddenDim, hiddenDim);
        _crossOutputProj = new Dense<T>(hiddenDim, hiddenDim);

        // FFN
        _ffn1 = new Dense<T>(hiddenDim, hiddenDim * 4);
        _ffn2 = new Dense<T>(hiddenDim * 4, hiddenDim);

        // Layer normalization with learnable gamma/beta parameters
        _norm1 = new TrOCRLayerNorm<T>(hiddenDim);
        _norm2 = new TrOCRLayerNorm<T>(hiddenDim);
        _norm3 = new TrOCRLayerNorm<T>(hiddenDim);
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> encoderOutput)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int encoderLen = encoderOutput.Shape[1];

        // Masked self-attention (causal mask for autoregressive decoding)
        var selfAttnOut = ApplyCausalSelfAttention(x, batch, seqLen);
        var residual1 = AddTensors(x, selfAttnOut, batch, seqLen);
        var x1 = _norm1.Forward(residual1);

        // Cross-attention to encoder output
        var crossAttnOut = ApplyCrossAttention(x1, encoderOutput, batch, seqLen, encoderLen);
        var residual2 = AddTensors(x1, crossAttnOut, batch, seqLen);
        var x2 = _norm2.Forward(residual2);

        // FFN
        var ffnOut = ApplyFFN(x2, batch, seqLen);
        var residual3 = AddTensors(x2, ffnOut, batch, seqLen);
        var output = _norm3.Forward(residual3);

        return output;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b, int batch, int seqLen)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < seqLen; j++)
            {
                for (int k = 0; k < _hiddenDim; k++)
                {
                    result[i, j, k] = _numOps.Add(a[i, j, k], b[i, j, k]);
                }
            }
        }
        return result;
    }

    private Tensor<T> ApplyCausalSelfAttention(Tensor<T> x, int batch, int seqLen)
    {
        // Project Q, K, V
        var q = ProjectSequence(x, _selfQueryProj);
        var k = ProjectSequence(x, _selfKeyProj);
        var v = ProjectSequence(x, _selfValueProj);

        // Compute masked attention (causal mask)
        var attnOutput = ComputeCausalAttention(q, k, v, batch, seqLen);

        // Output projection
        return ProjectSequence(attnOutput, _selfOutputProj);
    }

    private Tensor<T> ComputeCausalAttention(Tensor<T> q, Tensor<T> k, Tensor<T> v, int batch, int seqLen)
    {
        var output = new Tensor<T>(new[] { batch, seqLen, _hiddenDim });

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                int headOffset = h * _headDim;

                // Compute attention scores with causal mask
                var scores = new double[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (j > i)
                        {
                            // Future tokens are masked (set to -inf before softmax)
                            scores[i, j] = double.NegativeInfinity;
                        }
                        else
                        {
                            double score = 0;
                            for (int d = 0; d < _headDim; d++)
                            {
                                score += _numOps.ToDouble(q[b, i, headOffset + d]) *
                                         _numOps.ToDouble(k[b, j, headOffset + d]);
                            }
                            scores[i, j] = score * _scale;
                        }
                    }
                }

                // Softmax
                for (int i = 0; i < seqLen; i++)
                {
                    double maxScore = double.NegativeInfinity;
                    for (int j = 0; j <= i; j++) // Only look at non-masked positions
                    {
                        maxScore = Math.Max(maxScore, scores[i, j]);
                    }

                    double sumExp = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (j <= i)
                        {
                            scores[i, j] = Math.Exp(scores[i, j] - maxScore);
                            sumExp += scores[i, j];
                        }
                        else
                        {
                            scores[i, j] = 0; // Masked out
                        }
                    }

                    for (int j = 0; j <= i; j++)
                    {
                        scores[i, j] /= sumExp;
                    }
                }

                // Apply attention to values
                for (int i = 0; i < seqLen; i++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        double value = 0;
                        for (int j = 0; j <= i; j++)
                        {
                            value += scores[i, j] * _numOps.ToDouble(v[b, j, headOffset + d]);
                        }
                        output[b, i, headOffset + d] = _numOps.FromDouble(value);
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> ApplyCrossAttention(Tensor<T> x, Tensor<T> encoderOutput, int batch, int seqLen, int encoderLen)
    {
        // Query from decoder, Key/Value from encoder
        var q = ProjectSequence(x, _crossQueryProj);
        var k = ProjectSequence(encoderOutput, _crossKeyProj);
        var v = ProjectSequence(encoderOutput, _crossValueProj);

        // Compute cross-attention (no mask needed)
        var attnOutput = ComputeCrossAttention(q, k, v, batch, seqLen, encoderLen);

        // Output projection
        return ProjectSequence(attnOutput, _crossOutputProj);
    }

    private Tensor<T> ComputeCrossAttention(Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batch, int queryLen, int keyLen)
    {
        var output = new Tensor<T>(new[] { batch, queryLen, _hiddenDim });

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                int headOffset = h * _headDim;

                // Compute attention scores
                var scores = new double[queryLen, keyLen];
                for (int i = 0; i < queryLen; i++)
                {
                    for (int j = 0; j < keyLen; j++)
                    {
                        double score = 0;
                        for (int d = 0; d < _headDim; d++)
                        {
                            score += _numOps.ToDouble(q[b, i, headOffset + d]) *
                                     _numOps.ToDouble(k[b, j, headOffset + d]);
                        }
                        scores[i, j] = score * _scale;
                    }
                }

                // Softmax
                for (int i = 0; i < queryLen; i++)
                {
                    double maxScore = double.NegativeInfinity;
                    for (int j = 0; j < keyLen; j++)
                    {
                        maxScore = Math.Max(maxScore, scores[i, j]);
                    }

                    double sumExp = 0;
                    for (int j = 0; j < keyLen; j++)
                    {
                        scores[i, j] = Math.Exp(scores[i, j] - maxScore);
                        sumExp += scores[i, j];
                    }

                    for (int j = 0; j < keyLen; j++)
                    {
                        scores[i, j] /= sumExp;
                    }
                }

                // Apply attention to values
                for (int i = 0; i < queryLen; i++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        double value = 0;
                        for (int j = 0; j < keyLen; j++)
                        {
                            value += scores[i, j] * _numOps.ToDouble(v[b, j, headOffset + d]);
                        }
                        output[b, i, headOffset + d] = _numOps.FromDouble(value);
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> ProjectSequence(Tensor<T> x, Dense<T> proj)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int dim = x.Shape[2];
        int outDim = proj.OutputSize;

        var result = new Tensor<T>(new[] { batch, seqLen, outDim });

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                var feat = new Tensor<T>(new[] { 1, dim });
                for (int d = 0; d < dim; d++)
                {
                    feat[0, d] = x[b, s, d];
                }

                var projected = proj.Forward(feat);
                for (int d = 0; d < outDim; d++)
                {
                    result[b, s, d] = projected[0, d];
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyFFN(Tensor<T> x, int batch, int seqLen)
    {
        int ffnDim = _ffn1.OutputSize;
        var result = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                var feat = new Tensor<T>(new[] { 1, _hiddenDim });
                for (int d = 0; d < _hiddenDim; d++)
                {
                    feat[0, d] = x[b, s, d];
                }

                // FFN1 with GELU
                var h = _ffn1.Forward(feat);
                for (int d = 0; d < ffnDim; d++)
                {
                    double val = _numOps.ToDouble(h[0, d]);
                    h[0, d] = _numOps.FromDouble(GELU(val));
                }

                // FFN2
                var output = _ffn2.Forward(h);
                for (int d = 0; d < _hiddenDim; d++)
                {
                    result[b, s, d] = output[0, d];
                }
            }
        }

        return result;
    }

    private static double GELU(double x)
    {
        double c = Math.Sqrt(2.0 / Math.PI);
        return 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
    }

    public long GetParameterCount()
    {
        return _selfQueryProj.GetParameterCount() +
               _selfKeyProj.GetParameterCount() +
               _selfValueProj.GetParameterCount() +
               _selfOutputProj.GetParameterCount() +
               _crossQueryProj.GetParameterCount() +
               _crossKeyProj.GetParameterCount() +
               _crossValueProj.GetParameterCount() +
               _crossOutputProj.GetParameterCount() +
               _ffn1.GetParameterCount() +
               _ffn2.GetParameterCount() +
               _norm1.GetParameterCount() +
               _norm2.GetParameterCount() +
               _norm3.GetParameterCount();
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numHeads);
        _selfQueryProj.WriteParameters(writer);
        _selfKeyProj.WriteParameters(writer);
        _selfValueProj.WriteParameters(writer);
        _selfOutputProj.WriteParameters(writer);
        _crossQueryProj.WriteParameters(writer);
        _crossKeyProj.WriteParameters(writer);
        _crossValueProj.WriteParameters(writer);
        _crossOutputProj.WriteParameters(writer);
        _ffn1.WriteParameters(writer);
        _ffn2.WriteParameters(writer);
        _norm1.WriteParameters(writer);
        _norm2.WriteParameters(writer);
        _norm3.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        if (hiddenDim != _hiddenDim || numHeads != _numHeads)
        {
            throw new InvalidOperationException(
                $"TrOCRDecoderLayer configuration mismatch. Expected hiddenDim={_hiddenDim}, numHeads={_numHeads}, " +
                $"got hiddenDim={hiddenDim}, numHeads={numHeads}");
        }
        _selfQueryProj.ReadParameters(reader);
        _selfKeyProj.ReadParameters(reader);
        _selfValueProj.ReadParameters(reader);
        _selfOutputProj.ReadParameters(reader);
        _crossQueryProj.ReadParameters(reader);
        _crossKeyProj.ReadParameters(reader);
        _crossValueProj.ReadParameters(reader);
        _crossOutputProj.ReadParameters(reader);
        _ffn1.ReadParameters(reader);
        _ffn2.ReadParameters(reader);
        _norm1.ReadParameters(reader);
        _norm2.ReadParameters(reader);
        _norm3.ReadParameters(reader);
    }
}

/// <summary>
/// Layer normalization with learnable affine parameters for TrOCR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class TrOCRLayerNorm<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hiddenDim;
    private readonly Tensor<T> _gamma; // Scale parameter
    private readonly Tensor<T> _beta;  // Shift parameter
    private readonly double _eps;

    /// <summary>
    /// Gets the gamma (scale) parameter for weight loading.
    /// </summary>
    public Tensor<T> Gamma => _gamma;

    /// <summary>
    /// Gets the beta (shift) parameter for weight loading.
    /// </summary>
    public Tensor<T> Beta => _beta;

    public TrOCRLayerNorm(int hiddenDim, double eps = 1e-6)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _eps = eps;

        // Initialize gamma to 1 and beta to 0 (standard initialization)
        _gamma = new Tensor<T>(new[] { hiddenDim });
        _beta = new Tensor<T>(new[] { hiddenDim });

        for (int i = 0; i < hiddenDim; i++)
        {
            _gamma[i] = _numOps.FromDouble(1.0);
            _beta[i] = _numOps.FromDouble(0.0);
        }
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int hiddenDim = x.Shape[2];

        var result = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                // Compute mean
                double mean = 0;
                for (int d = 0; d < hiddenDim; d++)
                {
                    mean += _numOps.ToDouble(x[b, s, d]);
                }
                mean /= hiddenDim;

                // Compute variance
                double variance = 0;
                for (int d = 0; d < hiddenDim; d++)
                {
                    double diff = _numOps.ToDouble(x[b, s, d]) - mean;
                    variance += diff * diff;
                }
                variance /= hiddenDim;

                // Normalize and apply affine transformation
                double std = Math.Sqrt(variance + _eps);
                for (int d = 0; d < hiddenDim; d++)
                {
                    double normalized = (_numOps.ToDouble(x[b, s, d]) - mean) / std;
                    double gamma = _numOps.ToDouble(_gamma[d]);
                    double beta = _numOps.ToDouble(_beta[d]);
                    result[b, s, d] = _numOps.FromDouble(gamma * normalized + beta);
                }
            }
        }

        return result;
    }

    public long GetParameterCount()
    {
        return 2 * _hiddenDim; // gamma + beta
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        for (int i = 0; i < _hiddenDim; i++)
        {
            writer.Write(_numOps.ToDouble(_gamma[i]));
        }
        for (int i = 0; i < _hiddenDim; i++)
        {
            writer.Write(_numOps.ToDouble(_beta[i]));
        }
    }

    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        if (hiddenDim != _hiddenDim)
        {
            throw new InvalidOperationException($"TrOCRLayerNorm configuration mismatch. Expected hiddenDim={_hiddenDim}, got {hiddenDim}");
        }
        for (int i = 0; i < _hiddenDim; i++)
        {
            _gamma[i] = _numOps.FromDouble(reader.ReadDouble());
        }
        for (int i = 0; i < _hiddenDim; i++)
        {
            _beta[i] = _numOps.FromDouble(reader.ReadDouble());
        }
    }
}
