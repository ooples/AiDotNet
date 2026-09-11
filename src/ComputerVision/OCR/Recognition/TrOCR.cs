using AiDotNet.Tensors.Engines.Autodiff;
using System.IO;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models",
    "https://arxiv.org/abs/2109.10282",
    Year = 2023,
    Authors = "Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei")]
public partial class TrOCR<T> : OCRBase<T>
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

    /// <inheritdoc />
    /// <remarks>
    /// The encoder output concatenated with the logits of the first decoding step (the decoder run
    /// on the start token, cross-attending to the encoder). Autoregressive decoding has no single
    /// fixed-shape output, and the first step is deterministic and reaches every decoder weight - the
    /// token embedding, both attentions, the FFN, the norms and the output projection - where the
    /// encoder output alone (the convention of the Document/OCR TrOCR) would leave the whole decoder
    /// untrained.
    /// </remarks>
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Greedy autoregressive generation - the standard TrOCR inference (Li et al. 2021; Hugging Face
    /// <c>generate</c>): encode the image, then decode one token at a time from the start token,
    /// feeding each step's most likely token back in, until every sequence has produced the end token
    /// or <see cref="OCROptions{T}.MaxSequenceLength"/> is reached. Returns the logits of every step,
    /// <c>[batch, steps, vocabulary]</c>; a sequence that finished early keeps receiving the end token.
    /// </para>
    /// <para>
    /// Generation is not differentiated (it is <c>no_grad</c> in every reference implementation):
    /// <see cref="Train"/> uses teacher forcing instead.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardLogits(Tensor<T> image)
        => Generate(EncodeImage(PreprocessCrop(image))).Logits;

    private Tensor<T> EncodeImage(Tensor<T> image)
    {
        // Patch embedding, flattened to a token sequence, plus positional encoding.
        var x = AddPositionalEncoding(CvTensorOps<T>.FlattenSpatial(_patchEmbed.Forward(image)));
        for (int l = 0; l < _numLayers; l++)
        {
            x = ApplyEncoderLayer(x, l);
        }

        return x;
    }

    private (string text, T confidence) DecodeText(Tensor<T> encoderOutput)
    {
        var (_, tokens, confidences) = Generate(encoderOutput);

        // Convert tokens to text (the generated tokens exclude the start and end tokens)
        var textChars = new List<char>();
        foreach (int tokenId in tokens[0])
        {
            if (tokenId > 0 && tokenId < VocabularySize && IndexToChar.TryGetValue(tokenId, out char ch))
            {
                textChars.Add(ch);
            }
        }

        string text = new string(textChars.ToArray());
        double avgConf = confidences[0].Count > 0 ? confidences[0].Average() : 0;

        return (text, NumOps.FromDouble(avgConf));
    }

    /// <summary>
    /// Greedy generation with a key/value cache.
    /// </summary>
    /// <param name="encoderOutput">The encoder output <c>[batch, patches, hidden]</c>.</param>
    /// <returns>
    /// The logits of every step <c>[batch, steps, vocabulary]</c>, and per sequence the generated
    /// tokens (without the start and end tokens) and the probability of each.
    /// </returns>
    private (Tensor<T> Logits, List<int>[] Tokens, List<double>[] Confidences) Generate(Tensor<T> encoderOutput)
    {
        using var noGrad = new NoGradScope<T>();
        int batch = encoderOutput.Shape[0];
        int vocab = VocabularySize + 2;
        int maxSteps = Math.Max(1, Options.MaxSequenceLength - 1);

        var caches = new TrOCRLayerCache<T>[_numLayers];
        for (int l = 0; l < _numLayers; l++)
        {
            caches[l] = new TrOCRLayerCache<T>();
        }

        var tokens = new List<int>[batch];
        var confidences = new List<double>[batch];
        var finished = new bool[batch];
        var current = new int[batch];
        for (int b = 0; b < batch; b++)
        {
            tokens[b] = new List<int>();
            confidences[b] = new List<double>();
            current[b] = _startTokenId;
        }

        var steps = new List<Tensor<T>>();
        for (int step = 0; step < maxSteps; step++)
        {
            var x = EmbedTokens(current.Select(t => new[] { t }).ToArray(), step);
            for (int l = 0; l < _numLayers; l++)
            {
                x = _decoderLayers[l].ForwardStep(x, encoderOutput, caches[l]);
            }

            var logits = _outputProjection.ForwardTokens(x);                      // [batch, 1, vocab]
            steps.Add(logits);

            bool allFinished = true;
            for (int b = 0; b < batch; b++)
            {
                if (finished[b])
                {
                    current[b] = _endTokenId;
                    continue;
                }

                // Softmax over this step's logits; the first most likely token wins ties.
                double max = double.NegativeInfinity;
                for (int v = 0; v < vocab; v++)
                {
                    max = Math.Max(max, NumOps.ToDouble(logits[(b * vocab) + v]));
                }

                double sum = 0;
                int best = 0;
                double bestValue = double.NegativeInfinity;
                for (int v = 0; v < vocab; v++)
                {
                    double value = NumOps.ToDouble(logits[(b * vocab) + v]);
                    sum += Math.Exp(value - max);
                    if (value > bestValue)
                    {
                        bestValue = value;
                        best = v;
                    }
                }

                if (best == _endTokenId)
                {
                    finished[b] = true;
                    current[b] = _endTokenId;
                    continue;
                }

                tokens[b].Add(best);
                confidences[b].Add(Math.Exp(bestValue - max) / sum);
                current[b] = best;
                allFinished = false;
            }

            if (allFinished)
            {
                break;
            }
        }

        var all = steps.Count == 1 ? steps[0] : Engine.TensorConcatenate(steps.ToArray(), 1);
        return (all, tokens, confidences);
    }

    /// <summary>
    /// Embeds token ids <c>[batch][length]</c> as <c>[batch, length, hidden]</c>: the learned token
    /// embedding plus the sinusoidal position encoding, positions starting at <paramref name="startPosition"/>.
    /// </summary>
    private Tensor<T> EmbedTokens(int[][] tokens, int startPosition)
    {
        int batch = tokens.Length;
        int seqLen = tokens[0].Length;
        int vocabSize = VocabularySize + 2; // +2 for start/end tokens

        // One-hot token ids through the learned embedding projection, then positional encoding.
        var oneHot = new Tensor<T>(new[] { batch, seqLen, vocabSize });
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int tokenId = MathHelper.Clamp(tokens[b][t], 0, vocabSize - 1);
                oneHot[(((b * seqLen) + t) * vocabSize) + tokenId] = NumOps.FromDouble(1.0);
            }
        }

        return AddPositionalEncoding(_tokenEmbedding.ForwardTokens(oneHot), startPosition);
    }

    private Tensor<T> AddPositionalEncoding(Tensor<T> x, int startPosition = 0)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int hiddenDim = x.Shape[2];

        // The sinusoidal table is a constant; only the ADD must stay on the tape.
        var table = new Tensor<T>(new[] { 1, seqLen, hiddenDim });
        for (int pos = 0; pos < seqLen; pos++)
        {
            for (int i = 0; i < hiddenDim; i++)
            {
                int pairIndex = i / 2;
                double exponent = (2.0 * pairIndex) / hiddenDim;
                double angle = (pos + startPosition) / Math.Pow(10000.0, exponent);
                table[(pos * hiddenDim) + i] = NumOps.FromDouble((i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle));
            }
        }

        return Engine.TensorAdd(x, Engine.TensorBroadcastTo(table, new[] { batch, seqLen, hiddenDim }));
    }

    private Tensor<T> ApplyEncoderLayer(Tensor<T> x, int layerIdx)
    {
        // Apply proper transformer encoder layer with multi-head self-attention
        return _encoderLayers[layerIdx].Forward(x);
    }

    private Tensor<T> ApplyDecoder(Tensor<T> decoderInput, Tensor<T> encoderOutput)
    {
        var x = decoderInput;
        for (int l = 0; l < _numLayers; l++)
        {
            x = _decoderLayers[l].Forward(x, encoderOutput);
        }

        return _outputProjection.ForwardTokens(x);
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
                $"Source shape: [{string.Join(", ", source._shape)}], " +
                $"Destination shape: [{string.Join(", ", dest._shape)}]");
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

        if (name != Name)
        {
            throw new InvalidOperationException(
                $"TrOCR configuration mismatch. Expected name={Name}, got name={name}");
        }

        if (hiddenDim != _hiddenDim || numHeads != _numHeads || numLayers != _numLayers || patchSize != _patchSize ||
            vocabSize != VocabularySize || startTokenId != _startTokenId || endTokenId != _endTokenId)
        {
            throw new InvalidOperationException(
                $"TrOCR configuration mismatch. Expected hiddenDim={_hiddenDim}, numHeads={_numHeads}, " +
                $"numLayers={_numLayers}, patchSize={_patchSize}, vocabSize={VocabularySize}, " +
                $"startTokenId={_startTokenId}, endTokenId={_endTokenId}, " +
                $"got hiddenDim={hiddenDim}, numHeads={numHeads}, numLayers={numLayers}, patchSize={patchSize}, " +
                $"vocabSize={vocabSize}, startTokenId={startTokenId}, endTokenId={endTokenId}");
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

    /// <summary>
    /// Runs one teacher-forced training step with cross-entropy.
    /// </summary>
    /// <param name="input">The text-line image.</param>
    /// <param name="expectedOutput">
    /// The target text as token ids <c>[batch, length]</c>, or as scores <c>[batch, length, vocabulary]</c>
    /// (such as <see cref="Predict"/>'s output shape), whose most likely token at each position is the
    /// label. Positions after the first end token are padding and are ignored.
    /// </param>
    /// <remarks>
    /// The standard TrOCR recipe (Li et al. 2021; Hugging Face <c>VisionEncoderDecoderModel</c>): the
    /// decoder reads the labels shifted right behind the start token in ONE parallel causal pass, and
    /// the loss is the cross-entropy of each position's prediction of the next label. Every decoder
    /// weight is on the gradient path - including the self-attention query and key projections, which
    /// a single-step decode could never train (one key makes the attention weight exactly 1).
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (expectedOutput is null)
        {
            throw new ArgumentNullException(nameof(expectedOutput));
        }

        var labels = LabelsFrom(expectedOutput);
        var targets = LabelTargets(labels);
        RecordTrainingLoss(TensorModelTrainer<T>.Step(
            this, input, targets, NumOps.FromDouble(TrainingLearningRate),
            image => TeacherForcedLogits(image, labels),
            CrossEntropy));
    }

    /// <summary>
    /// Decoder logits <c>[batch, length, vocabulary]</c> under teacher forcing: position t reads the
    /// start token followed by labels 0..t-1.
    /// </summary>
    private Tensor<T> TeacherForcedLogits(Tensor<T> image, int[][] labels)
    {
        var encoderOutput = EncodeImage(PreprocessCrop(image));
        if (labels.Length != encoderOutput.Shape[0])
        {
            throw new ArgumentException(
                $"The target has {labels.Length} sequences but the input has {encoderOutput.Shape[0]} images.");
        }

        var shifted = labels.Select(row => new[] { _startTokenId }.Concat(row.Take(row.Length - 1)).ToArray()).ToArray();
        return ApplyDecoder(EmbedTokens(shifted, 0), encoderOutput);
    }

    /// <summary>
    /// Reads label ids from token ids <c>[batch, length]</c> or scores <c>[batch, length, vocabulary]</c>.
    /// </summary>
    private int[][] LabelsFrom(Tensor<T> target)
    {
        int vocab = VocabularySize + 2;
        if (target.Rank == 3 && target.Shape[2] == vocab)
        {
            int batch = target.Shape[0], length = target.Shape[1];
            var labels = new int[batch][];
            for (int b = 0; b < batch; b++)
            {
                labels[b] = new int[length];
                for (int t = 0; t < length; t++)
                {
                    int best = 0;
                    double bestValue = double.NegativeInfinity;
                    for (int v = 0; v < vocab; v++)
                    {
                        double value = NumOps.ToDouble(target[(((b * length) + t) * vocab) + v]);
                        if (value > bestValue)
                        {
                            bestValue = value;
                            best = v;
                        }
                    }

                    labels[b][t] = best;
                }
            }

            return labels;
        }

        if (target.Rank == 2)
        {
            int batch = target.Shape[0], length = target.Shape[1];
            var labels = new int[batch][];
            for (int b = 0; b < batch; b++)
            {
                labels[b] = new int[length];
                for (int t = 0; t < length; t++)
                {
                    double id = Math.Round(NumOps.ToDouble(target[(b * length) + t]));
                    if (id < 0 || id >= vocab)
                    {
                        throw new ArgumentException(
                            $"Label {id} at [{b}, {t}] is outside the vocabulary [0, {vocab}).", nameof(target));
                    }

                    labels[b][t] = (int)id;
                }
            }

            return labels;
        }

        throw new ArgumentException(
            $"TrOCR training targets are token ids [batch, length] or scores [batch, length, {vocab}]; " +
            $"got [{string.Join(", ", target.Shape.ToArray())}].", nameof(target));
    }

    /// <summary>
    /// One-hot targets <c>[batch, length, vocabulary]</c>; positions after the first end token are
    /// all zero, so they drop out of the loss.
    /// </summary>
    private Tensor<T> LabelTargets(int[][] labels)
    {
        int vocab = VocabularySize + 2;
        int batch = labels.Length, length = labels[0].Length;
        var targets = new Tensor<T>(new[] { batch, length, vocab });
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < length; t++)
            {
                targets[(((b * length) + t) * vocab) + labels[b][t]] = NumOps.One;
                if (labels[b][t] == _endTokenId)
                {
                    break;
                }
            }
        }

        return targets;
    }

    /// <summary>
    /// Mean cross-entropy over the labelled positions: <c>-sum(target * log_softmax(logits)) / count</c>.
    /// </summary>
    private static Tensor<T> CrossEntropy(Tensor<T> logits, Tensor<T> oneHotTargets)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();

        double labelled = 0;
        for (int i = 0; i < oneHotTargets.Length; i++)
        {
            labelled += ops.ToDouble(oneHotTargets[i]);
        }

        var logProbabilities = engine.TensorLogSoftmax(logits, axis: -1);
        var picked = engine.ReduceSum(engine.TensorMultiply(logProbabilities, oneHotTargets), null);
        return engine.TensorMultiplyScalar(picked, ops.FromDouble(-1.0 / Math.Max(1.0, labelled)));
    }
}

/// <summary>
/// Transformer encoder layer with proper multi-head self-attention for TrOCR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class TrOCREncoderLayer<T> : CvParameterModule<T>
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
        return AiDotNetEngine.Current.TensorAdd(a, b);
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
        => CvTensorOps<T>.MultiHeadAttention(q, k, v, _numHeads, _scale);

    private Tensor<T> ProjectSequence(Tensor<T> x, Dense<T> proj) => proj.ForwardTokens(x);

    private Tensor<T> ApplyFFN(Tensor<T> x, int batch, int seqLen)
        => _ffn2.ForwardTokens(AiDotNetEngine.Current.GELU(_ffn1.ForwardTokens(x)));



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

    /// <inheritdoc />
    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _queryProj;
        yield return _keyProj;
        yield return _valueProj;
        yield return _outputProj;
        yield return _ffn1;
        yield return _ffn2;
        yield return _norm1;
        yield return _norm2;
    }
}

/// <summary>
/// Incremental-decoding state of one <see cref="TrOCRDecoderLayer{T}"/>: the self-attention keys and
/// values of every token decoded so far, and the cross-attention keys and values of the encoder output.
/// </summary>
internal sealed class TrOCRLayerCache<T>
{
    public Tensor<T>? SelfKeys { get; set; }

    public Tensor<T>? SelfValues { get; set; }

    public Tensor<T>? CrossKeys { get; set; }

    public Tensor<T>? CrossValues { get; set; }
}

/// <summary>
/// Transformer decoder layer with proper multi-head self-attention and cross-attention for TrOCR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class TrOCRDecoderLayer<T> : CvParameterModule<T>
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
        return AiDotNetEngine.Current.TensorAdd(a, b);
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
        => CvTensorOps<T>.MultiHeadAttention(q, k, v, _numHeads, _scale, causal: true);

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
        => CvTensorOps<T>.MultiHeadAttention(q, k, v, _numHeads, _scale);

    private Tensor<T> ProjectSequence(Tensor<T> x, Dense<T> proj) => proj.ForwardTokens(x);

    private Tensor<T> ApplyFFN(Tensor<T> x, int batch, int seqLen)
        => _ffn2.ForwardTokens(AiDotNetEngine.Current.GELU(_ffn1.ForwardTokens(x)));



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

    /// <inheritdoc />
    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _selfQueryProj;
        yield return _selfKeyProj;
        yield return _selfValueProj;
        yield return _selfOutputProj;
        yield return _crossQueryProj;
        yield return _crossKeyProj;
        yield return _crossValueProj;
        yield return _crossOutputProj;
        yield return _ffn1;
        yield return _ffn2;
        yield return _norm1;
        yield return _norm2;
        yield return _norm3;
    }

    /// <summary>
    /// Runs this layer for ONE new decoder token, reusing the keys and values of every earlier token.
    /// </summary>
    /// <param name="x">The new token's hidden state <c>[batch, 1, hidden]</c>.</param>
    /// <param name="encoderOutput">The encoder output <c>[batch, patches, hidden]</c>.</param>
    /// <param name="cache">This layer's cache; the new token's keys and values are appended to it.</param>
    /// <returns>The new token's output <c>[batch, 1, hidden]</c>.</returns>
    /// <remarks>
    /// Incremental decoding with a key/value cache - the standard generate path (Hugging Face
    /// <c>use_cache</c>). Because self-attention is causal, the output for the newest token equals
    /// the last position of <see cref="Forward"/> over the whole prefix; the cache just avoids
    /// recomputing the earlier positions, making each step O(prefix) instead of O(prefix squared).
    /// The encoder's cross-attention keys and values are computed once, on the first step.
    /// </remarks>
    public Tensor<T> ForwardStep(Tensor<T> x, Tensor<T> encoderOutput, TrOCRLayerCache<T> cache)
    {
        var engine = AiDotNetEngine.Current;

        var q = ProjectSequence(x, _selfQueryProj);
        var k = ProjectSequence(x, _selfKeyProj);
        var v = ProjectSequence(x, _selfValueProj);
        cache.SelfKeys = cache.SelfKeys is null ? k : engine.TensorConcatenate(new[] { cache.SelfKeys, k }, 1);
        cache.SelfValues = cache.SelfValues is null ? v : engine.TensorConcatenate(new[] { cache.SelfValues, v }, 1);

        // The newest token may attend to every cached token, so no mask is needed.
        var selfAttn = ProjectSequence(
            CvTensorOps<T>.MultiHeadAttention(q, cache.SelfKeys, cache.SelfValues, _numHeads, _scale), _selfOutputProj);
        var x1 = _norm1.Forward(engine.TensorAdd(x, selfAttn));

        cache.CrossKeys ??= ProjectSequence(encoderOutput, _crossKeyProj);
        cache.CrossValues ??= ProjectSequence(encoderOutput, _crossValueProj);
        var crossAttn = ProjectSequence(
            CvTensorOps<T>.MultiHeadAttention(ProjectSequence(x1, _crossQueryProj), cache.CrossKeys, cache.CrossValues, _numHeads, _scale),
            _crossOutputProj);
        var x2 = _norm2.Forward(engine.TensorAdd(x1, crossAttn));

        return _norm3.Forward(engine.TensorAdd(x2, ApplyFFN(x2, x2.Shape[0], 1)));
    }
}

/// <summary>
/// Layer normalization with learnable affine parameters for TrOCR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class TrOCRLayerNorm<T> : CvParameterModule<T>
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

    public Tensor<T> Forward(Tensor<T> x) => CvTensorOps<T>.LayerNormLastAxis(x, _gamma, _beta, _eps);

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

    /// <inheritdoc />
    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() => Array.Empty<IParameterSource<T>?>();

    /// <inheritdoc />
    protected override IEnumerable<Tensor<T>> OwnParameterTensors()
    {
        yield return _gamma;
        yield return _beta;
    }
}
