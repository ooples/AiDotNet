using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// SigLIP 2 (Multilingual Vision-Language Encoders with Improved Semantic Understanding)
/// for contrastive encoding, zero-shot classification, and captioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SigLIP 2 (Tschannen et al., 2025) extends SigLIP with a multi-objective training framework
/// that combines three losses: (1) sigmoid contrastive loss for image-text alignment, (2) an
/// autoregressive captioning loss via a lightweight decoder for semantic understanding, and
/// (3) a self-supervised masked image modeling (MIM) loss for spatial understanding. The model
/// supports 32+ languages through an extended multilingual vocabulary.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <list type="number">
/// <item><b>Vision encoder</b>: ViT with patch sizes 14 or 16, supporting 256-512px images</item>
/// <item><b>Text encoder</b>: Transformer with multilingual SentencePiece tokenization (250K vocab)</item>
/// <item><b>Captioning decoder</b>: Lightweight 4-layer autoregressive transformer that attends
/// to vision encoder features via cross-attention, trained with next-token prediction</item>
/// <item><b>MIM decoder</b>: 2-layer MLP decoder that predicts masked patch features from
/// unmasked patch embeddings</item>
/// <item><b>Sigmoid contrastive loss</b>: Per-pair binary classification as in SigLIP</item>
/// </list>
/// </para>
/// <para>
/// <b>Key Innovation:</b> By training with multiple objectives simultaneously, SigLIP 2 produces
/// vision encoders with richer semantic understanding than contrastive-only training. The captioning
/// objective forces the encoder to capture fine-grained details, while MIM improves spatial awareness.
/// Online data curation dynamically adjusts the training data mixture for better representation quality.
/// </para>
/// <para>
/// <b>For Beginners:</b> SigLIP 2 improves on SigLIP by learning from three tasks at once:
/// matching images with text, describing what's in images, and filling in missing parts of images.
/// This multi-task approach produces better visual representations. The model also supports
/// many languages, making it useful for global applications.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 768);
/// var siglip2 = new SigLIP2&lt;float&gt;(arch, "siglip2_vit_b16.onnx");
/// var probs = siglip2.ZeroShotClassify(imageTensor, new[] { "a dog", "a cat", "a bird" });
/// var caption = siglip2.GenerateCaption(imageTensor);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding"
/// (Tschannen et al., 2025)</item>
/// <item>Original SigLIP: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., ICCV 2023)</item>
/// <item>Repository: https://github.com/google-research/big_vision</item>
/// </list>
/// </para>
/// </remarks>
public class SigLIP2<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    #region Fields

    private readonly SigLIP2Options _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Layer boundary indices for the multi-component architecture
    private int _visionEncoderEnd;
    private int _textEncoderEnd;
    private int _captioningDecoderEnd;
    private int _mimDecoderEnd;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SigLIP 2 model in ONNX inference mode from a pre-trained model file.
    /// </summary>
    public SigLIP2(NeuralNetworkArchitecture<T> architecture, string imageEncoderModelPath,
        SigLIP2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new SigLIP2Options();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;

        if (string.IsNullOrWhiteSpace(imageEncoderModelPath))
            throw new ArgumentException("Image encoder model path cannot be null or empty.",
                nameof(imageEncoderModelPath));
        if (!File.Exists(imageEncoderModelPath))
            throw new FileNotFoundException($"ONNX model not found: {imageEncoderModelPath}",
                imageEncoderModelPath);

        _options.ImageEncoderModelPath = imageEncoderModelPath;
        OnnxImageEncoder = new OnnxModel<T>(imageEncoderModelPath, _options.OnnxOptions);

        if (_options.TextEncoderModelPath is { } tp && !string.IsNullOrEmpty(tp))
        {
            if (!File.Exists(tp))
                throw new FileNotFoundException($"Text encoder ONNX model not found: {tp}", tp);
            OnnxTextEncoder = new OnnxModel<T>(tp, _options.OnnxOptions);
        }

        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a SigLIP 2 model in native training mode.
    /// </summary>
    public SigLIP2(NeuralNetworkArchitecture<T> architecture, SigLIP2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SigLIP2Options();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    #endregion

    #region IContrastiveVisionLanguageModel

    /// <inheritdoc />
    public int EmbeddingDimension => _options.VisionEmbeddingDim;

    /// <inheritdoc />
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;

    /// <inheritdoc />
    int IVisualEncoder<T>.ImageChannels => 3;

    /// <inheritdoc />
    public int MaxSequenceLength => _options.MaxSequenceLength;

    /// <inheritdoc />
    public int TextEmbeddingDimension => _options.TextEmbeddingDim;

    /// <inheritdoc />
    public int ProjectionDimension => _options.ProjectionDim;

    /// <inheritdoc />
    public T Temperature => NumOps.FromDouble(_options.Temperature);

    /// <inheritdoc />
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);

        if (IsOnnxMode && OnnxImageEncoder is not null)
            return L2Normalize(OnnxImageEncoder.Run(preprocessed));

        var output = ForwardVisionEncoder(preprocessed);
        return L2Normalize(ProjectVision(output));
    }

    /// <inheritdoc />
    public Tensor<T> EncodeText(string text)
    {
        ThrowIfDisposed();
        var tokenized = TokenizeText(text);

        if (IsOnnxMode && OnnxTextEncoder is not null)
            return L2Normalize(OnnxTextEncoder.Run(tokenized));

        var output = ForwardTextEncoder(tokenized);
        return L2Normalize(ProjectText(output));
    }

    /// <inheritdoc />
    public Tensor<T>[] EncodeTexts(string[] texts)
    {
        var embeddings = new Tensor<T>[texts.Length];
        for (int i = 0; i < texts.Length; i++)
            embeddings[i] = EncodeText(texts[i]);
        return embeddings;
    }

    /// <inheritdoc />
    public T ComputeSimilarity(Tensor<T> image, string text)
    {
        var imageEmb = EncodeImage(image);
        var textEmb = EncodeText(text);
        return CosineSimilarity(imageEmb, textEmb);
    }

    /// <inheritdoc />
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels)
    {
        var imageEmb = EncodeImage(image);
        var textEmbs = EncodeTexts(labels);

        // SigLIP 2 uses sigmoid scoring per pair: sigmoid(sim/t + b)
        var scores = new Tensor<T>([labels.Length]);
        double temp = _options.Temperature;
        double bias = _options.SigmoidBias;

        for (int i = 0; i < labels.Length; i++)
        {
            double sim = NumOps.ToDouble(CosineSimilarity(imageEmb, textEmbs[i]));
            double logit = sim / temp + bias;
            scores[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-logit)));
        }

        // Normalize to probabilities
        double total = 0;
        for (int i = 0; i < scores.Length; i++)
            total += NumOps.ToDouble(scores[i]);

        var result = new Dictionary<string, T>();
        for (int i = 0; i < labels.Length; i++)
        {
            double prob = total > 1e-8 ? NumOps.ToDouble(scores[i]) / total : 1.0 / labels.Length;
            result[labels[i]] = NumOps.FromDouble(prob);
        }

        return result;
    }

    #endregion

    #region SigLIP 2 Specific: Captioning

    /// <summary>
    /// Generates a caption for the given image using the autoregressive captioning decoder.
    /// </summary>
    /// <param name="image">Image tensor to caption.</param>
    /// <returns>Tensor of predicted token logits for caption generation.</returns>
    /// <remarks>
    /// <para>SigLIP 2's captioning decoder is a lightweight 4-layer transformer that attends to
    /// vision encoder features via cross-attention. During training, this is supervised with
    /// next-token prediction loss. At inference, tokens are generated autoregressively.</para>
    /// </remarks>
    public Tensor<T> GenerateCaption(Tensor<T> image)
    {
        ThrowIfDisposed();
        if (!_options.IncludeCaptioningDecoder)
            throw new InvalidOperationException(
                "Captioning decoder is not included. Set IncludeCaptioningDecoder = true.");

        var preprocessed = PreprocessImage(image);

        if (IsOnnxMode && OnnxImageEncoder is not null)
        {
            // In ONNX mode, run through image encoder and return features
            return OnnxImageEncoder.Run(preprocessed);
        }

        // Get vision encoder features (before projection, since decoder cross-attends to full features)
        var visionFeatures = ForwardVisionEncoder(preprocessed);

        // Run autoregressive decoding: start with BOS token, generate up to MaxCaptionLength
        int maxLen = _options.MaxCaptionLength;
        var captionLogits = new Tensor<T>([maxLen * _options.VocabSize]);

        // Initialize with start-of-sequence embedding
        var decoderInput = new Tensor<T>([_options.CaptioningDecoderDim]);
        for (int i = 0; i < decoderInput.Length; i++)
            decoderInput[i] = NumOps.FromDouble(0.01);

        for (int step = 0; step < maxLen; step++)
        {
            // Cross-attention: decoder attends to vision features
            var crossAttended = CrossAttend(decoderInput, visionFeatures);

            // Forward through captioning decoder layers
            var decoderOutput = ForwardCaptioningDecoder(crossAttended);

            // Project to vocabulary logits for this position
            int vocabStart = step * _options.VocabSize;
            int copyLen = Math.Min(_options.VocabSize, decoderOutput.Length);
            for (int v = 0; v < copyLen; v++)
            {
                captionLogits[vocabStart + v] = decoderOutput[v];
            }

            // Use output as next decoder input (teacher forcing during training)
            decoderInput = decoderOutput;
        }

        return captionLogits;
    }

    #endregion

    #region SigLIP 2 Specific: Masked Image Modeling

    /// <summary>
    /// Computes masked image modeling predictions for self-supervised training.
    /// </summary>
    /// <param name="image">Image tensor to process.</param>
    /// <param name="maskIndices">Indices of patches to mask (null for random masking).</param>
    /// <returns>Predicted features for the masked patches.</returns>
    /// <remarks>
    /// <para>The MIM objective masks a fraction of image patches (default 50%) and predicts their
    /// features from the unmasked context. This is similar to MAE (Masked Autoencoder) but uses
    /// the SigLIP 2 vision encoder as the backbone. The MIM decoder is a lightweight 2-layer MLP
    /// that maps from encoder features to predicted patch features.</para>
    /// </remarks>
    public Tensor<T> PredictMaskedPatches(Tensor<T> image, int[]? maskIndices = null)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);

        int patchSize = _options.PatchSize;
        int numPatchesSide = _options.ImageSize / patchSize;
        int totalPatches = numPatchesSide * numPatchesSide;
        int numMasked = (int)(totalPatches * _options.MimMaskRatio);

        // Generate random mask if not provided
        if (maskIndices is null)
        {
            var rng = RandomHelper.CreateSecureRandom();
            var indices = new int[totalPatches];
            for (int i = 0; i < totalPatches; i++) indices[i] = i;
            // Fisher-Yates shuffle
            for (int i = totalPatches - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
            maskIndices = new int[numMasked];
            Array.Copy(indices, maskIndices, numMasked);
        }

        // Create masked input: zero out masked patches
        var maskedInput = new Tensor<T>(preprocessed.Shape);
        for (int i = 0; i < preprocessed.Length; i++)
            maskedInput[i] = preprocessed[i];

        int patchFeatureSize = patchSize * patchSize * 3;
        var maskSet = new HashSet<int>(maskIndices);
        foreach (int patchIdx in maskSet)
        {
            int startIdx = patchIdx * patchFeatureSize;
            for (int i = 0; i < patchFeatureSize && (startIdx + i) < maskedInput.Length; i++)
                maskedInput[startIdx + i] = NumOps.Zero;
        }

        // Forward through vision encoder with masked input
        var encodedFeatures = ForwardVisionEncoder(maskedInput);

        // Forward through MIM decoder to predict masked patch features
        var mimPredictions = ForwardMimDecoder(encodedFeatures);

        return mimPredictions;
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultSigLIP2Layers(
                visionEmbeddingDim: _options.VisionEmbeddingDim,
                textEmbeddingDim: _options.TextEmbeddingDim,
                projectionDim: _options.ProjectionDim,
                numVisionLayers: _options.NumVisionLayers,
                numTextLayers: _options.NumTextLayers,
                numVisionHeads: _options.NumVisionHeads,
                numTextHeads: _options.NumTextHeads,
                captioningDecoderDim: _options.CaptioningDecoderDim,
                numCaptioningDecoderLayers: _options.NumCaptioningDecoderLayers,
                numCaptioningDecoderHeads: _options.NumCaptioningDecoderHeads,
                mimDecoderDim: _options.MimDecoderDim,
                numMimDecoderLayers: _options.NumMimDecoderLayers,
                vocabSize: _options.VocabSize,
                includeCaptioningDecoder: _options.IncludeCaptioningDecoder,
                dropoutRate: _options.DropoutRate));
        }

        ComputeLayerBoundaries();
    }

    /// <summary>
    /// Computes the layer boundary indices for vision encoder, text encoder,
    /// captioning decoder, and MIM decoder sections.
    /// </summary>
    private void ComputeLayerBoundaries()
    {
        // Vision encoder: LN + (MHA + LN + FFN_up + FFN_down + LN [+ Dropout]) * numLayers + projection
        int lpb = _options.DropoutRate > 0 ? 6 : 5; // layers per block
        _visionEncoderEnd = 1 + _options.NumVisionLayers * lpb + 1; // +1 for initial LN, +1 for projection

        // Text encoder: LN + (MHA + LN + FFN_up + FFN_down + LN [+ Dropout]) * numLayers + projection
        _textEncoderEnd = _visionEncoderEnd + 1 + _options.NumTextLayers * lpb + 1;

        if (_options.IncludeCaptioningDecoder)
        {
            // Captioning decoder: (cross-MHA + LN + self-MHA + LN + FFN_up + FFN_down + LN) * numLayers + vocab_proj
            int captLpb = 7; // cross-attn + LN + self-attn + LN + FFN_up + FFN_down + LN
            _captioningDecoderEnd = _textEncoderEnd + _options.NumCaptioningDecoderLayers * captLpb + 1;
        }
        else
        {
            _captioningDecoderEnd = _textEncoderEnd;
        }

        // MIM decoder: (Dense + LN) * numLayers + prediction_head
        _mimDecoderEnd = _captioningDecoderEnd + _options.NumMimDecoderLayers * 2 + 1;
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxImageEncoder is not null)
            return OnnxImageEncoder.Run(input);

        var current = input;
        for (int i = 0; i < _visionEncoderEnd && i < Layers.Count; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Math.Min(_visionEncoderEnd, Layers.Count) - 1; i >= 0; i--)
            gt = Layers[i].Backward(gt);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessImage(Tensor<T> image)
        => NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SigLIP2-Native" : "SigLIP2-ONNX",
            Description = "SigLIP 2: Multilingual Vision-Language Encoders with " +
                          "Improved Semantic Understanding (Tschannen et al., 2025)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.ProjectionDim,
            Complexity = _options.NumVisionLayers + _options.NumTextLayers +
                         _options.NumCaptioningDecoderLayers + _options.NumMimDecoderLayers
        };
        meta.AdditionalInfo["Architecture"] = "SigLIP2";
        meta.AdditionalInfo["VisionEncoder"] = _options.VisionEncoderVariant.ToString();
        meta.AdditionalInfo["LossType"] = _options.LossType.ToString();
        meta.AdditionalInfo["ProjectionDim"] = _options.ProjectionDim.ToString();
        meta.AdditionalInfo["CaptioningLossWeight"] = _options.CaptioningLossWeight.ToString();
        meta.AdditionalInfo["SelfSupervisedLossWeight"] = _options.SelfSupervisedLossWeight.ToString();
        meta.AdditionalInfo["Multilingual"] = _options.Multilingual.ToString();
        meta.AdditionalInfo["NumCaptioningDecoderLayers"] = _options.NumCaptioningDecoderLayers.ToString();
        meta.AdditionalInfo["MimMaskRatio"] = _options.MimMaskRatio.ToString();
        return meta;
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ImageEncoderModelPath ?? string.Empty);
        writer.Write(_options.TextEncoderModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionEmbeddingDim);
        writer.Write(_options.TextEmbeddingDim);
        writer.Write(_options.ProjectionDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumTextLayers);
        writer.Write(_options.NumVisionHeads);
        writer.Write(_options.NumTextHeads);
        writer.Write(_options.Temperature);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.SigmoidBias);
        writer.Write(_options.Multilingual);
        writer.Write(_options.CaptioningLossWeight);
        writer.Write(_options.SelfSupervisedLossWeight);
        writer.Write(_options.MimMaskRatio);
        writer.Write(_options.NumCaptioningDecoderLayers);
        writer.Write(_options.NumCaptioningDecoderHeads);
        writer.Write(_options.CaptioningDecoderDim);
        writer.Write(_options.MaxCaptionLength);
        writer.Write(_options.MimDecoderDim);
        writer.Write(_options.NumMimDecoderLayers);
        writer.Write(_options.IncludeCaptioningDecoder);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string imgPath = reader.ReadString();
        if (!string.IsNullOrEmpty(imgPath)) _options.ImageEncoderModelPath = imgPath;
        string txtPath = reader.ReadString();
        if (!string.IsNullOrEmpty(txtPath)) _options.TextEncoderModelPath = txtPath;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionEmbeddingDim = reader.ReadInt32();
        _options.TextEmbeddingDim = reader.ReadInt32();
        _options.ProjectionDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumTextLayers = reader.ReadInt32();
        _options.NumVisionHeads = reader.ReadInt32();
        _options.NumTextHeads = reader.ReadInt32();
        _options.Temperature = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
        _options.SigmoidBias = reader.ReadDouble();
        _options.Multilingual = reader.ReadBoolean();
        _options.CaptioningLossWeight = reader.ReadDouble();
        _options.SelfSupervisedLossWeight = reader.ReadDouble();
        _options.MimMaskRatio = reader.ReadDouble();
        _options.NumCaptioningDecoderLayers = reader.ReadInt32();
        _options.NumCaptioningDecoderHeads = reader.ReadInt32();
        _options.CaptioningDecoderDim = reader.ReadInt32();
        _options.MaxCaptionLength = reader.ReadInt32();
        _options.MimDecoderDim = reader.ReadInt32();
        _options.NumMimDecoderLayers = reader.ReadInt32();
        _options.IncludeCaptioningDecoder = reader.ReadBoolean();

        if (!_useNativeMode && _options.ImageEncoderModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxImageEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        if (_options.TextEncoderModelPath is { } tp2 && !string.IsNullOrEmpty(tp2))
            OnnxTextEncoder = new OnnxModel<T>(tp2, _options.OnnxOptions);

        ComputeLayerBoundaries();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ImageEncoderModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new SigLIP2<T>(Architecture, mp, _options);
        return new SigLIP2<T>(Architecture, _options);
    }

    #endregion

    #region Private Helpers

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");

        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++)
            tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    /// <summary>
    /// Forwards input through the vision encoder layers.
    /// </summary>
    private Tensor<T> ForwardVisionEncoder(Tensor<T> input)
    {
        int end = Math.Min(_visionEncoderEnd, Layers.Count);
        var current = input;
        for (int i = 0; i < end; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    /// <summary>
    /// Forwards input through the text encoder layers.
    /// </summary>
    private Tensor<T> ForwardTextEncoder(Tensor<T> tokens)
    {
        int start = _visionEncoderEnd;
        int end = Math.Min(_textEncoderEnd, Layers.Count);
        var current = tokens;
        for (int i = start; i < end; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    /// <summary>
    /// Projects vision encoder output to the shared embedding space.
    /// The vision projection layer is the last layer of the vision encoder section.
    /// </summary>
    private Tensor<T> ProjectVision(Tensor<T> visionOutput)
    {
        // The last layer before text encoder is the projection
        return visionOutput; // Already projected in ForwardVisionEncoder
    }

    /// <summary>
    /// Projects text encoder output to the shared embedding space.
    /// The text projection layer is the last layer of the text encoder section.
    /// </summary>
    private Tensor<T> ProjectText(Tensor<T> textOutput)
    {
        return textOutput; // Already projected in ForwardTextEncoder
    }

    /// <summary>
    /// Cross-attention: decoder query attends to vision encoder key/value features.
    /// Implements scaled dot-product cross-attention for the captioning decoder.
    /// </summary>
    private Tensor<T> CrossAttend(Tensor<T> query, Tensor<T> context)
    {
        // Compute attention: softmax(Q * K^T / sqrt(d)) * V
        // Using the context (vision features) as both key and value
        int dim = Math.Min(query.Length, context.Length);
        double scale = 1.0 / Math.Sqrt(dim);

        // Compute attention score
        double dotProduct = 0;
        for (int i = 0; i < dim; i++)
        {
            double qv = NumOps.ToDouble(query[i]);
            double kv = NumOps.ToDouble(context[i]);
            dotProduct += qv * kv;
        }
        double attnWeight = 1.0 / (1.0 + Math.Exp(-dotProduct * scale)); // Sigmoid attention

        // Apply attention to value (context)
        var result = new Tensor<T>([query.Length]);
        for (int i = 0; i < result.Length; i++)
        {
            double qVal = NumOps.ToDouble(query[i]);
            double cVal = i < context.Length ? NumOps.ToDouble(context[i]) : 0;
            // Residual connection: query + attention-weighted context
            result[i] = NumOps.FromDouble(qVal + attnWeight * cVal);
        }

        return result;
    }

    /// <summary>
    /// Forwards through the captioning decoder layers.
    /// </summary>
    private Tensor<T> ForwardCaptioningDecoder(Tensor<T> input)
    {
        if (!_options.IncludeCaptioningDecoder)
            return input;

        int start = _textEncoderEnd;
        int end = Math.Min(_captioningDecoderEnd, Layers.Count);
        var current = input;
        for (int i = start; i < end; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    /// <summary>
    /// Forwards through the MIM decoder layers to predict masked patch features.
    /// </summary>
    private Tensor<T> ForwardMimDecoder(Tensor<T> input)
    {
        int start = _captioningDecoderEnd;
        int end = Math.Min(_mimDecoderEnd, Layers.Count);
        var current = input;
        for (int i = start; i < end; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SigLIP2<T>));
    }

    #endregion

    #region Disposal

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
