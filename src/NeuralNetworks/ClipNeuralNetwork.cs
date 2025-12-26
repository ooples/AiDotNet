using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// CLIP (Contrastive Language-Image Pre-training) neural network for joint text-image embeddings.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLIP was introduced by OpenAI and trained on 400 million (image, text) pairs from the internet.
/// It learns to associate images with their textual descriptions in a shared embedding space.
/// This implementation uses pre-trained ONNX models for the vision and text encoders.
/// </para>
/// <para><b>For Beginners:</b> CLIP learned by looking at millions of images with captions.
///
/// Training process (simplified):
/// 1. Show CLIP an image of a dog and the caption "a golden retriever"
/// 2. CLIP learns that these should have similar embeddings
/// 3. Show CLIP the same dog image and "a sports car"
/// 4. CLIP learns that these should have different embeddings
/// 5. Repeat 400 million times!
///
/// Now CLIP can:
/// - Match any image to any text description
/// - Classify images without seeing examples (zero-shot)
/// - Power image search engines
/// - Find the best caption for any image
///
/// This implementation loads pre-trained ONNX models, so you get all this capability
/// without having to train the model yourself!
/// </para>
/// </remarks>
public class ClipNeuralNetwork<T> : NeuralNetworkBase<T>, IMultimodalEmbedding<T>
{
    /// <summary>
    /// The ONNX inference session for the image encoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The image encoder is typically a Vision Transformer (ViT) or ResNet that processes
    /// images and outputs embedding vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "eye" of CLIP - it looks at images
    /// and converts them into numbers (vectors) that represent their visual content.
    /// </para>
    /// </remarks>
    private readonly InferenceSession _imageEncoder;

    /// <summary>
    /// The ONNX inference session for the text encoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The text encoder is typically a Transformer that processes tokenized text
    /// and outputs embedding vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "reader" of CLIP - it reads text
    /// and converts it into numbers (vectors) that represent its meaning.
    /// </para>
    /// </remarks>
    private readonly InferenceSession _textEncoder;

    /// <summary>
    /// The tokenizer for converting text to token IDs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP uses a BPE (Byte Pair Encoding) tokenizer with a vocabulary of 49408 tokens.
    /// </para>
    /// <para><b>For Beginners:</b> Before CLIP can read text, it needs to break it
    /// into pieces called "tokens". A tokenizer is like a text splitter that breaks
    /// "Hello World" into ["Hello", "World"] or even smaller pieces.
    /// </para>
    /// </remarks>
    private readonly ITokenizer _tokenizer;

    /// <summary>
    /// The optimizer used for fine-tuning the network.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The loss function for contrastive learning.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The dimensionality of the embedding space.
    /// </summary>
    private readonly int _embeddingDimension;

    /// <summary>
    /// The maximum sequence length for text input.
    /// </summary>
    private readonly int _maxSequenceLength;

    /// <summary>
    /// The expected image size (height and width).
    /// </summary>
    private readonly int _imageSize;

    /// <summary>
    /// Path to the image encoder ONNX model (stored for CreateNewInstance).
    /// </summary>
    private readonly string _imageEncoderPath;

    /// <summary>
    /// Path to the text encoder ONNX model (stored for CreateNewInstance).
    /// </summary>
    private readonly string _textEncoderPath;

    /// <summary>
    /// Gets the dimensionality of the embedding vectors produced by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Both text and image embeddings will have this same dimension.
    /// Common values are 512 (CLIP ViT-B/32) or 768 (CLIP ViT-L/14).
    /// </para>
    /// </remarks>
    public int EmbeddingDimension => _embeddingDimension;

    /// <summary>
    /// Gets the maximum number of tokens the text encoder can process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP models typically have a maximum sequence length of 77 tokens.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <summary>
    /// Gets the expected image size (height and width) for the vision encoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP models expect square images of a specific size (e.g., 224x224).
    /// </para>
    /// </remarks>
    public int ImageSize => _imageSize;

    /// <summary>
    /// Initializes a new instance of the ClipNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="imageEncoderPath">Path to the ONNX model for the image encoder.</param>
    /// <param name="textEncoderPath">Path to the ONNX model for the text encoder.</param>
    /// <param name="tokenizer">The tokenizer for text processing. Required - use ClipTokenizerFactory to create one.</param>
    /// <param name="optimizer">The optimization algorithm. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function. If null, ContrastiveLoss is used.</param>
    /// <param name="embeddingDimension">The embedding dimension. Default is 512.</param>
    /// <param name="maxSequenceLength">The maximum sequence length. Default is 77.</param>
    /// <param name="imageSize">The expected image size. Default is 224.</param>
    /// <param name="maxGradNorm">The maximum gradient norm for gradient clipping.</param>
    /// <remarks>
    /// <para>
    /// This constructor loads pre-trained ONNX models for both the vision and text encoders.
    /// The models should be exported from a pre-trained CLIP model (e.g., from HuggingFace or OpenAI).
    /// </para>
    /// <para><b>For Beginners:</b> To use CLIP, you need two model files:
    /// 1. An image encoder (converts images to vectors)
    /// 2. A text encoder (converts text to vectors)
    ///
    /// These are typically downloaded from model repositories like HuggingFace.
    /// You also need a tokenizer to break text into tokens that the model understands.
    /// </para>
    /// </remarks>
    public ClipNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string imageEncoderPath,
        string textEncoderPath,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int embeddingDimension = 512,
        int maxSequenceLength = 77,
        int imageSize = 224,
        double maxGradNorm = 1.0)
        : base(architecture,
               lossFunction ?? new ContrastiveLoss<T>(),
               maxGradNorm)
    {
        if (string.IsNullOrWhiteSpace(imageEncoderPath))
            throw new ArgumentException("Image encoder path cannot be null or empty.", nameof(imageEncoderPath));
        if (string.IsNullOrWhiteSpace(textEncoderPath))
            throw new ArgumentException("Text encoder path cannot be null or empty.", nameof(textEncoderPath));
        if (!File.Exists(imageEncoderPath))
            throw new FileNotFoundException($"Image encoder model not found: {imageEncoderPath}");
        if (!File.Exists(textEncoderPath))
            throw new FileNotFoundException($"Text encoder model not found: {textEncoderPath}");
        if (embeddingDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(embeddingDimension), "Embedding dimension must be positive.");
        if (maxSequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSequenceLength), "Max sequence length must be positive.");
        if (imageSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageSize), "Image size must be positive.");

        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _imageEncoderPath = imageEncoderPath;
        _textEncoderPath = textEncoderPath;

        // Load ONNX models
        _imageEncoder = new InferenceSession(imageEncoderPath);
        _textEncoder = new InferenceSession(textEncoderPath);

        // Tokenizer is required; validate argument
        if (tokenizer is null)
            throw new ArgumentNullException(nameof(tokenizer));

        _tokenizer = tokenizer;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new ContrastiveLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP primarily uses ONNX encoders for inference. Layers are optional projection
    /// heads that can be used for fine-tuning or feature extraction.
    /// </para>
    /// <para><b>For Beginners:</b> Unlike traditional neural networks that build layers
    /// from scratch, CLIP loads pre-trained models from ONNX files. The layers here
    /// are optional additions for customization.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // CLIP uses ONNX encoders - layers are optional projection heads
            Layers.AddRange(LayerHelper<T>.CreateDefaultClipLayers(Architecture, _embeddingDimension));
        }
    }

    /// <summary>
    /// Converts a single text string into an embedding vector.
    /// </summary>
    /// <param name="text">The text to embed.</param>
    /// <returns>A normalized embedding vector.</returns>
    /// <remarks>
    /// <para>
    /// The text is tokenized, padded/truncated to MaxSequenceLength, processed through
    /// the text encoder, and L2-normalized.
    /// </para>
    /// </remarks>
    public Vector<T> GetTextEmbedding(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Text cannot be null or empty.", nameof(text));

        // Tokenize text with CLIP-specific options
        var encodingOptions = new EncodingOptions
        {
            MaxLength = _maxSequenceLength,
            Truncation = true,
            Padding = true,
            PaddingSide = "right",
            TruncationSide = "right",
            AddSpecialTokens = true,
            ReturnAttentionMask = true
        };
        var tokenResult = _tokenizer.Encode(text, encodingOptions);

        // Create input tensor for ONNX
        var inputIds = tokenResult.TokenIds.ToArray();
        var inputTensor = new OnnxTensors.DenseTensor<long>(inputIds.Select(i => (long)i).ToArray(), new[] { 1, inputIds.Length });

        // Run text encoder
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
        };

        using var results = _textEncoder.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Convert to Vector<T> and normalize
        var embedding = ConvertToVector(outputTensor);
        return NormalizeVector(embedding);
    }

    /// <summary>
    /// Converts multiple text strings into embedding vectors in a batch operation.
    /// </summary>
    /// <param name="texts">The texts to embed.</param>
    /// <returns>A collection of normalized embedding vectors.</returns>
    public IEnumerable<Vector<T>> GetTextEmbeddings(IEnumerable<string> texts)
    {
        if (texts == null)
            throw new ArgumentNullException(nameof(texts));

        var textList = texts.ToList();
        if (textList.Count == 0)
            return Enumerable.Empty<Vector<T>>();

        // Tokenize all texts with CLIP-specific options
        var encodingOptions = new EncodingOptions
        {
            MaxLength = _maxSequenceLength,
            Truncation = true,
            Padding = true,
            PaddingSide = "right",
            TruncationSide = "right",
            AddSpecialTokens = true,
            ReturnAttentionMask = true
        };
        var tokenResults = _tokenizer.EncodeBatch(textList, encodingOptions);

        // Process batch
        var embeddings = new List<Vector<T>>();
        foreach (var tokenResult in tokenResults)
        {
            var inputIds = tokenResult.TokenIds.ToArray();
            var inputTensor = new OnnxTensors.DenseTensor<long>(inputIds.Select(i => (long)i).ToArray(), new[] { 1, inputIds.Length });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
            };

            using var results = _textEncoder.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();

            var embedding = ConvertToVector(outputTensor);
            embeddings.Add(NormalizeVector(embedding));
        }

        return embeddings;
    }

    /// <summary>
    /// Converts a single image into an embedding vector.
    /// </summary>
    /// <param name="image">The preprocessed image tensor with shape [channels, height, width].</param>
    /// <returns>A normalized embedding vector.</returns>
    /// <remarks>
    /// <para>
    /// The image should be preprocessed (resized to ImageSize, normalized) before calling this method.
    /// </para>
    /// </remarks>
    public Vector<T> GetImageEmbedding(Tensor<T> image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));

        ValidateImageShape(image);

        // Convert to ONNX tensor format
        var onnxTensor = ConvertToOnnxTensor(image);

        // Run image encoder
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", onnxTensor)
        };

        using var results = _imageEncoder.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Convert to Vector<T> and normalize
        var embedding = ConvertToVector(outputTensor);
        return NormalizeVector(embedding);
    }

    /// <summary>
    /// Converts multiple images into embedding vectors in a batch operation.
    /// </summary>
    /// <param name="images">The preprocessed image tensors.</param>
    /// <returns>A collection of normalized embedding vectors.</returns>
    public IEnumerable<Vector<T>> GetImageEmbeddings(IEnumerable<Tensor<T>> images)
    {
        if (images == null)
            throw new ArgumentNullException(nameof(images));

        var imageList = images.ToList();
        if (imageList.Count == 0)
            return Enumerable.Empty<Vector<T>>();

        var embeddings = new List<Vector<T>>();
        foreach (var image in imageList)
        {
            embeddings.Add(GetImageEmbedding(image));
        }

        return embeddings;
    }

    /// <summary>
    /// Computes the similarity between a text embedding and an image embedding.
    /// </summary>
    /// <param name="textEmbedding">The text embedding vector.</param>
    /// <param name="imageEmbedding">The image embedding vector.</param>
    /// <returns>A similarity score (cosine similarity for normalized vectors).</returns>
    /// <remarks>
    /// <para>
    /// For L2-normalized embeddings, the dot product equals the cosine similarity.
    /// Values range from -1 (opposite) to 1 (identical), with 0 indicating orthogonality.
    /// </para>
    /// </remarks>
    public T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding)
    {
        if (textEmbedding == null)
            throw new ArgumentNullException(nameof(textEmbedding));
        if (imageEmbedding == null)
            throw new ArgumentNullException(nameof(imageEmbedding));
        if (textEmbedding.Length != imageEmbedding.Length)
            throw new ArgumentException("Embedding vectors must have the same dimension.");

        // Use Engine for vectorized dot product
        return Engine.DotProduct(textEmbedding, imageEmbedding);
    }

    /// <summary>
    /// Performs zero-shot image classification by comparing an image to a set of text labels.
    /// </summary>
    /// <param name="image">The preprocessed image tensor to classify.</param>
    /// <param name="classLabels">The candidate class labels.</param>
    /// <returns>A dictionary mapping each label to its probability score.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the similarity between the image and each label,
    /// then applies softmax to convert similarities into a probability distribution.
    /// </para>
    /// <para><b>For Beginners:</b> Zero-shot means we can classify images into
    /// categories we've never trained on! Just provide text descriptions of the
    /// categories and CLIP will figure out which one matches best.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, IEnumerable<string> classLabels)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (classLabels == null)
            throw new ArgumentNullException(nameof(classLabels));

        var labels = classLabels.ToList();
        if (labels.Count == 0)
            throw new ArgumentException("At least one class label is required.", nameof(classLabels));

        // Get image embedding
        var imageEmbedding = GetImageEmbedding(image);

        // Get text embeddings for all labels (with prompt template)
        var promptedLabels = labels.Select(label => $"a photo of a {label}").ToList();
        var textEmbeddings = GetTextEmbeddings(promptedLabels).ToList();

        // Compute similarities
        var similarities = new Vector<T>(labels.Count);
        for (int i = 0; i < labels.Count; i++)
        {
            similarities[i] = ComputeSimilarity(textEmbeddings[i], imageEmbedding);
        }

        // Apply softmax using Engine
        var probabilities = Engine.Softmax(similarities);

        // Create result dictionary
        var result = new Dictionary<string, T>();
        for (int i = 0; i < labels.Count; i++)
        {
            result[labels[i]] = probabilities[i];
        }

        return result;
    }

    /// <summary>
    /// Makes a prediction using the CLIP network for the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor (image data).</param>
    /// <returns>The predicted embedding tensor.</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        IsTrainingMode = false;

        var embedding = GetImageEmbedding(input);
        var result = Tensor<T>.FromVector(embedding);

        IsTrainingMode = true;
        return result;
    }

    /// <summary>
    /// Training is not fully supported for CLIP as it uses pre-trained ONNX models.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <remarks>
    /// <para>
    /// Fine-tuning CLIP requires modifying the projection layers only, as the
    /// ONNX encoders are frozen. Full training would require the original PyTorch models.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // CLIP training with ONNX models is limited to projection layer fine-tuning
        // Full training would require the original model format
        throw new NotSupportedException(
            "Full training is not supported for CLIP with ONNX models. " +
            "Only projection layer fine-tuning is available.");
    }

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.Slice(index, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                index += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Retrieves metadata about the CLIP neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.Clip,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "EmbeddingDimension", _embeddingDimension },
                { "MaxSequenceLength", _maxSequenceLength },
                { "ImageSize", _imageSize },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
                { "TaskType", Architecture.TaskType.ToString() },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Indicates whether this network supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP with ONNX models has limited training support (projection layers only).
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Serializes CLIP-specific data to a binary writer.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embeddingDimension);
        writer.Write(_maxSequenceLength);
        writer.Write(_imageSize);
        writer.Write(_optimizer.GetType().FullName ?? "AdamOptimizer");
        writer.Write(_lossFunction.GetType().FullName ?? "ContrastiveLoss");
    }

    /// <summary>
    /// Deserializes CLIP-specific data from a binary reader.
    /// </summary>
    /// <remarks>
    /// The readonly fields (_embeddingDimension, _maxSequenceLength, _imageSize) are set
    /// in the constructor and cannot be modified. This method validates that the deserialized
    /// values match the current instance configuration.
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int embeddingDim = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        int imgSize = reader.ReadInt32();
        string optimizerType = reader.ReadString();
        string lossFunctionType = reader.ReadString();

        // Validate that loaded values match current instance
        if (embeddingDim != _embeddingDimension)
        {
            throw new InvalidOperationException(
                $"Loaded embedding dimension ({embeddingDim}) doesn't match current ({_embeddingDimension}).");
        }

        if (maxSeqLen != _maxSequenceLength)
        {
            throw new InvalidOperationException(
                $"Loaded max sequence length ({maxSeqLen}) doesn't match current ({_maxSequenceLength}).");
        }

        if (imgSize != _imageSize)
        {
            throw new InvalidOperationException(
                $"Loaded image size ({imgSize}) doesn't match current ({_imageSize}).");
        }
    }

    /// <summary>
    /// Creates a new instance of ClipNeuralNetwork with the same configuration.
    /// </summary>
    /// <returns>A new ClipNeuralNetwork instance with the same configuration.</returns>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ClipNeuralNetwork<T>(
            Architecture,
            _imageEncoderPath,
            _textEncoderPath,
            _tokenizer,
            _optimizer,
            _lossFunction,
            _embeddingDimension,
            _maxSequenceLength,
            _imageSize
        );
    }

    /// <summary>
    /// Validates that the image tensor has the expected shape.
    /// </summary>
    private void ValidateImageShape(Tensor<T> image)
    {
        if (image.Shape.Length != 3 && image.Shape.Length != 4)
            throw new ArgumentException($"Image tensor must have 3 or 4 dimensions, got {image.Shape.Length}.");

        int channels, height, width;
        if (image.Shape.Length == 3)
        {
            channels = image.Shape[0];
            height = image.Shape[1];
            width = image.Shape[2];
        }
        else
        {
            channels = image.Shape[1];
            height = image.Shape[2];
            width = image.Shape[3];
        }

        if (channels != 3)
            throw new ArgumentException($"Image must have 3 channels (RGB), got {channels}.");
        if (height != _imageSize || width != _imageSize)
            throw new ArgumentException($"Image must be {_imageSize}x{_imageSize}, got {height}x{width}.");
    }

    /// <summary>
    /// Converts the internal Tensor to an ONNX DenseTensor.
    /// </summary>
    private OnnxTensors.DenseTensor<float> ConvertToOnnxTensor(Tensor<T> tensor)
    {
        var data = tensor.Data.Select(v => NumOps.ToFloat(v)).ToArray();

        // Add batch dimension if needed
        int[] shape;
        if (tensor.Shape.Length == 3)
        {
            shape = new[] { 1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2] };
        }
        else
        {
            shape = tensor.Shape;
        }

        return new OnnxTensors.DenseTensor<float>(data, shape);
    }

    /// <summary>
    /// Converts an ONNX output tensor to a Vector.
    /// </summary>
    private Vector<T> ConvertToVector(OnnxTensors.Tensor<float> onnxTensor)
    {
        var result = new Vector<T>(_embeddingDimension);

        // Get the embedding from the last token (CLS token) for text or pooled output for images
        // The output shape is typically [batch, seq_len, hidden] or [batch, hidden]
        int startIdx = 0;
        if (onnxTensor.Dimensions.Length > 2)
        {
            // For text encoder: use the embedding at position 0 (CLS token)
            startIdx = 0;
        }

        for (int i = 0; i < _embeddingDimension && i < onnxTensor.Length; i++)
        {
            result[i] = NumOps.FromDouble(onnxTensor.GetValue(startIdx + i));
        }

        return result;
    }

    /// <summary>
    /// L2-normalizes a vector using the Engine.
    /// </summary>
    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        return vector.Normalize();
    }

    /// <summary>
    /// Disposes of the ONNX inference sessions.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _imageEncoder?.Dispose();
            _textEncoder?.Dispose();
        }
        base.Dispose(disposing);
    }
}
