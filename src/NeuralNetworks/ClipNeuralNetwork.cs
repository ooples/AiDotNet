using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// CLIP (Contrastive Language-Image Pre-training) neural network that encodes both text
/// and images into a shared embedding space, enabling cross-modal similarity and zero-shot classification.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
public class ClipNeuralNetwork<T> : NeuralNetworkBase<T>, IMultimodalEmbedding<T>, IDisposable
{
    private string _imageEncoderPath;
    private string _textEncoderPath;
    private readonly ITokenizer _tokenizer;
    private int _embeddingDimension;
    private int _maxSequenceLength;
    private int _imageSize;
    private InferenceSession _imageSession;
    private InferenceSession _textSession;
    private bool _disposed;

    /// <summary>
    /// Gets the embedding dimension of the CLIP model.
    /// </summary>
    public int EmbeddingDimension => _embeddingDimension;

    /// <summary>
    /// Gets the maximum sequence length for text input.
    /// </summary>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <summary>
    /// Gets the expected image size (square images: ImageSize x ImageSize pixels).
    /// </summary>
    public int ImageSize => _imageSize;

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Initializes a new instance of the CLIP neural network.
    /// </summary>
    /// <param name="architecture">The network architecture configuration.</param>
    /// <param name="imageEncoderPath">Path to the ONNX image encoder model.</param>
    /// <param name="textEncoderPath">Path to the ONNX text encoder model.</param>
    /// <param name="tokenizer">The tokenizer for text processing.</param>
    /// <param name="lossFunction">The loss function (optional for inference-only use).</param>
    /// <param name="embeddingDimension">The embedding dimension (typically 512 or 768).</param>
    /// <param name="maxSequenceLength">Maximum sequence length for text (typically 77 for CLIP).</param>
    /// <param name="imageSize">Expected image size in pixels (typically 224 for CLIP).</param>
    public ClipNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string imageEncoderPath,
        string textEncoderPath,
        ITokenizer tokenizer,
        ILossFunction<T>? lossFunction = null,
        int embeddingDimension = 512,
        int maxSequenceLength = 77,
        int imageSize = 224)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(imageEncoderPath))
            throw new ArgumentException("Image encoder path cannot be null or empty.", nameof(imageEncoderPath));
        if (string.IsNullOrWhiteSpace(textEncoderPath))
            throw new ArgumentException("Text encoder path cannot be null or empty.", nameof(textEncoderPath));

        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

        if (!File.Exists(imageEncoderPath))
            throw new FileNotFoundException($"Image encoder model file not found: {imageEncoderPath}", imageEncoderPath);
        if (!File.Exists(textEncoderPath))
            throw new FileNotFoundException($"Text encoder model file not found: {textEncoderPath}", textEncoderPath);

        _imageEncoderPath = imageEncoderPath;
        _textEncoderPath = textEncoderPath;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;

        using (var sessionOptions = new SessionOptions())
        {
            try
            {
                _imageSession = new InferenceSession(imageEncoderPath, sessionOptions);
                try
                {
                    _textSession = new InferenceSession(textEncoderPath, sessionOptions);
                }
                catch
                {
                    _imageSession.Dispose();
                    throw;
                }
            }
            catch
            {
                throw;
            }
        }

        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        ClearLayers();
    }

    public override AiDotNet.Tensors.LinearAlgebra.Tensor<T> Predict(AiDotNet.Tensors.LinearAlgebra.Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        var imageData = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            imageData[i] = NumOps.ToDouble(input.Data.Span[i]);
        }

        var embedding = EncodeImage(imageData);
        var result = new AiDotNet.Tensors.LinearAlgebra.Tensor<T>(new[] { 1, _embeddingDimension });
        for (int i = 0; i < _embeddingDimension; i++)
        {
            result[0, i] = embedding[i];
        }

        return result;
    }

    public override void Train(AiDotNet.Tensors.LinearAlgebra.Tensor<T> input, AiDotNet.Tensors.LinearAlgebra.Tensor<T> expectedOutput)
    {
        throw new NotSupportedException("CLIP neural network in inference mode does not support training.");
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        // No-op for inference mode
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "ClipNeuralNetwork",
            ModelType = ModelType.Transformer,
            FeatureCount = 3 * _imageSize * _imageSize,
            Complexity = _embeddingDimension * _maxSequenceLength,
            Description = $"CLIP multimodal embedding model with {_embeddingDimension}-dimensional embeddings",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "EmbeddingDimension", _embeddingDimension },
                { "MaxSequenceLength", _maxSequenceLength },
                { "ImageSize", _imageSize },
                { "ImageEncoderPath", _imageEncoderPath },
                { "TextEncoderPath", _textEncoderPath }
            }
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_imageEncoderPath);
        writer.Write(_textEncoderPath);
        writer.Write(_embeddingDimension);
        writer.Write(_maxSequenceLength);
        writer.Write(_imageSize);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _imageEncoderPath = reader.ReadString();
        _textEncoderPath = reader.ReadString();
        _embeddingDimension = reader.ReadInt32();
        _maxSequenceLength = reader.ReadInt32();
        _imageSize = reader.ReadInt32();

        // Re-initialize sessions with loaded paths
        _imageSession.Dispose();
        _textSession.Dispose();

        var sessionOptions = new SessionOptions();
        _imageSession = new InferenceSession(_imageEncoderPath, sessionOptions);
        _textSession = new InferenceSession(_textEncoderPath, sessionOptions);
    }

    protected override IFullModel<T, AiDotNet.Tensors.LinearAlgebra.Tensor<T>, AiDotNet.Tensors.LinearAlgebra.Tensor<T>> CreateNewInstance()
    {
        return new ClipNeuralNetwork<T>(
            Architecture,
            _imageEncoderPath,
            _textEncoderPath,
            _tokenizer,
            LossFunction,
            _embeddingDimension,
            _maxSequenceLength,
            _imageSize);
    }

    /// <inheritdoc/>
    public Vector<T> EncodeText(string text)
    {
        var tokens = _tokenizer.Encode(text).TokenIds.ToArray();
        var attentionMask = new long[_maxSequenceLength];
        for (int i = 0; i < Math.Min(tokens.Length, _maxSequenceLength); i++)
            attentionMask[i] = 1;

        return GenerateTextEmbedding(PadOrTruncateTokens(tokens, _maxSequenceLength), attentionMask);
    }

    /// <inheritdoc/>
    public Task<Vector<T>> EmbedAsync(string text)
    {
        return Task.FromResult(EncodeText(text));
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeTextBatch(IEnumerable<string> texts)
    {
        var textList = texts.ToList();
        var result = new Matrix<T>(textList.Count, _embeddingDimension);
        for (int i = 0; i < textList.Count; i++)
        {
            var emb = EncodeText(textList[i]);
            for (int j = 0; j < _embeddingDimension; j++) result[i, j] = emb[j];
        }
        return result;
    }

    /// <inheritdoc/>
    public Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts)
    {
        return Task.FromResult(EncodeTextBatch(texts));
    }


    public Vector<T> EncodeImage(double[] imageData)
    {
        if (imageData == null || imageData.Length == 0)
            throw new ArgumentException("Image data cannot be null or empty", nameof(imageData));

        int expectedSize = 3 * _imageSize * _imageSize;
        if (imageData.Length != expectedSize)
            throw new ArgumentException($"Image data has {imageData.Length} elements but expected {expectedSize}", nameof(imageData));

        var embedding = GenerateImageEmbedding(imageData);
        return embedding.SafeNormalize();
    }

    public Matrix<T> EncodeImageBatch(IEnumerable<double[]> imageDataBatch)
    {
        if (imageDataBatch == null) throw new ArgumentNullException(nameof(imageDataBatch));
        var imageList = imageDataBatch.ToList();
        if (imageList.Count == 0) return new Matrix<T>(0, _embeddingDimension);

        var embeddings = new T[imageList.Count, _embeddingDimension];
        for (int i = 0; i < imageList.Count; i++)
        {
            var embedding = EncodeImage(imageList[i]);
            for (int j = 0; j < _embeddingDimension; j++)
                embeddings[i, j] = embedding[j];
        }

        return new Matrix<T>(embeddings);
    }

    public T ComputeSimilarity(Vector<T> embedding1, Vector<T> embedding2)
    {
        if (embedding1 == null || embedding2 == null)
            throw new ArgumentNullException(embedding1 == null ? nameof(embedding1) : nameof(embedding2));

        if (embedding1.Length != embedding2.Length)
            throw new ArgumentException($"Embeddings have different lengths: {embedding1.Length} vs {embedding2.Length}");

        return embedding1.DotProduct(embedding2);
    }

    public Dictionary<string, T> ZeroShotClassify(double[] imageData, IEnumerable<string> labels)
    {
        if (imageData == null || imageData.Length == 0)
            throw new ArgumentException("Image data cannot be null or empty", nameof(imageData));

        var labelList = labels?.ToList() ?? throw new ArgumentNullException(nameof(labels));
        if (labelList.Count == 0)
            throw new ArgumentException("Labels collection cannot be empty", nameof(labels));

        var imageEmbedding = EncodeImage(imageData);
        var textEmbeddings = labelList.Select(l => EncodeText($"a photo of a {l}")).ToList();

        var similarities = new double[labelList.Count];
        for (int i = 0; i < labelList.Count; i++)
        {
            similarities[i] = NumOps.ToDouble(ComputeSimilarity(imageEmbedding, textEmbeddings[i]));
        }

        var probabilities = Softmax(similarities);
        var result = new Dictionary<string, T>();
        for (int i = 0; i < labelList.Count; i++)
        {
            result[labelList[i]] = NumOps.FromDouble(probabilities[i]);
        }

        return result;
    }

    private Vector<T> GenerateTextEmbedding(int[] tokens, long[] attentionMask)
    {
        var inputIds = tokens.Select(t => (long)t).ToArray();
        var shape = new[] { 1, _maxSequenceLength };

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(inputIds, shape)),
            NamedOnnxValue.CreateFromTensor("attention_mask", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(attentionMask, shape))
        };

        using var results = _textSession.Run(inputs);
        var output = results.FirstOrDefault(r =>
            r.Name == "text_embeds" || r.Name == "pooler_output" || r.Name == "last_hidden_state")?.AsTensor<float>();

        if (output == null)
            throw new InvalidOperationException("Could not find suitable output in text encoder model.");

        var embedding = new T[_embeddingDimension];
        if (output.Dimensions.Length == 3)
        {
            int dim = output.Dimensions[2];
            if (dim < _embeddingDimension)
            {
                System.Diagnostics.Debug.WriteLine($"Warning: ONNX model output dimension ({dim}) is smaller than configured EmbeddingDimension ({_embeddingDimension}). Remaining values will be zero.");
            }

            for (int i = 0; i < Math.Min(_embeddingDimension, dim); i++)
                embedding[i] = NumOps.FromDouble(output[0, 0, i]);
        }
        else
        {
            int dim = output.Dimensions[1];
            if (dim < _embeddingDimension)
            {
                System.Diagnostics.Debug.WriteLine($"Warning: ONNX model output dimension ({dim}) is smaller than configured EmbeddingDimension ({_embeddingDimension}). Remaining values will be zero.");
            }

            for (int i = 0; i < Math.Min(_embeddingDimension, dim); i++)
                embedding[i] = NumOps.FromDouble(output[0, i]);
        }

        return new Vector<T>(embedding);
    }

    private Vector<T> GenerateImageEmbedding(double[] imageData)
    {
        var floatData = imageData.Select(d => (float)d).ToArray();
        var shape = new[] { 1, 3, _imageSize, _imageSize };
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(floatData, shape))
        };

        using var results = _imageSession.Run(inputs);
        var output = results.FirstOrDefault(r =>
            r.Name == "image_embeds" || r.Name == "pooler_output" || r.Name == "last_hidden_state")?.AsTensor<float>();

        if (output == null)
            throw new InvalidOperationException("Could not find suitable output in image encoder model.");

        var embedding = new T[_embeddingDimension];
        if (output.Dimensions.Length == 3)
        {
            int dim = output.Dimensions[2];
            if (dim < _embeddingDimension)
            {
                System.Diagnostics.Debug.WriteLine($"Warning: ONNX model output dimension ({dim}) is smaller than configured EmbeddingDimension ({_embeddingDimension}). Remaining values will be zero.");
            }

            for (int i = 0; i < Math.Min(_embeddingDimension, dim); i++)
                embedding[i] = NumOps.FromDouble(output[0, 0, i]);
        }
        else
        {
            int dim = output.Dimensions[1];
            if (dim < _embeddingDimension)
            {
                System.Diagnostics.Debug.WriteLine($"Warning: ONNX model output dimension ({dim}) is smaller than configured EmbeddingDimension ({_embeddingDimension}). Remaining values will be zero.");
            }

            for (int i = 0; i < Math.Min(_embeddingDimension, dim); i++)
                embedding[i] = NumOps.FromDouble(output[0, i]);
        }

        return new Vector<T>(embedding);
    }

    private static int[] PadOrTruncateTokens(int[] tokens, int targetLength)
    {
        var result = new int[targetLength];
        Array.Copy(tokens, result, Math.Min(tokens.Length, targetLength));
        return result;
    }

    private static double[] Softmax(double[] scores)
    {
        double temperature = 100.0;
        double max = scores.Max();
        var expScores = scores.Select(s => Math.Exp((s - max) * temperature)).ToArray();
        double sumExp = expScores.Sum();
        return expScores.Select(e => e / sumExp).ToArray();
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _imageSession?.Dispose();
                _textSession?.Dispose();
            }

            _disposed = true;
        }

        base.Dispose(disposing);
    }
}


