using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Models;

/// <summary>
/// Adapter that wraps a CLIP model implementing <see cref="IMultimodalEmbedding{T}"/>
/// to make it servable via the REST API.
/// </summary>
/// <typeparam name="T">The numeric type used by the model.</typeparam>
/// <remarks>
/// <para>
/// This adapter bridges the gap between the core <see cref="IMultimodalEmbedding{T}"/>
/// interface and the serving infrastructure's <see cref="IServableMultimodalModel{T}"/>
/// interface, enabling CLIP models to be served via HTTP endpoints.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a translator that makes a CLIP model
/// speak the same language as the web server. The CLIP model knows how to encode
/// text and images, and this adapter packages that functionality in a way the
/// web server understands.
/// </para>
/// </remarks>
public class ServableClipModel<T> : IServableMultimodalModel<T>
{
    private readonly IMultimodalEmbedding<T> _clipModel;
    private readonly string _modelName;

    /// <summary>
    /// Initializes a new instance of the <see cref="ServableClipModel{T}"/> class.
    /// </summary>
    /// <param name="clipModel">The underlying CLIP model.</param>
    /// <param name="modelName">The name to identify this model in the serving API.</param>
    /// <exception cref="ArgumentNullException">Thrown when clipModel or modelName is null.</exception>
    public ServableClipModel(IMultimodalEmbedding<T> clipModel, string modelName)
    {
        _clipModel = clipModel ?? throw new ArgumentNullException(nameof(clipModel));
        _modelName = modelName ?? throw new ArgumentNullException(nameof(modelName));
    }

    /// <inheritdoc/>
    public string ModelName => _modelName;

    /// <inheritdoc/>
    public int InputDimension => _clipModel.ImageSize * _clipModel.ImageSize * 3; // RGB image

    /// <inheritdoc/>
    public int OutputDimension => _clipModel.EmbeddingDimension;

    /// <inheritdoc/>
    public int EmbeddingDimension => _clipModel.EmbeddingDimension;

    /// <inheritdoc/>
    public int MaxSequenceLength => _clipModel.MaxSequenceLength;

    /// <inheritdoc/>
    public int ImageSize => _clipModel.ImageSize;

    /// <inheritdoc/>
    public IReadOnlyList<Modality> SupportedModalities { get; } = new List<Modality>
    {
        Modality.Text,
        Modality.Image
    }.AsReadOnly();

    /// <summary>
    /// Performs prediction by generating an image embedding from the input vector.
    /// </summary>
    /// <param name="input">The input image data as a flattened vector.</param>
    /// <returns>The image embedding as a vector.</returns>
    /// <remarks>
    /// For CLIP models, prediction treats the input as image data and returns
    /// the image embedding. For text-based predictions, use <see cref="EncodeText"/>.
    /// </remarks>
    public Vector<T> Predict(Vector<T> input)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        // Convert vector to double array for image encoding
        var imageData = new double[input.Length];
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < input.Length; i++)
        {
            imageData[i] = numOps.ToDouble(input[i]);
        }

        return _clipModel.EncodeImage(imageData);
    }

    /// <summary>
    /// Performs batch prediction by generating image embeddings from multiple inputs.
    /// </summary>
    /// <param name="inputs">The input images as rows in a matrix.</param>
    /// <returns>A matrix where each row is an embedding for the corresponding input.</returns>
    public Matrix<T> PredictBatch(Matrix<T> inputs)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var imageBatch = new List<double[]>();

        for (int i = 0; i < inputs.Rows; i++)
        {
            var imageData = new double[inputs.Columns];
            for (int j = 0; j < inputs.Columns; j++)
            {
                imageData[j] = numOps.ToDouble(inputs[i, j]);
            }
            imageBatch.Add(imageData);
        }

        return _clipModel.EncodeImageBatch(imageBatch);
    }

    /// <inheritdoc/>
    public Vector<T> EncodeText(string text)
    {
        return _clipModel.EncodeText(text);
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeTextBatch(IEnumerable<string> texts)
    {
        return _clipModel.EncodeTextBatch(texts);
    }

    /// <inheritdoc/>
    public Vector<T> EncodeImage(double[] imageData)
    {
        return _clipModel.EncodeImage(imageData);
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeImageBatch(IEnumerable<double[]> imageDataBatch)
    {
        return _clipModel.EncodeImageBatch(imageDataBatch);
    }

    /// <inheritdoc/>
    public T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding)
    {
        return _clipModel.ComputeSimilarity(textEmbedding, imageEmbedding);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(double[] imageData, IEnumerable<string> classLabels)
    {
        return _clipModel.ZeroShotClassify(imageData, classLabels);
    }
}
