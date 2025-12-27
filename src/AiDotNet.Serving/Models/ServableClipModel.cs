using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Models;

public class ServableClipModel<T> : IServableMultimodalModel<T>
{
    private readonly IMultimodalEmbedding<T> _clipModel;
    private readonly string _modelName;

    public ServableClipModel(IMultimodalEmbedding<T> clipModel, string modelName)
    {
        _clipModel = clipModel ?? throw new ArgumentNullException(nameof(clipModel));
        _modelName = modelName ?? throw new ArgumentNullException(nameof(modelName));
    }

    public string ModelName => _modelName;
    public int InputDimension => _clipModel.ImageSize * _clipModel.ImageSize * 3;
    public int OutputDimension => _clipModel.EmbeddingDimension;
    public int EmbeddingDimension => _clipModel.EmbeddingDimension;
    public int MaxSequenceLength => _clipModel.MaxSequenceLength;
    public int ImageSize => _clipModel.ImageSize;
    public IReadOnlyList<Modality> SupportedModalities { get; } = new List<Modality> { Modality.Text, Modality.Image }.AsReadOnly();

    public Vector<T> Predict(Vector<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        var imageTensor = ConvertToImageTensor(input);
        return _clipModel.GetImageEmbedding(imageTensor);
    }

    public Matrix<T> PredictBatch(Matrix<T> inputs)
    {
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        var imageTensors = new List<Tensor<T>>();
        for (int i = 0; i < inputs.Rows; i++)
        {
            var rowData = new T[inputs.Columns];
            for (int j = 0; j < inputs.Columns; j++) rowData[j] = inputs[i, j];
            var vector = new Vector<T>(rowData);
            imageTensors.Add(ConvertToImageTensor(vector));
        }
        var embeddings = _clipModel.GetImageEmbeddings(imageTensors).ToList();
        return ConvertEmbeddingsToMatrix(embeddings);
    }

    public Vector<T> EncodeText(string text) => _clipModel.GetTextEmbedding(text);
    public Matrix<T> EncodeTextBatch(IEnumerable<string> texts) => ConvertEmbeddingsToMatrix(_clipModel.GetTextEmbeddings(texts).ToList());
    public Vector<T> EncodeImage(Vector<T> imageData) => _clipModel.GetImageEmbedding(ConvertToImageTensor(imageData));
    public Matrix<T> EncodeImageBatch(IEnumerable<Vector<T>> imageDataBatch) => ConvertEmbeddingsToMatrix(_clipModel.GetImageEmbeddings(imageDataBatch.Select(ConvertToImageTensor)).ToList());
    public T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding) => _clipModel.ComputeSimilarity(textEmbedding, imageEmbedding);
    public Dictionary<string, T> ZeroShotClassify(Vector<T> imageData, IEnumerable<string> classLabels) => _clipModel.ZeroShotClassify(ConvertToImageTensor(imageData), classLabels);

    private Tensor<T> ConvertToImageTensor(Vector<T> input)
    {
        int imageSize = _clipModel.ImageSize;
        int channels = 3;
        int expectedSize = channels * imageSize * imageSize;
        if (input.Length != expectedSize) throw new ArgumentException("Input vector size mismatch", nameof(input));
        var tensor = new Tensor<T>(new[] { channels, imageSize, imageSize });
        for (int i = 0; i < input.Length; i++) tensor[i] = input[i];
        return tensor;
    }


    private Matrix<T> ConvertEmbeddingsToMatrix(List<Vector<T>> embeddings)
    {
        if (embeddings.Count == 0) return new Matrix<T>(0, _clipModel.EmbeddingDimension);
        int rows = embeddings.Count;
        int cols = embeddings[0].Length;
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = embeddings[i][j];
        return matrix;
    }
}

