using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Datasets;

/// <summary>
/// Base class for meta-learning datasets providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
public abstract class MetaDatasetBase<T> : IMetaDataset<T> where T : struct
{
    /// <summary>
    /// Storage for class data. Dictionary maps class index to tensor of examples.
    /// </summary>
    protected Dictionary<int, Tensor<T>> ClassData { get; set; } = new();

    /// <inheritdoc/>
    public abstract int NumClasses { get; }

    /// <inheritdoc/>
    public abstract int[] DataShape { get; }

    /// <inheritdoc/>
    public DatasetSplit Split { get; protected set; }

    /// <inheritdoc/>
    public virtual Tensor<T> GetClassData(int classIndex)
    {
        if (!ClassData.ContainsKey(classIndex))
            throw new ArgumentException($"Class {classIndex} not found in dataset");

        return ClassData[classIndex];
    }

    /// <inheritdoc/>
    public virtual int GetClassSize(int classIndex)
    {
        if (!ClassData.ContainsKey(classIndex))
            throw new ArgumentException($"Class {classIndex} not found in dataset");

        return ClassData[classIndex].Shape[0];
    }

    /// <inheritdoc/>
    public virtual int[] GetClassIndices()
    {
        return ClassData.Keys.ToArray();
    }

    /// <inheritdoc/>
    public virtual Tensor<T> SampleFromClass(int classIndex, int count, Random random)
    {
        if (!ClassData.ContainsKey(classIndex))
            throw new ArgumentException($"Class {classIndex} not found in dataset");

        var classData = ClassData[classIndex];
        var classSize = classData.Shape[0];

        if (count > classSize)
            throw new ArgumentException(
                $"Cannot sample {count} examples from class {classIndex} with only {classSize} examples");

        // Sample without replacement using Fisher-Yates
        var indices = Enumerable.Range(0, classSize).ToArray();
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var selectedIndices = indices.Take(count).ToArray();
        return ExtractExamples(classData, selectedIndices);
    }

    /// <summary>
    /// Extracts specific examples from a class tensor by indices.
    /// </summary>
    /// <param name="classData">Tensor containing all class examples</param>
    /// <param name="indices">Indices of examples to extract</param>
    /// <returns>Tensor containing selected examples</returns>
    protected virtual Tensor<T> ExtractExamples(Tensor<T> classData, int[] indices)
    {
        var shape = classData.Shape;
        var exampleShape = shape.Skip(1).ToArray();
        var exampleSize = exampleShape.Length > 0 ? exampleShape.Aggregate(1, (a, b) => a * b) : 1;

        var data = new T[indices.Length * exampleSize];
        var classDataArray = classData.ToVector().ToArray();

        for (int i = 0; i < indices.Length; i++)
        {
            var srcOffset = indices[i] * exampleSize;
            var dstOffset = i * exampleSize;
            Array.Copy(classDataArray, srcOffset, data, dstOffset, exampleSize);
        }

        var newShape = new[] { indices.Length }.Concat(exampleShape).ToArray();
        return new Tensor<T>(newShape, new Vector<T>(data));
    }

    /// <summary>
    /// Loads the dataset from storage. Must be implemented by derived classes.
    /// </summary>
    protected abstract void LoadDataset();

    /// <summary>
    /// Normalizes data to [0, 1] range or standardizes it.
    /// </summary>
    /// <param name="data">Data to normalize</param>
    /// <param name="min">Minimum value (for min-max normalization)</param>
    /// <param name="max">Maximum value (for min-max normalization)</param>
    /// <returns>Normalized data</returns>
    protected Tensor<T> NormalizeData(Tensor<T> data, double min = 0.0, double max = 255.0)
    {
        var dataArray = data.ToVector().ToArray();
        var normalized = new T[dataArray.Length];
        var range = max - min;

        for (int i = 0; i < dataArray.Length; i++)
        {
            var value = Convert.ToDouble(dataArray[i]);
            var normalizedValue = (value - min) / range;
            normalized[i] = (T)Convert.ChangeType(normalizedValue, typeof(T));
        }

        return new Tensor<T>(data.Shape, new Vector<T>(normalized));
    }
}
