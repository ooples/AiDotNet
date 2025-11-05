using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Datasets;

/// <summary>
/// Synthetic dataset for testing meta-learning algorithms.
/// Generates random data with configurable number of classes and examples.
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
/// <remarks>
/// For Beginners:
/// This dataset creates fake data for testing without needing real images.
/// It's useful for:
/// - Quick prototyping and debugging
/// - Unit tests
/// - Verifying that your meta-learning pipeline works
/// Each class has a random "prototype" and examples are variations of that prototype.
/// </remarks>
public class SyntheticDataset<T> : MetaDatasetBase<T> where T : struct
{
    private readonly int _numClasses;
    private readonly int _examplesPerClass;
    private readonly int[] _dataShape;
    private readonly Random _random;

    /// <inheritdoc/>
    public override int NumClasses => _numClasses;

    /// <inheritdoc/>
    public override int[] DataShape => _dataShape;

    /// <summary>
    /// Creates a new synthetic dataset with random data.
    /// </summary>
    /// <param name="numClasses">Number of classes to generate</param>
    /// <param name="examplesPerClass">Number of examples per class</param>
    /// <param name="dataShape">Shape of each data sample (e.g., [28, 28] for images)</param>
    /// <param name="split">Dataset split (train/val/test)</param>
    /// <param name="seed">Random seed for reproducibility</param>
    public SyntheticDataset(
        int numClasses,
        int examplesPerClass,
        int[] dataShape,
        DatasetSplit split = DatasetSplit.Train,
        int? seed = null)
    {
        if (numClasses <= 0)
            throw new ArgumentException("Number of classes must be positive", nameof(numClasses));
        if (examplesPerClass <= 0)
            throw new ArgumentException("Examples per class must be positive", nameof(examplesPerClass));
        if (dataShape == null || dataShape.Length == 0)
            throw new ArgumentException("Data shape must be specified", nameof(dataShape));

        _numClasses = numClasses;
        _examplesPerClass = examplesPerClass;
        _dataShape = dataShape;
        Split = split;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        LoadDataset();
    }

    /// <inheritdoc/>
    protected override void LoadDataset()
    {
        ClassData.Clear();

        var featureSize = _dataShape.Aggregate(1, (a, b) => a * b);

        for (int classIdx = 0; classIdx < _numClasses; classIdx++)
        {
            // Generate a random prototype for this class
            var prototype = GenerateRandomVector(featureSize);

            // Generate examples as variations of the prototype
            var classExamples = new List<T[]>();
            for (int exampleIdx = 0; exampleIdx < _examplesPerClass; exampleIdx++)
            {
                var example = GenerateVariation(prototype, noiseScale: 0.1);
                classExamples.Add(example);
            }

            // Stack examples into a tensor
            var classData = StackExamples(classExamples, _dataShape);
            ClassData[classIdx] = classData;
        }
    }

    private T[] GenerateRandomVector(int size)
    {
        var data = new T[size];
        for (int i = 0; i < size; i++)
        {
            // Generate values in [0, 1] range
            var value = _random.NextDouble();
            data[i] = (T)Convert.ChangeType(value, typeof(T));
        }
        return data;
    }

    private T[] GenerateVariation(T[] prototype, double noiseScale)
    {
        var variation = new T[prototype.Length];
        for (int i = 0; i < prototype.Length; i++)
        {
            // Add Gaussian noise to prototype
            var prototypeValue = Convert.ToDouble(prototype[i]);
            var noise = GenerateGaussianNoise() * noiseScale;
            var value = Math.Max(0.0, Math.Min(1.0, prototypeValue + noise)); // Clamp to [0, 1]
            variation[i] = (T)Convert.ChangeType(value, typeof(T));
        }
        return variation;
    }

    private double GenerateGaussianNoise()
    {
        // Box-Muller transform for Gaussian noise
        var u1 = _random.NextDouble();
        var u2 = _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private Tensor<T> StackExamples(List<T[]> examples, int[] exampleShape)
    {
        var batchSize = examples.Count;
        var exampleSize = exampleShape.Aggregate(1, (a, b) => a * b);
        var totalSize = batchSize * exampleSize;

        var data = new T[totalSize];
        for (int i = 0; i < batchSize; i++)
        {
            Array.Copy(examples[i], 0, data, i * exampleSize, exampleSize);
        }

        var shape = new[] { batchSize }.Concat(exampleShape).ToArray();
        return new Tensor<T>(shape, new Vector<T>(data));
    }
}
