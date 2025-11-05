using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Datasets;
using AiDotNet.MetaLearning.Tasks;

namespace AiDotNet.MetaLearning.Samplers;

/// <summary>
/// Default implementation of task sampler for N-way K-shot episodic sampling.
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
public class EpisodicTaskSampler<T> : ITaskSampler<T> where T : struct
{
    private readonly IMetaDataset<T> _dataset;
    private Random _random;

    /// <inheritdoc/>
    public int NumWays { get; }

    /// <inheritdoc/>
    public int NumShots { get; }

    /// <inheritdoc/>
    public int NumQueries { get; }

    /// <inheritdoc/>
    public IMetaDataset<T> Dataset => _dataset;

    /// <summary>
    /// Creates a new episodic task sampler.
    /// </summary>
    /// <param name="dataset">The meta-learning dataset</param>
    /// <param name="numWays">Number of classes per episode (N)</param>
    /// <param name="numShots">Number of support examples per class (K)</param>
    /// <param name="numQueries">Number of query examples per class</param>
    /// <param name="seed">Random seed for reproducibility</param>
    public EpisodicTaskSampler(
        IMetaDataset<T> dataset,
        int numWays,
        int numShots,
        int numQueries,
        int? seed = null)
    {
        _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));

        if (numWays <= 0)
            throw new ArgumentException("Number of ways must be positive", nameof(numWays));
        if (numShots <= 0)
            throw new ArgumentException("Number of shots must be positive", nameof(numShots));
        if (numQueries <= 0)
            throw new ArgumentException("Number of queries must be positive", nameof(numQueries));
        if (numWays > dataset.NumClasses)
            throw new ArgumentException(
                $"Number of ways ({numWays}) cannot exceed dataset classes ({dataset.NumClasses})");

        NumWays = numWays;
        NumShots = numShots;
        NumQueries = numQueries;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc/>
    public IEpisode<T> SampleEpisode()
    {
        // Sample N random classes
        var availableClasses = _dataset.GetClassIndices();
        var selectedClasses = SampleRandomClasses(availableClasses, NumWays);

        // Prepare storage for support and query sets
        var supportDataList = new List<Tensor<T>>();
        var supportLabelsList = new List<T>();
        var queryDataList = new List<Tensor<T>>();
        var queryLabelsList = new List<T>();

        // Sample K+Q examples from each selected class
        for (int i = 0; i < selectedClasses.Length; i++)
        {
            var classIdx = selectedClasses[i];
            var classSize = _dataset.GetClassSize(classIdx);

            if (classSize < NumShots + NumQueries)
                throw new InvalidOperationException(
                    $"Class {classIdx} has only {classSize} examples, " +
                    $"but needs {NumShots + NumQueries} (shots + queries)");

            // Sample examples for this class
            var classExamples = _dataset.SampleFromClass(
                classIdx,
                NumShots + NumQueries,
                _random);

            // Split into support and query
            for (int j = 0; j < NumShots; j++)
            {
                supportDataList.Add(GetExample(classExamples, j));
                supportLabelsList.Add(ConvertToT(i)); // Use relative label 0..N-1
            }

            for (int j = NumShots; j < NumShots + NumQueries; j++)
            {
                queryDataList.Add(GetExample(classExamples, j));
                queryLabelsList.Add(ConvertToT(i)); // Use relative label 0..N-1
            }
        }

        // Stack tensors
        var supportData = StackTensors(supportDataList);
        var supportLabels = CreateLabelTensor(supportLabelsList);
        var queryData = StackTensors(queryDataList);
        var queryLabels = CreateLabelTensor(queryLabelsList);

        return new Episode<T>(
            supportData,
            supportLabels,
            queryData,
            queryLabels,
            NumWays,
            NumShots,
            NumQueries,
            GenerateTaskId(selectedClasses));
    }

    /// <inheritdoc/>
    public IEpisode<T>[] SampleBatch(int batchSize)
    {
        if (batchSize <= 0)
            throw new ArgumentException("Batch size must be positive", nameof(batchSize));

        var episodes = new IEpisode<T>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            episodes[i] = SampleEpisode();
        }
        return episodes;
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _random = new Random(seed);
    }

    private int[] SampleRandomClasses(int[] availableClasses, int count)
    {
        // Fisher-Yates shuffle to sample without replacement
        var shuffled = availableClasses.ToArray();
        for (int i = shuffled.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
        }
        return shuffled.Take(count).ToArray();
    }

    private Tensor<T> GetExample(Tensor<T> batch, int index)
    {
        // Extract a single example from a batch
        // Assumes batch shape is [N, ...] where N is batch size
        var shape = batch.Shape;
        var exampleShape = shape.Skip(1).ToArray();
        var exampleSize = exampleShape.Length > 0 ? exampleShape.Aggregate(1, (a, b) => a * b) : 1;

        var data = new T[exampleSize];
        var batchData = batch.ToVector().ToArray();
        var offset = index * exampleSize;

        Array.Copy(batchData, offset, data, 0, exampleSize);

        return new Tensor<T>(exampleShape, new Vector<T>(data));
    }

    private Tensor<T> StackTensors(List<Tensor<T>> tensors)
    {
        if (tensors.Count == 0)
            throw new ArgumentException("Cannot stack empty tensor list");

        var shape = tensors[0].Shape;
        var tensorData = tensors[0].ToVector().ToArray();
        var totalSize = tensors.Count * tensorData.Length;
        var data = new T[totalSize];

        int offset = 0;
        foreach (var tensor in tensors)
        {
            var currentData = tensor.ToVector().ToArray();
            Array.Copy(currentData, 0, data, offset, currentData.Length);
            offset += currentData.Length;
        }

        var newShape = new[] { tensors.Count }.Concat(shape).ToArray();
        return new Tensor<T>(newShape, new Vector<T>(data));
    }

    private Tensor<T> CreateLabelTensor(List<T> labels)
    {
        return new Tensor<T>(new[] { labels.Count }, new Vector<T>(labels));
    }

    private T ConvertToT(int value)
    {
        return (T)Convert.ChangeType(value, typeof(T));
    }

    private string GenerateTaskId(int[] classIndices)
    {
        return $"Task_{string.Join("_", classIndices)}";
    }
}
