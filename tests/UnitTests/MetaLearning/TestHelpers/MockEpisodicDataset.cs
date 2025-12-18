using AiDotNet.MetaLearning.Data;

namespace AiDotNet.Tests.UnitTests.MetaLearning.TestHelpers;

/// <summary>
/// Mock episodic dataset for testing meta-learning algorithms.
/// </summary>
public class MockEpisodicDataset<T, TInput, TOutput> : IEpisodicDataset<T, TInput, TOutput>
{
    private readonly int _numClasses;
    private readonly int _examplesPerClass;
    private readonly int _inputDim;
    private Random _random;

    public MockEpisodicDataset(
        int numClasses = 20,
        int examplesPerClass = 50,
        int inputDim = 10,
        DatasetSplit split = DatasetSplit.Train)
    {
        _numClasses = numClasses;
        _examplesPerClass = examplesPerClass;
        _inputDim = inputDim;
        Split = split;
        _random = new Random(42);

        ClassCounts = new Dictionary<int, int>();
        for (int i = 0; i < _numClasses; i++)
        {
            ClassCounts[i] = _examplesPerClass;
        }
    }

    public int NumClasses => _numClasses;

    public Dictionary<int, int> ClassCounts { get; }

    public DatasetSplit Split { get; }

    public ITask<T, TInput, TOutput>[] SampleTasks(
        int numTasks,
        int numWays,
        int numShots,
        int numQueryPerClass)
    {
        if (numWays > _numClasses)
        {
            throw new ArgumentException($"numWays ({numWays}) cannot exceed NumClasses ({_numClasses})");
        }

        if (numShots + numQueryPerClass > _examplesPerClass)
        {
            throw new ArgumentException(
                $"numShots ({numShots}) + numQueryPerClass ({numQueryPerClass}) " +
                $"cannot exceed examples per class ({_examplesPerClass})");
        }

        var tasks = new ITask<T, TInput, TOutput>[numTasks];

        for (int taskIdx = 0; taskIdx < numTasks; taskIdx++)
        {
            // Sample random classes for this task
            var selectedClasses = SampleRandomClasses(numWays);

            // Create support and query sets
            int supportSize = numWays * numShots;
            int querySize = numWays * numQueryPerClass;

            var supportInput = CreateInput(supportSize);
            var supportOutput = CreateOutput(supportSize);
            var queryInput = CreateInput(querySize);
            var queryOutput = CreateOutput(querySize);

            // Fill with synthetic data
            int sampleIdx = 0;
            for (int classIdx = 0; classIdx < numWays; classIdx++)
            {
                int classLabel = selectedClasses[classIdx];

                // Support set
                for (int shot = 0; shot < numShots; shot++)
                {
                    FillInputWithClassData(supportInput, sampleIdx, classLabel);
                    SetOutputLabel(supportOutput, sampleIdx, classIdx);
                    sampleIdx++;
                }
            }

            sampleIdx = 0;
            for (int classIdx = 0; classIdx < numWays; classIdx++)
            {
                int classLabel = selectedClasses[classIdx];

                // Query set
                for (int query = 0; query < numQueryPerClass; query++)
                {
                    FillInputWithClassData(queryInput, sampleIdx, classLabel);
                    SetOutputLabel(queryOutput, sampleIdx, classIdx);
                    sampleIdx++;
                }
            }

            tasks[taskIdx] = new Task<T, TInput, TOutput>(
                supportInput,
                supportOutput,
                queryInput,
                queryOutput,
                numWays,
                numShots,
                numQueryPerClass,
                $"task_{taskIdx}"
            );
        }

        return tasks;
    }

    public void SetRandomSeed(int seed)
    {
        _random = new Random(seed);
    }

    private int[] SampleRandomClasses(int numWays)
    {
        var allClasses = Enumerable.Range(0, _numClasses).ToList();
        var selected = new int[numWays];

        for (int i = 0; i < numWays; i++)
        {
            int idx = _random.Next(allClasses.Count);
            selected[i] = allClasses[idx];
            allClasses.RemoveAt(idx);
        }

        return selected;
    }

    private TInput CreateInput(int batchSize)
    {
        if (typeof(TInput) == typeof(Matrix<double>))
        {
            return (TInput)(object)new Matrix<double>(batchSize, _inputDim);
        }
        else if (typeof(TInput) == typeof(Matrix<float>))
        {
            return (TInput)(object)new Matrix<float>(batchSize, _inputDim);
        }
        else if (typeof(TInput) == typeof(Tensor<double>))
        {
            return (TInput)(object)new Tensor<double>(new[] { batchSize, _inputDim });
        }
        else if (typeof(TInput) == typeof(Tensor<float>))
        {
            return (TInput)(object)new Tensor<float>(new[] { batchSize, _inputDim });
        }
        throw new NotSupportedException($"Input type {typeof(TInput)} not supported");
    }

    private TOutput CreateOutput(int batchSize)
    {
        if (typeof(TOutput) == typeof(Vector<double>))
        {
            return (TOutput)(object)new Vector<double>(batchSize);
        }
        else if (typeof(TOutput) == typeof(Vector<float>))
        {
            return (TOutput)(object)new Vector<float>(batchSize);
        }
        throw new NotSupportedException($"Output type {typeof(TOutput)} not supported");
    }

    private void FillInputWithClassData(TInput input, int rowIdx, int classLabel)
    {
        // Fill with synthetic data based on class label
        for (int col = 0; col < _inputDim; col++)
        {
            double value = classLabel + col * 0.1 + _random.NextDouble() * 0.01;

            if (input is Matrix<double> matrixDouble)
            {
                matrixDouble[rowIdx, col] = value;
            }
            else if (input is Matrix<float> matrixFloat)
            {
                matrixFloat[rowIdx, col] = (float)value;
            }
            else if (input is Tensor<double> tensorDouble)
            {
                tensorDouble[rowIdx, col] = value;
            }
            else if (input is Tensor<float> tensorFloat)
            {
                tensorFloat[rowIdx, col] = (float)value;
            }
        }
    }

    private void SetOutputLabel(TOutput output, int idx, int label)
    {
        if (output is Vector<double> vecDouble)
        {
            vecDouble[idx] = label;
        }
        else if (output is Vector<float> vecFloat)
        {
            vecFloat[idx] = label;
        }
    }
}
