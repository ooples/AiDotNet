global using System.Collections.Concurrent;

namespace AiDotNet.Optimizers;

public class TabuSearchOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private TabuSearchOptions _tabuOptions;
    private double _currentMutationRate;
    private int _currentTabuListSize;
    private int _currentNeighborhoodSize;

    public TabuSearchOptimizer(
        TabuSearchOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _random = new Random();
        _tabuOptions = options ?? new TabuSearchOptions();
        InitializeAdaptiveParameters();
    }

    private new void InitializeAdaptiveParameters()
    {
        _currentMutationRate = _tabuOptions.InitialMutationRate;
        _currentTabuListSize = _tabuOptions.InitialTabuListSize;
        _currentNeighborhoodSize = _tabuOptions.InitialNeighborhoodSize;
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var tabuList = new Queue<ISymbolicModel<T>>(_tabuOptions.TabuListSize);

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            var neighbors = GenerateNeighbors(currentSolution);
            var bestNeighbor = neighbors
                .Where(n => !IsTabu(n, tabuList))
                .OrderByDescending(n => EvaluateSolution(n, inputData).FitnessScore)
                .FirstOrDefault() ?? neighbors.First();

            currentSolution = bestNeighbor;

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateTabuList(tabuList, currentSolution);
            UpdateAdaptiveParameters(iteration);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break; // Early stopping criteria met, exit the loop
            }
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private void UpdateAdaptiveParameters(int iteration)
    {
        // Update mutation rate
        _currentMutationRate *= (iteration % 2 == 0) ? _tabuOptions.MutationRateDecay : _tabuOptions.MutationRateIncrease;
        _currentMutationRate = MathHelper.Clamp(_currentMutationRate, _tabuOptions.MinMutationRate, _tabuOptions.MaxMutationRate);

        // Update tabu list size
        _currentTabuListSize = (int)(_currentTabuListSize * ((iteration % 2 == 0) ? _tabuOptions.TabuListSizeDecay : _tabuOptions.TabuListSizeIncrease));
        _currentTabuListSize = MathHelper.Clamp(_currentTabuListSize, _tabuOptions.MinTabuListSize, _tabuOptions.MaxTabuListSize);

        // Update neighborhood size
        _currentNeighborhoodSize = (int)(_currentNeighborhoodSize * ((iteration % 2 == 0) ? _tabuOptions.NeighborhoodSizeDecay : _tabuOptions.NeighborhoodSizeIncrease));
        _currentNeighborhoodSize = MathHelper.Clamp(_currentNeighborhoodSize, _tabuOptions.MinNeighborhoodSize, _tabuOptions.MaxNeighborhoodSize);
    }

    private List<ISymbolicModel<T>> GenerateNeighbors(ISymbolicModel<T> currentSolution)
    {
        var neighbors = new List<ISymbolicModel<T>>();
        for (int i = 0; i < _currentNeighborhoodSize; i++)
        {
            neighbors.Add(currentSolution.Mutate(_currentMutationRate));
        }

        return neighbors;
    }

    private bool IsTabu(ISymbolicModel<T> solution, Queue<ISymbolicModel<T>> tabuList)
    {
        return tabuList.Any(tabuSolution => tabuSolution.Equals(solution));
    }

    private void UpdateTabuList(Queue<ISymbolicModel<T>> tabuList, ISymbolicModel<T> solution)
    {
        if (tabuList.Count >= _currentTabuListSize)
        {
            tabuList.Dequeue();
        }

        tabuList.Enqueue(solution);
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is TabuSearchOptions tabuOptions)
        {
            _tabuOptions = tabuOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected TabuSearchOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _tabuOptions;
    }

    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize Tabu Search-specific options
        string optionsJson = JsonConvert.SerializeObject(_tabuOptions);
        writer.Write(optionsJson);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize Tabu Search-specific options
        string optionsJson = reader.ReadString();
        _tabuOptions = JsonConvert.DeserializeObject<TabuSearchOptions>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        // Initialize adaptive parameters after deserialization
        InitializeAdaptiveParameters();
    }
}