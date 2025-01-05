global using System.Collections.Concurrent;

namespace AiDotNet.Optimizers;

public class TabuSearchOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private TabuSearchOptions _tabuOptions;

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

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break; // Early stopping criteria met, exit the loop
            }
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private List<int> RandomlySelectFeatures(int totalFeatures)
    {
        var selectedFeatures = new List<int>();
        var minFeatures = Math.Max(1, (int)(totalFeatures * _tabuOptions.MinFeatureRatio));
        var maxFeatures = Math.Min(totalFeatures, (int)(totalFeatures * _tabuOptions.MaxFeatureRatio));
    
        int featureCount = _random.Next(minFeatures, maxFeatures + 1);
    
        while (selectedFeatures.Count < featureCount)
        {
            int feature = _random.Next(0, totalFeatures);
            if (!selectedFeatures.Contains(feature))
            {
                selectedFeatures.Add(feature);
            }
        }
    
        return selectedFeatures;
    }

    private List<ISymbolicModel<T>> GenerateNeighbors(ISymbolicModel<T> currentSolution)
    {
        var neighbors = new List<ISymbolicModel<T>>();
        for (int i = 0; i < _tabuOptions.NeighborhoodSize; i++)
        {
            neighbors.Add(currentSolution.Mutate(_tabuOptions.MutationRate, NumOps));
        }

        return neighbors;
    }

    private bool IsTabu(ISymbolicModel<T> solution, Queue<ISymbolicModel<T>> tabuList)
    {
        return tabuList.Any(tabuSolution => tabuSolution.Equals(solution));
    }

    private void UpdateTabuList(Queue<ISymbolicModel<T>> tabuList, ISymbolicModel<T> solution)
    {
        if (tabuList.Count >= _tabuOptions.TabuListSize)
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
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize Tabu Search-specific options
            string optionsJson = JsonConvert.SerializeObject(_tabuOptions);
            writer.Write(optionsJson);

            return ms.ToArray();
        }
    }

    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize Tabu Search-specific options
            string optionsJson = reader.ReadString();
            _tabuOptions = JsonConvert.DeserializeObject<TabuSearchOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }
}