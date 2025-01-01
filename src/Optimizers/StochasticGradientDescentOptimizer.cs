namespace AiDotNet.Optimizers;

public class StochasticGradientDescentOptimizer<T> : OptimizerBase<T>, IGradientBasedOptimizer<T>
{
    private StochasticGradientDescentOptimizerOptions _options;

    public StochasticGradientDescentOptimizer(StochasticGradientDescentOptimizerOptions? options = null)
    {
        _options = options ?? new StochasticGradientDescentOptimizerOptions();
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, gradient);

            var currentStepData = PrepareAndEvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                break;
            }

            currentSolution = newSolution;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        Vector<T> updatedCoefficients = currentSolution.Coefficients.Subtract(gradient.Multiply(NumOps.FromDouble(_options.LearningRate)));
        return currentSolution.UpdateCoefficients(updatedCoefficients);
    }

    public Vector<T> UpdateVector(Vector<T> parameters, Vector<T> gradient)
    {
        return parameters.Subtract(gradient.Multiply(NumOps.FromDouble(_options.LearningRate)));
    }

    public Matrix<T> UpdateMatrix(Matrix<T> parameters, Matrix<T> gradient)
    {
        return parameters.Subtract(gradient.Multiply(NumOps.FromDouble(_options.LearningRate)));
    }

    public void Reset()
    {
        // SGD doesn't need to reset any state
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is StochasticGradientDescentOptimizerOptions sgdOptions)
        {
            _options = sgdOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected StochasticGradientDescentOptimizerOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _options;
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

            // Serialize SGD-specific options
            string optionsJson = JsonConvert.SerializeObject(_options);
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

            // Deserialize SGD-specific options
            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<StochasticGradientDescentOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }
}