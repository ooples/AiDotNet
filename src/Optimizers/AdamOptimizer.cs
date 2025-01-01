namespace AiDotNet.Optimizers;

public class AdamOptimizer<T> : OptimizerBase<T>, IGradientBasedOptimizer<T>
{
    private AdamOptimizerOptions _options;
    private Vector<T> _m;
    private Vector<T> _v;
    private int _t;

    public AdamOptimizer(AdamOptimizerOptions? options = null)
    {
        _options = options ?? new AdamOptimizerOptions();
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();

        _m = new Vector<T>(currentSolution.Coefficients.Length);
        _v = new Vector<T>(currentSolution.Coefficients.Length);
        _t = 0;

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _t++;
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
        if (_m == null || _v == null || _m.Length != gradient.Length)
        {
            _m = new Vector<T>(gradient.Length);
            _v = new Vector<T>(gradient.Length);
            _t = 0;
        }

        _t++;

        var updatedCoefficients = new Vector<T>(currentSolution.Coefficients.Length);

        for (int i = 0; i < gradient.Length; i++)
        {
            _m[i] = NumOps.Add(
                NumOps.Multiply(_m[i], NumOps.FromDouble(_options.Beta1)),
                NumOps.Multiply(gradient[i], NumOps.FromDouble(1 - _options.Beta1))
            );

            _v[i] = NumOps.Add(
                NumOps.Multiply(_v[i], NumOps.FromDouble(_options.Beta2)),
                NumOps.Multiply(NumOps.Multiply(gradient[i], gradient[i]), NumOps.FromDouble(1 - _options.Beta2))
            );

            T mHat = NumOps.Divide(_m[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));
            T vHat = NumOps.Divide(_v[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t)));

            T update = NumOps.Divide(
                mHat,
                NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon))
            );

            updatedCoefficients[i] = NumOps.Subtract(
                currentSolution.Coefficients[i],
                NumOps.Multiply(update, NumOps.FromDouble(_options.LearningRate))
            );
        }

        return currentSolution.UpdateCoefficients(updatedCoefficients);
    }

    public Vector<T> UpdateVector(Vector<T> parameters, Vector<T> gradient)
    {
        if (_m == null || _v == null || _m.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _v = new Vector<T>(parameters.Length);
            _t = 0;
        }

        _t++;

        for (int i = 0; i < parameters.Length; i++)
        {
            _m[i] = NumOps.Add(
                NumOps.Multiply(_m[i], NumOps.FromDouble(_options.Beta1)),
                NumOps.Multiply(gradient[i], NumOps.FromDouble(1 - _options.Beta1))
            );

            _v[i] = NumOps.Add(
                NumOps.Multiply(_v[i], NumOps.FromDouble(_options.Beta2)),
                NumOps.Multiply(NumOps.Multiply(gradient[i], gradient[i]), NumOps.FromDouble(1 - _options.Beta2))
            );

            T mHat = NumOps.Divide(_m[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));
            T vHat = NumOps.Divide(_v[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t)));

            T update = NumOps.Divide(
                mHat,
                NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon))
            );

            parameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(update, NumOps.FromDouble(_options.LearningRate))
            );
        }

        return parameters;
    }

    public Matrix<T> UpdateMatrix(Matrix<T> parameters, Matrix<T> gradient)
    {
        if (_m == null || _v == null || _m.Length != parameters.Rows * parameters.Columns)
        {
            _m = new Vector<T>(parameters.Rows * parameters.Columns);
            _v = new Vector<T>(parameters.Rows * parameters.Columns);
            _t = 0;
        }

        _t++;

        var updatedMatrix = new Matrix<T>(parameters.Rows, parameters.Columns);
        int index = 0;

        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                T g = gradient[i, j];

                _m[index] = NumOps.Add(
                    NumOps.Multiply(_m[index], NumOps.FromDouble(_options.Beta1)),
                    NumOps.Multiply(g, NumOps.FromDouble(1 - _options.Beta1))
                );

                _v[index] = NumOps.Add(
                    NumOps.Multiply(_v[index], NumOps.FromDouble(_options.Beta2)),
                    NumOps.Multiply(NumOps.Multiply(g, g), NumOps.FromDouble(1 - _options.Beta2))
                );

                T mHat = NumOps.Divide(_m[index], NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));
                T vHat = NumOps.Divide(_v[index], NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t)));

                T update = NumOps.Divide(
                    mHat,
                    NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon))
                );

                updatedMatrix[i, j] = NumOps.Subtract(
                    parameters[i, j],
                    NumOps.Multiply(update, NumOps.FromDouble(_options.LearningRate))
                );

                index++;
            }
        }

        return updatedMatrix;
    }

    public void Reset()
    {
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _t = 0;
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is AdamOptimizerOptions adamOptions)
        {
            _options = adamOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AdamOptimizerOptions.");
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

            // Serialize AdamOptimizerOptions
            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            // Serialize Adam-specific data
            writer.Write(_t);
            writer.Write(_m.Length);
            foreach (var value in _m)
            {
                writer.Write(Convert.ToDouble(value));
            }
            writer.Write(_v.Length);
            foreach (var value in _v)
            {
                writer.Write(Convert.ToDouble(value));
            }

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

            // Deserialize AdamOptimizerOptions
            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<AdamOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize Adam-specific data
            _t = reader.ReadInt32();
            int mLength = reader.ReadInt32();
            _m = new Vector<T>(mLength);
            for (int i = 0; i < mLength; i++)
            {
                _m[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            int vLength = reader.ReadInt32();
            _v = new Vector<T>(vLength);
            for (int i = 0; i < vLength; i++)
            {
                _v[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }
}