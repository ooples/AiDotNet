namespace AiDotNet.Regression;

public class GeneralizedAdditiveModel<T> : RegressionBase<T>
{
    private readonly GeneralizedAdditiveModelOptions<T> _options;
    private Matrix<T> _basisFunctions;
    private Vector<T> _coefficients;

    public GeneralizedAdditiveModel(
        GeneralizedAdditiveModelOptions<T>? options = null,
        IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new GeneralizedAdditiveModelOptions<T>();
        _basisFunctions = new Matrix<T>(0, 0, NumOps);
        _coefficients = new Vector<T>(0, NumOps);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);
        _basisFunctions = CreateBasisFunctions(x);
        FitModel(y);
    }

    private void ValidateInputs(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of rows in x must match the length of y.");
        }
    }

    private Matrix<T> CreateBasisFunctions(Matrix<T> x)
    {
        int numFeatures = x.Columns;
        int numBasisFunctions = _options.NumSplines * numFeatures;
        Matrix<T> basisFunctions = new Matrix<T>(x.Rows, numBasisFunctions, NumOps);

        for (int i = 0; i < numFeatures; i++)
        {
            Vector<T> feature = x.GetColumn(i);
            Vector<T> knots = CreateKnots(feature);

            for (int j = 0; j < _options.NumSplines; j++)
            {
                Vector<T> spline = CreateSpline(feature, knots[j], _options.Degree);
                basisFunctions.SetColumn(i * _options.NumSplines + j, spline);
            }
        }

        return basisFunctions;
    }

    private Vector<T> CreateKnots(Vector<T> feature)
    {
        Vector<T> sortedFeature = new Vector<T>(feature.OrderBy(v => v));
        int step = sortedFeature.Length / (_options.NumSplines + 1);
        return new Vector<T>([.. Enumerable.Range(1, _options.NumSplines).Select(i => sortedFeature[i * step])], NumOps);
    }

    private Vector<T> CreateSpline(Vector<T> feature, T knot, int degree)
    {
        return new Vector<T>(feature.Select(x => SplineFunction(x, knot, degree)));
    }

    private T SplineFunction(T x, T knot, int degree)
    {
        T diff = NumOps.Subtract(x, knot);
        return NumOps.GreaterThan(diff, NumOps.Zero)
            ? NumOps.Power(diff, NumOps.FromDouble(degree))
            : NumOps.Zero;
    }

    private void FitModel(Vector<T> y)
    {
        Matrix<T> penaltyMatrix = CreatePenaltyMatrix();
        Matrix<T> xTx = _basisFunctions.Transpose().Multiply(_basisFunctions);
        Matrix<T> regularizedXTX = Regularization.RegularizeMatrix(xTx);
        Vector<T> xTy = _basisFunctions.Transpose().Multiply(y);

        _coefficients = SolveSystem(regularizedXTX, xTy);
        _coefficients = Regularization.RegularizeCoefficients(_coefficients);
    }

    private Matrix<T> CreatePenaltyMatrix()
    {
        int size = _basisFunctions.Columns;
        Matrix<T> penaltyMatrix = Matrix<T>.CreateIdentity(size, NumOps);
        return penaltyMatrix;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        Matrix<T> inputBasisFunctions = CreateBasisFunctions(input);
        return inputBasisFunctions.Multiply(_coefficients);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Coefficients", _coefficients },
                { "FeatureImportance", CalculateFeatureImportances() },
                { "NumSplines", _options.NumSplines },
                { "Degree", _options.Degree }
            }
        };
    }

    protected override ModelType GetModelType() => ModelType.GeneralizedAdditiveModelRegression;

    protected override Vector<T> CalculateFeatureImportances()
    {
        int numFeatures = _basisFunctions.Columns / _options.NumSplines;
        Vector<T> importances = new Vector<T>(numFeatures, NumOps);

        for (int i = 0; i < numFeatures; i++)
        {
            T importance = NumOps.Zero;
            for (int j = 0; j < _options.NumSplines; j++)
            {
                importance = NumOps.Add(importance, NumOps.Abs(_coefficients[i * _options.NumSplines + j]));
            }
            importances[i] = importance;
        }

        return importances;
    }

    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Write base class data
        base.Serialize();

        // Write GAM-specific data
        writer.Write(_options.NumSplines);
        writer.Write(_options.Degree);

        // Write _basisFunctions
        writer.Write(_basisFunctions.Rows);
        writer.Write(_basisFunctions.Columns);
        for (int i = 0; i < _basisFunctions.Rows; i++)
        {
            for (int j = 0; j < _basisFunctions.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_basisFunctions[i, j]));
            }
        }

        // Write _coefficients
        writer.Write(_coefficients.Length);
        for (int i = 0; i < _coefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(_coefficients[i]));
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] modelData)
    {
        using MemoryStream ms = new MemoryStream(modelData);
        using BinaryReader reader = new BinaryReader(ms);

        // Read base class data
        base.Deserialize(modelData);

        // Read GAM-specific data
        _options.NumSplines = reader.ReadInt32();
        _options.Degree = reader.ReadInt32();

        // Read _basisFunctions
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _basisFunctions = new Matrix<T>(rows, cols, NumOps);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                _basisFunctions[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Read _coefficients
        int length = reader.ReadInt32();
        _coefficients = new Vector<T>(length, NumOps);
        for (int i = 0; i < length; i++)
        {
            _coefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}