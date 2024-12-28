namespace AiDotNet.Regression;

public class SplineRegression<T> : NonLinearRegressionBase<T>
{
    private readonly SplineRegressionOptions _options;
    private List<Vector<T>> _knots;
    private Vector<T> _coefficients;

    public SplineRegression(SplineRegressionOptions? options = null, IRegularization<T>? regularization = null)
    : base(options, regularization)
    {
        _options = options ?? new SplineRegressionOptions();
        _knots = [];
        _coefficients = new Vector<T>(0, NumOps);
    }

    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Generate knots for each feature
        _knots = [];
        for (int i = 0; i < x.Columns; i++)
        {
            _knots.Add(GenerateKnots(x.GetColumn(i)));
        }

        // Generate basis functions
        var basisFunctions = GenerateBasisFunctions(x);

        // Add regularization
        basisFunctions = Regularization.RegularizeMatrix(basisFunctions);

        // Solve for coefficients
        var xTx = basisFunctions.Transpose().Multiply(basisFunctions);
        var xTy = basisFunctions.Transpose().Multiply(y);
        _coefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _options.DecompositionType);

        // Apply regularization to coefficients
        _coefficients = Regularization.RegularizeCoefficients(_coefficients);
    }

    private Matrix<T> GenerateBasisFunctions(Matrix<T> x)
    {
        int totalBasis = 1; // Constant term
        for (int i = 0; i < x.Columns; i++)
        {
            totalBasis += _options.Degree + _knots[i].Length;
        }

        var basis = new Matrix<T>(x.Rows, totalBasis, NumOps);

        // Constant term
        for (int i = 0; i < x.Rows; i++)
            basis[i, 0] = NumOps.One;

        int columnIndex = 1;

        for (int feature = 0; feature < x.Columns; feature++)
        {
            var featureVector = x.GetColumn(feature);

            // Linear and higher-order terms
            for (int degree = 1; degree <= _options.Degree; degree++)
            {
                for (int i = 0; i < x.Rows; i++)
                    basis[i, columnIndex] = NumOps.Power(featureVector[i], NumOps.FromDouble(degree));
                columnIndex++;
            }

            // Knot terms
            for (int k = 0; k < _knots[feature].Length; k++)
            {
                for (int i = 0; i < x.Rows; i++)
                {
                    var diff = NumOps.Subtract(featureVector[i], _knots[feature][k]);
                    basis[i, columnIndex] = NumOps.GreaterThan(diff, NumOps.Zero)
                        ? NumOps.Power(diff, NumOps.FromDouble(_options.Degree))
                        : NumOps.Zero;
                }
                columnIndex++;
            }
        }

        return basis;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        var basisFunctions = GenerateBasisFunctions(input);
        return basisFunctions.Multiply(_coefficients);
    }

    protected override T PredictSingle(Vector<T> input)
    {
        var basisFunctions = GenerateBasisFunctions(new Matrix<T>([input]));
        return basisFunctions.Multiply(_coefficients)[0];
    }

    private Vector<T> GenerateKnots(Vector<T> x)
    {
        int numKnots = _options.NumberOfKnots;
        var sortedX = x.OrderBy(v => Convert.ToDouble(v)).ToArray();
        var knotIndices = Enumerable.Range(1, numKnots)
            .Select(i => (int)Math.Round((double)(i * (sortedX.Length - 1)) / (numKnots + 1)))
            .ToArray();

        return new Vector<T>([.. knotIndices.Select(i => sortedX[i])], NumOps);
    }

    protected override ModelType GetModelType() => ModelType.SplineRegression;

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize SplineRegression specific data
        writer.Write(_options.NumberOfKnots);
        writer.Write(_options.Degree);

        // Serialize knots
        writer.Write(_knots.Count);
        foreach (var knotVector in _knots)
        {
            writer.Write(knotVector.Length);
            for (int i = 0; i < knotVector.Length; i++)
                writer.Write(Convert.ToDouble(knotVector[i]));
        }

        // Serialize coefficients
        writer.Write(_coefficients.Length);
        for (int i = 0; i < _coefficients.Length; i++)
            writer.Write(Convert.ToDouble(_coefficients[i]));

        return ms.ToArray();
    }

    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize SplineRegression specific data
        _options.NumberOfKnots = reader.ReadInt32();
        _options.Degree = reader.ReadInt32();

        // Deserialize knots
        int knotsCount = reader.ReadInt32();
        _knots = new List<Vector<T>>();
        for (int j = 0; j < knotsCount; j++)
        {
            int knotsLength = reader.ReadInt32();
            var knotVector = new Vector<T>(knotsLength, NumOps);
            for (int i = 0; i < knotsLength; i++)
                knotVector[i] = NumOps.FromDouble(reader.ReadDouble());
            _knots.Add(knotVector);
        }

        // Deserialize coefficients
        int coefficientsLength = reader.ReadInt32();
        _coefficients = new Vector<T>(coefficientsLength, NumOps);
        for (int i = 0; i < coefficientsLength; i++)
            _coefficients[i] = NumOps.FromDouble(reader.ReadDouble());
    }
}