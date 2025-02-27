namespace AiDotNet.Regression;

public class MultinomialLogisticRegression<T> : RegressionBase<T>
{
    private readonly MultinomialLogisticRegressionOptions<T> _options;
    private Matrix<T>? _coefficients;
    private int _numClasses;

    public MultinomialLogisticRegression(MultinomialLogisticRegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new MultinomialLogisticRegressionOptions<T>();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidationHelper<T>.ValidateInputData(x, y);
        _numClasses = y.Distinct().Count();

        int numFeatures = x.Columns;
        _coefficients = new Matrix<T>(_numClasses, numFeatures + 1);

        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            Matrix<T> probabilities = ComputeProbabilities(xWithIntercept);
            Matrix<T> gradient = ComputeGradient(xWithIntercept, y, probabilities);
            Matrix<T> hessian = ComputeHessian(xWithIntercept, probabilities);

            if (Regularization != null)
            {
                gradient = Regularization.RegularizeMatrix(gradient);
                hessian = Regularization.RegularizeMatrix(hessian);
            }

            Vector<T> flattenedGradient = gradient.Flatten();
            Vector<T> update = MatrixSolutionHelper.SolveLinearSystem(hessian, flattenedGradient, MatrixDecompositionFactory.GetDecompositionType(_options.DecompositionMethod));

            Matrix<T> updateMatrix = new Matrix<T>(gradient.Rows, gradient.Columns);
            for (int i = 0; i < update.Length; i++)
            {
                updateMatrix[i / gradient.Columns, i % gradient.Columns] = update[i];
            }

            _coefficients = _coefficients.Subtract(updateMatrix);

            if (HasConverged(updateMatrix))
            {
                break;
            }
        }

        Coefficients = _coefficients.GetColumn(0);
        Intercept = _coefficients.GetColumn(_coefficients.Columns - 1)[0];
    }

    private Matrix<T> ComputeProbabilities(Matrix<T> x)
    {
        if (_coefficients == null)
            throw new InvalidOperationException("Coefficients have not been initialized.");

        Matrix<T> scores = x.Multiply(_coefficients.Transpose());
        Vector<T> maxScores = scores.RowWiseMax();
        Matrix<T> expScores = scores.Transform((s, i, j) => NumOps.Exp(NumOps.Subtract(s, maxScores[i])));
        Vector<T> sumExpScores = expScores.RowWiseSum();

        return expScores.PointwiseDivide(sumExpScores.ToColumnMatrix());
    }

    private Matrix<T> ComputeGradient(Matrix<T> x, Vector<T> y, Matrix<T> probabilities)
    {
        Matrix<T> yOneHot = CreateOneHotEncoding(y);
        return x.Transpose().Multiply(yOneHot.Subtract(probabilities));
    }

    private Matrix<T> ComputeHessian(Matrix<T> x, Matrix<T> probabilities)
    {
        int n = x.Rows;
        int p = x.Columns;
        Matrix<T> hessian = new(p * _numClasses, p * _numClasses);

        for (int i = 0; i < n; i++)
        {
            Vector<T> xi = x.GetRow(i);
            Vector<T> probs = probabilities.GetRow(i);
            Matrix<T> diagP = Matrix<T>.CreateDiagonal(probs);
            Matrix<T> ppt = probs.OuterProduct(probs);
            Matrix<T> h = diagP.Subtract(ppt);
            Matrix<T> xxt = xi.OuterProduct(xi);
            Matrix<T> block = xxt.KroneckerProduct(h);
            hessian = hessian.Add(block);
        }

        return hessian.Negate();
    }

    private Matrix<T> CreateOneHotEncoding(Vector<T> y)
    {
        Matrix<T> oneHot = new Matrix<T>(y.Length, _numClasses);
        for (int i = 0; i < y.Length; i++)
        {
            int classIndex = Convert.ToInt32(NumOps.ToInt32(y[i]));
            oneHot[i, classIndex] = NumOps.One;
        }

        return oneHot;
    }

    private bool HasConverged(Matrix<T> update)
    {
        T maxChange = update.Max(NumOps.Abs);
        return NumOps.LessThan(maxChange, NumOps.FromDouble(_options.Tolerance));
    }

    public override Vector<T> Predict(Matrix<T> x)
    {
        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));
        Matrix<T> probabilities = ComputeProbabilities(xWithIntercept);

        return probabilities.RowWiseArgmax();
    }

    public Matrix<T> PredictProbabilities(Matrix<T> x)
    {
        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));
        return ComputeProbabilities(xWithIntercept);
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize MultinomialLogisticRegression specific data
        writer.Write(_numClasses);

        // Write whether _coefficients is null
        writer.Write(_coefficients != null);

        if (_coefficients != null)
        {
            writer.Write(_coefficients.Rows);
            writer.Write(_coefficients.Columns);
            for (int i = 0; i < _coefficients.Rows; i++)
            {
                for (int j = 0; j < _coefficients.Columns; j++)
                {
                    writer.Write(Convert.ToDouble(_coefficients[i, j]));
                }
            }
        }

        // Serialize options
        writer.Write(_options.MaxIterations);
        writer.Write(Convert.ToDouble(_options.Tolerance));

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize MultinomialLogisticRegression specific data
        _numClasses = reader.ReadInt32();

        bool coefficientsExist = reader.ReadBoolean();
        if (coefficientsExist)
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            _coefficients = new Matrix<T>(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    _coefficients[i, j] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
        }
        else
        {
            _coefficients = null;
        }

        // Deserialize options
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = reader.ReadDouble();
    }

    protected override ModelType GetModelType()
    {
        return ModelType.MultinomialLogisticRegression;
    }
}