namespace AiDotNet.Regression;

public class RadialBasisFunctionRegression<T> : NonLinearRegressionBase<T>
{
    private readonly RadialBasisFunctionOptions _options;
    private Matrix<T> _centers;
    private Vector<T> _weights;

    public RadialBasisFunctionRegression(RadialBasisFunctionOptions? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new RadialBasisFunctionOptions();
        _centers = Matrix<T>.Empty();
        _weights = Vector<T>.Empty();
    }

    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Select centers
        _centers = SelectCenters(x);

        // Compute RBF features
        Matrix<T> rbfFeatures = ComputeRBFFeatures(x);

        // Apply regularization to the RBF features
        rbfFeatures = Regularization.RegularizeMatrix(rbfFeatures);

        // Solve for weights using linear regression
        _weights = SolveLinearRegression(rbfFeatures, y);

        // Apply regularization to the weights
        _weights = Regularization.RegularizeCoefficients(_weights);
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        Matrix<T> rbfFeatures = ComputeRBFFeatures(input);
        // Apply regularization to the RBF features before prediction
        rbfFeatures = Regularization.RegularizeMatrix(rbfFeatures);
        return rbfFeatures.Multiply(_weights);
    }

    protected override T PredictSingle(Vector<T> input)
    {
        Vector<T> rbfFeatures = ComputeRBFFeaturesSingle(input);
        // Apply regularization to the RBF features before prediction
        rbfFeatures = Regularization.RegularizeCoefficients(rbfFeatures);

        return rbfFeatures.DotProduct(_weights);
    }

   private Matrix<T> SelectCenters(Matrix<T> x)
    {
        int numCenters = Math.Min(_options.NumberOfCenters, x.Rows);
        var random = new Random(_options.Seed ?? Environment.TickCount);

        // Initialize centers randomly
        var centers = new Matrix<T>(numCenters, x.Columns, NumOps);
        var selectedIndices = new HashSet<int>();
        while (selectedIndices.Count < numCenters)
        {
            int index = random.Next(x.Rows);
            if (selectedIndices.Add(index))
            {
                centers.SetRow(selectedIndices.Count - 1, x.GetRow(index));
            }
        }

        // Perform K-means clustering
        const int maxIterations = 100;
        var assignments = new int[x.Rows];
        var newCenters = new Matrix<T>(numCenters, x.Columns, NumOps);

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            bool changed = false;

            // Assign points to nearest center
            for (int i = 0; i < x.Rows; i++)
            {
                int nearestCenter = 0;
                T minDistance = EuclideanDistance(x.GetRow(i), centers.GetRow(0));

                for (int j = 1; j < numCenters; j++)
                {
                    T distance = EuclideanDistance(x.GetRow(i), centers.GetRow(j));
                    if (NumOps.LessThan(distance, minDistance))
                    {
                        minDistance = distance;
                        nearestCenter = j;
                    }
                }

                if (assignments[i] != nearestCenter)
                {
                    assignments[i] = nearestCenter;
                    changed = true;
                }
            }

            if (!changed)
            {
                break; // Convergence reached
            }

            // Compute new centers
            var counts = new int[numCenters];
            for (int i = 0; i < numCenters; i++)
            {
                newCenters.SetRow(i, new Vector<T>(x.Columns));
            }

            for (int i = 0; i < x.Rows; i++)
            {
                int assignment = assignments[i];
                newCenters.SetRow(assignment, newCenters.GetRow(assignment).Add(x.GetRow(i)));
                counts[assignment]++;
            }

            for (int i = 0; i < numCenters; i++)
            {
                if (counts[i] > 0)
                {
                    newCenters.SetRow(i, newCenters.GetRow(i).Divide(NumOps.FromDouble(counts[i])));
                }
                else
                {
                    // If a center has no assigned points, reinitialize it randomly
                    int randomIndex = random.Next(x.Rows);
                    newCenters.SetRow(i, x.GetRow(randomIndex));
                }
            }

            centers = newCenters;
        }

        return centers;
    }

    private Matrix<T> ComputeRBFFeatures(Matrix<T> x)
    {
        var rbfFeatures = new Matrix<T>(x.Rows, _centers.Rows + 1, NumOps);

        for (int i = 0; i < x.Rows; i++)
        {
            var row = ComputeRBFFeaturesSingle(x.GetRow(i));
            for (int j = 0; j < row.Length; j++)
            {
                rbfFeatures[i, j] = row[j];
            }
        }

        return rbfFeatures;
    }

    private Vector<T> ComputeRBFFeaturesSingle(Vector<T> x)
    {
        var features = new Vector<T>(_centers.Rows + 1, NumOps)
        {
            [0] = NumOps.One // Bias term
        };

        for (int i = 0; i < _centers.Rows; i++)
        {
            T distance = EuclideanDistance(x, _centers.GetRow(i));
            features[i + 1] = RbfKernel(distance);
        }

        return features;
    }

    private T EuclideanDistance(Vector<T> x1, Vector<T> x2)
    {
        T sumSquared = NumOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            T diff = NumOps.Subtract(x1[i], x2[i]);
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sumSquared);
    }

    private T RbfKernel(T distance)
    {
        T gamma = NumOps.FromDouble(_options.Gamma);
        return NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, NumOps.Multiply(distance, distance))));
    }

    private Vector<T> SolveLinearRegression(Matrix<T> x, Vector<T> y)
    {
        // Use pseudo-inverse to solve for weights
        Matrix<T> xTranspose = x.Transpose();
        Matrix<T> xTx = xTranspose.Multiply(x);
        Matrix<T> xTxInverse = xTx.Inverse();
        Matrix<T> xTxInverseXT = xTxInverse.Multiply(xTranspose);
        return xTxInverseXT.Multiply(y);
    }

    protected override ModelType GetModelType()
    {
        return ModelType.RadialBasisFunctionRegression;
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize RBF specific data
        writer.Write(_options.NumberOfCenters);
        writer.Write(_options.Gamma);
        writer.Write(_options.Seed ?? -1);

        // Serialize centers
        writer.Write(_centers.Rows);
        writer.Write(_centers.Columns);
        for (int i = 0; i < _centers.Rows; i++)
        {
            for (int j = 0; j < _centers.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_centers[i, j]));
            }
        }

        // Serialize weights
        writer.Write(_weights.Length);
        for (int i = 0; i < _weights.Length; i++)
        {
            writer.Write(Convert.ToDouble(_weights[i]));
        }

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

        // Deserialize RBF specific data
        _options.NumberOfCenters = reader.ReadInt32();
        _options.Gamma = reader.ReadDouble();
        int seed = reader.ReadInt32();
        _options.Seed = seed == -1 ? null : seed;

        // Deserialize centers
        int centerRows = reader.ReadInt32();
        int centerColumns = reader.ReadInt32();
        _centers = new Matrix<T>(centerRows, centerColumns, NumOps);
        for (int i = 0; i < centerRows; i++)
        {
            for (int j = 0; j < centerColumns; j++)
            {
                _centers[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize weights
        int weightsLength = reader.ReadInt32();
        _weights = new Vector<T>(weightsLength, NumOps);
        for (int i = 0; i < weightsLength; i++)
        {
            _weights[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}