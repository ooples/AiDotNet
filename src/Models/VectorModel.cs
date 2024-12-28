namespace AiDotNet.Models;

public class VectorModel<T> : ISymbolicModel<T>
{
    public Vector<T> Coefficients { get; }
    private readonly INumericOperations<T> _numOps;

    public VectorModel(Vector<T> coefficients, INumericOperations<T> numOps)
    {
        Coefficients = coefficients;
        _numOps = numOps;
    }

    public int FeatureCount => Coefficients.Length;

    public int Complexity => throw new NotImplementedException();

    public bool IsFeatureUsed(int featureIndex)
    {
        return !_numOps.Equals(Coefficients[featureIndex], _numOps.Zero);
    }

    public T Evaluate(Vector<T> input)
    {
        if (input.Length != Coefficients.Length)
        {
            throw new ArgumentException("Input vector length must match coefficients length.");
        }

        T result = _numOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            result = _numOps.Add(result, _numOps.Multiply(Coefficients[i], input[i]));
        }
        return result;
    }

    public ISymbolicModel<T> Mutate(double mutationRate, INumericOperations<T> numOps)
    {
        Vector<T> mutatedCoefficients = new Vector<T>(Coefficients.Length, numOps);
        Random random = new Random();

        for (int i = 0; i < Coefficients.Length; i++)
        {
            if (random.NextDouble() < mutationRate)
            {
                // Mutate the coefficient by adding a small random value
                T mutation = numOps.FromDouble((random.NextDouble() - 0.5) * 0.1);
                mutatedCoefficients[i] = numOps.Add(Coefficients[i], mutation);
            }
            else
            {
                mutatedCoefficients[i] = Coefficients[i];
            }
        }

        return new VectorModel<T>(mutatedCoefficients, numOps);
    }

    public ISymbolicModel<T> Crossover(ISymbolicModel<T> other, double crossoverRate, INumericOperations<T> numOps)
    {
        if (!(other is VectorModel<T> otherVector))
        {
            throw new ArgumentException("Crossover can only be performed with another VectorModel.");
        }

        if (Coefficients.Length != otherVector.Coefficients.Length)
        {
            throw new ArgumentException("Vector lengths must match for crossover.");
        }

        Vector<T> childCoefficients = new Vector<T>(Coefficients.Length, numOps);
        Random random = new Random();

        for (int i = 0; i < Coefficients.Length; i++)
        {
            if (random.NextDouble() < crossoverRate)
            {
                // Perform crossover by taking the average of the two coefficients
                childCoefficients[i] = numOps.Divide(
                    numOps.Add(Coefficients[i], otherVector.Coefficients[i]),
                    numOps.FromDouble(2.0)
                );
            }
            else
            {
                // Randomly choose from either parent
                childCoefficients[i] = random.NextDouble() < 0.5 ? Coefficients[i] : otherVector.Coefficients[i];
            }
        }

        return new VectorModel<T>(childCoefficients, numOps);
    }

    public ISymbolicModel<T> Clone()
    {
        Vector<T> clonedCoefficients = new(Coefficients.Length, _numOps);
        for (int i = 0; i < Coefficients.Length; i++)
        {
            clonedCoefficients[i] = Coefficients[i];
        }
        return new VectorModel<T>(clonedCoefficients, _numOps);
    }

    public void Fit(Matrix<T> X, Vector<T> y)
    {
        // For VectorModel, Fit is the same as Train
        Train(X, y);
    }

    public void Train(Matrix<T> X, Vector<T> y)
    {
        if (X.Rows != y.Length)
        {
            throw new ArgumentException("Number of rows in X must match the length of y.");
        }

        if (X.Columns != FeatureCount)
        {
            throw new ArgumentException($"Number of columns in X ({X.Columns}) must match the FeatureCount ({FeatureCount}).");
        }

        // Implement a simple linear regression using the normal equation
        // (X^T * X)^-1 * X^T * y
        Matrix<T> XTranspose = X.Transpose();
        Matrix<T> XTX = XTranspose * X;
        Matrix<T> XTXInverse = XTX.Inverse();
        Matrix<T> XTY = XTranspose * Matrix<T>.FromVector(y);
        Vector<T> newCoefficients = (XTXInverse * XTY).GetColumn(0);

        // Update the coefficients
        for (int i = 0; i < FeatureCount; i++)
        {
            Coefficients[i] = newCoefficients[i];
        }
    }

    public Vector<T> Predict(Matrix<T> input)
    {
        if (input.Columns != FeatureCount)
        {
            throw new ArgumentException($"Input matrix has {input.Columns} columns, but the model expects {FeatureCount} features.");
        }

        Vector<T> predictions = new Vector<T>(input.Rows, _numOps);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = Evaluate(input.GetRow(i));
        }

        return predictions;
    }

    public ModelMetadata<T> GetModelMetadata()
    {
        T norm = Coefficients.Norm();
        norm ??= _numOps.Zero;

        return new ModelMetadata<T>
        {
            ModelType = ModelType.Vector,
            FeatureCount = FeatureCount,
            Complexity = FeatureCount, // For a vector model, complexity is the number of features
            Description = $"Vector model with {FeatureCount} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "CoefficientNorm", norm! },
                { "NonZeroCoefficients", Coefficients.Count(c => !_numOps.Equals(c, _numOps.Zero)) },
                { "MeanCoefficient", Coefficients.Mean()! },
                { "MaxCoefficient", Coefficients.Max()! },
                { "MinCoefficient", Coefficients.Min()! }
            }
        };
    }

    public byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Write the number of coefficients
        writer.Write(Coefficients.Length);

        // Write each coefficient
        for (int i = 0; i < Coefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(Coefficients[i]));
        }

        return ms.ToArray();
    }

    public void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        // Read the number of coefficients
        int length = reader.ReadInt32();

        // Create a new Vector<T> to hold the deserialized coefficients
        Vector<T> newCoefficients = new Vector<T>(length, _numOps);

        // Read each coefficient
        for (int i = 0; i < length; i++)
        {
            newCoefficients[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        // Update the Coefficients property
        for (int i = 0; i < length; i++)
        {
            Coefficients[i] = newCoefficients[i];
        }
    }
}