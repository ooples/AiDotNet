namespace AiDotNet.Models;

public class VectorModel<T> : ISymbolicModel<T>
{
    public Vector<T> Coefficients { get; }
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public VectorModel(Vector<T> coefficients)
    {
        Coefficients = coefficients;
    }

    public int FeatureCount => Coefficients.Length;

    public int Complexity => throw new NotImplementedException();

    public bool IsFeatureUsed(int featureIndex)
    {
        return !NumOps.Equals(Coefficients[featureIndex], NumOps.Zero);
    }

    public T Evaluate(Vector<T> input)
    {
        if (input.Length != Coefficients.Length)
        {
            throw new ArgumentException("Input vector length must match coefficients length.");
        }

        T result = NumOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            result = NumOps.Add(result, NumOps.Multiply(Coefficients[i], input[i]));
        }

        return result;
    }

    public ISymbolicModel<T> Mutate(double mutationRate)
    {
        Vector<T> mutatedCoefficients = new Vector<T>(Coefficients.Length);
        Random random = new Random();

        for (int i = 0; i < Coefficients.Length; i++)
        {
            if (random.NextDouble() < mutationRate)
            {
                // Mutate the coefficient by adding a small random value
                T mutation = NumOps.FromDouble((random.NextDouble() - 0.5) * 0.1);
                mutatedCoefficients[i] = NumOps.Add(Coefficients[i], mutation);
            }
            else
            {
                mutatedCoefficients[i] = Coefficients[i];
            }
        }

        return new VectorModel<T>(mutatedCoefficients);
    }

    public ISymbolicModel<T> Crossover(ISymbolicModel<T> other, double crossoverRate)
    {
        if (!(other is VectorModel<T> otherVector))
        {
            throw new ArgumentException("Crossover can only be performed with another VectorModel.");
        }

        if (Coefficients.Length != otherVector.Coefficients.Length)
        {
            throw new ArgumentException("Vector lengths must match for crossover.");
        }

        Vector<T> childCoefficients = new Vector<T>(Coefficients.Length);
        Random random = new Random();

        for (int i = 0; i < Coefficients.Length; i++)
        {
            if (random.NextDouble() < crossoverRate)
            {
                // Perform crossover by taking the average of the two coefficients
                childCoefficients[i] = NumOps.Divide(
                    NumOps.Add(Coefficients[i], otherVector.Coefficients[i]),
                    NumOps.FromDouble(2.0)
                );
            }
            else
            {
                // Randomly choose from either parent
                childCoefficients[i] = random.NextDouble() < 0.5 ? Coefficients[i] : otherVector.Coefficients[i];
            }
        }

        return new VectorModel<T>(childCoefficients);
    }

    public ISymbolicModel<T> Copy()
    {
        Vector<T> clonedCoefficients = new(Coefficients.Length);
        for (int i = 0; i < Coefficients.Length; i++)
        {
            clonedCoefficients[i] = Coefficients[i];
        }
        return new VectorModel<T>(clonedCoefficients);
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

        Vector<T> predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = Evaluate(input.GetRow(i));
        }

        return predictions;
    }

    public ModelMetadata<T> GetModelMetadata()
    {
        T norm = Coefficients.Norm();
        norm ??= NumOps.Zero;

        return new ModelMetadata<T>
        {
            ModelType = ModelType.Vector,
            FeatureCount = FeatureCount,
            Complexity = FeatureCount, // For a vector model, complexity is the number of features
            Description = $"Vector model with {FeatureCount} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "CoefficientNorm", norm! },
                { "NonZeroCoefficients", Coefficients.Count(c => !NumOps.Equals(c, NumOps.Zero)) },
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
        Vector<T> newCoefficients = new Vector<T>(length);

        // Read each coefficient
        for (int i = 0; i < length; i++)
        {
            newCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Update the Coefficients property
        for (int i = 0; i < length; i++)
        {
            Coefficients[i] = newCoefficients[i];
        }
    }

    public ISymbolicModel<T> UpdateCoefficients(Vector<T> newCoefficients)
    {
        if (newCoefficients.Length != this.FeatureCount)
        {
            throw new ArgumentException($"The number of new coefficients ({newCoefficients.Length}) must match the current feature count ({this.FeatureCount}).");
        }

        // Create a new VectorModel with the updated coefficients
        return new VectorModel<T>(newCoefficients);
    }
}