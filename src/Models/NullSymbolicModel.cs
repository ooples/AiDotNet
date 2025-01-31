namespace AiDotNet.Models;

public class NullSymbolicModel<T> : ISymbolicModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public int Complexity => 0;

    public int FeatureCount => 0;

    public Vector<T> Coefficients => Vector<T>.Empty();

    public T Intercept => NumOps.Zero;

    public ISymbolicModel<T> Copy()
    {
        return new NullSymbolicModel<T>();
    }

    public ISymbolicModel<T> Crossover(ISymbolicModel<T> other, double crossoverRate)
    {
        return new NullSymbolicModel<T>();
    }

    public void Deserialize(byte[] data)
    {
    }

    public T Evaluate(Vector<T> input)
    {
        return NumOps.Zero;
    }

    public void Fit(Matrix<T> X, Vector<T> y)
    {
    }

    public ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.None
        };
    }

    public bool IsFeatureUsed(int featureIndex)
    {
        return false;
    }

    public ISymbolicModel<T> Mutate(double mutationRate)
    {
        return new NullSymbolicModel<T>();
    }

    public T Predict(Vector<T> input)
    {
        return NumOps.Zero;
    }

    public Vector<T> Predict(Matrix<T> input)
    {
        return Vector<T>.Empty();
    }

    public Vector<T> PredictMany(Matrix<T> inputs)
    {
        return Vector<T>.Empty();
    }

    public byte[] Serialize()
    {
        return [];
    }

    public void Train(Matrix<T> x, Vector<T> y)
    {
    }

    public ISymbolicModel<T> UpdateCoefficients(Vector<T> newCoefficients)
    {
        return new NullSymbolicModel<T>();
    }
}