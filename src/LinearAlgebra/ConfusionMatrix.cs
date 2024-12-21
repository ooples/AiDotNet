namespace AiDotNet.LinearAlgebra;

public class ConfusionMatrix<T> : MatrixBase<T>
{
    public T TruePositives => this[0, 0];
    public T TrueNegatives => this[1, 1];
    public T FalsePositives => this[1, 0];
    public T FalseNegatives => this[0, 1];

    public ConfusionMatrix(T truePositives, T trueNegatives, T falsePositives, T falseNegatives)
        : base(2, 2)
    {
        this[0, 0] = truePositives;
        this[1, 1] = trueNegatives;
        this[1, 0] = falsePositives;
        this[0, 1] = falseNegatives;
    }

    protected override MatrixBase<T> CreateInstance(int rows, int cols)
    {
        return new Matrix<T>(rows, cols, ops);
    }

    public T Accuracy
    {
        get
        {
            T numerator = ops.Add(TruePositives, TrueNegatives);
            T denominator = ops.Add(ops.Add(ops.Add(TruePositives, TrueNegatives), FalsePositives), FalseNegatives);

            return ops.Divide(numerator, denominator);
        }
    }

    public T Precision
    {
        get
        {
            T denominator = ops.Add(TruePositives, FalsePositives);
            return ops.Equals(denominator, ops.Zero) ? ops.Zero : ops.Divide(TruePositives, denominator);
        }
    }

    public T Recall
    {
        get
        {
            T denominator = ops.Add(TruePositives, FalseNegatives);
            return ops.Equals(denominator, ops.Zero) ? ops.Zero : ops.Divide(TruePositives, denominator);
        }
    }

    public T F1Score
    {
        get
        {
            T precision = Precision;
            T recall = Recall;
            T numerator = ops.Multiply(ops.FromDouble(2), ops.Multiply(precision, recall));
            T denominator = ops.Add(precision, recall);
            return ops.Equals(denominator, ops.Zero) ? ops.Zero : ops.Divide(numerator, denominator);
        }
    }

    public T Specificity
    {
        get
        {
            T denominator = ops.Add(TrueNegatives, FalsePositives);
            return ops.Equals(denominator, ops.Zero) ? ops.Zero : ops.Divide(TrueNegatives, denominator);
        }
    }
}