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
        return new Matrix<T>(rows, cols);
    }

    public T Accuracy
    {
        get
        {
            T numerator = NumOps.Add(TruePositives, TrueNegatives);
            T denominator = NumOps.Add(NumOps.Add(NumOps.Add(TruePositives, TrueNegatives), FalsePositives), FalseNegatives);

            return NumOps.Divide(numerator, denominator);
        }
    }

    public T Precision
    {
        get
        {
            T denominator = NumOps.Add(TruePositives, FalsePositives);
            return NumOps.Equals(denominator, NumOps.Zero) ? NumOps.Zero : NumOps.Divide(TruePositives, denominator);
        }
    }

    public T Recall
    {
        get
        {
            T denominator = NumOps.Add(TruePositives, FalseNegatives);
            return NumOps.Equals(denominator, NumOps.Zero) ? NumOps.Zero : NumOps.Divide(TruePositives, denominator);
        }
    }

    public T F1Score
    {
        get
        {
            T precision = Precision;
            T recall = Recall;
            T numerator = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(precision, recall));
            T denominator = NumOps.Add(precision, recall);
            return NumOps.Equals(denominator, NumOps.Zero) ? NumOps.Zero : NumOps.Divide(numerator, denominator);
        }
    }

    public T Specificity
    {
        get
        {
            T denominator = NumOps.Add(TrueNegatives, FalsePositives);
            return NumOps.Equals(denominator, NumOps.Zero) ? NumOps.Zero : NumOps.Divide(TrueNegatives, denominator);
        }
    }
}