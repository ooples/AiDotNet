namespace AiDotNet.Regularization;

public class L2Regularization<T> : RegularizationBase<T>
{
    public L2Regularization(RegularizationOptions? options = null) : base(options ?? new RegularizationOptions
        {
            Type = RegularizationType.L2,
            Strength = 0.01, // Default L2 regularization strength
            L1Ratio = 0.0  // For L2, this should always be 0.0
        })
    {
    }

    public override Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        // L2 regularization typically doesn't modify the input matrix
        return matrix;
    }

    public override Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        return coefficients.Multiply(NumOps.Subtract(NumOps.One, regularizationStrength));
    }

    public override Vector<T> RegularizeGradient(Vector<T> gradient, Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        return gradient.Add(coefficients.Multiply(regularizationStrength));
    }
}