namespace AiDotNet.Regularization;

public class L1Regularization<T> : RegularizationBase<T>
{
    public L1Regularization(RegularizationOptions? options = null)
        : base(options ?? new RegularizationOptions
        {
            Type = RegularizationType.L1,
            Strength = 0.1, // Default L1 regularization strength
            L1Ratio = 1.0  // For L1, this should always be 1.0
        })
    {
    }

    public override Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        // L1 regularization typically doesn't modify the input matrix
        return matrix;
    }

    public override Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        return coefficients.Transform(c =>
        {
            var sub = NumOps.Subtract(NumOps.Abs(c), regularizationStrength);
            return NumOps.Multiply(
                NumOps.SignOrZero(c),
                NumOps.GreaterThan(sub, NumOps.Zero) ? sub : NumOps.Zero
            );
        });
    }

    public override Vector<T> RegularizeGradient(Vector<T> gradient, Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        return gradient.Add(coefficients.Transform(c => 
            NumOps.Multiply(regularizationStrength, NumOps.SignOrZero(c))
        ));
    }
}