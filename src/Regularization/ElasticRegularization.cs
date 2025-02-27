namespace AiDotNet.Regularization;

public class ElasticNetRegularization<T> : RegularizationBase<T>
{
    public ElasticNetRegularization(RegularizationOptions? options = null) : base(options ?? new RegularizationOptions
        {
            Type = RegularizationType.ElasticNet,
            Strength = 0.1, // Default Elastic Net regularization strength
            L1Ratio = 0.5  // Default balance between L1 and L2
        })
    {
    }

    public override Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        // Elastic Net regularization typically doesn't modify the input matrix
        return matrix;
    }

    public override Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        var l1Ratio = NumOps.FromDouble(Options.L1Ratio);
        var l2Ratio = NumOps.FromDouble(1 - Options.L1Ratio);

        return coefficients.Transform(c =>
        {
            var subPart = NumOps.Subtract(NumOps.Abs(c), NumOps.Multiply(regularizationStrength, l1Ratio));
            var l1Part = NumOps.Multiply(
                NumOps.SignOrZero(c),
                NumOps.GreaterThan(subPart, NumOps.Zero) ? subPart : NumOps.Zero
            );
            var l2Part = NumOps.Multiply(
                c,
                NumOps.Subtract(NumOps.One, NumOps.Multiply(regularizationStrength, l2Ratio))
            );
            return NumOps.Add(l1Part, l2Part);
        });
    }

    public override Vector<T> RegularizeGradient(Vector<T> gradient, Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        var l1Ratio = NumOps.FromDouble(Options.L1Ratio);
        var l2Ratio = NumOps.FromDouble(1 - Options.L1Ratio);

        return gradient.Add(coefficients.Transform(c =>
        {
            var l1Part = NumOps.Multiply(regularizationStrength, NumOps.Multiply(l1Ratio, NumOps.SignOrZero(c)));
            var l2Part = NumOps.Multiply(regularizationStrength, NumOps.Multiply(l2Ratio, c));
            return NumOps.Add(l1Part, l2Part);
        }));
    }
}