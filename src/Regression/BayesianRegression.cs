namespace AiDotNet.Regression;

public class BayesianRegression<T> : RegressionBase<T>
{
    private readonly BayesianRegressionOptions<T> _bayesOptions;
    private Matrix<T> _posteriorCovariance;

    public BayesianRegression(BayesianRegressionOptions<T>? bayesianOptions = null, 
                              IRegularization<T>? regularization = null)
        : base(bayesianOptions, regularization)
    {
        _bayesOptions = bayesianOptions ?? new BayesianRegressionOptions<T>();
        _posteriorCovariance = new Matrix<T>(0, 0);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;

        // Add bias term if using intercept
        if (Options.UseIntercept)
        {
            x = x.AddConstantColumn(NumOps.One);
            d++;
        }

        // Apply kernel if specified
        if (_bayesOptions.KernelType != KernelType.Linear)
        {
            x = ApplyKernel(x);
        }

        // Apply regularization
        x = Regularization.RegularizeMatrix(x);
        y = Regularization.RegularizeCoefficients(y);

        // Compute prior precision (inverse of prior covariance)
        var priorPrecision = Matrix<T>.CreateIdentity(d).Multiply(NumOps.FromDouble(_bayesOptions.Alpha));

        // Compute the design matrix precision
        var noisePrecision = NumOps.FromDouble(_bayesOptions.Beta);
        var designPrecision = x.Transpose().Multiply(x).Multiply(noisePrecision);

        // Compute posterior precision and covariance
        var posteriorPrecision = priorPrecision.Add(designPrecision);
        
        // Use the factory to create the appropriate decomposition
        var decomposition = MatrixDecompositionFactory.CreateDecomposition(posteriorPrecision, _bayesOptions.DecompositionType);
        _posteriorCovariance = MatrixHelper<T>.InvertUsingDecomposition(decomposition);

        // Compute posterior mean (coefficients)
        var xTy = x.Transpose().Multiply(y).Multiply(noisePrecision);
        var coeffs = _posteriorCovariance.Multiply(xTy);

        if (Options.UseIntercept)
        {
            Intercept = coeffs[0];
            Coefficients = new Vector<T>([.. coeffs.Skip(1)]);
        }
        else
        {
            Coefficients = coeffs;
        }
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        if (Options.UseIntercept)
        {
            input = input.AddConstantColumn(NumOps.One);
        }

        if (_bayesOptions.KernelType != KernelType.Linear)
        {
            input = ApplyKernel(input);
        }

        var coefficientsMatrix = Coefficients.AppendAsMatrix(Intercept);
        return input.Multiply(coefficientsMatrix).GetColumn(0);
    }

    public (Vector<T> Mean, Vector<T> Variance) PredictWithUncertainty(Matrix<T> input)
    {
        if (Options.UseIntercept)
        {
            input = input.AddConstantColumn(NumOps.One);
        }

        if (_bayesOptions.KernelType != KernelType.Linear)
        {
            input = ApplyKernel(input);
        }

        var mean = Predict(input);
        var variance = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            var x = input.GetRow(i);
            var xCov = x.DotProduct(_posteriorCovariance.Multiply(x));
            variance[i] = NumOps.Add(xCov, NumOps.FromDouble(1.0 / _bayesOptions.Beta));
        }

        return (mean, variance);
    }

    private Matrix<T> ApplyKernel(Matrix<T> input)
    {
        return _bayesOptions.KernelType switch
        {
            KernelType.RBF => ApplyRBFKernel(input),
            KernelType.Polynomial => ApplyPolynomialKernel(input),
            KernelType.Sigmoid => ApplySigmoidKernel(input),
            KernelType.Linear => input,// Linear kernel (no change)
            KernelType.Laplacian => ApplyLaplacianKernel(input),
            _ => throw new ArgumentException($"Unsupported kernel type: {_bayesOptions.KernelType}"),
        };
    }

    private Matrix<T> ApplyLaplacianKernel(Matrix<T> input)
    {
        int n = input.Rows;
        var output = new Matrix<T>(n, n);
        var gamma = NumOps.FromDouble(_bayesOptions.LaplacianGamma); // Kernel width parameter

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++) // We only need to compute half of the matrix due to symmetry
            {
                if (i == j)
                {
                    output[i, j] = NumOps.One; // The kernel of a point with itself is always 1
                }
                else
                {
                    var distance = CalculateManhattanDistance(input.GetRow(i), input.GetRow(j));
                    var kernelValue = NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, distance)));
                    output[i, j] = kernelValue;
                    output[j, i] = kernelValue; // The kernel matrix is symmetric
                }
            }
        }

        return output;
    }

    private T CalculateManhattanDistance(Vector<T> x, Vector<T> y)
    {
        T distance = NumOps.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            distance = NumOps.Add(distance, NumOps.Abs(NumOps.Subtract(x[i], y[i])));
        }

        return distance;
    }

    private Matrix<T> ApplyRBFKernel(Matrix<T> input)
    {
        int n = input.Rows;
        var result = new Matrix<T>(n, n);
        var gamma = NumOps.FromDouble(_bayesOptions.Gamma);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                var diff = input.GetRow(i).Subtract(input.GetRow(j));
                var squaredDistance = diff.DotProduct(diff);
                var value = NumOps.Exp(NumOps.Multiply(NumOps.Negate(gamma), squaredDistance));
                result[i, j] = result[j, i] = value;
            }
        }

        return result;
    }

    private Matrix<T> ApplyPolynomialKernel(Matrix<T> input)
    {
        int n = input.Rows;
        var result = new Matrix<T>(n, n);
        var gamma = NumOps.FromDouble(_bayesOptions.Gamma);
        var coef0 = NumOps.FromDouble(_bayesOptions.Coef0);
        var degree = _bayesOptions.PolynomialDegree;

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                var dot = input.GetRow(i).DotProduct(input.GetRow(j));
                var value = NumOps.Power(NumOps.Add(NumOps.Multiply(gamma, dot), coef0), NumOps.FromDouble(degree));
                result[i, j] = result[j, i] = value;
            }
        }

        return result;
    }

    private Matrix<T> ApplySigmoidKernel(Matrix<T> input)
    {
        int n = input.Rows;
        var result = new Matrix<T>(n, n);
        var gamma = NumOps.FromDouble(_bayesOptions.Gamma);
        var coef0 = NumOps.FromDouble(_bayesOptions.Coef0);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                var dot = input.GetRow(i).DotProduct(input.GetRow(j));
                var value = MathHelper.Tanh(NumOps.Add(NumOps.Multiply(gamma, dot), coef0));
                result[i, j] = result[j, i] = value;
            }
        }

        return result;
    }

    protected override ModelType GetModelType()
    {
        return ModelType.BayesianRegression;
    }
}