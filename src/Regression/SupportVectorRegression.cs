
namespace AiDotNet.Regression;

public class SupportVectorRegression<T> : NonLinearRegressionBase<T>
{
    private readonly SupportVectorRegressionOptions _options;

    public SupportVectorRegression(SupportVectorRegressionOptions? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new SupportVectorRegressionOptions();
    }

    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Apply regularization to the input matrix
        Matrix<T> regularizedX = Regularization.RegularizeMatrix(x);

        // Implement SMO algorithm for SVR with regularized input
        SequentialMinimalOptimization(regularizedX, y);

        // Apply regularization to the coefficients (alphas)
        Alphas = Regularization.RegularizeCoefficients(Alphas);
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        // Apply regularization to the input matrix for prediction
        Matrix<T> regularizedInput = Regularization.RegularizeMatrix(input);

        var predictions = new Vector<T>(regularizedInput.Rows, NumOps);
        for (int i = 0; i < regularizedInput.Rows; i++)
        {
            predictions[i] = PredictSingle(regularizedInput.GetRow(i));
        }

        return predictions;
    }

    protected override T PredictSingle(Vector<T> input)
    {
        T result = B;
        for (int i = 0; i < SupportVectors.Rows; i++)
        {
            result = NumOps.Add(result, NumOps.Multiply(Alphas[i], 
                KernelFunction(SupportVectors.GetRow(i), input)));
        }

        return result;
    }

    private void SequentialMinimalOptimization(Matrix<T> x, Vector<T> y)
    {
        int m = x.Rows;
        Alphas = new Vector<T>(m, NumOps);
        Vector<T> errors = new(m, NumOps);

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            int numChangedAlphas = 0;
            for (int i = 0; i < m; i++)
            {
                T Ei = NumOps.Subtract(PredictSingle(x.GetRow(i)), y[i]);
                if ((NumOps.LessThan(y[i], NumOps.Subtract(PredictSingle(x.GetRow(i)), NumOps.FromDouble(_options.Epsilon))) && NumOps.LessThan(Alphas[i], NumOps.FromDouble(_options.C))) ||
                    (NumOps.GreaterThan(y[i], NumOps.Add(PredictSingle(x.GetRow(i)), NumOps.FromDouble(_options.Epsilon))) && NumOps.GreaterThan(Alphas[i], NumOps.Zero)))
                {
                    int j = SelectSecondAlpha(i, m);
                    T Ej = NumOps.Subtract(PredictSingle(x.GetRow(j)), y[j]);

                    T oldAi = Alphas[i];
                    T oldAj = Alphas[j];

                    (T L, T H) = ComputeBounds(y[i], y[j], Alphas[i], Alphas[j]);

                    if (NumOps.Equals(L, H)) continue;

                    T eta = NumOps.Subtract(
                        NumOps.Subtract(
                            NumOps.Multiply(NumOps.FromDouble(2), KernelFunction(x.GetRow(i), x.GetRow(j))),
                            KernelFunction(x.GetRow(i), x.GetRow(i))
                        ),
                        KernelFunction(x.GetRow(j), x.GetRow(j))
                    );

                    if (NumOps.GreaterThanOrEquals(eta, NumOps.Zero)) continue;

                    Alphas[j] = NumOps.Subtract(Alphas[j], NumOps.Divide(NumOps.Multiply(y[j], NumOps.Subtract(Ei, Ej)), eta));
                    Alphas[j] = Clip(Alphas[j], L, H);

                    if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(Alphas[j], oldAj)), NumOps.FromDouble(1e-5)))
                        continue;

                    Alphas[i] = NumOps.Add(Alphas[i], NumOps.Multiply(NumOps.Multiply(y[i], y[j]), NumOps.Subtract(oldAj, Alphas[j])));

                    T b1 = NumOps.Add(B, NumOps.Subtract(Ei, NumOps.Multiply(NumOps.Multiply(y[i], NumOps.Subtract(Alphas[i], oldAi)), KernelFunction(x.GetRow(i), x.GetRow(i)))));
                    b1 = NumOps.Subtract(b1, NumOps.Multiply(NumOps.Multiply(y[j], NumOps.Subtract(Alphas[j], oldAj)), KernelFunction(x.GetRow(i), x.GetRow(j))));

                    T b2 = NumOps.Add(B, NumOps.Subtract(Ej, NumOps.Multiply(NumOps.Multiply(y[i], NumOps.Subtract(Alphas[i], oldAi)), KernelFunction(x.GetRow(i), x.GetRow(j)))));
                    b2 = NumOps.Subtract(b2, NumOps.Multiply(NumOps.Multiply(y[j], NumOps.Subtract(Alphas[j], oldAj)), KernelFunction(x.GetRow(j), x.GetRow(j))));

                    if (NumOps.GreaterThan(Alphas[i], NumOps.Zero) && NumOps.LessThan(Alphas[i], NumOps.FromDouble(_options.C)))
                        B = b1;
                    else if (NumOps.GreaterThan(Alphas[j], NumOps.Zero) && NumOps.LessThan(Alphas[j], NumOps.FromDouble(_options.C)))
                        B = b2;
                    else
                        B = NumOps.Divide(NumOps.Add(b1, b2), NumOps.FromDouble(2));

                    numChangedAlphas++;
                }
            }

            if (numChangedAlphas == 0)
                break;
        }

        // Store support vectors
        SupportVectors = x.GetRows(Enumerable.Range(0, m).Where(i => NumOps.GreaterThan(Alphas[i], NumOps.Zero)).ToArray());
    }

    private readonly Random _random = new();

    private int SelectSecondAlpha(int i, int m)
    {
        int j;
        do
        {
            j = _random.Next(m);
        } while (j == i);

        return j;
    }

    private (T L, T H) ComputeBounds(T yi, T yj, T ai, T aj)
    {
        if (NumOps.Equals(yi, yj))
        {
            T L = MathHelper.Max(NumOps.Zero, NumOps.Subtract(NumOps.Add(ai, aj), NumOps.FromDouble(_options.C)));
            T H = MathHelper.Min(NumOps.FromDouble(_options.C), NumOps.Add(ai, aj));

            return (L, H);
        }
        else
        {
            T L = MathHelper.Max(NumOps.Zero, NumOps.Subtract(aj, ai));
            T H = MathHelper.Min(NumOps.FromDouble(_options.C), NumOps.Add(NumOps.Subtract(NumOps.FromDouble(_options.C), ai), aj));

            return (L, H);
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Epsilon"] = _options.Epsilon;
        metadata.AdditionalInfo["C"] = _options.C;
        metadata.AdditionalInfo["RegularizationType"] = Regularization.GetType().Name;

        return metadata;
    }

    protected override ModelType GetModelType()
    {
        return ModelType.SupportVectorRegression;
    }
}