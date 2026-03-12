using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// GraN-DAG — Gradient-based Neural DAG Learning.
/// </summary>
/// <remarks>
/// <para>
/// GraN-DAG parameterizes each structural equation f_j as a neural network with sigmoid
/// activations. The weighted adjacency matrix A[i,j] = ||W1_j[:,i]||_2 is derived from
/// the first-layer input weights. Path-specific connectivity through the MLP gives a
/// refined adjacency measure. The NOTEARS acyclicity constraint h(A) = tr(e^(A*A)) - d
/// is enforced via augmented Lagrangian.
/// </para>
/// <para>
/// <b>For Beginners:</b> GraN-DAG trains a separate neural network for each variable to
/// predict it from the others. The "importance" of each input connection tells us the
/// causal strength, while a mathematical constraint ensures no circular causation.
/// </para>
/// <para>
/// Reference: Lachapelle et al. (2020), "Gradient-Based Neural DAG Learning", ICLR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Gradient-Based Neural DAG Learning", "https://openreview.net/forum?id=rklbKA4YDS", Year = 2020, Authors = "Sebastien Lachapelle, Philippe Brouillard, Tristan Deleu, Simon Lacoste-Julien")]
public class GraNDAGAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "GraN-DAG";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public GraNDAGAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / d));

        // Per-variable MLPs: W1[j] is d x h, W2[j] is h x 1
        var W1 = new Matrix<T>[d];
        var W2 = new Matrix<T>[d];
        for (int j = 0; j < d; j++)
        {
            W1[j] = new Matrix<T>(d, h);
            W2[j] = new Matrix<T>(h, 1);
            for (int i = 0; i < d; i++)
                if (i != j)
                    for (int k = 0; k < h; k++)
                        W1[j][i, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
            for (int k = 0; k < h; k++)
                W2[j][k, 0] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
        }

        T lr = NumOps.FromDouble(LearningRate);
        T alpha = NumOps.Zero;
        T rho = NumOps.One;
        T rhoMax = NumOps.FromDouble(1e+16);

        for (int outer = 0; outer < MaxEpochs; outer++)
        {
            for (int inner = 0; inner < 20; inner++)
            {
                // Compute loss gradients per sample per variable
                var gW1 = new Matrix<T>[d];
                var gW2 = new Matrix<T>[d];
                for (int j = 0; j < d; j++)
                {
                    gW1[j] = new Matrix<T>(d, h);
                    gW2[j] = new Matrix<T>(h, 1);
                }

                T invN = NumOps.FromDouble(1.0 / n);
                for (int s = 0; s < n; s++)
                {
                    for (int j = 0; j < d; j++)
                    {
                        // Forward: hidden = sigmoid(W1^T * x), output = W2^T * hidden
                        var hidden = new T[h];
                        for (int k = 0; k < h; k++)
                        {
                            T sum = NumOps.Zero;
                            for (int i = 0; i < d; i++)
                                sum = NumOps.Add(sum, NumOps.Multiply(data[s, i], W1[j][i, k]));
                            double sv = NumOps.ToDouble(sum);
                            hidden[k] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                        }

                        T pred = NumOps.Zero;
                        for (int k = 0; k < h; k++)
                            pred = NumOps.Add(pred, NumOps.Multiply(hidden[k], W2[j][k, 0]));

                        T residual = NumOps.Multiply(NumOps.Subtract(pred, data[s, j]), invN);

                        // Backprop
                        for (int k = 0; k < h; k++)
                        {
                            gW2[j][k, 0] = NumOps.Add(gW2[j][k, 0], NumOps.Multiply(residual, hidden[k]));

                            T sigDeriv = NumOps.Multiply(hidden[k], NumOps.Subtract(NumOps.One, hidden[k]));
                            T dHidden = NumOps.Multiply(residual, NumOps.Multiply(W2[j][k, 0], sigDeriv));
                            for (int i = 0; i < d; i++)
                                gW1[j][i, k] = NumOps.Add(gW1[j][i, k], NumOps.Multiply(dHidden, data[s, i]));
                        }
                    }
                }

                // Acyclicity gradient on adjacency A[i,j] = ||W1[j][:,i]||_2
                var A = ExtractAdjacency(W1, d, h);
                T hVal = ComputeTraceExpConstraint(A, d);
                T augCoeff = NumOps.Add(alpha, NumOps.Multiply(rho, hVal));

                // Chain rule: dh/dW1 via dh/dA * dA/dW1
                var (_, hGrad) = ComputeExpGradient(A, d);
                for (int j = 0; j < d; j++)
                    for (int i = 0; i < d; i++)
                    {
                        if (i == j) continue;
                        T aij = A[i, j];
                        if (!NumOps.GreaterThan(aij, NumOps.FromDouble(1e-12))) continue;

                        T dhda = NumOps.Multiply(augCoeff, hGrad[i, j]);
                        for (int k = 0; k < h; k++)
                        {
                            T w1val = W1[j][i, k];
                            T grad = NumOps.Divide(NumOps.Multiply(dhda, w1val), aij);
                            gW1[j][i, k] = NumOps.Add(gW1[j][i, k], grad);
                        }
                    }

                // Update
                for (int j = 0; j < d; j++)
                {
                    for (int i = 0; i < d; i++)
                        for (int k = 0; k < h; k++)
                            W1[j][i, k] = NumOps.Subtract(W1[j][i, k], NumOps.Multiply(lr, gW1[j][i, k]));
                    for (int k = 0; k < h; k++)
                        W2[j][k, 0] = NumOps.Subtract(W2[j][k, 0], NumOps.Multiply(lr, gW2[j][k, 0]));

                    // Zero diagonal
                    for (int k = 0; k < h; k++)
                        W1[j][j, k] = NumOps.Zero;
                }
            }

            // Outer: update augmented Lagrangian
            var Afinal = ExtractAdjacency(W1, d, h);
            T hFinal = ComputeTraceExpConstraint(Afinal, d);
            alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hFinal));
            if (NumOps.GreaterThan(hFinal, NumOps.FromDouble(0.25)))
                rho = NumOps.Multiply(rho, NumOps.FromDouble(10));
            if (NumOps.GreaterThan(rho, rhoMax)) break;
            if (!NumOps.GreaterThan(hFinal, NumOps.FromDouble(1e-8))) break;
        }

        var result = ExtractAdjacency(W1, d, h);
        // Threshold
        T wThreshold = NumOps.FromDouble(0.3);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (!NumOps.GreaterThan(NumOps.Abs(result[i, j]), wThreshold))
                    result[i, j] = NumOps.Zero;

        return result;
    }

    private Matrix<T> ExtractAdjacency(Matrix<T>[] W1, int d, int h)
    {
        var A = new Matrix<T>(d, d);
        for (int j = 0; j < d; j++)
            for (int i = 0; i < d; i++)
            {
                if (i == j) continue;
                T norm = NumOps.Zero;
                for (int k = 0; k < h; k++)
                    norm = NumOps.Add(norm, NumOps.Multiply(W1[j][i, k], W1[j][i, k]));
                A[i, j] = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(norm)));
            }
        return A;
    }

    private T ComputeTraceExpConstraint(Matrix<T> A, int d)
    {
        // h(A) = tr(e^(A∘A)) - d using power series: exp(M) = I + M + M^2/2! + ...
        var AA = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                AA[i, j] = NumOps.Multiply(A[i, j], A[i, j]);

        var power = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++) power[i, i] = NumOps.One;
        var expM = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++) expM[i, i] = NumOps.One;

        for (int p = 1; p <= 10; p++)
        {
            // power = power * AA (unscaled M^p)
            power = MatMul(power, AA);
            T fact = NumOps.FromDouble(1.0 / Factorial(p));
            // expM += power / p!
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    expM[i, j] = NumOps.Add(expM[i, j], NumOps.Multiply(power[i, j], fact));
        }

        T trace = NumOps.Zero;
        for (int i = 0; i < d; i++)
            trace = NumOps.Add(trace, expM[i, i]);
        return NumOps.Subtract(trace, NumOps.FromDouble(d));
    }

    private (T h, Matrix<T> grad) ComputeExpGradient(Matrix<T> A, int d)
    {
        // Gradient: dh/dA[i,j] = 2 * A[i,j] * (e^(A∘A))^T[i,j]
        var AA = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                AA[i, j] = NumOps.Multiply(A[i, j], A[i, j]);

        var expM = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++) expM[i, i] = NumOps.One;
        var power = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++) power[i, i] = NumOps.One;

        for (int p = 1; p <= 10; p++)
        {
            // power = power * AA (unscaled M^p)
            power = MatMul(power, AA);
            T fact = NumOps.FromDouble(1.0 / Factorial(p));
            // expM += power / p!
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    expM[i, j] = NumOps.Add(expM[i, j], NumOps.Multiply(power[i, j], fact));
        }

        T trace = NumOps.Zero;
        for (int i = 0; i < d; i++)
            trace = NumOps.Add(trace, expM[i, i]);
        T h = NumOps.Subtract(trace, NumOps.FromDouble(d));

        var grad = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                grad[i, j] = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(A[i, j], expM[j, i]));

        return (h, grad);
    }

    private static double Factorial(int n)
    {
        double result = 1;
        for (int i = 2; i <= n; i++) result *= i;
        return result;
    }
}
