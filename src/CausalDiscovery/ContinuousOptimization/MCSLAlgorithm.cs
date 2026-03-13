using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// MCSL — Masked Gradient-Based Causal Structure Learning.
/// </summary>
/// <remarks>
/// <para>
/// MCSL learns causal structure by maintaining a separate mask matrix M alongside the
/// weight matrix W. The effective adjacency is W * sigmoid(M/tau), where tau is a
/// temperature parameter that anneals from soft to hard masks. This separation of
/// structure (M) and weights (W) enables cleaner sparsity than L1 alone.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize W from pairwise OLS and M (mask logits) = 0</item>
/// <item>Compute soft mask: mask = sigmoid(M / temperature)</item>
/// <item>Effective adjacency: W_eff = W * mask (element-wise)</item>
/// <item>Compute L2 loss on W_eff and NOTEARS acyclicity constraint</item>
/// <item>Update W and M via gradient descent with augmented Lagrangian</item>
/// <item>Anneal temperature: decrease over iterations (soft → hard mask)</item>
/// <item>Threshold final W * sigmoid(M / tau_final)</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> MCSL adds a clever trick on top of NOTEARS. Instead of learning edge
/// weights directly and then thresholding, it learns a separate "switch" for each edge (on/off)
/// along with the weight. This makes it easier for the algorithm to decide which edges should
/// exist vs. not exist, leading to sparser and often more accurate graphs.
/// </para>
/// <para>
/// Reference: Ng et al. (2021), "Masked Gradient-Based Causal Structure Learning", SDM.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Optimization)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Masked Gradient-Based Causal Structure Learning", "https://doi.org/10.1137/1.9781611976700.63", Year = 2021, Authors = "Ignavier Ng, Shengyu Zhu, Zhitang Chen, Zhuangyan Fang")]
public class MCSLAlgorithm<T> : ContinuousOptimizationBase<T>
{
    private readonly double _learningRateValue;
    private readonly int _innerIterations;
    private double _rhoMax = 1e+16;

    /// <inheritdoc/>
    public override string Name => "MCSL";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes MCSL with optional configuration.
    /// </summary>
    public MCSLAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyOptions(options);
        _learningRateValue = options?.LearningRate ?? 0.001;
        _innerIterations = options?.InnerIterations ?? 30;
        if (options?.MaxPenalty is { } maxPenalty) _rhoMax = maxPenalty;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        var X = StandardizeData(data);

        // Initialize W from pairwise OLS and M (mask logits) = 0
        var W = new Matrix<T>(d, d);
        var M = new Matrix<T>(d, d); // mask logits, initialized to zero

        // Initialize W from covariance
        var (_, initGrad) = ComputeL2Loss(X, W);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (i != j)
                    W[i, j] = NumOps.Negate(initGrad[i, j]); // rough OLS initialization

        T lr = NumOps.FromDouble(_learningRateValue);
        double rho = 1.0, alpha = 0.0, prevH = double.MaxValue;

        for (int outerIter = 0; outerIter < MaxIterations; outerIter++)
        {
            // Temperature annealing: tau starts at 1.0, decreases to 0.1
            double tau = Math.Max(0.1, Math.Pow(0.95, outerIter));

            for (int innerIter = 0; innerIter < _innerIterations; innerIter++)
            {
                // Compute effective adjacency: W_eff[i,j] = W[i,j] * sigmoid(M[i,j] / tau)
                var Weff = new Matrix<T>(d, d);
                var mask = new Matrix<T>(d, d);

                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                    {
                        double mVal = NumOps.ToDouble(M[i, j]) / tau;
                        double sigmoidVal = Sigmoid(mVal);
                        mask[i, j] = NumOps.FromDouble(sigmoidVal);
                        Weff[i, j] = NumOps.Multiply(W[i, j], mask[i, j]);
                    }

                // Loss and constraint on W_eff
                var (loss, lossGrad) = ComputeL2Loss(X, Weff);
                var (h, hGrad) = ComputeNOTEARSConstraint(Weff);

                T augCoeff = NumOps.FromDouble(alpha + rho * h);

                // Total gradient w.r.t. W_eff
                var totalGrad = new Matrix<T>(d, d);
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        totalGrad[i, j] = NumOps.Add(lossGrad[i, j],
                                          NumOps.Multiply(augCoeff, hGrad[i, j]));

                // Chain rule: dL/dW = dL/dWeff * mask, dL/dM = dL/dWeff * W * sigmoid'(M/tau) / tau
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                    {
                        if (i == j) continue;

                        T gradW = NumOps.Multiply(totalGrad[i, j], mask[i, j]);
                        W[i, j] = NumOps.Subtract(W[i, j], NumOps.Multiply(lr, gradW));

                        // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x)) / tau
                        double sigVal = NumOps.ToDouble(mask[i, j]);
                        double sigmoidDeriv = sigVal * (1.0 - sigVal) / tau;
                        T gradM = NumOps.Multiply(totalGrad[i, j],
                                  NumOps.Multiply(W[i, j], NumOps.FromDouble(sigmoidDeriv)));
                        M[i, j] = NumOps.Subtract(M[i, j], NumOps.Multiply(lr, gradM));
                    }
            }

            // Evaluate constraint
            var Wfinal = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    Wfinal[i, j] = NumOps.Multiply(W[i, j],
                                   NumOps.FromDouble(Sigmoid(NumOps.ToDouble(M[i, j]) / tau)));

            var (hVal, _) = ComputeNOTEARSConstraint(Wfinal);
            alpha += rho * hVal;
            if (hVal > 0.25 * prevH) rho = Math.Min(rho * 10, _rhoMax);
            prevH = hVal;

            if (hVal < HTolerance || rho >= _rhoMax) break;
        }

        // Final: hard mask and threshold
        var result = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                double maskVal = Sigmoid(NumOps.ToDouble(M[i, j]) / 0.1);
                T weight = NumOps.Multiply(W[i, j], NumOps.FromDouble(maskVal));
                if (Math.Abs(NumOps.ToDouble(weight)) >= WThreshold)
                    result[i, j] = weight;
            }

        return result;
    }

    private static double Sigmoid(double x)
    {
        if (x > 20) return 1.0;
        if (x < -20) return 0.0;
        return 1.0 / (1.0 + Math.Exp(-x));
    }
}
