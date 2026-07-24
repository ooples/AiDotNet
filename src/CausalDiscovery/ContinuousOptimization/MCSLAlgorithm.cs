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
[ResearchPaper("Masked Gradient-Based Causal Structure Learning", "https://doi.org/10.1137/1.9781611976700.63", Year = 2021, Authors = "Ignavier Ng, Shengyu Zhu, Zhitang Chen, Zhuangyan Fang")]
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
        // Paper (Ng et al. 2021, §4) trains each augmented-Lagrangian subproblem with
        // Adam at learning rate 3e-2 for ~1000 steps. Plain SGD at 1e-3 for 30 steps
        // (the previous defaults) never moves the weights before the penalty rho
        // escalates, collapsing every edge to zero.
        _learningRateValue = options?.LearningRate ?? 0.03;
        _innerIterations = options?.InnerIterations ?? 100;
        if (options?.MaxPenalty is { } maxPenalty) _rhoMax = maxPenalty;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        var X = StandardizeData(data);

        // Initialize W = 0 and M (mask logits U) = 0, matching NOTEARS and the paper.
        // A dense OLS/correlation warm start biases the learned mask toward spurious
        // strongly-correlated pairs (e.g. two effects of a common cause) and prevents
        // recovery of the true sparse structure — the mask locks onto whichever edges
        // start large. Growing the edges from zero lets the L1 + acyclicity terms
        // select the correct sparse DAG, exactly as the sibling NOTEARS solver does.
        var W = new Matrix<T>(d, d);
        var M = new Matrix<T>(d, d);

        // Fixed temperature. The paper uses tau = 0.2 throughout (not annealed):
        // a small constant temperature keeps sigmoid(U/tau) close to a hard {0,1}
        // mask while remaining differentiable.
        const double tau = 0.2;

        // Augmented-Lagrangian schedule following the paper: rho0 = 10^(-ceil(0.3 d)),
        // gamma = 0.25, alpha0 = 0. rho only escalates when the acyclicity constraint
        // fails to make sufficient progress.
        double rho = Math.Pow(10.0, -Math.Ceiling(0.3 * d));
        double alpha = 0.0, prevH = double.MaxValue;
        const double gamma = 0.25;

        // Adam optimizer state (paper trains each subproblem with Adam). Adam's
        // per-coordinate step normalization is essential here: when rho grows large
        // the raw acyclicity gradient dominates, and plain SGD either stalls (tiny
        // lr) or explodes (large lr). Adam keeps a stable ~lr-sized step regardless.
        double lr = _learningRateValue;
        const double beta1 = 0.9, beta2 = 0.999, epsAdam = 1e-8;
        var mW = new double[d, d];
        var vW = new double[d, d];
        var mM = new double[d, d];
        var vM = new double[d, d];
        int adamT = 0;

        for (int outerIter = 0; outerIter < MaxIterations; outerIter++)
        {
            for (int innerIter = 0; innerIter < _innerIterations; innerIter++)
            {
                adamT++;

                // Effective adjacency: W_eff[i,j] = W[i,j] * sigmoid(M[i,j] / tau)
                var Weff = new Matrix<T>(d, d);
                var mask = new Matrix<T>(d, d);
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                    {
                        double sigmoidVal = Sigmoid(NumOps.ToDouble(M[i, j]) / tau);
                        mask[i, j] = NumOps.FromDouble(sigmoidVal);
                        Weff[i, j] = NumOps.Multiply(W[i, j], mask[i, j]);
                    }

                // Loss and acyclicity constraint on W_eff
                var (_, lossGrad) = ComputeL2Loss(X, Weff);
                var (h, hGrad) = ComputeNOTEARSConstraint(Weff);
                double augCoeff = alpha + rho * h;

                double biasCorr1 = 1.0 - Math.Pow(beta1, adamT);
                double biasCorr2 = 1.0 - Math.Pow(beta2, adamT);

                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                    {
                        if (i == j) continue;

                        double wVal = NumOps.ToDouble(W[i, j]);
                        double maskVal = NumOps.ToDouble(mask[i, j]);

                        // Gradient w.r.t. W_eff = loss grad + augmented acyclicity grad
                        // plus an L1 sparsity subgradient on the effective edge
                        // (paper penalizes lambda * ||g_tau(U)||_1).
                        double gEff = NumOps.ToDouble(lossGrad[i, j])
                                    + augCoeff * NumOps.ToDouble(hGrad[i, j])
                                    + Lambda1 * Math.Sign(wVal * maskVal);

                        // Chain rule to the underlying parameters:
                        //   dL/dW = gEff * mask
                        //   dL/dM = gEff * W * sigmoid'(M/tau)/tau,  sigmoid' = s(1-s)
                        double gradW = gEff * maskVal;
                        double gradM = gEff * wVal * (maskVal * (1.0 - maskVal) / tau);

                        // Adam step for W
                        mW[i, j] = beta1 * mW[i, j] + (1.0 - beta1) * gradW;
                        vW[i, j] = beta2 * vW[i, j] + (1.0 - beta2) * gradW * gradW;
                        double mHatW = mW[i, j] / biasCorr1;
                        double vHatW = vW[i, j] / biasCorr2;
                        W[i, j] = NumOps.FromDouble(wVal - lr * mHatW / (Math.Sqrt(vHatW) + epsAdam));

                        // Adam step for M (mask logits)
                        mM[i, j] = beta1 * mM[i, j] + (1.0 - beta1) * gradM;
                        vM[i, j] = beta2 * vM[i, j] + (1.0 - beta2) * gradM * gradM;
                        double mHatM = mM[i, j] / biasCorr1;
                        double vHatM = vM[i, j] / biasCorr2;
                        double mCur = NumOps.ToDouble(M[i, j]);
                        M[i, j] = NumOps.FromDouble(mCur - lr * mHatM / (Math.Sqrt(vHatM) + epsAdam));
                    }
            }

            // Evaluate acyclicity on the masked weights and update the multipliers.
            var Wfinal = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    Wfinal[i, j] = NumOps.Multiply(W[i, j],
                                   NumOps.FromDouble(Sigmoid(NumOps.ToDouble(M[i, j]) / tau)));

            var (hVal, _) = ComputeNOTEARSConstraint(Wfinal);
            alpha += rho * hVal;
            if (hVal > gamma * prevH) rho = Math.Min(rho * 10.0, _rhoMax);
            prevH = hVal;

            if (hVal < HTolerance || rho >= _rhoMax) break;
        }

        // Final effective adjacency using the learned soft mask, then prune by weight.
        var maskedW = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                double maskVal = Sigmoid(NumOps.ToDouble(M[i, j]) / tau);
                maskedW[i, j] = NumOps.Multiply(W[i, j], NumOps.FromDouble(maskVal));
            }

        return ThresholdWithFallback(maskedW, WThreshold, data);
    }

    private static double Sigmoid(double x)
    {
        if (x > 20) return 1.0;
        if (x < -20) return 0.0;
        return 1.0 / (1.0 + Math.Exp(-x));
    }
}
