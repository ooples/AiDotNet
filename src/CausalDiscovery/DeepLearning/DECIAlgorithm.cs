using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// DECI — Deep End-to-end Causal Inference.
/// </summary>
/// <remarks>
/// <para>
/// DECI is a flow-based variational inference method that jointly learns the causal graph
/// and the functional relationships between variables. It uses normalizing flows to model
/// flexible conditional distributions and a variational distribution over DAGs.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize edge logits Z[i,j] and per-variable MLPs (location networks)</item>
/// <item>Sample soft adjacency: A[i,j] = sigmoid(Z[i,j] / tau) with temperature annealing</item>
/// <item>For each target j: predict location f_j = MLP_j(masked input by A[:,j])</item>
/// <item>Compute log-likelihood using Gaussian noise model: log p(x_j | pa(j))</item>
/// <item>Add KL divergence for edge distribution and NOTEARS acyclicity penalty</item>
/// <item>Optimize ELBO via gradient descent on Z and MLP weights</item>
/// <item>Threshold final edge probabilities to get DAG</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> DECI simultaneously learns "which variables cause which" and
/// "how they cause each other." It's particularly good at handling complex, non-standard
/// relationships and can also estimate intervention effects.
/// </para>
/// <para>
/// Reference: Geffner et al. (2022), "Deep End-to-end Causal Inference", arXiv.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Deep End-to-end Causal Inference", "https://arxiv.org/abs/2202.02195", Year = 2022, Authors = "Tomas Geffner, Javier Antoran, Adam Foster, Wenbo Gong, Chao Ma, Emre Kiciman, Amit Sharma, Angus Lamb, Martin Kukla, Nick Pawlowski, Miltiadis Allamanis, Cheng Zhang")]
public class DECIAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "DECI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public DECIAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / d));
        var cov = ComputeCovarianceMatrix(data);
        T eps = NumOps.FromDouble(1e-10);

        // Edge logits Z[i,j] for variational distribution over graph
        var Z = new Matrix<T>(d, d);

        // Per-variable location networks: W1[j] (d x h), W2[j] (h x 1)
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

        // Per-variable log-variance for Gaussian noise model
        var logVar = new T[d];
        for (int j = 0; j < d; j++)
            logVar[j] = NumOps.Zero;

        T lr = NumOps.FromDouble(LearningRate);
        T alpha = NumOps.Zero;
        T rho = NumOps.One;
        T lambda1 = NumOps.FromDouble(0.1); // Sparsity prior

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            double tau = Math.Max(0.1, 1.0 * Math.Pow(0.95, epoch));
            T tauT = NumOps.FromDouble(tau);
            T invN = NumOps.FromDouble(1.0 / n);

            // Compute edge probabilities
            var P = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T zScaled = NumOps.Divide(Z[i, j], tauT);
                    double sv = NumOps.ToDouble(zScaled);
                    P[i, j] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                }

            var gZ = new Matrix<T>(d, d);
            var gW1 = new Matrix<T>[d];
            var gW2 = new Matrix<T>[d];
            var gLogVar = new T[d];
            for (int j = 0; j < d; j++)
            {
                gW1[j] = new Matrix<T>(d, h);
                gW2[j] = new Matrix<T>(h, 1);
                gLogVar[j] = NumOps.Zero;
            }

            // Forward pass and gradient computation
            for (int s = 0; s < n; s++)
            {
                for (int j = 0; j < d; j++)
                {
                    // Masked input: x_masked[i] = x[i] * P[i,j]
                    var hidden = new T[h];
                    for (int k = 0; k < h; k++)
                    {
                        T sum = NumOps.Zero;
                        for (int i = 0; i < d; i++)
                            sum = NumOps.Add(sum, NumOps.Multiply(
                                NumOps.Multiply(data[s, i], P[i, j]), W1[j][i, k]));
                        double sv = NumOps.ToDouble(sum);
                        hidden[k] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                    }

                    T pred = NumOps.Zero;
                    for (int k = 0; k < h; k++)
                        pred = NumOps.Add(pred, NumOps.Multiply(hidden[k], W2[j][k, 0]));

                    // Gaussian log-likelihood gradient: d/d_mu [ -0.5 * (x-mu)^2 / var ]
                    T noiseVar = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logVar[j])));
                    T invVar = NumOps.Divide(NumOps.One, NumOps.Add(noiseVar, eps));
                    T residual = NumOps.Subtract(pred, data[s, j]);
                    T scaledResidual = NumOps.Multiply(NumOps.Multiply(residual, invVar), invN);

                    // Log-variance gradient of NLL: d/d_logvar [ 0.5*logvar + 0.5*(x-mu)^2*exp(-logvar) ]
                    // = 0.5 - 0.5*(x-mu)^2/var
                    T resSq = NumOps.Multiply(residual, residual);
                    gLogVar[j] = NumOps.Add(gLogVar[j], NumOps.Multiply(invN,
                        NumOps.Subtract(NumOps.FromDouble(0.5),
                            NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(resSq, invVar)))));

                    // Backprop through MLP
                    for (int k = 0; k < h; k++)
                    {
                        gW2[j][k, 0] = NumOps.Add(gW2[j][k, 0], NumOps.Multiply(scaledResidual, hidden[k]));

                        T sigD = NumOps.Multiply(hidden[k], NumOps.Subtract(NumOps.One, hidden[k]));
                        T dH = NumOps.Multiply(scaledResidual, NumOps.Multiply(W2[j][k, 0], sigD));

                        for (int i = 0; i < d; i++)
                        {
                            T maskedInput = NumOps.Multiply(data[s, i], P[i, j]);
                            gW1[j][i, k] = NumOps.Add(gW1[j][i, k], NumOps.Multiply(dH, maskedInput));

                            // Gradient w.r.t. edge probability
                            T dMask = NumOps.Multiply(dH, NumOps.Multiply(W1[j][i, k], data[s, i]));
                            T pSigD = NumOps.Multiply(P[i, j], NumOps.Subtract(NumOps.One, P[i, j]));
                            gZ[i, j] = NumOps.Add(gZ[i, j],
                                NumOps.Divide(NumOps.Multiply(dMask, pSigD), tauT));
                        }
                    }
                }
            }

            // NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
            var PSq = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    PSq[i, j] = NumOps.Multiply(P[i, j], P[i, j]);
            var expPSq = MatrixExponentialTaylor(PSq, d);
            T hVal = NumOps.Zero;
            for (int i = 0; i < d; i++)
                hVal = NumOps.Add(hVal, expPSq[i, i]);
            hVal = NumOps.Subtract(hVal, NumOps.FromDouble(d));

            // KL divergence + acyclicity gradients on edge logits
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T pij = P[i, j];
                    T oneMinusP = NumOps.Subtract(NumOps.One, pij);
                    // KL(Bernoulli(p) || Bernoulli(prior)) where prior is small
                    T klGrad = NumOps.Multiply(lambda1, pij);
                    // Acyclicity gradient: (alpha + rho*h) * [exp(P∘P)^T ∘ 2P][i,j]
                    T acycGrad = NumOps.Multiply(
                        NumOps.Add(alpha, NumOps.Multiply(rho, hVal)),
                        NumOps.Multiply(expPSq[j, i], NumOps.Multiply(NumOps.FromDouble(2), pij)));
                    T totalGrad = NumOps.Add(klGrad, acycGrad);
                    T pSigD = NumOps.Divide(NumOps.Multiply(pij, oneMinusP),
                        NumOps.Add(tauT, eps));
                    gZ[i, j] = NumOps.Add(gZ[i, j], NumOps.Multiply(totalGrad, pSigD));
                }

            // Apply gradients
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    if (i != j)
                        Z[i, j] = NumOps.Subtract(Z[i, j], NumOps.Multiply(lr, gZ[i, j]));

            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < d; i++)
                    for (int k = 0; k < h; k++)
                        W1[j][i, k] = NumOps.Subtract(W1[j][i, k], NumOps.Multiply(lr, gW1[j][i, k]));
                for (int k = 0; k < h; k++)
                    W2[j][k, 0] = NumOps.Subtract(W2[j][k, 0], NumOps.Multiply(lr, gW2[j][k, 0]));
                logVar[j] = NumOps.Subtract(logVar[j],
                    NumOps.Multiply(NumOps.FromDouble(LearningRate * 0.1), gLogVar[j]));
                // Zero diagonal
                for (int k = 0; k < h; k++)
                    W1[j][j, k] = NumOps.Zero;
            }

            // Update augmented Lagrangian with NOTEARS h(P) = tr(exp(P∘P)) - d
            alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hVal));
            T rhoMax = NumOps.FromDouble(1e+16);
            if (NumOps.GreaterThan(hVal, NumOps.FromDouble(0.25)) && !NumOps.GreaterThan(rho, rhoMax))
                rho = NumOps.Multiply(rho, NumOps.FromDouble(10));
        }

        // Final output: threshold edge probabilities and compute OLS weights
        var result = new Matrix<T>(d, d);
        T edgeThreshold = NumOps.FromDouble(0.5);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                T zScaled = NumOps.Divide(Z[i, j], NumOps.FromDouble(0.1));
                double sv = NumOps.ToDouble(zScaled);
                double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                T prob = NumOps.FromDouble(sigVal);

                if (NumOps.GreaterThan(prob, edgeThreshold))
                {
                    T varI = cov[i, i];
                    if (NumOps.GreaterThan(varI, eps))
                    {
                        T weight = NumOps.Divide(cov[i, j], varI);
                        if (NumOps.GreaterThan(NumOps.Abs(weight), NumOps.FromDouble(0.1)))
                            result[i, j] = weight;
                    }
                }
            }

        return result;
    }

}
