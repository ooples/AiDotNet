using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// CASTLE — Causal Structure Learning via neural networks with shared masked architecture.
/// </summary>
/// <remarks>
/// <para>
/// CASTLE trains a neural network to predict each variable from the others, using a shared
/// mask layer M that represents the adjacency matrix. For each target variable j, the
/// input is masked by column j of M, then passed through shared hidden layers. The mask
/// M is learned jointly with the network weights via L1 regularization and NOTEARS
/// acyclicity constraint on M.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize mask M (d x d) with sigmoid logits and shared MLP weights</item>
/// <item>For target j: input = X * diag(sigmoid(M[:,j])), excluding j</item>
/// <item>Pass masked input through shared hidden layers (sigmoid activation)</item>
/// <item>Compute MSE loss between prediction and X[:,j]</item>
/// <item>Add L1 penalty on M and NOTEARS h(sigmoid(M)) = 0</item>
/// <item>Update M and MLP weights via gradient descent</item>
/// <item>Threshold final sigmoid(M) to get adjacency</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CASTLE uses a neural network with a special "mask" that learns
/// which inputs matter for predicting each variable. The mask ends up representing the
/// causal graph — connections that help prediction stay, others are removed.
/// </para>
/// <para>
/// Reference: Kyono et al. (2020), "CASTLE: Regularization via Auxiliary Causal Graph Discovery", NeurIPS.
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
[ModelPaper("CASTLE: Regularization via Auxiliary Causal Graph Discovery", "https://proceedings.neurips.cc/paper/2020/hash/1f8d87e1461a3d422a3e0eaa8e945e19-Abstract.html", Year = 2020, Authors = "Trent Kyono, Yao Zhang, Mihaela van der Schaar")]
public class CASTLEAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CASTLE";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public CASTLEAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T initScale = NumOps.FromDouble(Math.Sqrt(2.0 / d));

        // Mask logits M[i,j]: sigmoid(M[i,j]) = probability of edge i→j
        var M = new Matrix<T>(d, d);

        // Shared MLP: W_h (d x h), W_o (h x 1)
        var Wh = new Matrix<T>(d, h);
        var Wo = new Matrix<T>(h, 1);
        for (int i = 0; i < d; i++)
            for (int k = 0; k < h; k++)
                Wh[i, k] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));
        for (int k = 0; k < h; k++)
            Wo[k, 0] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));

        T lr = NumOps.FromDouble(LearningRate);
        T lambda1 = NumOps.FromDouble(0.1);
        T rho = NumOps.One;

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            var gM = new Matrix<T>(d, d);
            var gWh = new Matrix<T>(d, h);
            var gWo = new Matrix<T>(h, 1);
            T invN = NumOps.FromDouble(1.0 / n);

            // Compute mask probabilities
            var mask = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) { mask[i, j] = NumOps.Zero; continue; }
                    double sv = NumOps.ToDouble(M[i, j]);
                    mask[i, j] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                }

            for (int s = 0; s < n; s++)
            {
                for (int j = 0; j < d; j++)
                {
                    // Masked input: x_masked[i] = x[i] * mask[i,j]
                    var xMasked = new Vector<T>(d);
                    for (int i = 0; i < d; i++)
                        xMasked[i] = NumOps.Multiply(data[s, i], mask[i, j]);

                    var hidden = new Vector<T>(h);
                    for (int k = 0; k < h; k++)
                    {
                        var whCol = new Vector<T>(d);
                        for (int i = 0; i < d; i++) whCol[i] = Wh[i, k];
                        double sv = NumOps.ToDouble(Engine.DotProduct(xMasked, whCol));
                        hidden[k] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                    }

                    var woCol = new Vector<T>(h);
                    for (int k = 0; k < h; k++) woCol[k] = Wo[k, 0];
                    T pred = Engine.DotProduct(hidden, woCol);

                    T residual = NumOps.Multiply(NumOps.Subtract(pred, data[s, j]), invN);

                    // Backprop through shared MLP
                    for (int k = 0; k < h; k++)
                    {
                        gWo[k, 0] = NumOps.Add(gWo[k, 0], NumOps.Multiply(residual, hidden[k]));

                        T sigD = NumOps.Multiply(hidden[k], NumOps.Subtract(NumOps.One, hidden[k]));
                        T dH = NumOps.Multiply(residual, NumOps.Multiply(Wo[k, 0], sigD));

                        for (int i = 0; i < d; i++)
                        {
                            T maskedInput = NumOps.Multiply(data[s, i], mask[i, j]);
                            gWh[i, k] = NumOps.Add(gWh[i, k], NumOps.Multiply(dH, maskedInput));

                            // Gradient w.r.t. mask
                            T dMask = NumOps.Multiply(dH, NumOps.Multiply(Wh[i, k], data[s, i]));
                            T maskSigD = NumOps.Multiply(mask[i, j], NumOps.Subtract(NumOps.One, mask[i, j]));
                            gM[i, j] = NumOps.Add(gM[i, j], NumOps.Multiply(dMask, maskSigD));
                        }
                    }
                }
            }

            // NOTEARS acyclicity: h(mask) = tr(exp(mask∘mask)) - d
            var maskSq = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    maskSq[i, j] = NumOps.Multiply(mask[i, j], mask[i, j]);
            var expMaskSq = MatrixExponentialTaylor(maskSq, d);

            // L1 and acyclicity gradients on mask
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    // L1 on mask
                    T l1Grad = NumOps.Multiply(lambda1,
                        NumOps.FromDouble(Math.Sign(NumOps.ToDouble(mask[i, j]))));
                    // Acyclicity gradient: d/dM[i,j] tr(exp(M∘M)) = [exp(M∘M)^T ∘ 2M][i,j]
                    T acycGrad = NumOps.Multiply(rho,
                        NumOps.Multiply(expMaskSq[j, i], NumOps.Multiply(NumOps.FromDouble(2), mask[i, j])));
                    T mSigD = NumOps.Multiply(mask[i, j], NumOps.Subtract(NumOps.One, mask[i, j]));
                    gM[i, j] = NumOps.Add(gM[i, j],
                        NumOps.Multiply(NumOps.Add(l1Grad, acycGrad), mSigD));
                }

            // Apply gradients
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    if (i != j)
                        M[i, j] = NumOps.Subtract(M[i, j], NumOps.Multiply(lr, gM[i, j]));

            for (int i = 0; i < d; i++)
                for (int k = 0; k < h; k++)
                    Wh[i, k] = NumOps.Subtract(Wh[i, k], NumOps.Multiply(lr, gWh[i, k]));
            for (int k = 0; k < h; k++)
                Wo[k, 0] = NumOps.Subtract(Wo[k, 0], NumOps.Multiply(lr, gWo[k, 0]));

            rho = NumOps.Multiply(rho, NumOps.FromDouble(1.05));
        }

        // Compute final mask and threshold
        var cov = ComputeCovarianceMatrix(data);
        var learnedP = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                double sv = NumOps.ToDouble(M[i, j]);
                learnedP[i, j] = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
            }

        return BuildFinalAdjacency(learnedP, cov, d);
    }

}
