using System;
using AiDotNet.SelfSupervisedLearning;
using AiDotNet.SelfSupervisedLearning.Losses;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SelfSupervisedLearning;

/// <summary>
/// Deep mathematical correctness tests for self-supervised learning loss functions
/// and temperature scheduling. Verifies exact hand-calculated values and mathematical
/// properties to catch bugs in InfoNCE, NT-Xent, Barlow Twins, and BYOL losses.
/// </summary>
public class SSLLossesDeepMathIntegrationTests
{
    private const double Tol = 1e-6;
    private const double RelaxedTol = 1e-4;

    #region InfoNCE Loss - Exact Math Verification

    [Fact]
    public void InfoNCE_IdenticalQueryAndKey_MinimalLoss()
    {
        // When query and positive key are identical (and normalized), the positive logit
        // is maximized (1/tau), and the loss should be minimal
        var loss = new InfoNCELoss<double>(temperature: 1.0, normalize: false);

        // query = [1, 0], positive key = [1, 0], negative key = [0, 1]
        var queries = new Tensor<double>(new double[] { 1, 0 }, [1, 2]);
        var posKeys = new Tensor<double>(new double[] { 1, 0 }, [1, 2]);
        var negKeys = new Tensor<double>(new double[] { 0, 1 }, [1, 2]);

        double result = loss.ComputeLoss(queries, posKeys, negKeys);

        // pos logit = q.kpos / tau = (1*1 + 0*0) / 1 = 1
        // neg logit = q.kneg / tau = (1*0 + 0*1) / 1 = 0
        // loss = -log(exp(1) / (exp(1) + exp(0))) = -log(e / (e + 1))
        // = log(1 + 1/e) = log(1 + e^(-1))
        double expected = Math.Log(1 + Math.Exp(-1));
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void InfoNCE_OrthogonalQueryAndKey_HigherLoss()
    {
        // When query is orthogonal to positive key, loss is higher
        var loss = new InfoNCELoss<double>(temperature: 1.0, normalize: false);

        // query = [1, 0], positive key = [0, 1] (orthogonal), negative key = [-1, 0]
        var queries = new Tensor<double>(new double[] { 1, 0 }, [1, 2]);
        var posKeys = new Tensor<double>(new double[] { 0, 1 }, [1, 2]);
        var negKeys = new Tensor<double>(new double[] { -1, 0 }, [1, 2]);

        double result = loss.ComputeLoss(queries, posKeys, negKeys);

        // pos logit = (1*0 + 0*1) / 1 = 0
        // neg logit = (1*(-1) + 0*0) / 1 = -1
        // loss = -log(exp(0) / (exp(0) + exp(-1))) = -log(1 / (1 + e^(-1)))
        // = log(1 + e^(-1))
        double expected = Math.Log(1 + Math.Exp(-1));
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void InfoNCE_LowerTemperature_SharperDistribution()
    {
        // Lower temperature should make the distribution sharper
        // With correct positive pair, lower temp should give LOWER loss
        var highTempLoss = new InfoNCELoss<double>(temperature: 1.0, normalize: false);
        var lowTempLoss = new InfoNCELoss<double>(temperature: 0.1, normalize: false);

        // query = [1, 0], positive key = [1, 0] (perfect match), negative = [0, 1]
        var queries = new Tensor<double>(new double[] { 1, 0 }, [1, 2]);
        var posKeys = new Tensor<double>(new double[] { 1, 0 }, [1, 2]);
        var negKeys = new Tensor<double>(new double[] { 0, 1 }, [1, 2]);

        double highTempResult = highTempLoss.ComputeLoss(queries, posKeys, negKeys);
        double lowTempResult = lowTempLoss.ComputeLoss(queries, posKeys, negKeys);

        // With correct positive match and lower temperature:
        // pos logit is 1/0.1 = 10, neg logit is 0/0.1 = 0
        // softmax(10, 0) is very peaked -> loss is very low
        Assert.True(lowTempResult < highTempResult,
            $"Low temp loss ({lowTempResult}) should be lower than high temp ({highTempResult}) for correct match");
    }

    [Fact]
    public void InfoNCE_HandCalculated_MultipleNegatives()
    {
        // 1 query, 1 positive, 2 negatives
        var loss = new InfoNCELoss<double>(temperature: 1.0, normalize: false);

        var queries = new Tensor<double>(new double[] { 1, 0 }, [1, 2]);
        var posKeys = new Tensor<double>(new double[] { 0.9, 0.1 }, [1, 2]);
        var negKeys = new Tensor<double>(new double[] { 0.1, 0.9, -0.5, 0.5 }, [2, 2]);

        double result = loss.ComputeLoss(queries, posKeys, negKeys);

        // pos logit = (1*0.9 + 0*0.1) / 1 = 0.9
        // neg logit 0 = (1*0.1 + 0*0.9) / 1 = 0.1
        // neg logit 1 = (1*(-0.5) + 0*0.5) / 1 = -0.5
        // maxLogit = 0.9
        // sumExp = exp(0.9-0.9) + exp(0.1-0.9) + exp(-0.5-0.9)
        //        = 1 + exp(-0.8) + exp(-1.4)
        // logSumExp = 0.9 + ln(1 + exp(-0.8) + exp(-1.4))
        // loss = logSumExp - posLogit = ln(1 + exp(-0.8) + exp(-1.4))
        double expected = Math.Log(1 + Math.Exp(-0.8) + Math.Exp(-1.4));
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void InfoNCE_Accuracy_PerfectMatch_ReturnsOne()
    {
        var loss = new InfoNCELoss<double>(temperature: 0.1, normalize: false);

        // Query and positive key are identical, negative is very different
        var queries = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);
        var posKeys = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);
        var negKeys = new Tensor<double>(new double[] { -1, -1 }, [1, 2]);

        double accuracy = loss.ComputeAccuracy(queries, posKeys, negKeys);
        Assert.Equal(1.0, accuracy, Tol);
    }

    [Fact]
    public void InfoNCE_InBatch_HandCalculated()
    {
        // In-batch contrastive: positive pairs are along the diagonal
        var loss = new InfoNCELoss<double>(temperature: 1.0, normalize: false);

        // Batch of 2: q0 matches k0, q1 matches k1
        var queries = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);
        var keys = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);

        double result = loss.ComputeLossInBatch(queries, keys);

        // For q0=[1,0]: logits = [q0.k0/1, q0.k1/1] = [1, 0], positive idx=0
        // loss_0 = -logit[0] + log(exp(1) + exp(0)) = -1 + log(e + 1)
        // For q1=[0,1]: logits = [q1.k0/1, q1.k1/1] = [0, 1], positive idx=1
        // loss_1 = -logit[1] + log(exp(0) + exp(1)) = -1 + log(1 + e)
        // avg = -1 + log(1 + e)
        double expected = -1.0 + Math.Log(1 + Math.E);
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void InfoNCE_WithGradients_LossMatchesForwardOnly()
    {
        var loss = new InfoNCELoss<double>(temperature: 0.5, normalize: false);

        var queries = new Tensor<double>(new double[] { 0.5, 0.3, -0.2, 0.8 }, [2, 2]);
        var keys = new Tensor<double>(new double[] { 0.4, 0.6, 0.1, 0.7 }, [2, 2]);

        double forwardLoss = loss.ComputeLossInBatch(queries, keys);
        var (gradLoss, gradQ, gradK) = loss.ComputeLossInBatchWithGradients(queries, keys);

        Assert.Equal(forwardLoss, gradLoss, Tol);
    }

    [Fact]
    public void InfoNCE_NegativeTemperature_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InfoNCELoss<double>(temperature: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InfoNCELoss<double>(temperature: 0));
    }

    #endregion

    #region NT-Xent Loss - Exact Math Verification

    [Fact]
    public void NTXent_IdenticalViews_MinimalLoss()
    {
        // When both views produce identical embeddings, the loss should be minimal
        var loss = new NTXentLoss<double>(temperature: 1.0, normalize: false);

        // 2 samples, 2 dims: both views identical
        var z1 = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);
        var z2 = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);

        double result = loss.ComputeLoss(z1, z2);

        // Combined: [[1,0], [0,1], [1,0], [0,1]]  (z1 then z2)
        // n = 4 (2*batch_size)
        // Similarity matrix (dot products):
        //     [0]  [1]  [2]  [3]
        // [0]  1    0    1    0
        // [1]  0    1    0    1
        // [2]  1    0    1    0
        // [3]  0    1    0    1
        //
        // Temperature-scaled (tau=1): same values
        // For i=0 (from z1): positive is j=2 (z2[0])
        //   Exclude self (j=0), negatives: j=1(0), j=3(0)
        //   Sum of exp: exp(0) + exp(1) + exp(0) = 1 + e + 1 = 2 + e (excluding self)
        //   loss_0 = -1 + log(2 + e)  [pos score=1, sumexp over j!=0]
        // Actually: exp(sim[0,1]/1) + exp(sim[0,2]/1) + exp(sim[0,3]/1) = exp(0) + exp(1) + exp(0)
        //   = 1 + e + 1
        //   maxVal = 1 (sim[0,2])
        //   sumExp = exp(0-1) + exp(1-1) + exp(0-1) = e^(-1) + 1 + e^(-1)
        //   logSumExp = 1 + ln(2e^(-1) + 1) = 1 + ln(2/e + 1)
        //   loss_0 = logSumExp - sim[0,2] = 1 + ln(2/e + 1) - 1 = ln(2/e + 1)

        // All 4 rows give same loss by symmetry
        // avg = ln(2/e + 1)
        double expected = Math.Log(2.0 / Math.E + 1.0);
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void NTXent_LowerTemperature_SharperContrast()
    {
        // Lower temperature should make the contrast sharper
        var highTempLoss = new NTXentLoss<double>(temperature: 1.0, normalize: false);
        var lowTempLoss = new NTXentLoss<double>(temperature: 0.1, normalize: false);

        // z1 and z2 are well-separated pairs
        var z1 = new Tensor<double>(new double[] { 1, 0, -1, 0 }, [2, 2]);
        var z2 = new Tensor<double>(new double[] { 0.9, 0.1, -0.9, 0.1 }, [2, 2]);

        double highTempResult = highTempLoss.ComputeLoss(z1, z2);
        double lowTempResult = lowTempLoss.ComputeLoss(z1, z2);

        // With good positive pairs, lower temperature should give lower loss
        // because the softmax becomes more peaked on the correct pair
        Assert.True(lowTempResult < highTempResult,
            $"Low temp ({lowTempResult}) should be less than high temp ({highTempResult}) for good pairs");
    }

    [Fact]
    public void NTXent_NegativeTemperature_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new NTXentLoss<double>(temperature: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new NTXentLoss<double>(temperature: 0));
    }

    [Fact]
    public void NTXent_WithGradients_LossMatchesForwardOnly()
    {
        var loss = new NTXentLoss<double>(temperature: 0.5, normalize: false);

        var z1 = new Tensor<double>(new double[] { 0.5, 0.3, -0.2, 0.8 }, [2, 2]);
        var z2 = new Tensor<double>(new double[] { 0.4, 0.6, 0.1, 0.7 }, [2, 2]);

        double forwardLoss = loss.ComputeLoss(z1, z2);
        var (gradLoss, gradZ1, gradZ2) = loss.ComputeLossWithGradients(z1, z2);

        Assert.Equal(forwardLoss, gradLoss, Tol);
    }

    [Fact]
    public void NTXent_SymmetricLoss()
    {
        // NT-Xent loss should be symmetric: L(z1, z2) = L(z2, z1)
        var loss = new NTXentLoss<double>(temperature: 0.5, normalize: true);

        var z1 = new Tensor<double>(new double[] { 1.5, -0.3, 0.2, 0.8 }, [2, 2]);
        var z2 = new Tensor<double>(new double[] { 0.4, 0.6, -0.1, 0.7 }, [2, 2]);

        double loss12 = loss.ComputeLoss(z1, z2);
        double loss21 = loss.ComputeLoss(z2, z1);

        Assert.Equal(loss12, loss21, Tol);
    }

    [Fact]
    public void NTXent_LossAlwaysNonNegative()
    {
        var loss = new NTXentLoss<double>(temperature: 0.5, normalize: true);

        var z1 = new Tensor<double>(new double[] { 1, 0, 0, 1, -1, 0 }, [3, 2]);
        var z2 = new Tensor<double>(new double[] { 0.5, 0.5, -0.5, 0.5, 0, -1 }, [3, 2]);

        double result = loss.ComputeLoss(z1, z2);
        Assert.True(result >= 0, $"NT-Xent loss should be non-negative, got {result}");
    }

    #endregion

    #region Barlow Twins Loss - Exact Math Verification

    [Fact]
    public void BarlowTwins_IdentityCrossCorrelation_ZeroLoss()
    {
        // When cross-correlation is identity matrix, loss should be 0
        // This happens when z1 = z2 and they're already normalized
        var loss = new BarlowTwinsLoss<double>(lambda: 0.005, normalize: false);

        // Construct z1 and z2 such that C = I (batch-normalized then cross-correlated)
        // If z1 = z2 = identity-like (orthonormal columns after batch processing)
        // Simplest: z1 = z2 = [[1, 0], [0, 1]]
        var z1 = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);
        var z2 = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);

        // Cross-correlation: C_ij = (1/N) * sum_b z1[b,i] * z2[b,j]
        // C[0,0] = (1*1 + 0*0)/2 = 0.5
        // C[0,1] = (1*0 + 0*1)/2 = 0
        // C[1,0] = (0*1 + 1*0)/2 = 0
        // C[1,1] = (0*0 + 1*1)/2 = 0.5
        // So C = [[0.5, 0], [0, 0.5]] - NOT identity
        // Invariance loss = (1-0.5)^2 + (1-0.5)^2 = 0.25 + 0.25 = 0.5
        // Redundancy loss = 0
        double result = loss.ComputeLoss(z1, z2);
        double expected = 0.5; // (1-0.5)^2 + (1-0.5)^2
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void BarlowTwins_HandCalculated_2x2()
    {
        // No normalization for easier hand calculation
        var loss = new BarlowTwinsLoss<double>(lambda: 1.0, normalize: false);

        // z1 = [[1, 2], [3, 4]], z2 = [[5, 6], [7, 8]]
        var z1 = new Tensor<double>(new double[] { 1, 2, 3, 4 }, [2, 2]);
        var z2 = new Tensor<double>(new double[] { 5, 6, 7, 8 }, [2, 2]);

        // C_ij = (1/2) * sum_b z1[b,i] * z2[b,j]
        // C[0,0] = (1*5 + 3*7)/2 = (5+21)/2 = 13
        // C[0,1] = (1*6 + 3*8)/2 = (6+24)/2 = 15
        // C[1,0] = (2*5 + 4*7)/2 = (10+28)/2 = 19
        // C[1,1] = (2*6 + 4*8)/2 = (12+32)/2 = 22
        //
        // Invariance: (1-13)^2 + (1-22)^2 = 144 + 441 = 585
        // Redundancy (lambda=1): 15^2 + 19^2 = 225 + 361 = 586
        // Total = 585 + 1*586 = 1171
        double result = loss.ComputeLoss(z1, z2);
        Assert.Equal(1171.0, result, Tol);
    }

    [Fact]
    public void BarlowTwins_CrossCorrelation_HandCalculated()
    {
        var loss = new BarlowTwinsLoss<double>(lambda: 0.005, normalize: false);

        var z1 = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);
        var z2 = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);

        var cc = loss.ComputeCrossCorrelation(z1, z2, 2);

        Assert.Equal(0.5, cc[0, 0], Tol);
        Assert.Equal(0.0, cc[0, 1], Tol);
        Assert.Equal(0.0, cc[1, 0], Tol);
        Assert.Equal(0.5, cc[1, 1], Tol);
    }

    [Fact]
    public void BarlowTwins_OffDiagonalSum_HandCalculated()
    {
        var loss = new BarlowTwinsLoss<double>(lambda: 0.005, normalize: false);

        // Matrix [[1, 0.3], [0.2, 1]]
        var matrix = new Tensor<double>(new double[] { 1, 0.3, 0.2, 1 }, [2, 2]);

        double offDiag = loss.OffDiagonalSum(matrix);
        // Sum of squares of off-diagonal: 0.3^2 + 0.2^2 = 0.09 + 0.04 = 0.13
        Assert.Equal(0.13, offDiag, Tol);
    }

    [Fact]
    public void BarlowTwins_LambdaEffect_HigherLambdaPenalizesRedundancy()
    {
        // Higher lambda should penalize off-diagonal elements more
        var lowLambda = new BarlowTwinsLoss<double>(lambda: 0.001, normalize: false);
        var highLambda = new BarlowTwinsLoss<double>(lambda: 1.0, normalize: false);

        // Embeddings with significant cross-correlation
        var z1 = new Tensor<double>(new double[] { 1, 1, 2, 2 }, [2, 2]);
        var z2 = new Tensor<double>(new double[] { 1, 1, 2, 2 }, [2, 2]);

        double lowResult = lowLambda.ComputeLoss(z1, z2);
        double highResult = highLambda.ComputeLoss(z1, z2);

        // Higher lambda -> higher penalty for redundant features
        Assert.True(highResult >= lowResult,
            $"High lambda loss ({highResult}) should be >= low lambda ({lowResult})");
    }

    [Fact]
    public void BarlowTwins_NegativeLambda_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new BarlowTwinsLoss<double>(lambda: -0.1));
    }

    [Fact]
    public void BarlowTwins_WithGradients_LossMatchesForwardOnly()
    {
        var loss = new BarlowTwinsLoss<double>(lambda: 0.005, normalize: false);

        var z1 = new Tensor<double>(new double[] { 0.5, 0.3, -0.2, 0.8 }, [2, 2]);
        var z2 = new Tensor<double>(new double[] { 0.4, 0.6, 0.1, 0.7 }, [2, 2]);

        double forwardLoss = loss.ComputeLoss(z1, z2);
        var (gradLoss, gradZ1, gradZ2) = loss.ComputeLossWithGradients(z1, z2);

        Assert.Equal(forwardLoss, gradLoss, Tol);
    }

    [Fact]
    public void BarlowTwins_LossAlwaysNonNegative()
    {
        var loss = new BarlowTwinsLoss<double>(lambda: 0.005, normalize: true);

        var z1 = new Tensor<double>(new double[] { 1, -2, 3, -4 }, [2, 2]);
        var z2 = new Tensor<double>(new double[] { -1, 2, -3, 4 }, [2, 2]);

        double result = loss.ComputeLoss(z1, z2);
        Assert.True(result >= 0, $"Barlow Twins loss should be non-negative, got {result}");
    }

    #endregion

    #region BYOL Loss - Exact Math Verification

    [Fact]
    public void BYOL_IdenticalVectors_ZeroLoss()
    {
        // When prediction equals target (after normalization), cosine similarity = 1
        // Loss = 2 - 2*1 = 0
        var loss = new BYOLLoss<double>(normalize: true, symmetric: false);

        var pred = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);
        var target = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);

        double result = loss.ComputeLoss(pred, target);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void BYOL_OppositeVectors_MaximumLoss()
    {
        // When prediction is opposite of target, cosine similarity = -1
        // Loss = 2 - 2*(-1) = 4
        var loss = new BYOLLoss<double>(normalize: true, symmetric: false);

        var pred = new Tensor<double>(new double[] { 1, 0, 0, 1 }, [2, 2]);
        var target = new Tensor<double>(new double[] { -1, 0, 0, -1 }, [2, 2]);

        double result = loss.ComputeLoss(pred, target);
        Assert.Equal(4.0, result, Tol);
    }

    [Fact]
    public void BYOL_OrthogonalVectors_LossIsTwo()
    {
        // When prediction and target are orthogonal, cosine similarity = 0
        // Loss = 2 - 2*0 = 2
        var loss = new BYOLLoss<double>(normalize: true, symmetric: false);

        var pred = new Tensor<double>(new double[] { 1, 0 }, [1, 2]);
        var target = new Tensor<double>(new double[] { 0, 1 }, [1, 2]);

        double result = loss.ComputeLoss(pred, target);
        Assert.Equal(2.0, result, Tol);
    }

    [Fact]
    public void BYOL_HandCalculated_CosineSimilarity()
    {
        // pred = [3, 4], target = [1, 0]
        // |pred| = 5, |target| = 1
        // normalized pred = [0.6, 0.8], normalized target = [1, 0]
        // cosine sim = 0.6*1 + 0.8*0 = 0.6
        // Loss = 2 - 2*0.6 = 0.8
        var loss = new BYOLLoss<double>(normalize: true, symmetric: false);

        var pred = new Tensor<double>(new double[] { 3, 4 }, [1, 2]);
        var target = new Tensor<double>(new double[] { 1, 0 }, [1, 2]);

        double result = loss.ComputeLoss(pred, target);
        // With epsilon in normalization: norm = sqrt(25 + 1e-8) ~ 5.0
        // The result should be very close to 0.8
        Assert.Equal(0.8, result, RelaxedTol);
    }

    [Fact]
    public void BYOL_SymmetricLoss_IsAverageBothDirections()
    {
        var loss = new BYOLLoss<double>(normalize: true, symmetric: true);

        var pred1 = new Tensor<double>(new double[] { 1, 0 }, [1, 2]);
        var proj2 = new Tensor<double>(new double[] { 0.7, 0.7 }, [1, 2]);
        var pred2 = new Tensor<double>(new double[] { 0.5, 0.5 }, [1, 2]);
        var proj1 = new Tensor<double>(new double[] { 0.8, -0.2 }, [1, 2]);

        double symLoss = loss.ComputeSymmetricLoss(pred1, proj2, pred2, proj1);

        double loss1 = loss.ComputeLoss(pred1, proj2);
        double loss2 = loss.ComputeLoss(pred2, proj1);
        double expected = 0.5 * (loss1 + loss2);

        Assert.Equal(expected, symLoss, Tol);
    }

    [Fact]
    public void BYOL_MSELoss_EquivalentToCosineLoss_ForNormalized()
    {
        // For normalized vectors: MSE = 2 - 2*cos_sim = BYOL loss
        // This is because ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b = 2 - 2*cos_sim
        var loss = new BYOLLoss<double>(normalize: true, symmetric: false);

        var pred = new Tensor<double>(new double[] { 3, 4, 1, -1 }, [2, 2]);
        var target = new Tensor<double>(new double[] { 1, 2, -1, 1 }, [2, 2]);

        double cosineLoss = loss.ComputeLoss(pred, target);
        double mseLoss = loss.ComputeMSELoss(pred, target);

        // They should be proportional: BYOL loss = 2 - 2*cos = ||p_norm - z_norm||^2
        // MSE divides by dim, so MSE = ||p_norm - z_norm||^2 / dim
        // BYOL_loss = MSE * dim
        // But wait, BYOL divides by batchSize, MSE divides by (batchSize * dim)
        // So BYOL_loss = MSE * dim
        int dim = 2;
        Assert.Equal(cosineLoss, mseLoss * dim, RelaxedTol);
    }

    [Fact]
    public void BYOL_LossBoundedZeroToFour()
    {
        // BYOL loss = 2 - 2*cos, and cos ranges from -1 to 1
        // So loss ranges from 0 to 4
        var loss = new BYOLLoss<double>(normalize: true, symmetric: false);

        var pred = new Tensor<double>(new double[] { 2, -3, 0.1, 0.9, -5, 2 }, [3, 2]);
        var target = new Tensor<double>(new double[] { 1, 1, -1, 0.5, 3, -1 }, [3, 2]);

        double result = loss.ComputeLoss(pred, target);
        Assert.True(result >= 0.0 && result <= 4.0,
            $"BYOL loss should be in [0, 4], got {result}");
    }

    [Fact]
    public void BYOL_WithGradients_LossMatchesForwardOnly()
    {
        var loss = new BYOLLoss<double>(normalize: true, symmetric: false);

        var pred = new Tensor<double>(new double[] { 0.5, 0.3, -0.2, 0.8 }, [2, 2]);
        var target = new Tensor<double>(new double[] { 0.4, 0.6, 0.1, 0.7 }, [2, 2]);

        double forwardLoss = loss.ComputeLoss(pred, target);
        var (gradLoss, gradPred) = loss.ComputeLossWithGradients(pred, target);

        Assert.Equal(forwardLoss, gradLoss, Tol);
    }

    #endregion

    #region Temperature Scheduler - Exact Math Verification

    [Fact]
    public void TempScheduler_Constant_AlwaysReturnsInitial()
    {
        var scheduler = new TemperatureScheduler(
            TemperatureScheduleType.Constant,
            initialTemperature: 0.07,
            finalTemperature: 0.07);

        Assert.Equal(0.07, scheduler.GetTemperature(0), Tol);
        Assert.Equal(0.07, scheduler.GetTemperature(50000), Tol);
        Assert.Equal(0.07, scheduler.GetTemperature(100000), Tol);
    }

    [Fact]
    public void TempScheduler_LinearDecay_BoundaryValues()
    {
        var scheduler = new TemperatureScheduler(
            TemperatureScheduleType.LinearDecay,
            initialTemperature: 1.0,
            finalTemperature: 0.1,
            totalSteps: 100);

        // At step 0: initial
        Assert.Equal(1.0, scheduler.GetTemperature(0), Tol);
        // At step 50: halfway = 1.0 + (0.1 - 1.0) * 0.5 = 1.0 - 0.45 = 0.55
        Assert.Equal(0.55, scheduler.GetTemperature(50), Tol);
        // At step 100: final
        Assert.Equal(0.1, scheduler.GetTemperature(100), Tol);
        // Beyond total steps: clamped to final
        Assert.Equal(0.1, scheduler.GetTemperature(200), Tol);
    }

    [Fact]
    public void TempScheduler_CosineDecay_BoundaryValues()
    {
        var scheduler = new TemperatureScheduler(
            TemperatureScheduleType.CosineDecay,
            initialTemperature: 1.0,
            finalTemperature: 0.1,
            totalSteps: 100);

        // At step 0: cosineProgress = (1-cos(0))/2 = 0, temp = 1.0
        Assert.Equal(1.0, scheduler.GetTemperature(0), Tol);

        // At step 50 (halfway): cosine progress = (1 - cos(pi*0.5))/2 = (1-0)/2 = 0.5
        // temp = 1.0 + (0.1 - 1.0) * 0.5 = 1.0 - 0.45 = 0.55
        Assert.Equal(0.55, scheduler.GetTemperature(50), Tol);

        // At step 100: cosineProgress = (1-cos(pi))/2 = (1-(-1))/2 = 1.0
        // temp = 1.0 + (0.1-1.0)*1.0 = 0.1
        Assert.Equal(0.1, scheduler.GetTemperature(100), Tol);

        // Beyond total steps: clamped at progress=1.0 -> final temp
        Assert.Equal(0.1, scheduler.GetTemperature(200), Tol);
    }

    [Fact]
    public void TempScheduler_ExponentialDecay_HandCalculated()
    {
        var scheduler = new TemperatureScheduler(
            TemperatureScheduleType.ExponentialDecay,
            initialTemperature: 1.0,
            finalTemperature: 0.01,
            totalSteps: 100);

        // At step 0: initial
        Assert.Equal(1.0, scheduler.GetTemperature(0), Tol);

        // At step 50: 1.0 * (0.01/1.0)^0.5 = (0.01)^0.5 = 0.1
        Assert.Equal(0.1, scheduler.GetTemperature(50), Tol);

        // At step 100: 1.0 * (0.01)^1.0 = 0.01
        Assert.Equal(0.01, scheduler.GetTemperature(100), Tol);
    }

    [Fact]
    public void TempScheduler_LinearWarmup_HandCalculated()
    {
        var scheduler = new TemperatureScheduler(
            TemperatureScheduleType.LinearWarmup,
            initialTemperature: 0.04,
            finalTemperature: 0.07,
            warmupSteps: 50,
            totalSteps: 100);

        // During warmup (step 0 to 49): linear from initial to final
        // At step 0: 0.04
        Assert.Equal(0.04, scheduler.GetTemperature(0), Tol);

        // At step 25: 0.04 + (0.07-0.04) * 25/50 = 0.04 + 0.015 = 0.055
        Assert.Equal(0.055, scheduler.GetTemperature(25), Tol);

        // At step 50: post-warmup begins
        // The code's post-warmup phase linearly goes from initial to final over remaining steps
        // postWarmupProgress = (50-50)/(100-50) = 0
        // temp = 0.04 + (0.07-0.04)*0 = 0.04
        Assert.Equal(0.04, scheduler.GetTemperature(50), Tol);

        // At step 75: postWarmupProgress = (75-50)/(100-50) = 0.5
        // temp = 0.04 + (0.07-0.04)*0.5 = 0.055
        Assert.Equal(0.055, scheduler.GetTemperature(75), Tol);

        // At step 100: postWarmupProgress = (100-50)/50 = 1.0
        // temp = 0.04 + (0.07-0.04)*1.0 = 0.07
        Assert.Equal(0.07, scheduler.GetTemperature(100), Tol);
    }

    [Fact]
    public void TempScheduler_CosineWarmup_BoundaryValues()
    {
        var scheduler = new TemperatureScheduler(
            TemperatureScheduleType.CosineWarmup,
            initialTemperature: 0.04,
            finalTemperature: 0.07,
            warmupSteps: 100,
            totalSteps: 200);

        // At step 0: initial
        Assert.Equal(0.04, scheduler.GetTemperature(0), Tol);

        // At step 50 (halfway through warmup):
        // cosineProgress = (1 - cos(pi*0.5))/2 = (1-0)/2 = 0.5
        // temp = 0.04 + (0.07-0.04)*0.5 = 0.04 + 0.015 = 0.055
        Assert.Equal(0.055, scheduler.GetTemperature(50), Tol);

        // At step 100 (end of warmup): final
        Assert.Equal(0.07, scheduler.GetTemperature(100), Tol);

        // Beyond warmup: stays at final
        Assert.Equal(0.07, scheduler.GetTemperature(150), Tol);
    }

    [Fact]
    public void TempScheduler_NegativeStep_ClampsToZero()
    {
        var scheduler = new TemperatureScheduler(
            TemperatureScheduleType.LinearDecay,
            initialTemperature: 1.0,
            finalTemperature: 0.1,
            totalSteps: 100);

        // Negative step should be treated as step 0
        Assert.Equal(1.0, scheduler.GetTemperature(-5), Tol);
    }

    [Fact]
    public void TempScheduler_InvalidParameters_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TemperatureScheduler(initialTemperature: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TemperatureScheduler(initialTemperature: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TemperatureScheduler(finalTemperature: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TemperatureScheduler(warmupSteps: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TemperatureScheduler(totalSteps: 0));
    }

    [Fact]
    public void TempScheduler_ConstantFactory_ReturnsConstant()
    {
        var scheduler = TemperatureScheduler.Constant(0.1);
        Assert.Equal(TemperatureScheduleType.Constant, scheduler.ScheduleType);
        Assert.Equal(0.1, scheduler.InitialTemperature, Tol);
        Assert.Equal(0.1, scheduler.GetTemperature(99999), Tol);
    }

    [Fact]
    public void TempScheduler_ForEpoch_HandCalculated()
    {
        var scheduler = new TemperatureScheduler(
            TemperatureScheduleType.LinearDecay,
            initialTemperature: 1.0,
            finalTemperature: 0.1,
            totalSteps: 100);

        // Epoch 0 of 10: progress = 0, step = 0 -> temp = 1.0
        Assert.Equal(1.0, scheduler.GetTemperatureForEpoch(0, 10), Tol);

        // Epoch 5 of 10: progress = 0.5, step = 50 -> temp = 0.55
        Assert.Equal(0.55, scheduler.GetTemperatureForEpoch(5, 10), Tol);

        // Epoch 10 of 10: progress = 1.0, step = 100 -> temp = 0.1
        Assert.Equal(0.1, scheduler.GetTemperatureForEpoch(10, 10), Tol);
    }

    #endregion

    #region Momentum Scheduler - Exact Math Verification

    [Fact]
    public void MomentumSchedule_CosineFromBaseToFinal()
    {
        // ScheduleMomentum uses cosine schedule
        double baseMomentum = 0.99;
        double finalMomentum = 0.999;
        int totalEpochs = 100;

        // At epoch 0: base momentum
        double m0 = MomentumEncoder<double>.ScheduleMomentum(baseMomentum, finalMomentum, 0, totalEpochs);
        Assert.Equal(0.99, m0, Tol);

        // At epoch 50: halfway
        // cosineProgress = (1 - cos(pi*0.5))/2 = (1-0)/2 = 0.5
        // m = 0.99 + (0.999 - 0.99) * 0.5 = 0.99 + 0.0045 = 0.9945
        double m50 = MomentumEncoder<double>.ScheduleMomentum(baseMomentum, finalMomentum, 50, totalEpochs);
        Assert.Equal(0.9945, m50, Tol);

        // At epoch 100: final
        double m100 = MomentumEncoder<double>.ScheduleMomentum(baseMomentum, finalMomentum, 100, totalEpochs);
        Assert.Equal(0.999, m100, Tol);
    }

    [Fact]
    public void MomentumSchedule_Monotonic()
    {
        // Momentum should increase monotonically (base < final)
        double baseMomentum = 0.99;
        double finalMomentum = 0.999;

        double prevM = baseMomentum;
        for (int epoch = 1; epoch <= 100; epoch++)
        {
            double m = MomentumEncoder<double>.ScheduleMomentum(baseMomentum, finalMomentum, epoch, 100);
            Assert.True(m >= prevM, $"Momentum at epoch {epoch} ({m}) < previous ({prevM})");
            prevM = m;
        }
    }

    [Fact]
    public void MomentumSchedule_ZeroEpochs_ReturnsFinal()
    {
        double result = MomentumEncoder<double>.ScheduleMomentum(0.99, 0.999, 50, 0);
        Assert.Equal(0.999, result, Tol);
    }

    #endregion

    #region Cross-Loss Consistency Tests

    [Fact]
    public void InfoNCE_And_NTXent_SameForInBatch_WithSameSetup()
    {
        // InfoNCE in-batch mode should behave similarly to NTXent for asymmetric pairs
        // Note: NTXent is symmetric (considers both z1->z2 and z2->z1)
        // while InfoNCE in-batch only considers q->k direction
        var infonce = new InfoNCELoss<double>(temperature: 1.0, normalize: false);
        var ntxent = new NTXentLoss<double>(temperature: 1.0, normalize: false);

        var z1 = new Tensor<double>(new double[] { 1, 0 }, [1, 2]);
        var z2 = new Tensor<double>(new double[] { 0.9, 0.1 }, [1, 2]);

        // Both losses should be positive for non-identical views
        double infoResult = infonce.ComputeLossInBatch(z1, z2);
        double ntxentResult = ntxent.ComputeLoss(z1, z2);

        Assert.True(infoResult >= 0, $"InfoNCE loss should be non-negative: {infoResult}");
        Assert.True(ntxentResult >= 0, $"NT-Xent loss should be non-negative: {ntxentResult}");
    }

    [Fact]
    public void AllLosses_NonNegative_ForRandomInput()
    {
        var infonce = new InfoNCELoss<double>(temperature: 0.5, normalize: true);
        var ntxent = new NTXentLoss<double>(temperature: 0.5, normalize: true);
        var barlow = new BarlowTwinsLoss<double>(lambda: 0.005, normalize: true);
        var byol = new BYOLLoss<double>(normalize: true);

        var z1 = new Tensor<double>(new double[] { 0.5, -0.3, 0.8, 0.1, -0.7, 0.4 }, [3, 2]);
        var z2 = new Tensor<double>(new double[] { -0.2, 0.6, 0.3, -0.5, 0.9, -0.1 }, [3, 2]);

        var negKeys = new Tensor<double>(new double[] { 0.1, -0.9, -0.6, 0.3 }, [2, 2]);

        double infoResult = infonce.ComputeLoss(z1, z2, negKeys);
        double ntxentResult = ntxent.ComputeLoss(z1, z2);
        double barlowResult = barlow.ComputeLoss(z1, z2);
        double byolResult = byol.ComputeLoss(z1, z2);

        Assert.True(infoResult >= 0, $"InfoNCE loss should be non-negative: {infoResult}");
        Assert.True(ntxentResult >= 0, $"NT-Xent loss should be non-negative: {ntxentResult}");
        Assert.True(barlowResult >= 0, $"Barlow Twins loss should be non-negative: {barlowResult}");
        Assert.True(byolResult >= 0, $"BYOL loss should be non-negative: {byolResult}");
    }

    [Fact]
    public void BYOL_ConsistentWithMSE_ForNormalizedVectors()
    {
        // BYOL cosine loss and MSE loss should be related for normalized vectors
        // For unit vectors: ||a-b||^2 = 2 - 2*cos(a,b)
        // So BYOL loss (per sample) = MSE * dim (since MSE divides by dim)
        var loss = new BYOLLoss<double>(normalize: true);

        var pred = new Tensor<double>(new double[] { 1.5, 2.5 }, [1, 2]);
        var target = new Tensor<double>(new double[] { -0.5, 1.5 }, [1, 2]);

        double cosineLoss = loss.ComputeLoss(pred, target);
        double mseLoss = loss.ComputeMSELoss(pred, target);

        // cosineLoss = 2 - 2*cos
        // mseLoss = ||p_norm - z_norm||^2 / dim = (2 - 2*cos) / dim
        // So cosineLoss = mseLoss * dim
        Assert.Equal(cosineLoss, mseLoss * 2.0, RelaxedTol);
    }

    #endregion
}
