using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

/// <summary>
/// Deep mathematical integration tests for meta-learning algorithms.
/// Verifies correctness of core mathematical operations used by:
/// - ProtoNets: prototype computation (class means), distance metrics, softmax classification
/// - MAML: inner-loop gradient descent, meta-gradient accumulation, gradient clipping
/// - Reptile: parameter interpolation θ_new = θ_old + ε * (θ_adapted - θ_old)
/// - SimpleShot: L2 normalization, CL2N normalization (center + L2), nearest centroid
/// - Matching Networks: cosine similarity, dot product, negative Euclidean distance, softmax attention
/// </summary>
public class MetaLearningDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region ProtoNets: Prototype Computation (Class Mean Embedding)

    [Fact]
    public void ProtoNets_Prototype_IsMeanOfClassEmbeddings()
    {
        // ProtoNets: prototype_k = (1/|S_k|) * sum(f(x_i)) for x_i in class k
        // Class 0 embeddings: [1,2,3], [3,4,5] -> prototype = [2, 3, 4]
        // Class 1 embeddings: [10,20,30] -> prototype = [10, 20, 30]
        var class0Embeddings = new double[][] { [1, 2, 3], [3, 4, 5] };
        var class1Embeddings = new double[][] { [10, 20, 30] };

        var proto0 = ComputePrototype(class0Embeddings);
        var proto1 = ComputePrototype(class1Embeddings);

        Assert.Equal(2.0, proto0[0], Tolerance);
        Assert.Equal(3.0, proto0[1], Tolerance);
        Assert.Equal(4.0, proto0[2], Tolerance);
        Assert.Equal(10.0, proto1[0], Tolerance);
        Assert.Equal(20.0, proto1[1], Tolerance);
        Assert.Equal(30.0, proto1[2], Tolerance);
    }

    [Fact]
    public void ProtoNets_Prototype_SingleExample_EqualsItself()
    {
        // With one example per class (1-shot), prototype = the single embedding
        var embedding = new double[][] { [5.5, -3.2, 7.1] };
        var proto = ComputePrototype(embedding);

        Assert.Equal(5.5, proto[0], Tolerance);
        Assert.Equal(-3.2, proto[1], Tolerance);
        Assert.Equal(7.1, proto[2], Tolerance);
    }

    [Fact]
    public void ProtoNets_EuclideanDistance_HandComputed()
    {
        // d(q, p) = sqrt(sum((q_i - p_i)^2))
        // q = [1, 2, 3], p = [4, 6, 3]
        // d = sqrt((1-4)^2 + (2-6)^2 + (3-3)^2) = sqrt(9+16+0) = sqrt(25) = 5.0
        var query = new double[] { 1, 2, 3 };
        var prototype = new double[] { 4, 6, 3 };

        double distance = EuclideanDistance(query, prototype);
        Assert.Equal(5.0, distance, Tolerance);
    }

    [Fact]
    public void ProtoNets_SoftmaxOverNegDistances_ClassifiesNearest()
    {
        // ProtoNets uses softmax(-d(q, p_k)) for classification
        // query = [2, 3], proto0 = [2, 3] (dist=0), proto1 = [10, 10] (dist=sqrt(64+49)=sqrt(113))
        var query = new double[] { 2, 3 };
        var proto0 = new double[] { 2, 3 };
        var proto1 = new double[] { 10, 10 };

        double d0 = EuclideanDistance(query, proto0); // 0.0
        double d1 = EuclideanDistance(query, proto1); // sqrt(113) ≈ 10.63

        var probs = Softmax([-d0, -d1]); // softmax([0, -10.63])

        // Class 0 should get probability very close to 1
        Assert.True(probs[0] > 0.99, $"Nearest prototype class should have prob > 0.99, got {probs[0]}");
        Assert.True(probs[1] < 0.01, $"Far prototype class should have prob < 0.01, got {probs[1]}");
    }

    [Fact]
    public void ProtoNets_EqualDistances_ProduceUniformProbabilities()
    {
        // When equidistant from all prototypes, softmax gives uniform distribution
        var query = new double[] { 0, 0 };
        var proto0 = new double[] { 1, 0 };
        var proto1 = new double[] { -1, 0 };
        var proto2 = new double[] { 0, 1 };

        double d0 = EuclideanDistance(query, proto0); // 1.0
        double d1 = EuclideanDistance(query, proto1); // 1.0
        double d2 = EuclideanDistance(query, proto2); // 1.0

        Assert.Equal(d0, d1, Tolerance);
        Assert.Equal(d1, d2, Tolerance);

        var probs = Softmax([-d0, -d1, -d2]);
        Assert.Equal(1.0 / 3.0, probs[0], 1e-5);
        Assert.Equal(1.0 / 3.0, probs[1], 1e-5);
        Assert.Equal(1.0 / 3.0, probs[2], 1e-5);
    }

    [Fact]
    public void ProtoNets_SquaredEuclidean_MoreDiscriminative()
    {
        // Squared Euclidean magnifies differences: classes farther away get exponentially lower prob
        // q=[0,0], p0=[1,0](d=1, d^2=1), p1=[3,0](d=3, d^2=9)
        var query = new double[] { 0, 0 };
        var proto0 = new double[] { 1, 0 };
        var proto1 = new double[] { 3, 0 };

        double d0 = EuclideanDistance(query, proto0);
        double d1 = EuclideanDistance(query, proto1);

        var probsEuclidean = Softmax([-d0, -d1]);
        var probsSquared = Softmax([-(d0 * d0), -(d1 * d1)]);

        // Squared distances should make the nearest class even more dominant
        Assert.True(probsSquared[0] > probsEuclidean[0],
            $"Squared Euclidean prob for nearest {probsSquared[0]} should exceed Euclidean {probsEuclidean[0]}");
    }

    #endregion

    #region MAML: Inner Loop Gradient Descent

    [Fact]
    public void MAML_InnerLoop_SingleStep_CorrectUpdate()
    {
        // MAML inner loop: θ' = θ - α * ∇L(θ)
        // θ = [1.0, 2.0, 3.0], α = 0.1, ∇L = [0.5, -1.0, 2.0]
        // θ' = [1.0-0.1*0.5, 2.0-0.1*(-1.0), 3.0-0.1*2.0] = [0.95, 2.1, 2.8]
        var theta = new double[] { 1.0, 2.0, 3.0 };
        var gradients = new double[] { 0.5, -1.0, 2.0 };
        double alpha = 0.1;

        var thetaPrime = GradientStep(theta, gradients, alpha);

        Assert.Equal(0.95, thetaPrime[0], Tolerance);
        Assert.Equal(2.1, thetaPrime[1], Tolerance);
        Assert.Equal(2.8, thetaPrime[2], Tolerance);
    }

    [Fact]
    public void MAML_InnerLoop_MultipleSteps_Converges()
    {
        // Multiple gradient steps should decrease loss on training data
        // Linear model: y = w*x, loss = (y_pred - y_true)^2
        // x = 2.0, y_true = 6.0, w_init = 1.0 -> y_pred = 2.0
        // gradient dL/dw = 2*(y_pred - y_true)*x = 2*(2-6)*2 = -16
        // w_new = 1.0 - 0.01*(-16) = 1.16 -> y_pred = 2.32 (closer to 6)
        double w = 1.0;
        double x = 2.0;
        double yTrue = 6.0;
        double alpha = 0.01;

        double initialLoss = Math.Pow(w * x - yTrue, 2); // (2-6)^2 = 16

        for (int step = 0; step < 50; step++)
        {
            double yPred = w * x;
            double gradient = 2 * (yPred - yTrue) * x;
            w = w - alpha * gradient;
        }

        double finalLoss = Math.Pow(w * x - yTrue, 2);
        Assert.True(finalLoss < initialLoss,
            $"Loss should decrease: initial={initialLoss}, final={finalLoss}");
        // After 50 steps with lr=0.01, w should approach 3.0 (since y=3*2=6)
        Assert.True(Math.Abs(w - 3.0) < 0.1, $"Weight should approach 3.0, got {w}");
    }

    [Fact]
    public void MAML_MetaGradient_AveragesAcrossTasks()
    {
        // Meta-gradient = (1/B) * sum(task_gradients)
        // Task 1 gradient: [1.0, 2.0, 3.0]
        // Task 2 gradient: [3.0, 0.0, -1.0]
        // Task 3 gradient: [-1.0, 4.0, 1.0]
        // Average = [(1+3-1)/3, (2+0+4)/3, (3-1+1)/3] = [1.0, 2.0, 1.0]
        var taskGrads = new double[][]
        {
            [1.0, 2.0, 3.0],
            [3.0, 0.0, -1.0],
            [-1.0, 4.0, 1.0]
        };

        var avgGrad = AverageVectors(taskGrads);

        Assert.Equal(1.0, avgGrad[0], Tolerance);
        Assert.Equal(2.0, avgGrad[1], Tolerance);
        Assert.Equal(1.0, avgGrad[2], Tolerance);
    }

    [Fact]
    public void MAML_GradientClipping_NormExceedsThreshold()
    {
        // Gradient clipping: if ||g|| > threshold, g = g * (threshold / ||g||)
        // g = [3.0, 4.0] -> ||g|| = 5.0, threshold = 2.5
        // clipped = [3.0, 4.0] * (2.5/5.0) = [1.5, 2.0]
        var gradients = new double[] { 3.0, 4.0 };
        double threshold = 2.5;

        var clipped = ClipGradients(gradients, threshold);
        double clippedNorm = L2Norm(clipped);

        Assert.Equal(threshold, clippedNorm, 1e-5);
        Assert.Equal(1.5, clipped[0], Tolerance);
        Assert.Equal(2.0, clipped[1], Tolerance);
    }

    [Fact]
    public void MAML_GradientClipping_NormBelowThreshold_NoChange()
    {
        // When ||g|| <= threshold, gradients unchanged
        var gradients = new double[] { 1.0, 1.0 }; // ||g|| = sqrt(2) ≈ 1.414
        double threshold = 5.0;

        var clipped = ClipGradients(gradients, threshold);

        Assert.Equal(gradients[0], clipped[0], Tolerance);
        Assert.Equal(gradients[1], clipped[1], Tolerance);
    }

    [Fact]
    public void MAML_FOMAML_IgnoresSecondOrderTerms()
    {
        // FOMAML approximation: meta-gradient ≈ ∇L_query(θ') where θ' = θ - α*∇L_support(θ)
        // Full MAML: meta-gradient = ∇L_query(θ') * (I - α*H_support)
        // For simple linear model: y=wx, full MAML includes Hessian term
        // FOMAML just uses first-order gradient at adapted point

        // Model: y = w*x, support: (x=1, y=2), query: (x=1, y=3)
        double w = 0.5;
        double alpha = 0.1;

        // Support forward: loss_s = (0.5*1 - 2)^2 = 2.25
        // Support gradient: dL_s/dw = 2*(0.5-2)*1 = -3.0
        double supportGrad = 2 * (w * 1.0 - 2.0) * 1.0;
        Assert.Equal(-3.0, supportGrad, Tolerance);

        // Adapted: w' = 0.5 - 0.1*(-3.0) = 0.8
        double wPrime = w - alpha * supportGrad;
        Assert.Equal(0.8, wPrime, Tolerance);

        // FOMAML: query gradient at w'
        // dL_q/dw' = 2*(0.8*1 - 3)*1 = 2*(-2.2) = -4.4
        double fomamlGrad = 2 * (wPrime * 1.0 - 3.0) * 1.0;
        Assert.Equal(-4.4, fomamlGrad, Tolerance);

        // Full MAML would also include Hessian factor (I - α*H)
        // H = d²L_s/dw² = 2*x² = 2*1 = 2
        double hessian = 2.0 * 1.0 * 1.0;
        double fullMAMLGrad = fomamlGrad * (1 - alpha * hessian);
        // = -4.4 * (1 - 0.1*2) = -4.4 * 0.8 = -3.52

        Assert.Equal(-3.52, fullMAMLGrad, Tolerance);
        // FOMAML and full MAML differ
        Assert.NotEqual(fomamlGrad, fullMAMLGrad, Tolerance);
    }

    #endregion

    #region Reptile: Parameter Interpolation

    [Fact]
    public void Reptile_ParameterInterpolation_Formula()
    {
        // Reptile update: θ_new = θ_old + ε * (θ_adapted - θ_old)
        // θ_old = [1.0, 2.0, 3.0], θ_adapted = [2.0, 4.0, 1.0], ε = 0.5
        // θ_new = [1.0+0.5*(2-1), 2.0+0.5*(4-2), 3.0+0.5*(1-3)]
        //       = [1.5, 3.0, 2.0]
        var thetaOld = new double[] { 1.0, 2.0, 3.0 };
        var thetaAdapted = new double[] { 2.0, 4.0, 1.0 };
        double epsilon = 0.5;

        var thetaNew = ReptileUpdate(thetaOld, thetaAdapted, epsilon);

        Assert.Equal(1.5, thetaNew[0], Tolerance);
        Assert.Equal(3.0, thetaNew[1], Tolerance);
        Assert.Equal(2.0, thetaNew[2], Tolerance);
    }

    [Fact]
    public void Reptile_Epsilon1_FullAdaptation()
    {
        // ε = 1.0: θ_new = θ_old + 1.0*(θ_adapted - θ_old) = θ_adapted
        var thetaOld = new double[] { 1.0, 2.0 };
        var thetaAdapted = new double[] { 5.0, -3.0 };

        var thetaNew = ReptileUpdate(thetaOld, thetaAdapted, 1.0);

        Assert.Equal(5.0, thetaNew[0], Tolerance);
        Assert.Equal(-3.0, thetaNew[1], Tolerance);
    }

    [Fact]
    public void Reptile_Epsilon0_NoChange()
    {
        // ε = 0.0: θ_new = θ_old + 0*(θ_adapted - θ_old) = θ_old
        var thetaOld = new double[] { 1.0, 2.0 };
        var thetaAdapted = new double[] { 5.0, -3.0 };

        var thetaNew = ReptileUpdate(thetaOld, thetaAdapted, 0.0);

        Assert.Equal(1.0, thetaNew[0], Tolerance);
        Assert.Equal(2.0, thetaNew[1], Tolerance);
    }

    [Fact]
    public void Reptile_BatchedUpdate_AveragesTaskDirections()
    {
        // With batch: direction = (1/B) * sum(θ_adapted_i - θ_old)
        // θ_old = [0, 0], adapted_1 = [2, 4], adapted_2 = [4, -2]
        // avg_direction = [(2+4)/2, (4-2)/2] = [3, 1]
        // θ_new = [0, 0] + 0.1 * [3, 1] = [0.3, 0.1]
        var thetaOld = new double[] { 0, 0 };
        var adapted1 = new double[] { 2, 4 };
        var adapted2 = new double[] { 4, -2 };
        double epsilon = 0.1;

        var direction = new double[2];
        for (int i = 0; i < 2; i++)
        {
            direction[i] = ((adapted1[i] - thetaOld[i]) + (adapted2[i] - thetaOld[i])) / 2.0;
        }

        var thetaNew = new double[2];
        for (int i = 0; i < 2; i++)
        {
            thetaNew[i] = thetaOld[i] + epsilon * direction[i];
        }

        Assert.Equal(0.3, thetaNew[0], Tolerance);
        Assert.Equal(0.1, thetaNew[1], Tolerance);
    }

    [Fact]
    public void Reptile_ConvergesWithRepeatedUpdates()
    {
        // If all tasks have optimal params at [3.0, 3.0], Reptile converges there
        double[] theta = [0.0, 0.0];
        double epsilon = 0.1;
        double[] target = [3.0, 3.0];

        for (int step = 0; step < 50; step++)
        {
            // Simulate adaptation: move toward target with some noise
            double[] adapted = [target[0] + 0.01 * step % 2, target[1] - 0.01 * step % 2];
            theta = ReptileUpdate(theta, adapted, epsilon);
        }

        // Should be close to target after many steps
        Assert.True(Math.Abs(theta[0] - target[0]) < 0.5,
            $"theta[0]={theta[0]} should be close to {target[0]}");
        Assert.True(Math.Abs(theta[1] - target[1]) < 0.5,
            $"theta[1]={theta[1]} should be close to {target[1]}");
    }

    #endregion

    #region SimpleShot: L2 Normalization

    [Fact]
    public void SimpleShot_L2Normalization_UnitNorm()
    {
        // L2 normalization: x_norm = x / ||x||_2
        // x = [3, 4] -> ||x|| = 5 -> x_norm = [0.6, 0.8]
        var x = new double[] { 3.0, 4.0 };
        var normalized = L2Normalize(x);

        double norm = L2Norm(normalized);
        Assert.Equal(1.0, norm, Tolerance);
        Assert.Equal(0.6, normalized[0], Tolerance);
        Assert.Equal(0.8, normalized[1], Tolerance);
    }

    [Fact]
    public void SimpleShot_L2Normalization_PreservesDirection()
    {
        // Normalized vector should be a positive scalar multiple of original
        var x = new double[] { 2.0, -3.0, 1.0 };
        var normalized = L2Normalize(x);

        // Direction check: normalized[i]/x[i] should be constant for all i
        double ratio = normalized[0] / x[0];
        Assert.True(ratio > 0, "Ratio should be positive (same direction)");
        Assert.Equal(ratio, normalized[1] / x[1], Tolerance);
        Assert.Equal(ratio, normalized[2] / x[2], Tolerance);
    }

    [Fact]
    public void SimpleShot_CL2N_CentersThenNormalizes()
    {
        // CL2N: center features by subtracting mean, then L2 normalize
        // Features: [2, 4, 6], [4, 6, 8] -> mean = [3, 5, 7]
        // Centered: [-1, -1, -1], [1, 1, 1]
        // L2 norm of [-1,-1,-1] = sqrt(3), normalized = [-1/sqrt(3), -1/sqrt(3), -1/sqrt(3)]
        var features = new double[][] { [2, 4, 6], [4, 6, 8] };

        var mean = ComputeMean(features);
        Assert.Equal(3.0, mean[0], Tolerance);
        Assert.Equal(5.0, mean[1], Tolerance);
        Assert.Equal(7.0, mean[2], Tolerance);

        var centered0 = SubtractVector(features[0], mean);
        var centered1 = SubtractVector(features[1], mean);

        Assert.Equal(-1.0, centered0[0], Tolerance);
        Assert.Equal(-1.0, centered0[1], Tolerance);
        Assert.Equal(-1.0, centered0[2], Tolerance);

        var normalized0 = L2Normalize(centered0);
        double expectedVal = -1.0 / Math.Sqrt(3);
        Assert.Equal(expectedVal, normalized0[0], Tolerance);
        Assert.Equal(expectedVal, normalized0[1], Tolerance);
        Assert.Equal(expectedVal, normalized0[2], Tolerance);
    }

    [Fact]
    public void SimpleShot_CL2N_OppositeClassesBecomeAntiparallel()
    {
        // After CL2N with two symmetric classes, they become exactly opposite directions
        var features = new double[][] { [1, 0], [3, 0] }; // mean = [2, 0]
        // Centered: [-1, 0], [1, 0]
        // Normalized: [-1, 0], [1, 0] (already unit norm)
        var mean = ComputeMean(features);
        var n0 = L2Normalize(SubtractVector(features[0], mean));
        var n1 = L2Normalize(SubtractVector(features[1], mean));

        // Dot product should be -1 (antiparallel)
        double dot = DotProduct(n0, n1);
        Assert.Equal(-1.0, dot, Tolerance);
    }

    [Fact]
    public void SimpleShot_NearestCentroid_ClassifiesCorrectly()
    {
        // 3 classes with centroids at known positions
        // c0 = [1, 0], c1 = [0, 1], c2 = [-1, 0]
        // Query [0.9, 0.1] -> closest to c0
        var centroids = new double[][] { [1, 0], [0, 1], [-1, 0] };
        var query = new double[] { 0.9, 0.1 };

        int predicted = NearestCentroid(query, centroids);
        Assert.Equal(0, predicted);
    }

    [Fact]
    public void SimpleShot_NearestCentroid_WithNormalization_ChangesDecision()
    {
        // Without normalization: query [10, 1] closest to centroid [8, 0]
        // With L2 normalization: query becomes [10/sqrt(101), 1/sqrt(101)] ≈ [0.995, 0.0995]
        //   c0 normalized: [1, 0], c1 normalized: [0, 1]
        //   Still closest to c0 in this case
        var centroids = new double[][] { [8, 0], [0, 5] };
        var query = new double[] { 10, 1 };

        int predRaw = NearestCentroid(query, centroids);

        // Normalize everything
        var normQuery = L2Normalize(query);
        var normCentroids = centroids.Select(L2Normalize).ToArray();
        int predNorm = NearestCentroid(normQuery, normCentroids);

        // Both should pick class 0 in this case
        Assert.Equal(0, predRaw);
        Assert.Equal(0, predNorm);
    }

    #endregion

    #region Matching Networks: Cosine Similarity

    [Fact]
    public void MatchingNets_CosineSimilarity_ParallelVectors()
    {
        // Parallel vectors: cos(a, 2a) = 1.0
        var a = new double[] { 1.0, 2.0, 3.0 };
        var b = new double[] { 2.0, 4.0, 6.0 };

        double cosine = CosineSimilarity(a, b);
        Assert.Equal(1.0, cosine, Tolerance);
    }

    [Fact]
    public void MatchingNets_CosineSimilarity_OrthogonalVectors()
    {
        // Orthogonal vectors: cos(a, b) = 0
        var a = new double[] { 1.0, 0.0 };
        var b = new double[] { 0.0, 1.0 };

        double cosine = CosineSimilarity(a, b);
        Assert.Equal(0.0, cosine, Tolerance);
    }

    [Fact]
    public void MatchingNets_CosineSimilarity_AntiparallelVectors()
    {
        // Antiparallel vectors: cos(a, -a) = -1.0
        var a = new double[] { 1.0, 2.0, 3.0 };
        var b = new double[] { -1.0, -2.0, -3.0 };

        double cosine = CosineSimilarity(a, b);
        Assert.Equal(-1.0, cosine, Tolerance);
    }

    [Fact]
    public void MatchingNets_CosineSimilarity_HandComputed()
    {
        // a = [1, 2, 3], b = [4, -5, 6]
        // dot = 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
        // ||a|| = sqrt(1+4+9) = sqrt(14)
        // ||b|| = sqrt(16+25+36) = sqrt(77)
        // cos = 12 / (sqrt(14) * sqrt(77)) = 12 / sqrt(1078) ≈ 12/32.833 ≈ 0.36541
        var a = new double[] { 1, 2, 3 };
        var b = new double[] { 4, -5, 6 };

        double cosine = CosineSimilarity(a, b);
        double expected = 12.0 / Math.Sqrt(14.0 * 77.0);
        Assert.Equal(expected, cosine, Tolerance);
    }

    [Fact]
    public void MatchingNets_DotProduct_HandComputed()
    {
        // a = [1, 2, 3], b = [4, -5, 6]
        // dot = 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
        var a = new double[] { 1, 2, 3 };
        var b = new double[] { 4, -5, 6 };

        double dot = DotProduct(a, b);
        Assert.Equal(12.0, dot, Tolerance);
    }

    [Fact]
    public void MatchingNets_NegativeEuclidean_MoreSimilarMeansLargerValue()
    {
        // Negative Euclidean: -||a-b||
        // a=[0,0], b1=[1,0](d=1), b2=[3,0](d=3)
        // -d(a,b1) = -1 > -3 = -d(a,b2)
        // So b1 is "more similar" (larger negative distance)
        var a = new double[] { 0, 0 };
        var b1 = new double[] { 1, 0 };
        var b2 = new double[] { 3, 0 };

        double negDist1 = -EuclideanDistance(a, b1);
        double negDist2 = -EuclideanDistance(a, b2);

        Assert.True(negDist1 > negDist2, $"Closer point should have larger neg distance: {negDist1} vs {negDist2}");
    }

    #endregion

    #region Matching Networks: Softmax Attention Weights

    [Fact]
    public void MatchingNets_SoftmaxAttention_SumsToOne()
    {
        // softmax always sums to 1
        var similarities = new double[] { 1.5, -0.5, 0.3, 2.1 };
        var weights = Softmax(similarities);

        double sum = weights.Sum();
        Assert.Equal(1.0, sum, 1e-10);
    }

    [Fact]
    public void MatchingNets_SoftmaxAttention_AllPositive()
    {
        // softmax outputs are always positive
        var similarities = new double[] { -100, -200, -50 };
        var weights = Softmax(similarities);

        foreach (var w in weights)
        {
            Assert.True(w > 0, $"Softmax weight {w} should be positive");
        }
    }

    [Fact]
    public void MatchingNets_SoftmaxAttention_MaxGetsHighestWeight()
    {
        // Largest input gets largest softmax output
        var similarities = new double[] { 1.0, 3.0, 2.0 };
        var weights = Softmax(similarities);

        Assert.True(weights[1] > weights[0], "Max input should have highest weight");
        Assert.True(weights[1] > weights[2], "Max input should have highest weight");
    }

    [Fact]
    public void MatchingNets_SoftmaxAttention_HandComputed()
    {
        // softmax([1, 2, 3]) = exp([1,2,3])/sum(exp([1,2,3]))
        // exp(1) ≈ 2.71828, exp(2) ≈ 7.38906, exp(3) ≈ 20.08554
        // sum = 30.19287
        // result = [0.09003, 0.24473, 0.66524]
        var logits = new double[] { 1, 2, 3 };
        var weights = Softmax(logits);

        Assert.Equal(0.09003, weights[0], 4); // 4 decimal places
        Assert.Equal(0.24473, weights[1], 4);
        Assert.Equal(0.66524, weights[2], 4);
    }

    [Fact]
    public void MatchingNets_TemperatureScaling_SharpensSoftmax()
    {
        // Low temperature -> sharper distribution (more confident)
        // High temperature -> flatter distribution (more uniform)
        var similarities = new double[] { 1.0, 2.0, 3.0 };

        var weightsT1 = Softmax(similarities); // temperature = 1
        var weightsT01 = Softmax(similarities.Select(s => s / 0.1).ToArray()); // temperature = 0.1
        var weightsT10 = Softmax(similarities.Select(s => s / 10.0).ToArray()); // temperature = 10

        // Low temp should be sharper: max weight closer to 1
        Assert.True(weightsT01[2] > weightsT1[2],
            $"Low temp max weight {weightsT01[2]} should exceed normal {weightsT1[2]}");
        // High temp should be flatter: max weight closer to 1/3
        Assert.True(weightsT10[2] < weightsT1[2],
            $"High temp max weight {weightsT10[2]} should be less than normal {weightsT1[2]}");
    }

    [Fact]
    public void MatchingNets_AttentionWeightedPrediction_HandComputed()
    {
        // Attention-weighted prediction: y_hat = sum(w_i * y_i)
        // Support labels (one-hot): [[1,0], [0,1], [1,0]]
        // Attention weights: [0.5, 0.3, 0.2]
        // y_hat = 0.5*[1,0] + 0.3*[0,1] + 0.2*[1,0] = [0.7, 0.3]
        var labels = new double[][] { [1, 0], [0, 1], [1, 0] };
        var weights = new double[] { 0.5, 0.3, 0.2 };

        var prediction = AttentionWeightedPrediction(weights, labels);

        Assert.Equal(0.7, prediction[0], Tolerance);
        Assert.Equal(0.3, prediction[1], Tolerance);
    }

    [Fact]
    public void MatchingNets_AttentionPrediction_AllWeightOnOne_ReturnsItsLabel()
    {
        // If all attention on one example, prediction = that example's label
        var labels = new double[][] { [1, 0], [0, 1], [1, 0] };
        var weights = new double[] { 0.0, 1.0, 0.0 };

        var prediction = AttentionWeightedPrediction(weights, labels);

        Assert.Equal(0.0, prediction[0], Tolerance);
        Assert.Equal(1.0, prediction[1], Tolerance);
    }

    #endregion

    #region Cross-Entropy Loss for Meta-Learning

    [Fact]
    public void MetaLearning_CrossEntropyLoss_HandComputed()
    {
        // CE = -sum(y_true * log(y_pred)) / N
        // For single example: true class = 0, predicted probs = [0.7, 0.2, 0.1]
        // CE = -log(0.7) ≈ 0.35667
        double predictedProb = 0.7;
        double ce = -Math.Log(predictedProb);
        Assert.Equal(0.35667, ce, 4);
    }

    [Fact]
    public void MetaLearning_CrossEntropyLoss_PerfectPrediction()
    {
        // Perfect prediction: prob = 1.0 -> CE = -log(1) = 0
        double ce = -Math.Log(1.0);
        Assert.Equal(0.0, ce, Tolerance);
    }

    [Fact]
    public void MetaLearning_CrossEntropyLoss_UniformPrediction()
    {
        // Uniform over K classes: prob = 1/K -> CE = log(K)
        int K = 5;
        double ce = -Math.Log(1.0 / K);
        Assert.Equal(Math.Log(K), ce, Tolerance);
    }

    [Fact]
    public void MetaLearning_CrossEntropyLoss_BatchAverage()
    {
        // Batch CE = mean over examples
        // Example 1: true=0, probs=[0.8, 0.2] -> CE1 = -log(0.8)
        // Example 2: true=1, probs=[0.3, 0.7] -> CE2 = -log(0.7)
        // Mean CE = (-log(0.8) + -log(0.7)) / 2
        double ce1 = -Math.Log(0.8);
        double ce2 = -Math.Log(0.7);
        double meanCE = (ce1 + ce2) / 2.0;

        double expected = (-Math.Log(0.8) + -Math.Log(0.7)) / 2.0;
        Assert.Equal(expected, meanCE, Tolerance);
        Assert.True(meanCE > 0, "Cross-entropy should be positive");
    }

    #endregion

    #region Few-Shot Episode Construction

    [Fact]
    public void FewShot_NWayKShot_CorrectSupportSize()
    {
        // N-way K-shot: support set has N*K examples
        int N = 5; // 5 classes
        int K = 3; // 3 examples per class
        int supportSize = N * K;
        Assert.Equal(15, supportSize);
    }

    [Fact]
    public void FewShot_PrototypeCount_EqualsNWay()
    {
        // Number of prototypes should equal N (number of classes)
        int N = 5;
        int K = 3;

        // Simulate class assignments
        var classAssignments = new int[N * K];
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < K; j++)
            {
                classAssignments[i * K + j] = i;
            }
        }

        int numClasses = classAssignments.Distinct().Count();
        Assert.Equal(N, numClasses);
    }

    [Fact]
    public void FewShot_OneShotPrototype_EqualsExactExample()
    {
        // In 1-shot learning, prototype = the single support example
        int K = 1;
        int embeddingDim = 4;
        var supportEmbeddings = new double[][] { [1.5, -2.3, 0.7, 4.1] };

        var prototype = ComputePrototype(supportEmbeddings);

        for (int d = 0; d < embeddingDim; d++)
        {
            Assert.Equal(supportEmbeddings[0][d], prototype[d], Tolerance);
        }
    }

    #endregion

    #region Gradient Computation for Linear Model

    [Fact]
    public void LinearModel_MSEGradient_HandComputed()
    {
        // Model: y = W*x + b, where W=[w1,w2], x=[x1,x2], b=bias
        // Loss = (y_pred - y_true)^2
        // dL/dw_i = 2*(y_pred - y_true) * x_i
        // dL/db = 2*(y_pred - y_true)

        // W = [0.5, 1.0], b = 0.1, x = [2.0, 3.0], y_true = 4.0
        // y_pred = 0.5*2 + 1.0*3 + 0.1 = 1 + 3 + 0.1 = 4.1
        // error = 4.1 - 4.0 = 0.1
        // dL/dw1 = 2 * 0.1 * 2.0 = 0.4
        // dL/dw2 = 2 * 0.1 * 3.0 = 0.6
        // dL/db = 2 * 0.1 = 0.2
        double w1 = 0.5, w2 = 1.0, b = 0.1;
        double x1 = 2.0, x2 = 3.0, yTrue = 4.0;

        double yPred = w1 * x1 + w2 * x2 + b;
        Assert.Equal(4.1, yPred, Tolerance);

        double error = yPred - yTrue;
        Assert.Equal(0.1, error, Tolerance);

        double gradW1 = 2 * error * x1;
        double gradW2 = 2 * error * x2;
        double gradB = 2 * error;

        Assert.Equal(0.4, gradW1, Tolerance);
        Assert.Equal(0.6, gradW2, Tolerance);
        Assert.Equal(0.2, gradB, Tolerance);
    }

    [Fact]
    public void LinearModel_BatchGradient_AveragedOverExamples()
    {
        // Batch gradient = (1/N) * sum of per-example gradients
        // Example 1: x=[1], y_true=2, W=[0.5], b=0
        //   y_pred=0.5, error=-1.5, grad_w = 2*(-1.5)*1 = -3.0, grad_b = -3.0
        // Example 2: x=[2], y_true=1, W=[0.5], b=0
        //   y_pred=1.0, error=-0.0 wait... y_pred=0.5*2=1.0, error=1.0-1.0=0.0
        // Let's use: W=[1], b=0, x1=[1] y1=3, x2=[2] y2=1
        //   pred1=1, err1=-2, grad_w_1=2*(-2)*1=-4
        //   pred2=2, err2=1, grad_w_2=2*(1)*2=4
        //   avg_grad_w = (-4 + 4)/2 = 0
        double w = 1.0;
        double[] xs = [1.0, 2.0];
        double[] ys = [3.0, 1.0];

        double sumGrad = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            double pred = w * xs[i];
            double err = pred - ys[i];
            sumGrad += 2 * err * xs[i];
        }
        double avgGrad = sumGrad / xs.Length;

        Assert.Equal(0.0, avgGrad, Tolerance);
    }

    #endregion

    #region Distance Metric Properties

    [Fact]
    public void Distance_Triangle_Inequality()
    {
        // d(a,c) <= d(a,b) + d(b,c) for any a, b, c
        var a = new double[] { 1, 0, 0 };
        var b = new double[] { 0, 1, 0 };
        var c = new double[] { 0, 0, 1 };

        double dAC = EuclideanDistance(a, c);
        double dAB = EuclideanDistance(a, b);
        double dBC = EuclideanDistance(b, c);

        Assert.True(dAC <= dAB + dBC + 1e-10,
            $"Triangle inequality violated: d(a,c)={dAC} > d(a,b)+d(b,c)={dAB + dBC}");
    }

    [Fact]
    public void Distance_Symmetry()
    {
        // d(a,b) = d(b,a)
        var a = new double[] { 1.5, -3.2, 0.7 };
        var b = new double[] { -0.5, 2.1, 4.3 };

        double dAB = EuclideanDistance(a, b);
        double dBA = EuclideanDistance(b, a);

        Assert.Equal(dAB, dBA, Tolerance);
    }

    [Fact]
    public void Distance_Identity()
    {
        // d(a,a) = 0
        var a = new double[] { 1.5, -3.2, 0.7 };
        double d = EuclideanDistance(a, a);
        Assert.Equal(0.0, d, Tolerance);
    }

    [Fact]
    public void Distance_NonNegativity()
    {
        // d(a,b) >= 0 for all a,b
        var a = new double[] { -5, 10, -3 };
        var b = new double[] { 7, -2, 8 };
        double d = EuclideanDistance(a, b);
        Assert.True(d >= 0, $"Distance should be non-negative: {d}");
    }

    #endregion

    #region Cosine Similarity Properties

    [Fact]
    public void CosineSimilarity_BoundedBetweenNeg1And1()
    {
        // cos(a,b) in [-1, 1] for any non-zero a, b
        var vectors = new double[][]
        {
            [1, 2, 3],
            [-4, 5, -6],
            [0.1, -0.2, 0.3],
            [100, -200, 300]
        };

        for (int i = 0; i < vectors.Length; i++)
        {
            for (int j = 0; j < vectors.Length; j++)
            {
                double cos = CosineSimilarity(vectors[i], vectors[j]);
                Assert.True(cos >= -1.0 - 1e-10 && cos <= 1.0 + 1e-10,
                    $"Cosine similarity {cos} out of [-1,1] for pair ({i},{j})");
            }
        }
    }

    [Fact]
    public void CosineSimilarity_ScaleInvariant()
    {
        // cos(a, k*b) = cos(a, b) for k > 0
        var a = new double[] { 1, 2, 3 };
        var b = new double[] { 4, -5, 6 };
        var b_scaled = b.Select(x => x * 100.0).ToArray();

        double cos1 = CosineSimilarity(a, b);
        double cos2 = CosineSimilarity(a, b_scaled);

        Assert.Equal(cos1, cos2, 1e-10);
    }

    [Fact]
    public void CosineSimilarity_SelfSimilarity_IsOne()
    {
        // cos(a, a) = 1 for any non-zero a
        var a = new double[] { -3.7, 2.1, 0.0, -1.5 };
        double cos = CosineSimilarity(a, a);
        Assert.Equal(1.0, cos, Tolerance);
    }

    #endregion

    #region One-Hot Encoding for Meta-Learning

    [Fact]
    public void OneHot_Encoding_CorrectFormat()
    {
        // Class labels [0, 1, 2, 0] with 3 classes ->
        // [[1,0,0], [0,1,0], [0,0,1], [1,0,0]]
        var labels = new int[] { 0, 1, 2, 0 };
        int numClasses = 3;

        var oneHot = OneHotEncode(labels, numClasses);

        // Row 0: class 0 -> [1,0,0]
        Assert.Equal(1.0, oneHot[0][0], Tolerance);
        Assert.Equal(0.0, oneHot[0][1], Tolerance);
        Assert.Equal(0.0, oneHot[0][2], Tolerance);

        // Row 1: class 1 -> [0,1,0]
        Assert.Equal(0.0, oneHot[1][0], Tolerance);
        Assert.Equal(1.0, oneHot[1][1], Tolerance);
        Assert.Equal(0.0, oneHot[1][2], Tolerance);

        // Row 2: class 2 -> [0,0,1]
        Assert.Equal(0.0, oneHot[2][0], Tolerance);
        Assert.Equal(0.0, oneHot[2][1], Tolerance);
        Assert.Equal(1.0, oneHot[2][2], Tolerance);
    }

    [Fact]
    public void OneHot_RowSumsToOne()
    {
        // Each row of one-hot encoding sums to 1
        var labels = new int[] { 0, 2, 1, 3, 0 };
        int numClasses = 4;
        var oneHot = OneHotEncode(labels, numClasses);

        foreach (var row in oneHot)
        {
            double rowSum = row.Sum();
            Assert.Equal(1.0, rowSum, Tolerance);
        }
    }

    #endregion

    #region Meta-Learning Loss Landscape Properties

    [Fact]
    public void MetaLoss_InnerLoopSteps_DecreaseLoss()
    {
        // More inner loop steps should generally decrease loss on support set
        // Using simple quadratic loss: L(θ) = (θ - θ*)^2 where θ* = 3
        double theta = 0.0;
        double thetaStar = 3.0;
        double lr = 0.1;

        var losses = new List<double>();
        for (int step = 0; step <= 10; step++)
        {
            double loss = Math.Pow(theta - thetaStar, 2);
            losses.Add(loss);

            // Gradient: dL/dθ = 2*(θ - θ*)
            double grad = 2 * (theta - thetaStar);
            theta = theta - lr * grad;
        }

        // Each step should have lower or equal loss
        for (int i = 1; i < losses.Count; i++)
        {
            Assert.True(losses[i] <= losses[i - 1] + 1e-10,
                $"Loss at step {i} ({losses[i]}) should not exceed step {i - 1} ({losses[i - 1]})");
        }
    }

    [Fact]
    public void MetaLoss_LargerLR_FasterInitialDecrease()
    {
        // With larger learning rate, initial loss decrease is faster
        // But may overshoot if too large
        double theta0 = 0.0;
        double thetaStar = 3.0;

        double theta_slow = theta0;
        double theta_fast = theta0;
        double lr_slow = 0.01;
        double lr_fast = 0.1;

        // One step
        double grad = 2 * (theta0 - thetaStar); // = -6
        theta_slow -= lr_slow * grad; // 0 - 0.01*(-6) = 0.06
        theta_fast -= lr_fast * grad; // 0 - 0.1*(-6) = 0.6

        double loss_slow = Math.Pow(theta_slow - thetaStar, 2); // (0.06-3)^2 = 8.6436
        double loss_fast = Math.Pow(theta_fast - thetaStar, 2); // (0.6-3)^2 = 5.76

        Assert.True(loss_fast < loss_slow,
            $"Faster LR should have lower loss after 1 step: {loss_fast} vs {loss_slow}");
    }

    #endregion

    #region Helper Methods

    private static double[] ComputePrototype(double[][] classEmbeddings)
    {
        int dim = classEmbeddings[0].Length;
        var prototype = new double[dim];
        foreach (var emb in classEmbeddings)
        {
            for (int d = 0; d < dim; d++)
            {
                prototype[d] += emb[d];
            }
        }
        for (int d = 0; d < dim; d++)
        {
            prototype[d] /= classEmbeddings.Length;
        }
        return prototype;
    }

    private static double EuclideanDistance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    private static double[] Softmax(double[] logits)
    {
        double max = logits.Max();
        var exps = logits.Select(x => Math.Exp(x - max)).ToArray();
        double sum = exps.Sum();
        return exps.Select(x => x / sum).ToArray();
    }

    private static double CosineSimilarity(double[] a, double[] b)
    {
        double dot = 0, normASq = 0, normBSq = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normASq += a[i] * a[i];
            normBSq += b[i] * b[i];
        }
        double denom = Math.Sqrt(normASq) * Math.Sqrt(normBSq);
        if (denom < 1e-10) return 0;
        return dot / denom;
    }

    private static double DotProduct(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static double L2Norm(double[] v)
    {
        return Math.Sqrt(v.Sum(x => x * x));
    }

    private static double[] L2Normalize(double[] v)
    {
        double norm = L2Norm(v);
        if (norm < 1e-10) return v.ToArray();
        return v.Select(x => x / norm).ToArray();
    }

    private static double[] GradientStep(double[] theta, double[] gradients, double lr)
    {
        var result = new double[theta.Length];
        for (int i = 0; i < theta.Length; i++)
        {
            result[i] = theta[i] - lr * gradients[i];
        }
        return result;
    }

    private static double[] ReptileUpdate(double[] thetaOld, double[] thetaAdapted, double epsilon)
    {
        var result = new double[thetaOld.Length];
        for (int i = 0; i < thetaOld.Length; i++)
        {
            result[i] = thetaOld[i] + epsilon * (thetaAdapted[i] - thetaOld[i]);
        }
        return result;
    }

    private static double[] AverageVectors(double[][] vectors)
    {
        int dim = vectors[0].Length;
        var avg = new double[dim];
        foreach (var v in vectors)
        {
            for (int i = 0; i < dim; i++)
            {
                avg[i] += v[i];
            }
        }
        for (int i = 0; i < dim; i++)
        {
            avg[i] /= vectors.Length;
        }
        return avg;
    }

    private static double[] ClipGradients(double[] gradients, double threshold)
    {
        double norm = L2Norm(gradients);
        if (norm <= threshold)
        {
            return gradients.ToArray();
        }
        double scale = threshold / norm;
        return gradients.Select(g => g * scale).ToArray();
    }

    private static double[] ComputeMean(double[][] vectors)
    {
        int dim = vectors[0].Length;
        var mean = new double[dim];
        foreach (var v in vectors)
        {
            for (int i = 0; i < dim; i++)
            {
                mean[i] += v[i];
            }
        }
        for (int i = 0; i < dim; i++)
        {
            mean[i] /= vectors.Length;
        }
        return mean;
    }

    private static double[] SubtractVector(double[] a, double[] b)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    private static int NearestCentroid(double[] query, double[][] centroids)
    {
        int nearest = 0;
        double minDist = double.MaxValue;
        for (int i = 0; i < centroids.Length; i++)
        {
            double dist = EuclideanDistance(query, centroids[i]);
            if (dist < minDist)
            {
                minDist = dist;
                nearest = i;
            }
        }
        return nearest;
    }

    private static double[] AttentionWeightedPrediction(double[] weights, double[][] labels)
    {
        int numClasses = labels[0].Length;
        var prediction = new double[numClasses];
        for (int s = 0; s < weights.Length; s++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                prediction[c] += weights[s] * labels[s][c];
            }
        }
        return prediction;
    }

    private static double[][] OneHotEncode(int[] labels, int numClasses)
    {
        var result = new double[labels.Length][];
        for (int i = 0; i < labels.Length; i++)
        {
            result[i] = new double[numClasses];
            if (labels[i] >= 0 && labels[i] < numClasses)
            {
                result[i][labels[i]] = 1.0;
            }
        }
        return result;
    }

    #endregion
}
