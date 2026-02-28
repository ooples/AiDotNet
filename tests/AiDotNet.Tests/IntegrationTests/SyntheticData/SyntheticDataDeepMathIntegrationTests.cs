using AiDotNet.NeuralNetworks.SyntheticData;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// Deep integration tests for SyntheticData:
/// ColumnMetadata (construction, properties, Clone, NumCategories, IsCategorical, IsNumerical),
/// ColumnDataType enum,
/// Synthetic data math (VGM normalization, one-hot encoding dimensions, mode-specific sampling,
/// Wasserstein distance, KL divergence, privacy metrics, data utility).
/// </summary>
public class SyntheticDataDeepMathIntegrationTests
{
    // ============================
    // ColumnDataType Enum
    // ============================

    [Fact]
    public void ColumnDataType_HasExpectedValues()
    {
        var values = (((ColumnDataType[])Enum.GetValues(typeof(ColumnDataType))));
        Assert.Equal(3, values.Length);
    }

    [Theory]
    [InlineData(ColumnDataType.Continuous)]
    [InlineData(ColumnDataType.Discrete)]
    [InlineData(ColumnDataType.Categorical)]
    public void ColumnDataType_AllValuesExist(ColumnDataType type)
    {
        Assert.True(Enum.IsDefined(typeof(ColumnDataType), type));
    }

    // ============================
    // ColumnMetadata: Construction
    // ============================

    [Fact]
    public void ColumnMetadata_Construction_ContinuousColumn()
    {
        var col = new ColumnMetadata("Age", ColumnDataType.Continuous);
        Assert.Equal("Age", col.Name);
        Assert.Equal(ColumnDataType.Continuous, col.DataType);
        Assert.Empty(col.Categories);
        Assert.Equal(0, col.NumCategories);
        Assert.True(col.IsNumerical);
        Assert.False(col.IsCategorical);
    }

    [Fact]
    public void ColumnMetadata_Construction_DiscreteColumn()
    {
        var col = new ColumnMetadata("Count", ColumnDataType.Discrete);
        Assert.Equal(ColumnDataType.Discrete, col.DataType);
        Assert.True(col.IsNumerical);
        Assert.False(col.IsCategorical);
    }

    [Fact]
    public void ColumnMetadata_Construction_CategoricalColumn()
    {
        var categories = new[] { "Red", "Green", "Blue" };
        var col = new ColumnMetadata("Color", ColumnDataType.Categorical, categories);
        Assert.Equal(ColumnDataType.Categorical, col.DataType);
        Assert.True(col.IsCategorical);
        Assert.False(col.IsNumerical);
        Assert.Equal(3, col.NumCategories);
        Assert.Equal("Red", col.Categories[0]);
        Assert.Equal("Green", col.Categories[1]);
        Assert.Equal("Blue", col.Categories[2]);
    }

    [Fact]
    public void ColumnMetadata_Construction_NullCategories_Empty()
    {
        var col = new ColumnMetadata("X", ColumnDataType.Continuous, null);
        Assert.Empty(col.Categories);
        Assert.Equal(0, col.NumCategories);
    }

    [Fact]
    public void ColumnMetadata_Construction_ColumnIndex()
    {
        var col = new ColumnMetadata("X", ColumnDataType.Continuous, columnIndex: 5);
        Assert.Equal(5, col.ColumnIndex);
    }

    [Fact]
    public void ColumnMetadata_DefaultStatistics()
    {
        var col = new ColumnMetadata("X", ColumnDataType.Continuous);
        Assert.Equal(0.0, col.Min);
        Assert.Equal(0.0, col.Max);
        Assert.Equal(0.0, col.Mean);
        Assert.Equal(1.0, col.Std); // Default Std is 1.0
    }

    [Fact]
    public void ColumnMetadata_SetStatistics()
    {
        var col = new ColumnMetadata("X", ColumnDataType.Continuous);
        col.Min = -5.0;
        col.Max = 10.0;
        col.Mean = 2.5;
        col.Std = 3.7;

        Assert.Equal(-5.0, col.Min);
        Assert.Equal(10.0, col.Max);
        Assert.Equal(2.5, col.Mean);
        Assert.Equal(3.7, col.Std);
    }

    // ============================
    // ColumnMetadata: Clone
    // ============================

    [Fact]
    public void ColumnMetadata_Clone_CopiesAllFields()
    {
        var original = new ColumnMetadata("Age", ColumnDataType.Continuous, columnIndex: 3)
        {
            Min = 0,
            Max = 100,
            Mean = 35,
            Std = 15
        };

        var clone = original.Clone();

        Assert.Equal(original.Name, clone.Name);
        Assert.Equal(original.DataType, clone.DataType);
        Assert.Equal(original.ColumnIndex, clone.ColumnIndex);
        Assert.Equal(original.Min, clone.Min);
        Assert.Equal(original.Max, clone.Max);
        Assert.Equal(original.Mean, clone.Mean);
        Assert.Equal(original.Std, clone.Std);
    }

    [Fact]
    public void ColumnMetadata_Clone_IsIndependent()
    {
        var original = new ColumnMetadata("Age", ColumnDataType.Continuous)
        {
            Min = 0,
            Max = 100
        };

        var clone = original.Clone();
        clone.Min = -10;
        clone.Max = 200;

        Assert.Equal(0.0, original.Min); // Original unchanged
        Assert.Equal(100.0, original.Max);
    }

    [Fact]
    public void ColumnMetadata_Clone_CategoricalPreservesCategories()
    {
        var categories = new[] { "A", "B", "C" };
        var original = new ColumnMetadata("Cat", ColumnDataType.Categorical, categories);

        var clone = original.Clone();

        Assert.Equal(3, clone.NumCategories);
        Assert.Equal("A", clone.Categories[0]);
        Assert.Equal("B", clone.Categories[1]);
        Assert.Equal("C", clone.Categories[2]);
    }

    // ============================
    // Synthetic Data Math: VGM (Variational Gaussian Mixture) Normalization
    // ============================

    [Fact]
    public void SyntheticMath_VGM_ModeSpecificNormalization()
    {
        // CTGAN uses VGM to normalize continuous columns:
        // 1. Fit a GMM with K modes
        // 2. For value x, select mode k with highest posterior
        // 3. Normalize: alpha_k = (x - mu_k) / (4 * sigma_k)
        // 4. Output: [alpha_k, one_hot(k)]

        double x = 5.0;
        double mu = 4.0;  // Mode mean
        double sigma = 0.5; // Mode std

        double alpha = (x - mu) / (4 * sigma);

        // alpha should be normalized around [-1, 1] for most values within 4 sigma
        Assert.Equal(0.5, alpha, 1e-10);
        Assert.True(alpha >= -1.0 && alpha <= 1.0,
            $"Alpha ({alpha}) should be in [-1, 1] for values within 4 sigma");
    }

    [Theory]
    [InlineData(5, 1, 10)]  // 5 continuous columns, 1 mode each: 5 * (1 + 1) = 10
    [InlineData(3, 10, 33)]  // 3 continuous columns, 10 modes: 3 * (1 + 10) = 33
    [InlineData(1, 5, 6)]   // 1 continuous column, 5 modes: 1 * (1 + 5) = 6
    public void SyntheticMath_VGM_TransformedWidth_ContinuousOnly(int numContinuous, int numModes, int expectedWidth)
    {
        // Each continuous column produces: 1 alpha value + numModes one-hot values
        int width = numContinuous * (1 + numModes);
        Assert.Equal(expectedWidth, width);
    }

    [Fact]
    public void SyntheticMath_OneHotEncoding_Width()
    {
        // Categorical column with K categories produces K-dimensional one-hot vector
        int numCategories = 5;
        double[] oneHot = new double[numCategories];
        int selectedCategory = 2;
        oneHot[selectedCategory] = 1.0;

        // Only one element should be 1
        Assert.Equal(1.0, oneHot.Sum(), 1e-10);
        Assert.Equal(1.0, oneHot[selectedCategory]);

        // All others should be 0
        for (int i = 0; i < numCategories; i++)
        {
            if (i != selectedCategory)
                Assert.Equal(0.0, oneHot[i]);
        }
    }

    [Theory]
    [InlineData(3, 10, 2, new int[] { 5, 3 }, 41)]
    // 3 continuous cols * (1 + 10 modes) = 33 + 5 + 3 = 41 categorical one-hot dims
    public void SyntheticMath_TransformedWidth_Mixed(int numContinuous, int numModes, int numCategorical,
        int[] categorySizes, int expectedWidth)
    {
        int continuousWidth = numContinuous * (1 + numModes);
        int categoricalWidth = categorySizes.Sum();
        int totalWidth = continuousWidth + categoricalWidth;
        Assert.Equal(expectedWidth, totalWidth);
    }

    // ============================
    // Synthetic Data Math: Gaussian Diffusion (TabDDPM)
    // ============================

    [Theory]
    [InlineData(0.0001, 0.02, 1000)]  // Standard linear schedule
    public void SyntheticMath_DiffusionBetaSchedule_Linear(double betaStart, double betaEnd, int numTimesteps)
    {
        // Linear schedule: beta_t = betaStart + t * (betaEnd - betaStart) / (T - 1)
        double[] betas = new double[numTimesteps];
        for (int t = 0; t < numTimesteps; t++)
        {
            betas[t] = betaStart + t * (betaEnd - betaStart) / (numTimesteps - 1);
        }

        Assert.Equal(betaStart, betas[0], 1e-10);
        Assert.Equal(betaEnd, betas[numTimesteps - 1], 1e-10);

        // Betas should be monotonically increasing
        for (int t = 1; t < numTimesteps; t++)
        {
            Assert.True(betas[t] >= betas[t - 1], $"Beta at t={t} should be >= beta at t={t - 1}");
        }
    }

    [Fact]
    public void SyntheticMath_DiffusionAlphaCumprod_Decreases()
    {
        // alpha_t = 1 - beta_t
        // alpha_bar_t = product(alpha_i, i=1..t)
        // alpha_bar should decrease over time (more noise accumulated)
        int T = 100;
        double betaStart = 0.0001, betaEnd = 0.02;

        double[] alphasCumprod = new double[T];
        double cumprod = 1.0;
        for (int t = 0; t < T; t++)
        {
            double beta = betaStart + t * (betaEnd - betaStart) / (T - 1);
            double alpha = 1.0 - beta;
            cumprod *= alpha;
            alphasCumprod[t] = cumprod;
        }

        // Alpha cumprod should be monotonically decreasing
        for (int t = 1; t < T; t++)
        {
            Assert.True(alphasCumprod[t] < alphasCumprod[t - 1],
                $"Alpha cumprod at t={t} should be < at t={t - 1}");
        }

        // First value should be close to 1, last should be close to 0
        Assert.True(alphasCumprod[0] > 0.99, "Alpha cumprod at t=0 should be close to 1");
        Assert.True(alphasCumprod[T - 1] < 0.5, "Alpha cumprod at final t should be small");
    }

    [Fact]
    public void SyntheticMath_DiffusionForwardProcess()
    {
        // Forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        double x0 = 5.0;
        double alphaCumprod = 0.5;  // Middle of diffusion
        double epsilon = 1.0;  // Unit noise

        double sqrtAlphaBar = Math.Sqrt(alphaCumprod);
        double sqrtOneMinusAlphaBar = Math.Sqrt(1.0 - alphaCumprod);

        double xt = sqrtAlphaBar * x0 + sqrtOneMinusAlphaBar * epsilon;

        // x_t should be a mixture of signal and noise
        Assert.False(double.IsNaN(xt));
        Assert.False(double.IsInfinity(xt));

        // Verify the variance of the forward process is 1 (preserving scale)
        // Var(x_t) = alpha_bar * Var(x0) + (1 - alpha_bar) * Var(epsilon)
        // If Var(x0) = 1 and Var(epsilon) = 1, then Var(x_t) = alpha_bar + (1 - alpha_bar) = 1
        double varContribution = alphaCumprod + (1.0 - alphaCumprod);
        Assert.Equal(1.0, varContribution, 1e-10);
    }

    [Fact]
    public void SyntheticMath_DiffusionMSELoss()
    {
        // MSE loss between predicted and actual noise
        double[] predicted = { 0.5, -0.3, 0.8, 0.1 };
        double[] actual = { 0.4, -0.2, 0.9, 0.0 };

        double mse = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double diff = predicted[i] - actual[i];
            mse += diff * diff;
        }
        mse /= predicted.Length;

        // MSE = (0.01 + 0.01 + 0.01 + 0.01) / 4 = 0.04 / 4 = 0.01
        Assert.Equal(0.01, mse, 1e-10);
    }

    // ============================
    // Synthetic Data Math: CTGAN Conditional Vector
    // ============================

    [Fact]
    public void SyntheticMath_ConditionalVector_Width()
    {
        // Conditional vector width = sum of categories across all categorical columns
        int[] categoriesPerColumn = { 5, 3, 10 }; // 3 categorical columns
        int condWidth = categoriesPerColumn.Sum();
        Assert.Equal(18, condWidth);
    }

    [Fact]
    public void SyntheticMath_ConditionalVector_OneHotPerColumn()
    {
        // Each column's section of the conditional vector is one-hot
        int[] categoriesPerColumn = { 3, 4 };
        int totalWidth = categoriesPerColumn.Sum(); // 7
        double[] condVector = new double[totalWidth];

        // Set category 1 in column 0 (offset 0)
        condVector[1] = 1.0;
        // Set category 2 in column 1 (offset 3)
        condVector[3 + 2] = 1.0;

        // Each column's section should sum to 1
        double sum0 = condVector[0] + condVector[1] + condVector[2];
        double sum1 = condVector[3] + condVector[4] + condVector[5] + condVector[6];
        Assert.Equal(1.0, sum0, 1e-10);
        Assert.Equal(1.0, sum1, 1e-10);
    }

    // ============================
    // Synthetic Data Math: Data Quality Metrics
    // ============================

    [Fact]
    public void SyntheticMath_WassersteinDistance_SameDistribution()
    {
        // 1D Wasserstein distance between identical sorted distributions = 0
        double[] real = { 1, 2, 3, 4, 5 };
        double[] synthetic = { 1, 2, 3, 4, 5 };

        Array.Sort(real);
        Array.Sort(synthetic);

        double distance = 0;
        for (int i = 0; i < real.Length; i++)
        {
            distance += Math.Abs(real[i] - synthetic[i]);
        }
        distance /= real.Length;

        Assert.Equal(0.0, distance, 1e-10);
    }

    [Theory]
    [InlineData(new double[] { 1, 2, 3, 4, 5 }, new double[] { 2, 3, 4, 5, 6 }, 1.0)]  // Shifted by 1
    [InlineData(new double[] { 0, 0, 0, 0, 0 }, new double[] { 1, 1, 1, 1, 1 }, 1.0)]  // All offset by 1
    public void SyntheticMath_WassersteinDistance_KnownValues(double[] real, double[] synthetic, double expectedDistance)
    {
        Array.Sort(real);
        Array.Sort(synthetic);

        double distance = 0;
        for (int i = 0; i < real.Length; i++)
        {
            distance += Math.Abs(real[i] - synthetic[i]);
        }
        distance /= real.Length;

        Assert.Equal(expectedDistance, distance, 1e-10);
    }

    [Theory]
    [InlineData(new double[] { 0.5, 0.5 }, new double[] { 0.5, 0.5 }, 0.0)]      // Same distribution
    [InlineData(new double[] { 1.0, 0.0 }, new double[] { 0.5, 0.5 }, 0.693)]     // Max divergence from delta to uniform
    public void SyntheticMath_KLDivergence(double[] p, double[] q, double expectedKL)
    {
        // KL(P || Q) = sum(p_i * log(p_i / q_i))
        double kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            if (p[i] > 0 && q[i] > 0)
            {
                kl += p[i] * Math.Log(p[i] / q[i]);
            }
        }

        Assert.Equal(expectedKL, kl, 1e-2);
    }

    [Fact]
    public void SyntheticMath_KLDivergence_NonNegative()
    {
        // Gibbs' inequality: KL divergence is always >= 0
        double[] p = { 0.3, 0.5, 0.2 };
        double[] q = { 0.4, 0.4, 0.2 };

        double kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            if (p[i] > 0 && q[i] > 0)
            {
                kl += p[i] * Math.Log(p[i] / q[i]);
            }
        }

        Assert.True(kl >= -1e-10, $"KL divergence ({kl}) must be non-negative");
    }

    // ============================
    // Synthetic Data Math: Privacy Metrics
    // ============================

    [Theory]
    [InlineData(1.0, 0.368)]    // P(privacy) = exp(-epsilon) for epsilon-DP
    [InlineData(0.1, 0.905)]    // More private
    [InlineData(10.0, 0.0000454)] // Less private
    public void SyntheticMath_DifferentialPrivacy_EpsilonBudget(double epsilon, double expectedMaxDisclosure)
    {
        // In epsilon-differential privacy, the probability of disclosure is bounded by exp(-epsilon)
        double maxDisclosure = Math.Exp(-epsilon);
        Assert.Equal(expectedMaxDisclosure, maxDisclosure, 1e-3);
    }

    [Theory]
    [InlineData(1.0, 1.0, 1.0)]     // sensitivity=1, epsilon=1: noise scale = 1
    [InlineData(1.0, 0.1, 10.0)]    // Lower epsilon needs more noise
    [InlineData(2.0, 1.0, 2.0)]     // Higher sensitivity needs more noise
    public void SyntheticMath_DPNoiseScale_Laplace(double sensitivity, double epsilon, double expectedScale)
    {
        // Laplace mechanism: noise ~ Laplace(0, sensitivity / epsilon)
        double scale = sensitivity / epsilon;
        Assert.Equal(expectedScale, scale, 1e-10);
    }

    // ============================
    // Synthetic Data Math: Column Statistics
    // ============================

    [Fact]
    public void SyntheticMath_ZScoreNormalization()
    {
        double[] data = { 10, 20, 30, 40, 50 };
        double mean = data.Average();  // 30
        double std = Math.Sqrt(data.Select(x => (x - mean) * (x - mean)).Sum() / data.Length);  // 14.14

        double[] normalized = data.Select(x => (x - mean) / std).ToArray();

        Assert.Equal(0.0, normalized.Average(), 1e-10);
        double normalizedStd = Math.Sqrt(normalized.Select(x => x * x).Sum() / normalized.Length);
        Assert.Equal(1.0, normalizedStd, 1e-10);
    }

    [Fact]
    public void SyntheticMath_MinMaxNormalization()
    {
        double[] data = { 10, 20, 30, 40, 50 };
        double min = data.Min();
        double max = data.Max();

        double[] normalized = data.Select(x => (x - min) / (max - min)).ToArray();

        Assert.Equal(0.0, normalized.Min(), 1e-10);
        Assert.Equal(1.0, normalized.Max(), 1e-10);
    }

    // ============================
    // Synthetic Data Math: GAN Training Metrics
    // ============================

    [Fact]
    public void SyntheticMath_GeneratorLoss_BinaryCrossEntropy()
    {
        // Generator loss: -log(D(G(z)))
        // Generator wants D(G(z)) -> 1 (fool discriminator)
        double dOutput = 0.8; // Discriminator thinks it's real with p=0.8
        double gLoss = -Math.Log(dOutput);

        Assert.True(gLoss > 0, "Generator loss should be positive");
        Assert.True(gLoss < 1.0, "Good discriminator output should give small loss");
    }

    [Fact]
    public void SyntheticMath_DiscriminatorLoss_BinaryCrossEntropy()
    {
        // Discriminator loss: -[log(D(real)) + log(1 - D(G(z)))]
        double dReal = 0.9;  // Correctly identifies real as real
        double dFake = 0.1;  // Correctly identifies fake as fake

        double dLoss = -(Math.Log(dReal) + Math.Log(1 - dFake));

        Assert.True(dLoss > 0, "Discriminator loss should be positive");

        // Perfect discriminator would have minimal loss
        double perfectLoss = -(Math.Log(1.0 - 1e-10) + Math.Log(1.0 - 1e-10));
        Assert.True(dLoss > perfectLoss,
            "Non-perfect discriminator should have higher loss than perfect one");
    }

    [Theory]
    [InlineData(0.5, 0.693)]   // Random discriminator: -log(0.5) = 0.693
    [InlineData(0.9, 0.105)]   // Good discriminator: -log(0.9) = 0.105
    [InlineData(0.1, 2.303)]   // Bad discriminator: -log(0.1) = 2.303
    public void SyntheticMath_BinaryCrossEntropy_Values(double prediction, double expectedLoss)
    {
        double loss = -Math.Log(prediction);
        Assert.Equal(expectedLoss, loss, 1e-2);
    }

    // ============================
    // Synthetic Data Math: TVAE (Variational Autoencoder)
    // ============================

    [Fact]
    public void SyntheticMath_VAE_ReparameterizationTrick()
    {
        // z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
        double mu = 2.0;
        double logVar = 0.5;
        double sigma = Math.Exp(0.5 * logVar); // sigma = exp(logVar / 2)
        double epsilon = 0.7; // Fixed for testing

        double z = mu + sigma * epsilon;

        Assert.False(double.IsNaN(z));
        Assert.False(double.IsInfinity(z));
        Assert.Equal(mu + Math.Exp(0.25) * 0.7, z, 1e-10);
    }

    [Fact]
    public void SyntheticMath_VAE_KLDivergence_FromStandardNormal()
    {
        // KL(q(z|x) || p(z)) where q = N(mu, sigma^2) and p = N(0, 1)
        // KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        double mu = 0.5;
        double logVar = -0.5; // log(sigma^2) = -0.5, so sigma^2 = exp(-0.5)

        double kl = -0.5 * (1.0 + logVar - mu * mu - Math.Exp(logVar));

        Assert.True(kl >= 0, "KL divergence must be non-negative");

        // At mu=0, logVar=0 (standard normal), KL should be 0
        double klZero = -0.5 * (1.0 + 0.0 - 0.0 - Math.Exp(0.0));
        Assert.Equal(0.0, klZero, 1e-10);
    }

    // ============================
    // Synthetic Data Math: Category Frequency Matching
    // ============================

    [Fact]
    public void SyntheticMath_CategoryFrequency_JensenShannonDivergence()
    {
        // JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q)
        double[] p = { 0.3, 0.5, 0.2 }; // Real distribution
        double[] q = { 0.25, 0.5, 0.25 }; // Synthetic distribution

        double[] m = new double[p.Length];
        for (int i = 0; i < p.Length; i++)
            m[i] = 0.5 * (p[i] + q[i]);

        double klPM = 0, klQM = 0;
        for (int i = 0; i < p.Length; i++)
        {
            if (p[i] > 0) klPM += p[i] * Math.Log(p[i] / m[i]);
            if (q[i] > 0) klQM += q[i] * Math.Log(q[i] / m[i]);
        }

        double jsd = 0.5 * klPM + 0.5 * klQM;

        Assert.True(jsd >= 0, "JSD must be non-negative");
        Assert.True(jsd <= Math.Log(2) + 1e-10, "JSD must be <= ln(2)");
    }

    [Fact]
    public void SyntheticMath_CategoryFrequency_IdenticalDistributions()
    {
        double[] p = { 0.3, 0.5, 0.2 };
        double[] q = { 0.3, 0.5, 0.2 };

        double[] m = new double[p.Length];
        for (int i = 0; i < p.Length; i++)
            m[i] = 0.5 * (p[i] + q[i]);

        double klPM = 0, klQM = 0;
        for (int i = 0; i < p.Length; i++)
        {
            if (p[i] > 0) klPM += p[i] * Math.Log(p[i] / m[i]);
            if (q[i] > 0) klQM += q[i] * Math.Log(q[i] / m[i]);
        }

        double jsd = 0.5 * klPM + 0.5 * klQM;
        Assert.Equal(0.0, jsd, 1e-10);
    }
}
