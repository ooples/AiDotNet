using System;
using AiDotNet.Engines;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Metrics;

/// <summary>
/// Kernel Inception Distance (KID) - A metric for evaluating the quality of generated images.
/// </summary>
/// <remarks>
/// <para>
/// KID measures how similar generated images are to real images using the Maximum Mean Discrepancy (MMD)
/// with a polynomial kernel in the feature space of an Inception network.
/// </para>
/// <para>
/// Advantages over FID:
/// - Unbiased estimator (FID is biased for small sample sizes)
/// - Provides variance estimates for the metric
/// - Works well with smaller datasets
/// - More robust to sample size variations
/// </para>
/// <para>
/// Formula: KID = MMD^2(F_real, F_generated) using polynomial kernel k(x,y) = (x^T y / d + 1)^3
/// </para>
/// <para>
/// Typical KID scores (multiplied by 1000 for readability):
/// - KID &lt; 0.5: Excellent quality
/// - KID 0.5-2.0: Good quality
/// - KID 2.0-5.0: Moderate quality
/// - KID &gt; 5.0: Poor quality
/// </para>
/// <para>
/// Based on "Demystifying MMD GANs" by Binkowski et al. (2018)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations</typeparam>
public class KernelInceptionDistance<T>
{
    private readonly INumericOperations<T> _numOps;
    private IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the feature extraction network used for computing image representations.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> FeatureNetwork { get; }

    /// <summary>
    /// Gets the dimensionality of extracted features.
    /// </summary>
    public int FeatureDimension { get; }

    /// <summary>
    /// Gets or sets the polynomial degree for the kernel. Default is 3.
    /// </summary>
    public int PolynomialDegree { get; set; }

    /// <summary>
    /// Gets or sets the number of subsets for computing variance estimates.
    /// </summary>
    public int NumSubsets { get; set; }

    /// <summary>
    /// Gets or sets the subset size for variance computation.
    /// </summary>
    public int SubsetSize { get; set; }

    /// <summary>
    /// Initializes a new instance of KID calculator.
    /// </summary>
    /// <param name="featureNetwork">Pre-trained network for feature extraction</param>
    /// <param name="featureDimension">Dimension of extracted features (default 2048 for InceptionV3)</param>
    /// <param name="polynomialDegree">Polynomial kernel degree (default 3)</param>
    /// <param name="numSubsets">Number of subsets for variance estimation (default 100)</param>
    /// <param name="subsetSize">Size of each subset (default 1000)</param>
    public KernelInceptionDistance(
        ConvolutionalNeuralNetwork<T> featureNetwork,
        int featureDimension = 2048,
        int polynomialDegree = 3,
        int numSubsets = 100,
        int subsetSize = 1000)
    {
        if (featureNetwork == null)
        {
            throw new ArgumentNullException(nameof(featureNetwork),
                "A pre-trained feature extraction network is required for KID computation");
        }
        if (featureDimension < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(featureDimension),
                "Feature dimension must be at least 1");
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        FeatureNetwork = featureNetwork;
        FeatureDimension = featureDimension;
        PolynomialDegree = polynomialDegree;
        NumSubsets = numSubsets;
        SubsetSize = subsetSize;
    }

    /// <summary>
    /// Computes the KID score between real and generated images.
    /// </summary>
    /// <param name="realImages">Tensor of real images [N, C, H, W]</param>
    /// <param name="generatedImages">Tensor of generated images [N, C, H, W]</param>
    /// <returns>KID score (lower is better)</returns>
    public double ComputeKID(Tensor<T> realImages, Tensor<T> generatedImages)
    {
        var (mean, _) = ComputeKIDWithVariance(realImages, generatedImages);
        return mean;
    }

    /// <summary>
    /// Computes KID score with variance estimate using subset sampling.
    /// </summary>
    /// <param name="realImages">Tensor of real images [N, C, H, W]</param>
    /// <param name="generatedImages">Tensor of generated images [N, C, H, W]</param>
    /// <returns>Tuple of (mean KID, standard deviation)</returns>
    public (double mean, double std) ComputeKIDWithVariance(Tensor<T> realImages, Tensor<T> generatedImages)
    {
        if (realImages.Shape.Length < 4 || generatedImages.Shape.Length < 4)
        {
            throw new ArgumentException("Images must be 4D tensors [N, C, H, W]");
        }

        // Extract features
        var realFeatures = ExtractFeatures(realImages);
        var genFeatures = ExtractFeatures(generatedImages);

        // Compute KID using subsets
        int numReal = realFeatures.Rows;
        int numGen = genFeatures.Rows;
        int actualSubsetSize = Math.Min(SubsetSize, Math.Min(numReal, numGen));
        int actualNumSubsets = Math.Min(NumSubsets, Math.Min(numReal, numGen) / actualSubsetSize);

        if (actualNumSubsets < 1)
        {
            // Not enough samples for subset sampling, compute single MMD
            double singleMmd = ComputeMMD(realFeatures, genFeatures);
            return (singleMmd, 0.0);
        }

        var kidScores = new double[actualNumSubsets];
        var random = new Random(42); // Fixed seed for reproducibility

        for (int i = 0; i < actualNumSubsets; i++)
        {
            // Sample random subsets
            var realSubset = SampleSubset(realFeatures, actualSubsetSize, random);
            var genSubset = SampleSubset(genFeatures, actualSubsetSize, random);

            // Compute MMD for this subset
            kidScores[i] = ComputeMMD(realSubset, genSubset);
        }

        // Compute mean and standard deviation
        double mean = 0.0;
        foreach (double score in kidScores)
        {
            mean += score;
        }
        mean /= actualNumSubsets;

        double variance = 0.0;
        foreach (double score in kidScores)
        {
            variance += (score - mean) * (score - mean);
        }
        variance /= actualNumSubsets;
        double std = Math.Sqrt(variance);

        return (mean, std);
    }

    /// <summary>
    /// Computes KID using pre-computed feature statistics.
    /// </summary>
    /// <param name="realFeatures">Pre-computed features of real images [N, D]</param>
    /// <param name="generatedImages">Generated images to evaluate</param>
    /// <returns>KID score</returns>
    public double ComputeKIDWithFeatures(Matrix<T> realFeatures, Tensor<T> generatedImages)
    {
        var genFeatures = ExtractFeatures(generatedImages);
        return ComputeMMD(realFeatures, genFeatures);
    }

    /// <summary>
    /// Extracts features from images using the feature network.
    /// </summary>
    private Matrix<T> ExtractFeatures(Tensor<T> images)
    {
        var numImages = images.Shape[0];
        var features = new Matrix<T>(numImages, FeatureDimension);

        bool originalTrainingMode = FeatureNetwork.IsTrainingMode;
        FeatureNetwork.SetTrainingMode(false);

        try
        {
            var imageSize = images.Length / numImages;
            var singleImageShape = new[] { 1, images.Shape[1], images.Shape[2], images.Shape[3] };

            for (int i = 0; i < numImages; i++)
            {
                var singleImage = new Tensor<T>(singleImageShape);

                for (int k = 0; k < imageSize; k++)
                {
                    singleImage.SetFlat(k, images.GetFlat(i * imageSize + k));
                }

                var output = FeatureNetwork.Predict(singleImage);

                var featureCount = Math.Min(output.Length, FeatureDimension);
                for (int j = 0; j < featureCount; j++)
                {
                    features[i, j] = output.GetFlat(j);
                }
            }

            return features;
        }
        finally
        {
            FeatureNetwork.SetTrainingMode(originalTrainingMode);
        }
    }

    /// <summary>
    /// Computes Maximum Mean Discrepancy (MMD) using polynomial kernel.
    /// </summary>
    private double ComputeMMD(Matrix<T> features1, Matrix<T> features2)
    {
        int n1 = features1.Rows;
        int n2 = features2.Rows;
        int dim = features1.Columns;

        // Compute kernel matrices
        // K(x, y) = ((x^T y) / d + 1)^3
        T dimT = _numOps.FromDouble(dim);

        // E[k(x, x')] for x, x' from features1
        double k11 = 0.0;
        for (int i = 0; i < n1; i++)
        {
            for (int j = i + 1; j < n1; j++)
            {
                double dot = ComputeDotProduct(features1, i, features1, j);
                double k = Math.Pow(dot / dim + 1.0, PolynomialDegree);
                k11 += 2.0 * k;
            }
        }
        k11 /= (n1 * (n1 - 1));

        // E[k(y, y')] for y, y' from features2
        double k22 = 0.0;
        for (int i = 0; i < n2; i++)
        {
            for (int j = i + 1; j < n2; j++)
            {
                double dot = ComputeDotProduct(features2, i, features2, j);
                double k = Math.Pow(dot / dim + 1.0, PolynomialDegree);
                k22 += 2.0 * k;
            }
        }
        k22 /= (n2 * (n2 - 1));

        // E[k(x, y)] for x from features1, y from features2
        double k12 = 0.0;
        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < n2; j++)
            {
                double dot = ComputeDotProduct(features1, i, features2, j);
                double k = Math.Pow(dot / dim + 1.0, PolynomialDegree);
                k12 += k;
            }
        }
        k12 /= (n1 * n2);

        // MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        double mmdSquared = k11 + k22 - 2.0 * k12;

        // Return MMD (not squared) to match common usage
        return Math.Max(0, mmdSquared);
    }

    /// <summary>
    /// Computes dot product between two feature vectors.
    /// </summary>
    private double ComputeDotProduct(Matrix<T> features1, int idx1, Matrix<T> features2, int idx2)
    {
        double dot = 0.0;
        for (int k = 0; k < features1.Columns; k++)
        {
            double v1 = _numOps.ToDouble(features1[idx1, k]);
            double v2 = _numOps.ToDouble(features2[idx2, k]);
            dot += v1 * v2;
        }
        return dot;
    }

    /// <summary>
    /// Samples a random subset of features.
    /// </summary>
    private Matrix<T> SampleSubset(Matrix<T> features, int subsetSize, Random random)
    {
        int n = features.Rows;
        int d = features.Columns;

        // Generate random indices
        var indices = new int[subsetSize];
        var used = new bool[n];
        for (int i = 0; i < subsetSize; i++)
        {
            int idx;
            do
            {
                idx = random.Next(n);
            } while (used[idx]);
            used[idx] = true;
            indices[i] = idx;
        }

        // Create subset matrix
        var subset = new Matrix<T>(subsetSize, d);
        for (int i = 0; i < subsetSize; i++)
        {
            for (int j = 0; j < d; j++)
            {
                subset[i, j] = features[indices[i], j];
            }
        }

        return subset;
    }

    /// <summary>
    /// Pre-computes features for a set of images for efficient repeated evaluation.
    /// </summary>
    /// <param name="images">Images to compute features for</param>
    /// <returns>Feature matrix [N, D]</returns>
    public Matrix<T> PrecomputeFeatures(Tensor<T> images)
    {
        return ExtractFeatures(images);
    }
}
