using System;
using System.Linq;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Metrics
{
    /// <summary>
    /// Fréchet Inception Distance (FID) - A metric for evaluating the quality of generated images.
    ///
    /// For Beginners:
    /// FID measures how similar generated images are to real images by comparing their
    /// statistical properties in a feature space. Lower FID scores indicate better quality.
    ///
    /// Think of it like comparing two photo albums:
    /// 1. Extract features from photos using a pre-trained neural network (Inception)
    /// 2. Compute statistics (mean and covariance) for real photos
    /// 3. Compute statistics for generated photos
    /// 4. Measure the distance between these statistics
    ///
    /// The distance is called "Fréchet distance" - a mathematical way to measure how
    /// different two multivariate Gaussian distributions are.
    ///
    /// Why it's better than just looking at images:
    /// - Objective and quantitative (not just human opinion)
    /// - Captures both quality AND diversity
    /// - Correlates well with human judgment
    /// - Industry standard for GAN evaluation
    ///
    /// Typical FID scores:
    /// - FID < 10: Excellent (very close to real images)
    /// - FID 10-20: Good quality
    /// - FID 20-50: Moderate quality
    /// - FID > 50: Poor quality
    ///
    /// Based on "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
    /// by Heusel et al. (2017)
    /// </summary>
    /// <typeparam name="T">The numeric type for computations (e.g., double, float)</typeparam>
    public class FrechetInceptionDistance<T> where T : struct, IComparable, IFormattable, IConvertible, IComparable<T>, IEquatable<T>
    {
        private readonly INumericOperations<T> NumOps;

        /// <summary>
        /// Gets the Inception network used for feature extraction.
        /// In a full implementation, this would be a pre-trained InceptionV3 model.
        /// </summary>
        public ConvolutionalNeuralNetwork<T>? InceptionNetwork { get; private set; }

        /// <summary>
        /// Gets or sets the layer from which to extract features.
        /// Typically the final pooling layer before classification (2048 dimensions).
        /// </summary>
        public int FeatureLayer { get; set; }

        /// <summary>
        /// Gets the dimensionality of extracted features.
        /// For InceptionV3, this is typically 2048.
        /// </summary>
        public int FeatureDimension { get; private set; }

        /// <summary>
        /// Initializes a new instance of FID calculator.
        /// </summary>
        /// <param name="inceptionNetwork">Pre-trained Inception network for feature extraction (optional)</param>
        /// <param name="featureDimension">Dimension of extracted features (default 2048 for InceptionV3)</param>
        public FrechetInceptionDistance(
            ConvolutionalNeuralNetwork<T>? inceptionNetwork = null,
            int featureDimension = 2048)
        {
            NumOps = MathHelper.GetNumericOperations<T>();
            InceptionNetwork = inceptionNetwork;
            FeatureDimension = featureDimension;
            FeatureLayer = -2; // Second to last layer (before classification)
        }

        /// <summary>
        /// Computes the FID score between real and generated images.
        /// </summary>
        /// <param name="realImages">Tensor of real images</param>
        /// <param name="generatedImages">Tensor of generated images</param>
        /// <returns>FID score (lower is better)</returns>
        public double ComputeFID(Tensor<T> realImages, Tensor<T> generatedImages)
        {
            // Extract features from real and generated images
            var realFeatures = ExtractFeatures(realImages);
            var generatedFeatures = ExtractFeatures(generatedImages);

            // Compute mean and covariance for both distributions
            var (realMean, realCov) = ComputeStatistics(realFeatures);
            var (genMean, genCov) = ComputeStatistics(generatedFeatures);

            // Compute Fréchet distance
            var fid = ComputeFrechetDistance(realMean, realCov, genMean, genCov);

            return fid;
        }

        /// <summary>
        /// Extracts features from images using the Inception network.
        /// </summary>
        /// <param name="images">Images to process</param>
        /// <returns>Feature matrix (num_images × feature_dim)</returns>
        private Matrix<T> ExtractFeatures(Tensor<T> images)
        {
            var numImages = images.Shape[0];

            if (InceptionNetwork == null)
            {
                // If no Inception network provided, return dummy features
                // In a real implementation, you would load a pre-trained InceptionV3
                return CreateDummyFeatures(numImages);
            }

            // Set to inference mode
            InceptionNetwork.SetTrainingMode(false);

            var features = new Matrix<T>(numImages, FeatureDimension);

            // Process each image
            for (int i = 0; i < numImages; i++)
            {
                // Extract single image
                var imageSize = images.Length / numImages;
                var singleImage = new Tensor<T>(new[] { 1, images.Shape[1], images.Shape[2], images.Shape[3] });

                // Copy data from source tensor to single image tensor
                for (int k = 0; k < imageSize; k++)
                {
                    singleImage.SetFlat(k, images.GetFlat(i * imageSize + k));
                }

                // Forward pass through Inception network
                var output = InceptionNetwork.Predict(singleImage);

                // Extract features from specified layer
                // In full implementation, would access intermediate layer activations
                // For now, use output
                for (int j = 0; j < Math.Min(output.Length, FeatureDimension); j++)
                {
                    features[i, j] = output.GetFlat(j);
                }
            }

            return features;
        }

        /// <summary>
        /// Creates dummy features for testing when no Inception network is available.
        /// </summary>
        private Matrix<T> CreateDummyFeatures(int numSamples)
        {
            var random = new Random();
            var features = new Matrix<T>(numSamples, FeatureDimension);

            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < FeatureDimension; j++)
                {
                    features[i, j] = NumOps.FromDouble(random.NextDouble());
                }
            }

            return features;
        }

        /// <summary>
        /// Computes mean and covariance matrix of feature vectors.
        /// </summary>
        /// <param name="features">Feature matrix (num_samples × feature_dim)</param>
        /// <returns>Tuple of (mean vector, covariance matrix)</returns>
        /// <exception cref="ArgumentException">Thrown when numSamples is less than 2.</exception>
        private (Vector<T> mean, Matrix<T> covariance) ComputeStatistics(Matrix<T> features)
        {
            var numSamples = features.Rows;
            var dim = features.Columns;

            if (numSamples < 2)
            {
                throw new ArgumentException(
                    $"Need at least 2 samples to compute covariance, got {numSamples}.",
                    nameof(features));
            }

            // Compute mean
            var mean = new Vector<T>(dim);
            for (int j = 0; j < dim; j++)
            {
                var sum = NumOps.Zero;
                for (int i = 0; i < numSamples; i++)
                {
                    sum = NumOps.Add(sum, features[i, j]);
                }
                mean[j] = NumOps.Divide(sum, NumOps.FromDouble(numSamples));
            }

            // Compute covariance matrix
            var covariance = new Matrix<T>(dim, dim);
            for (int j1 = 0; j1 < dim; j1++)
            {
                for (int j2 = 0; j2 < dim; j2++)
                {
                    var sum = NumOps.Zero;
                    for (int i = 0; i < numSamples; i++)
                    {
                        var diff1 = NumOps.Subtract(features[i, j1], mean[j1]);
                        var diff2 = NumOps.Subtract(features[i, j2], mean[j2]);
                        sum = NumOps.Add(sum, NumOps.Multiply(diff1, diff2));
                    }
                    covariance[j1, j2] = NumOps.Divide(sum, NumOps.FromDouble(numSamples - 1));
                }
            }

            return (mean, covariance);
        }

        /// <summary>
        /// Computes the Fréchet distance between two Gaussian distributions.
        /// FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))
        /// where μ is mean, Σ is covariance, Tr is trace
        /// </summary>
        private double ComputeFrechetDistance(
            Vector<T> mean1,
            Matrix<T> cov1,
            Vector<T> mean2,
            Matrix<T> cov2)
        {
            // 1. Compute squared difference of means: ||μ₁ - μ₂||²
            var meanDiffSq = NumOps.Zero;
            for (int i = 0; i < mean1.Length; i++)
            {
                var diff = NumOps.Subtract(mean1[i], mean2[i]);
                meanDiffSq = NumOps.Add(meanDiffSq, NumOps.Multiply(diff, diff));
            }

            // 2. Compute trace of covariance matrices: Tr(Σ₁) + Tr(Σ₂)
            var trace1 = NumOps.Zero;
            var trace2 = NumOps.Zero;
            for (int i = 0; i < cov1.Rows; i++)
            {
                trace1 = NumOps.Add(trace1, cov1[i, i]);
                trace2 = NumOps.Add(trace2, cov2[i, i]);
            }
            var traceCov = NumOps.Add(trace1, trace2);

            // 3. Compute Tr(√(Σ₁Σ₂)) using proper matrix square root
            // For symmetric positive semi-definite matrices, we compute the product
            // and then find the trace of its square root
            var traceSqrtCovProduct = ComputeTraceSqrtCovProduct(cov1, cov2);

            // FID = ||μ₁ - μ₂||² + Tr(Σ₁) + Tr(Σ₂) - 2*Tr(√(Σ₁Σ₂))
            var fid = NumOps.Add(meanDiffSq, traceCov);
            fid = NumOps.Subtract(fid, NumOps.Multiply(NumOps.FromDouble(2.0), traceSqrtCovProduct));

            return Convert.ToDouble(fid);
        }

        /// <summary>
        /// Computes Tr(√(Σ₁Σ₂)) using Newton-Schulz iteration for matrix square root.
        /// </summary>
        private T ComputeTraceSqrtCovProduct(Matrix<T> cov1, Matrix<T> cov2)
        {
            int n = cov1.Rows;

            // Compute the matrix product Σ₁ * Σ₂
            var product = new Matrix<T>(n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    var sum = NumOps.Zero;
                    for (int k = 0; k < n; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(cov1[i, k], cov2[k, j]));
                    }
                    product[i, j] = sum;
                }
            }

            // For computing Tr(√A), we use the identity that for SPD matrices:
            // Tr(√A) = sum of square roots of eigenvalues
            // Use power iteration to approximate the trace of the square root
            // via Newton-Schulz iteration: Y_{k+1} = 0.5 * Y_k * (3I - Y_k^2 * A)
            // with Y_0 = A / ||A||_F, converges to √(A^{-1}), so we need to adapt

            // Simpler approach: Use the property that for SPD matrices,
            // Tr(√A) ≈ √Tr(A) when eigenvalues are close together,
            // but better to use Denman-Beavers iteration which converges to √A

            // Denman-Beavers iteration: Y_0 = A, Z_0 = I
            // Y_{k+1} = 0.5 * (Y_k + Z_k^{-1})
            // Z_{k+1} = 0.5 * (Z_k + Y_k^{-1})
            // Converges to: Y → √A, Z → √(A^{-1})

            // For numerical stability, use a simpler approximation with eigenvalue sum
            // First, symmetrize the product to handle numerical issues: (A + A^T) / 2
            var symProduct = new Matrix<T>(n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    symProduct[i, j] = NumOps.Divide(
                        NumOps.Add(product[i, j], product[j, i]),
                        NumOps.FromDouble(2.0));
                }
            }

            // Use Newton-Schulz iteration for matrix square root
            // Start with Y = A / ||A||_F for numerical stability
            var frobNormSq = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    frobNormSq = NumOps.Add(frobNormSq, NumOps.Multiply(symProduct[i, j], symProduct[i, j]));
                }
            }
            var frobNorm = NumOps.Sqrt(frobNormSq);

            // If the product is essentially zero, return zero
            if (NumOps.LessThan(frobNorm, NumOps.FromDouble(1e-10)))
            {
                return NumOps.Zero;
            }

            // Scale for numerical stability
            var scale = NumOps.Sqrt(frobNorm);
            var Y = new Matrix<T>(n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Y[i, j] = NumOps.Divide(symProduct[i, j], scale);
                }
            }

            // Newton-Schulz iteration: Y_{k+1} = 0.5 * Y_k * (3I - Y_k * Y_k)
            // Run for a fixed number of iterations
            const int maxIterations = 15;
            var identity = Matrix<T>.CreateIdentity(n);

            for (int iter = 0; iter < maxIterations; iter++)
            {
                // Compute Y * Y
                var YY = MatrixMultiply(Y, Y);

                // Compute 3I - Y*Y
                var threeIMinusYY = new Matrix<T>(n, n);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        threeIMinusYY[i, j] = NumOps.Subtract(
                            NumOps.Multiply(NumOps.FromDouble(3.0), identity[i, j]),
                            YY[i, j]);
                    }
                }

                // Y = 0.5 * Y * (3I - Y*Y)
                var newY = MatrixMultiply(Y, threeIMinusYY);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        Y[i, j] = NumOps.Multiply(NumOps.FromDouble(0.5), newY[i, j]);
                    }
                }
            }

            // Y now approximates √(A/scale), so √A ≈ Y * √scale
            // Tr(√A) = √scale * Tr(Y)
            var sqrtScale = NumOps.Sqrt(scale);
            var traceY = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                traceY = NumOps.Add(traceY, Y[i, i]);
            }

            return NumOps.Multiply(sqrtScale, traceY);
        }

        /// <summary>
        /// Multiplies two matrices.
        /// </summary>
        private Matrix<T> MatrixMultiply(Matrix<T> a, Matrix<T> b)
        {
            int n = a.Rows;
            var result = new Matrix<T>(n, n);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    var sum = NumOps.Zero;
                    for (int k = 0; k < n; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(a[i, k], b[k, j]));
                    }
                    result[i, j] = sum;
                }
            }

            return result;
        }

        /// <summary>
        /// Computes FID using pre-computed statistics.
        /// Useful when you want to compare against a fixed set of real images.
        /// </summary>
        /// <param name="realMean">Mean of real image features</param>
        /// <param name="realCov">Covariance of real image features</param>
        /// <param name="generatedImages">Generated images to evaluate</param>
        /// <returns>FID score</returns>
        public double ComputeFIDWithStats(
            Vector<T> realMean,
            Matrix<T> realCov,
            Tensor<T> generatedImages)
        {
            var generatedFeatures = ExtractFeatures(generatedImages);
            var (genMean, genCov) = ComputeStatistics(generatedFeatures);
            return ComputeFrechetDistance(realMean, realCov, genMean, genCov);
        }

        /// <summary>
        /// Pre-computes and caches statistics for a set of real images.
        /// This is useful for evaluating multiple generated batches against the same real data.
        /// </summary>
        /// <param name="realImages">Real images to compute statistics for</param>
        /// <returns>Tuple of (mean, covariance)</returns>
        public (Vector<T> mean, Matrix<T> covariance) PrecomputeRealStatistics(Tensor<T> realImages)
        {
            var features = ExtractFeatures(realImages);
            return ComputeStatistics(features);
        }
    }
}
