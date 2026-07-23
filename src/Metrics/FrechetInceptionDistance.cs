using System;
using AiDotNet.Engines;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Metrics
{
    /// <summary>
    /// Fréchet Inception Distance (FID) - A metric for evaluating the quality of generated images.
    ///
    /// FID measures how similar generated images are to real images by comparing their
    /// statistical properties in a feature space. Lower FID scores indicate better quality.
    ///
    /// The algorithm:
    /// 1. Extract features from images using a pre-trained neural network
    /// 2. Compute statistics (mean and covariance) for real and generated image features
    /// 3. Compute the Fréchet distance between the two Gaussian distributions
    ///
    /// Formula: FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))
    /// where μ is mean, Σ is covariance, Tr is trace
    ///
    /// Typical FID scores:
    /// - FID less than 10: Excellent quality
    /// - FID 10-20: Good quality
    /// - FID 20-50: Moderate quality
    /// - FID greater than 50: Poor quality
    ///
    /// Based on "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
    /// by Heusel et al. (2017)
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class FrechetInceptionDistance<T>
    {
        private readonly INumericOperations<T> _numOps;
        private IEngine Engine => AiDotNetEngine.Current;

        /// <summary>
        /// Gets the feature extraction network used for computing image representations.
        /// </summary>
        public ConvolutionalNeuralNetwork<T> FeatureNetwork { get; }

        /// <summary>
        /// Gets or sets the layer index from which to extract features.
        /// Use -1 for the last layer, -2 for second to last, etc.
        /// </summary>
        public int FeatureLayer { get; set; }

        /// <summary>
        /// Gets the dimensionality of extracted features.
        /// </summary>
        public int FeatureDimension { get; }

        /// <summary>
        /// Initializes a new instance of FID calculator.
        /// </summary>
        /// <param name="featureNetwork">Pre-trained network for feature extraction</param>
        /// <param name="featureDimension">Dimension of extracted features</param>
        /// <exception cref="ArgumentNullException">Thrown when featureNetwork is null</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when featureDimension is less than 1</exception>
        public FrechetInceptionDistance(
            ConvolutionalNeuralNetwork<T> featureNetwork,
            int featureDimension = 2048)
        {
            if (featureNetwork == null)
            {
                throw new ArgumentNullException(nameof(featureNetwork),
                    "A pre-trained feature extraction network is required for FID computation");
            }
            if (featureDimension < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(featureDimension),
                    "Feature dimension must be at least 1");
            }

            _numOps = MathHelper.GetNumericOperations<T>();
            FeatureNetwork = featureNetwork;
            FeatureDimension = featureDimension;
            FeatureLayer = -2;
        }

        /// <summary>
        /// Computes the FID score between real and generated images.
        /// </summary>
        /// <param name="realImages">Tensor of real images [N, C, H, W]</param>
        /// <param name="generatedImages">Tensor of generated images [N, C, H, W]</param>
        /// <returns>FID score (lower is better)</returns>
        /// <exception cref="ArgumentException">Thrown when images have incompatible shapes</exception>
        public double ComputeFID(Tensor<T> realImages, Tensor<T> generatedImages)
        {
            if (realImages.Shape.Length < 4 || generatedImages.Shape.Length < 4)
            {
                throw new ArgumentException("Images must be 4D tensors [N, C, H, W]");
            }

            var realFeatures = ExtractFeatures(realImages);
            var generatedFeatures = ExtractFeatures(generatedImages);

            var (realMean, realCov) = ComputeStatistics(realFeatures);
            var (genMean, genCov) = ComputeStatistics(generatedFeatures);

            return ComputeFrechetDistance(realMean, realCov, genMean, genCov);
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
        /// Computes mean and covariance matrix of feature vectors using vectorized operations.
        /// </summary>
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

            var mean = new Vector<T>(dim);
            var nInv = _numOps.FromDouble(1.0 / numSamples);

            for (int j = 0; j < dim; j++)
            {
                var column = features.GetColumn(j);
                var sum = Engine.Sum(column);
                mean[j] = _numOps.Multiply(sum, nInv);
            }

            var centered = new Matrix<T>(numSamples, dim);
            for (int i = 0; i < numSamples; i++)
            {
                var row = features.GetRow(i);
                var centeredRow = (Vector<T>)Engine.Subtract(row, mean);
                centered.SetRow(i, centeredRow);
            }

            var nMinusOneInv = _numOps.FromDouble(1.0 / (numSamples - 1));
            var centeredT = centered.Transpose();
            var covariance = (Matrix<T>)Engine.MatrixMultiply(centeredT, centered);

            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    covariance[i, j] = _numOps.Multiply(covariance[i, j], nMinusOneInv);
                }
            }

            return (mean, covariance);
        }

        /// <summary>
        /// Computes the Fréchet distance between two Gaussian distributions.
        /// </summary>
        private double ComputeFrechetDistance(
            Vector<T> mean1,
            Matrix<T> cov1,
            Vector<T> mean2,
            Matrix<T> cov2)
        {
            var meanDiff = (Vector<T>)Engine.Subtract(mean1, mean2);
            var meanDiffSq = Engine.DotProduct(meanDiff, meanDiff);

            var trace1 = ComputeTrace(cov1);
            var trace2 = ComputeTrace(cov2);
            var traceCov = _numOps.Add(trace1, trace2);

            var traceSqrtCovProduct = ComputeTraceSqrtCovProduct(cov1, cov2);

            var fid = _numOps.Add(meanDiffSq, traceCov);
            fid = _numOps.Subtract(fid, _numOps.Multiply(_numOps.FromDouble(2.0), traceSqrtCovProduct));

            return _numOps.ToDouble(fid);
        }

        /// <summary>
        /// Computes the trace of a matrix (sum of diagonal elements).
        /// </summary>
        private T ComputeTrace(Matrix<T> matrix)
        {
            var trace = _numOps.Zero;
            var n = Math.Min(matrix.Rows, matrix.Columns);
            for (int i = 0; i < n; i++)
            {
                trace = _numOps.Add(trace, matrix[i, i]);
            }
            return trace;
        }

        /// <summary>
        /// Computes Tr(sqrt(cov1 * cov2)) using Newton-Schulz iteration for matrix square root.
        /// </summary>
        private T ComputeTraceSqrtCovProduct(Matrix<T> cov1, Matrix<T> cov2)
        {
            int n = cov1.Rows;

            var product = (Matrix<T>)Engine.MatrixMultiply(cov1, cov2);

            var symProduct = new Matrix<T>(n, n);
            var half = _numOps.FromDouble(0.5);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    symProduct[i, j] = _numOps.Multiply(
                        _numOps.Add(product[i, j], product[j, i]),
                        half);
                }
            }

            // Compute Frobenius norm first to determine regularization scale
            var preFrobNormSq = _numOps.Zero;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    preFrobNormSq = _numOps.Add(preFrobNormSq,
                        _numOps.Multiply(symProduct[i, j], symProduct[i, j]));
                }
            }
            var preFrobNorm = _numOps.Sqrt(preFrobNormSq);

            // Tikhonov regularization for numerical stability
            // Add small value to diagonal to improve conditioning
            var eps = _numOps.FromDouble(Math.Max(1e-12, 1e-6 * _numOps.ToDouble(preFrobNorm)));
            for (int i = 0; i < n; i++)
            {
                symProduct[i, i] = _numOps.Add(symProduct[i, i], eps);
            }

            var frobNormSq = _numOps.Zero;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    frobNormSq = _numOps.Add(frobNormSq,
                        _numOps.Multiply(symProduct[i, j], symProduct[i, j]));
                }
            }
            var frobNorm = _numOps.Sqrt(frobNormSq);

            if (_numOps.LessThan(frobNorm, _numOps.FromDouble(1e-10)))
            {
                return _numOps.Zero;
            }

            var scale = _numOps.Sqrt(frobNorm);
            var scaleInv = _numOps.Divide(_numOps.One, scale);

            var A = new Matrix<T>(n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    A[i, j] = _numOps.Multiply(symProduct[i, j], scaleInv);
                }
            }

            var Y = Matrix<T>.CreateIdentity(n);
            var identity = Matrix<T>.CreateIdentity(n);
            var three = _numOps.FromDouble(3.0);

            const int maxIterations = 15;
            const double convergenceTolerance = 1e-10;
            const double divergenceThreshold = 1e10;

            for (int iter = 0; iter < maxIterations; iter++)
            {
                var YY = (Matrix<T>)Engine.MatrixMultiply(Y, Y);
                var AYY = (Matrix<T>)Engine.MatrixMultiply(A, YY);

                var threeIMinusAYY = new Matrix<T>(n, n);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        threeIMinusAYY[i, j] = _numOps.Subtract(
                            _numOps.Multiply(three, identity[i, j]),
                            AYY[i, j]);
                    }
                }

                var newY = (Matrix<T>)Engine.MatrixMultiply(Y, threeIMinusAYY);

                // Compute update magnitude for convergence check
                var updateNormSq = _numOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        var newVal = _numOps.Multiply(half, newY[i, j]);
                        var diff = _numOps.Subtract(newVal, Y[i, j]);
                        updateNormSq = _numOps.Add(updateNormSq, _numOps.Multiply(diff, diff));
                        Y[i, j] = newVal;
                    }
                }

                var updateNorm = _numOps.ToDouble(_numOps.Sqrt(updateNormSq));

                // Check for divergence (values growing too large)
                if (updateNorm > divergenceThreshold)
                {
                    // Newton-Schulz failed; use eigenvalue-based fallback
                    // For trace(sqrt(A)), compute sum of sqrt of eigenvalues
                    return ComputeTraceSqrtViaEigenvalues(symProduct, n);
                }

                // Check for convergence (change is negligible)
                if (updateNorm < convergenceTolerance)
                {
                    break;
                }
            }

            var AY = (Matrix<T>)Engine.MatrixMultiply(A, Y);

            var sqrtScale = _numOps.Sqrt(scale);
            var traceAY = ComputeTrace(AY);

            return _numOps.Multiply(sqrtScale, traceAY);
        }

        /// <summary>
        /// Computes FID using pre-computed statistics for the real distribution.
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
        /// Pre-computes statistics for a set of images.
        /// </summary>
        /// <param name="images">Images to compute statistics for</param>
        /// <returns>Tuple of (mean, covariance)</returns>
        public (Vector<T> mean, Matrix<T> covariance) PrecomputeStatistics(Tensor<T> images)
        {
            var features = ExtractFeatures(images);
            return ComputeStatistics(features);
        }

        /// <summary>
        /// Computes trace(sqrt(A)) using eigenvalue decomposition via Jacobi iteration.
        /// Used as fallback when Newton-Schulz iteration diverges.
        /// </summary>
        private T ComputeTraceSqrtViaEigenvalues(Matrix<T> A, int n)
        {
            // Create a working copy for Jacobi iteration
            var matrix = new Matrix<T>(n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    matrix[i, j] = A[i, j];
                }
            }

            const int maxJacobiIterations = 50;
            const double jacobiTolerance = 1e-10;

            // Jacobi eigenvalue algorithm for symmetric matrices
            for (int iter = 0; iter < maxJacobiIterations; iter++)
            {
                // Find largest off-diagonal element
                int p = 0, q = 1;
                var maxOffDiag = _numOps.Zero;
                bool foundOffDiag = false;

                for (int i = 0; i < n; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        var absVal = _numOps.GreaterThanOrEquals(matrix[i, j], _numOps.Zero)
                            ? matrix[i, j]
                            : _numOps.Negate(matrix[i, j]);

                        if (!foundOffDiag || _numOps.GreaterThan(absVal, maxOffDiag))
                        {
                            maxOffDiag = absVal;
                            p = i;
                            q = j;
                            foundOffDiag = true;
                        }
                    }
                }

                // Check convergence (off-diagonal elements small enough)
                if (_numOps.ToDouble(maxOffDiag) < jacobiTolerance)
                {
                    break;
                }

                // Compute Jacobi rotation angle
                var diff = _numOps.Subtract(matrix[q, q], matrix[p, p]);
                T theta;
                if (Math.Abs(_numOps.ToDouble(diff)) < 1e-15)
                {
                    theta = _numOps.FromDouble(Math.PI / 4.0);
                }
                else
                {
                    var tau = _numOps.Divide(
                        _numOps.Multiply(_numOps.FromDouble(2.0), matrix[p, q]),
                        diff);
                    var tauDouble = _numOps.ToDouble(tau);
                    var t = Math.Sign(tauDouble) / (Math.Abs(tauDouble) + Math.Sqrt(1.0 + tauDouble * tauDouble));
                    theta = _numOps.FromDouble(Math.Atan(t));
                }

                var cosTheta = _numOps.FromDouble(Math.Cos(_numOps.ToDouble(theta)));
                var sinTheta = _numOps.FromDouble(Math.Sin(_numOps.ToDouble(theta)));

                // Apply Jacobi rotation: A' = J^T * A * J
                for (int i = 0; i < n; i++)
                {
                    if (i != p && i != q)
                    {
                        var aip = matrix[i, p];
                        var aiq = matrix[i, q];

                        matrix[i, p] = _numOps.Add(
                            _numOps.Multiply(cosTheta, aip),
                            _numOps.Multiply(sinTheta, aiq));
                        matrix[p, i] = matrix[i, p];

                        matrix[i, q] = _numOps.Subtract(
                            _numOps.Multiply(cosTheta, aiq),
                            _numOps.Multiply(sinTheta, aip));
                        matrix[q, i] = matrix[i, q];
                    }
                }

                var app = matrix[p, p];
                var aqq = matrix[q, q];
                var apq = matrix[p, q];

                matrix[p, p] = _numOps.Add(
                    _numOps.Add(
                        _numOps.Multiply(_numOps.Multiply(cosTheta, cosTheta), app),
                        _numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(cosTheta, sinTheta)), apq)),
                    _numOps.Multiply(_numOps.Multiply(sinTheta, sinTheta), aqq));

                matrix[q, q] = _numOps.Add(
                    _numOps.Subtract(
                        _numOps.Multiply(_numOps.Multiply(sinTheta, sinTheta), app),
                        _numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(cosTheta, sinTheta)), apq)),
                    _numOps.Multiply(_numOps.Multiply(cosTheta, cosTheta), aqq));

                matrix[p, q] = _numOps.Zero;
                matrix[q, p] = _numOps.Zero;
            }

            // Sum sqrt of eigenvalues (diagonal elements after Jacobi)
            // Only include positive eigenvalues (covariance matrices should be PSD)
            var traceSqrt = _numOps.Zero;
            for (int i = 0; i < n; i++)
            {
                var eigenvalue = matrix[i, i];
                if (_numOps.GreaterThan(eigenvalue, _numOps.Zero))
                {
                    traceSqrt = _numOps.Add(traceSqrt, _numOps.Sqrt(eigenvalue));
                }
            }

            return traceSqrt;
        }
    }
}
