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
        public T ComputeFID(Tensor<T> realImages, Tensor<T> generatedImages)
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

                    Engine.SetRow(features, i, FrechetDistanceHelper<T>.FeatureRow(output, FeatureDimension));
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
            return FrechetDistanceHelper<T>.ComputeStatistics(features);
        }

        /// <summary>
        /// Computes the Fréchet distance between two Gaussian distributions.
        /// </summary>
        private T ComputeFrechetDistance(
            Vector<T> mean1,
            Matrix<T> cov1,
            Vector<T> mean2,
            Matrix<T> cov2)
        {
            return FrechetDistanceHelper<T>.FrechetDistance(mean1, cov1, mean2, cov2);
        }

        /// <summary>
        /// Computes FID using pre-computed statistics for the real distribution.
        /// </summary>
        /// <param name="realMean">Mean of real image features</param>
        /// <param name="realCov">Covariance of real image features</param>
        /// <param name="generatedImages">Generated images to evaluate</param>
        /// <returns>FID score</returns>
        public T ComputeFIDWithStats(
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
    }
}
