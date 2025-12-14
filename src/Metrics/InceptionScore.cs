using System;
using System.Linq;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Metrics
{
    /// <summary>
    /// Inception Score (IS) - A metric for evaluating the quality and diversity of generated images.
    ///
    /// For Beginners:
    /// Inception Score measures two important properties of generated images:
    /// 1. Quality: Each image should clearly belong to one class (low entropy)
    /// 2. Diversity: The collection should cover many different classes (high entropy)
    ///
    /// Think of it like evaluating an art student's portfolio:
    /// - Quality: Each drawing should be clearly recognizable (is it a cat? a dog? a car?)
    /// - Diversity: The portfolio should have many different subjects, not just cats
    ///
    /// How it works:
    /// 1. Pass generated images through a pre-trained Inception classifier
    /// 2. Get probability distributions over classes (e.g., "70% cat, 20% dog, 10% other")
    /// 3. For quality: Check if predictions are confident (peaked distributions)
    /// 4. For diversity: Check if many different classes are represented
    /// 5. Combine using KL divergence: IS = exp(E[KL(p(y|x) || p(y))])
    ///
    /// The math behind it:
    /// - p(y|x): Probability of class y given image x (conditional distribution)
    /// - p(y): Overall probability of class y across all images (marginal distribution)
    /// - KL divergence measures how different these two distributions are
    /// - Higher KL = more confident predictions + diverse outputs = Better IS
    ///
    /// Typical Inception Scores:
    /// - IS > 10: Excellent (ImageNet-quality)
    /// - IS 5-10: Good quality
    /// - IS 2-5: Moderate quality
    /// - IS < 2: Poor quality (random noise ≈ 1)
    ///
    /// Important notes:
    /// - IS can be "fooled" by mode collapse (generating same high-quality image repeatedly)
    /// - FID is generally considered more reliable, but IS is still widely used
    /// - Scores only meaningful within same dataset (can't compare IS on different datasets)
    ///
    /// Based on "Improved Techniques for Training GANs" by Salimans et al. (2016)
    /// </summary>
    /// <typeparam name="T">The numeric type for computations (e.g., double, float)</typeparam>
    public class InceptionScore<T> where T : struct, IComparable, IFormattable, IConvertible, IComparable<T>, IEquatable<T>
    {
        private readonly INumericOperations<T> NumOps;

        /// <summary>
        /// Gets the Inception network used for classification.
        /// Should be pre-trained on ImageNet (1000 classes).
        /// </summary>
        public ConvolutionalNeuralNetwork<T>? InceptionNetwork { get; private set; }

        /// <summary>
        /// Gets the number of classes in the classifier.
        /// For ImageNet InceptionV3, this is 1000.
        /// </summary>
        public int NumClasses { get; private set; }

        /// <summary>
        /// Gets or sets the number of splits for computing IS with uncertainty.
        /// The dataset is split into this many parts and IS is computed for each,
        /// allowing calculation of mean and standard deviation.
        /// </summary>
        public int NumSplits { get; set; }

        /// <summary>
        /// Initializes a new instance of Inception Score calculator.
        /// </summary>
        /// <param name="inceptionNetwork">Pre-trained Inception network for classification (optional)</param>
        /// <param name="numClasses">Number of classes in the classifier (default 1000 for ImageNet)</param>
        /// <param name="numSplits">Number of splits for computing mean/std (default 10)</param>
        public InceptionScore(
            ConvolutionalNeuralNetwork<T>? inceptionNetwork = null,
            int numClasses = 1000,
            int numSplits = 10)
        {
            NumOps = MathHelper.GetNumericOperations<T>();
            InceptionNetwork = inceptionNetwork;
            NumClasses = numClasses;
            NumSplits = numSplits;
        }

        /// <summary>
        /// Computes the Inception Score for a set of generated images.
        /// </summary>
        /// <param name="generatedImages">Tensor of generated images</param>
        /// <returns>Inception Score (higher is better, typical range 1-15+)</returns>
        public double ComputeIS(Tensor<T> generatedImages)
        {
            var (mean, _) = ComputeISWithUncertainty(generatedImages);
            return mean;
        }

        /// <summary>
        /// Computes Inception Score with mean and standard deviation across splits.
        /// This provides uncertainty estimation for the IS metric.
        /// </summary>
        /// <param name="generatedImages">Tensor of generated images</param>
        /// <returns>Tuple of (mean IS, standard deviation)</returns>
        public (double mean, double std) ComputeISWithUncertainty(Tensor<T> generatedImages)
        {
            var numImages = generatedImages.Shape[0];
            var splitSize = numImages / NumSplits;

            if (splitSize == 0)
            {
                throw new ArgumentException($"Not enough images for {NumSplits} splits. Need at least {NumSplits} images.");
            }

            var scores = new double[NumSplits];

            // Compute IS for each split
            for (int split = 0; split < NumSplits; split++)
            {
                var startIdx = split * splitSize;
                var endIdx = (split == NumSplits - 1) ? numImages : (split + 1) * splitSize;
                var currentSplitSize = endIdx - startIdx;

                // Extract split images
                var splitImages = ExtractImageSubset(generatedImages, startIdx, currentSplitSize);

                // Compute IS for this split
                scores[split] = ComputeISForSplit(splitImages);
            }

            // Compute mean and standard deviation
            var mean = scores.Average();
            var variance = scores.Select(s => Math.Pow(s - mean, 2)).Average();
            var std = Math.Sqrt(variance);

            return (mean, std);
        }

        /// <summary>
        /// Computes Inception Score for a single split of images.
        /// </summary>
        private double ComputeISForSplit(Tensor<T> images)
        {
            // Get predicted class probabilities for all images
            var predictions = GetPredictions(images);

            // Compute marginal distribution p(y) = mean of p(y|x) over all x
            var marginal = ComputeMarginalDistribution(predictions);

            // Compute KL divergence for each image and take mean
            var klDivergences = new double[predictions.Rows];
            for (int i = 0; i < predictions.Rows; i++)
            {
                klDivergences[i] = ComputeKLDivergence(predictions, i, marginal);
            }

            var meanKL = klDivergences.Average();

            // IS = exp(E[KL(p(y|x) || p(y))])
            var inceptionScore = Math.Exp(meanKL);

            return inceptionScore;
        }

        /// <summary>
        /// Gets class probability predictions for all images using Inception network.
        /// </summary>
        /// <param name="images">Images to classify</param>
        /// <returns>Matrix of predictions (num_images × num_classes)</returns>
        private Matrix<T> GetPredictions(Tensor<T> images)
        {
            var numImages = images.Shape[0];
            var predictions = new Matrix<T>(numImages, NumClasses);

            if (InceptionNetwork == null)
            {
                // No network provided, return dummy predictions
                return CreateDummyPredictions(numImages);
            }

            InceptionNetwork.SetTrainingMode(false);

            for (int i = 0; i < numImages; i++)
            {
                // Extract single image
                var imageSize = images.Length / numImages;
                var singleImage = new Tensor<T>(new[] { 1, images.Shape[1], images.Shape[2], images.Shape[3] });
                for (int idx = 0; idx < imageSize; idx++)
                {
                    singleImage.SetFlat(idx, images.GetFlat(i * imageSize + idx));
                }

                // Forward pass
                var output = InceptionNetwork.Predict(singleImage);

                // Apply softmax to get probabilities
                var probs = Softmax(output);

                // Store predictions
                for (int j = 0; j < Math.Min(probs.Length, NumClasses); j++)
                {
                    predictions[i, j] = probs.GetFlat(j);
                }
            }

            return predictions;
        }

        /// <summary>
        /// Applies softmax activation to convert logits to probabilities.
        /// </summary>
        private Tensor<T> Softmax(Tensor<T> logits)
        {
            var result = new Tensor<T>(logits.Shape);
            var maxLogit = NumOps.Zero;

            // Find max for numerical stability
            for (int i = 0; i < logits.Length; i++)
            {
                if (NumOps.GreaterThan(logits.GetFlat(i), maxLogit))
                {
                    maxLogit = logits.GetFlat(i);
                }
            }

            // Compute exp(x - max) and sum
            var sum = NumOps.Zero;
            for (int i = 0; i < logits.Length; i++)
            {
                var shifted = NumOps.Subtract(logits.GetFlat(i), maxLogit);
                var expVal = NumOps.Exp(shifted);
                result.SetFlat(i, expVal);
                sum = NumOps.Add(sum, expVal);
            }

            // Normalize
            for (int i = 0; i < result.Length; i++)
            {
                result.SetFlat(i, NumOps.Divide(result.GetFlat(i), sum));
            }

            return result;
        }

        /// <summary>
        /// Creates dummy predictions for testing when no Inception network is available.
        /// </summary>
        private Matrix<T> CreateDummyPredictions(int numSamples)
        {
            var random = RandomHelper.ThreadSafeRandom;
            var predictions = new Matrix<T>(numSamples, NumClasses);

            for (int i = 0; i < numSamples; i++)
            {
                // Create random probabilities that sum to 1
                var sum = 0.0;
                var values = new double[NumClasses];

                for (int j = 0; j < NumClasses; j++)
                {
                    values[j] = random.NextDouble();
                    sum += values[j];
                }

                // Normalize
                for (int j = 0; j < NumClasses; j++)
                {
                    predictions[i, j] = NumOps.FromDouble(values[j] / sum);
                }
            }

            return predictions;
        }

        /// <summary>
        /// Computes the marginal distribution p(y) by averaging over all images.
        /// p(y) = (1/N) * Σ p(y|x_i)
        /// </summary>
        private Vector<T> ComputeMarginalDistribution(Matrix<T> predictions)
        {
            var numImages = predictions.Rows;
            var marginal = new Vector<T>(NumClasses);

            for (int j = 0; j < NumClasses; j++)
            {
                var sum = NumOps.Zero;
                for (int i = 0; i < numImages; i++)
                {
                    sum = NumOps.Add(sum, predictions[i, j]);
                }
                marginal[j] = NumOps.Divide(sum, NumOps.FromDouble(numImages));
            }

            return marginal;
        }

        /// <summary>
        /// Computes KL divergence between conditional p(y|x) and marginal p(y).
        /// KL(p(y|x) || p(y)) = Σ p(y|x) * log(p(y|x) / p(y))
        /// </summary>
        private double ComputeKLDivergence(Matrix<T> predictions, int imageIdx, Vector<T> marginal)
        {
            var kl = 0.0;
            var epsilon = 1e-10; // Small value to avoid log(0)

            for (int j = 0; j < NumClasses; j++)
            {
                var pYgivenX = Convert.ToDouble(predictions[imageIdx, j]) + epsilon;
                var pY = Convert.ToDouble(marginal[j]) + epsilon;

                // KL divergence: p(y|x) * log(p(y|x) / p(y))
                kl += pYgivenX * Math.Log(pYgivenX / pY);
            }

            return kl;
        }

        /// <summary>
        /// Extracts a subset of images from the tensor.
        /// </summary>
        private Tensor<T> ExtractImageSubset(Tensor<T> images, int startIdx, int count)
        {
            var imageSize = images.Length / images.Shape[0];
            var subsetShape = new int[images.Shape.Length];
            subsetShape[0] = count;
            for (int i = 1; i < images.Shape.Length; i++)
            {
                subsetShape[i] = images.Shape[i];
            }

            var subset = new Tensor<T>(subsetShape);
            for (int idx = 0; idx < count * imageSize; idx++)
            {
                subset.SetFlat(idx, images.GetFlat(startIdx * imageSize + idx));
            }

            return subset;
        }

        /// <summary>
        /// Computes both IS and FID if a FID calculator is provided.
        /// Useful for comprehensive evaluation.
        /// </summary>
        /// <param name="generatedImages">Generated images to evaluate</param>
        /// <param name="realImages">Real images for FID comparison (optional)</param>
        /// <returns>Tuple of (IS mean, IS std, FID score if real images provided)</returns>
        public (double isMean, double isStd, double? fid) ComputeComprehensiveMetrics(
            Tensor<T> generatedImages,
            Tensor<T>? realImages = null)
        {
            var (isMean, isStd) = ComputeISWithUncertainty(generatedImages);

            double? fid = null;
            if (realImages != null && InceptionNetwork != null)
            {
                var fidCalculator = new FrechetInceptionDistance<T>(InceptionNetwork);
                fid = fidCalculator.ComputeFID(realImages, generatedImages);
            }

            return (isMean, isStd, fid);
        }
    }
}
