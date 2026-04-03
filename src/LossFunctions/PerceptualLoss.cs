using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Perceptual Loss function for comparing high-level features of images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Perceptual Loss is a type of loss function used primarily in image processing
/// and generative models. Unlike pixel-wise losses (like MSE) that compare images pixel by pixel,
/// perceptual loss compares high-level features extracted from the images.
///
/// The key idea is to:
/// 1. Pass both the generated image and target image through a pre-trained network (like VGG)
/// 2. Extract features from various layers of this network
/// 3. Compare these features rather than raw pixels
///
/// This approach is more aligned with human perception because:
/// - It focuses on semantic content rather than exact pixel values
/// - It captures textures, patterns, and structures that are perceptually important
/// - It allows for some flexibility in pixel-level details while preserving overall appearance
///
/// Perceptual Loss is commonly used in:
/// - Style transfer algorithms
/// - Super-resolution models
/// - Image-to-image translation
/// - Any task where the "look" of an image is more important than exact pixel reproduction
/// </para>
/// </remarks>
[LossCategory(LossCategory.Generation)]
[LossCategory(LossCategory.Reconstruction)]
[LossTask(LossTask.ImageGeneration)]
[LossTask(LossTask.SuperResolution)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = true, ApiShape = LossApiShape.ImageMatrix, ExpectedOutput = OutputType.Continuous)]
public class PerceptualLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The feature extractor function that converts images to feature representations.
    /// </summary>
    private readonly Func<Matrix<T>, Vector<Vector<T>>> _featureExtractor;

    /// <summary>
    /// The weights for each feature layer.
    /// </summary>
    private readonly Vector<T> _layerWeights;

    /// <summary>
    /// Initializes a new instance of the PerceptualLoss class.
    /// </summary>
    /// <param name="featureExtractor">Function that extracts features from images.</param>
    /// <param name="layerWeights">Weights for each feature layer.</param>
    public PerceptualLoss(Func<Matrix<T>, Vector<Vector<T>>> featureExtractor, Vector<T> layerWeights)
    {
        Guard.NotNull(featureExtractor);
        _featureExtractor = featureExtractor;
        _layerWeights = layerWeights;
    }

    /// <summary>
    /// Calculates the Perceptual Loss between generated and target images.
    /// </summary>
    /// <param name="generated">The generated image as a matrix.</param>
    /// <param name="target">The target image as a matrix.</param>
    /// <returns>The perceptual loss value.</returns>
    public T Calculate(Matrix<T> generated, Matrix<T> target)
    {
        // Extract features from both images
        Vector<Vector<T>> generatedFeatures = _featureExtractor(generated);
        Vector<Vector<T>> targetFeatures = _featureExtractor(target);

        // Ensure we have the same number of feature layers
        if (generatedFeatures.Length != targetFeatures.Length)
        {
            throw new ArgumentException("Generated and target feature counts do not match.");
        }

        // If layer weights are not provided or have incorrect length, use uniform weights
        Vector<T> weights = _layerWeights;
        if (weights == null || weights.Length != generatedFeatures.Length)
        {
            weights = new Vector<T>(generatedFeatures.Length);
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = NumOps.One;
            }
        }

        // Calculate weighted MSE loss for each feature layer
        T totalLoss = NumOps.Zero;
        for (int layer = 0; layer < generatedFeatures.Length; layer++)
        {
            Vector<T> genFeatures = generatedFeatures[layer];
            Vector<T> tgtFeatures = targetFeatures[layer];

            // Calculate MSE for this feature layer
            T layerLoss = MeanSquaredError(genFeatures, tgtFeatures);

            // Apply weight to this layer's loss
            totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(weights[layer], layerLoss));
        }

        return totalLoss;
    }

    /// <summary>
    /// Calculates the Mean Squared Error between two vectors.
    /// </summary>
    private T MeanSquaredError(Vector<T> v1, Vector<T> v2)
    {
        if (v1.Length != v2.Length)
        {
            throw new ArgumentException("Vectors must have the same length.");
        }

        T sum = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            T diff = NumOps.Subtract(v1[i], v2[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(v1.Length));
    }

    /// <summary>
    /// This method is not used for Perceptual Loss as it requires image matrices.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (target) values vector.</param>
    /// <returns>Throws NotSupportedException.</returns>
    /// <exception cref="NotSupportedException">Always thrown as Perceptual Loss requires image matrices.</exception>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "Perceptual Loss requires image matrices. " +
            "Use the Calculate(Matrix<T>, Matrix<T>) method instead."
        );
    }

    /// <summary>
    /// This method is not used for Perceptual Loss as it requires image matrices.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (target) values vector.</param>
    /// <returns>Throws NotSupportedException.</returns>
    /// <exception cref="NotSupportedException">Always thrown as Perceptual Loss requires image matrices.</exception>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "Perceptual Loss requires image matrices and is typically calculated using automatic differentiation."
        );
    }

    /// <summary>
    /// Optional neural network feature extractor for tape-based training.
    /// Set via <see cref="SetFeatureExtractorNetwork"/> before calling <see cref="ComputeTapeLoss"/>.
    /// </summary>
    private INeuralNetwork<T>? _featureExtractorNetwork;

    /// <summary>
    /// Sets the neural network used for tape-differentiable feature extraction.
    /// The network should be pre-trained and frozen (its weights won't be updated).
    /// Its output is treated as the feature representation for perceptual comparison.
    /// </summary>
    public void SetFeatureExtractorNetwork(INeuralNetwork<T> network)
    {
        _featureExtractorNetwork = network ?? throw new ArgumentNullException(nameof(network));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Computes perceptual loss by passing both predicted and target through the feature
    /// extractor network, then computing MSE between the feature representations.
    /// The feature extractor's output is detached via StopGradient on the target features
    /// so only the predicted path contributes gradients.
    /// Requires <see cref="SetFeatureExtractorNetwork"/> to be called first.
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (_featureExtractorNetwork is null)
            throw new InvalidOperationException(
                "Feature extractor network not set. Call SetFeatureExtractorNetwork() before training.");

        // Extract features from predicted (tape-tracked) and target (detached)
        var predFeatures = _featureExtractorNetwork.Predict(predicted);
        var targetFeatures = Engine.StopGradient(_featureExtractorNetwork.Predict(target));

        // MSE between feature representations
        var diff = Engine.TensorSubtract(predFeatures, targetFeatures);
        var sq = Engine.TensorMultiply(diff, diff);
        var allAxes = Enumerable.Range(0, sq.Shape.Length).ToArray();
        return Engine.ReduceMean(sq, allAxes, keepDims: false);
    }
}
