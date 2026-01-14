using AiDotNet.Helpers;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Combined loss function for Real-ESRGAN super-resolution training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Real-ESRGAN uses a combination of three loss functions for training:
/// - L1 (pixel-wise) loss: Ensures pixel-level accuracy
/// - Perceptual (VGG) loss: Ensures perceptual quality using deep features
/// - GAN (adversarial) loss: Ensures realistic details and textures
/// </para>
/// <para>
/// The total loss is computed as:
/// <code>
/// L_total = λ_L1 * L_L1 + λ_perceptual * L_perceptual + λ_GAN * L_GAN
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> This loss function guides Real-ESRGAN training by balancing three goals:
///
/// 1. **L1 Loss (pixel accuracy)**: Makes sure each pixel is close to the ground truth.
///    Like comparing photos pixel-by-pixel.
///
/// 2. **Perceptual Loss (looks right)**: Uses a pre-trained network (VGG) to compare
///    high-level features. Ensures the output "looks right" even if pixels aren't exact.
///
/// 3. **GAN Loss (realistic details)**: The discriminator judges if output looks real.
///    This adds fine details and textures that make images look natural.
///
/// The weights control how much each goal matters:
/// - Higher L1 weight = more pixel-accurate but potentially blurry
/// - Higher perceptual weight = better visual quality
/// - Higher GAN weight = more realistic textures but potential artifacts
///
/// The default weights (1.0, 1.0, 0.1) are from the Real-ESRGAN paper.
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
/// with Pure Synthetic Data", ICCV 2021. https://arxiv.org/abs/2107.10833
/// </para>
/// </remarks>
public class RealESRGANLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The L1 (pixel-wise) loss function.
    /// </summary>
    private readonly MeanAbsoluteErrorLoss<T> _l1Loss;

    /// <summary>
    /// Weight for the L1 loss component.
    /// </summary>
    private readonly double _l1Weight;

    /// <summary>
    /// Weight for the perceptual loss component.
    /// </summary>
    private readonly double _perceptualWeight;

    /// <summary>
    /// Weight for the GAN loss component.
    /// </summary>
    private readonly double _ganWeight;

    /// <summary>
    /// The feature extractor for perceptual loss (VGG features).
    /// </summary>
    private readonly Func<Tensor<T>, Tensor<T>>? _featureExtractor;

    /// <summary>
    /// Initializes a new instance of the RealESRGANLoss class.
    /// </summary>
    /// <param name="l1Weight">Weight for L1 loss. Default: 1.0 (from Real-ESRGAN paper).</param>
    /// <param name="perceptualWeight">Weight for perceptual loss. Default: 1.0 (from Real-ESRGAN paper).</param>
    /// <param name="ganWeight">Weight for GAN loss. Default: 0.1 (from Real-ESRGAN paper).</param>
    /// <param name="featureExtractor">Optional VGG feature extractor for perceptual loss.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create this loss with default weights from the paper:
    /// <code>
    /// var loss = new RealESRGANLoss&lt;double&gt;();
    /// </code>
    ///
    /// Or customize weights for different trade-offs:
    /// <code>
    /// // More pixel-accurate (potentially blurrier)
    /// var loss = new RealESRGANLoss&lt;double&gt;(l1Weight: 2.0, ganWeight: 0.05);
    ///
    /// // More realistic textures (potential artifacts)
    /// var loss = new RealESRGANLoss&lt;double&gt;(ganWeight: 0.2);
    /// </code>
    /// </para>
    /// </remarks>
    public RealESRGANLoss(
        double l1Weight = 1.0,
        double perceptualWeight = 1.0,
        double ganWeight = 0.1,
        Func<Tensor<T>, Tensor<T>>? featureExtractor = null)
    {
        if (l1Weight < 0)
            throw new ArgumentOutOfRangeException(nameof(l1Weight), l1Weight, "L1 weight must be non-negative.");
        if (perceptualWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(perceptualWeight), perceptualWeight, "Perceptual weight must be non-negative.");
        if (ganWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(ganWeight), ganWeight, "GAN weight must be non-negative.");

        _l1Weight = l1Weight;
        _perceptualWeight = perceptualWeight;
        _ganWeight = ganWeight;
        _featureExtractor = featureExtractor;
        _l1Loss = new MeanAbsoluteErrorLoss<T>();
    }

    /// <summary>
    /// Gets the L1 weight.
    /// </summary>
    public double L1Weight => _l1Weight;

    /// <summary>
    /// Gets the perceptual loss weight.
    /// </summary>
    public double PerceptualWeight => _perceptualWeight;

    /// <summary>
    /// Gets the GAN loss weight.
    /// </summary>
    public double GANWeight => _ganWeight;

    /// <summary>
    /// Calculates the combined Real-ESRGAN loss.
    /// </summary>
    /// <param name="predicted">The predicted (super-resolved) output.</param>
    /// <param name="actual">The ground truth high-resolution target.</param>
    /// <returns>The combined loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the L1 and perceptual components of the loss.
    /// The GAN loss component should be computed separately during training
    /// using the discriminator output.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This calculates how "wrong" the prediction is.
    /// Lower values mean the prediction is closer to the ground truth.
    /// </para>
    /// </remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Calculate L1 loss
        T l1Loss = _l1Loss.CalculateLoss(predicted, actual);
        T weightedL1 = NumOps.Multiply(l1Loss, NumOps.FromDouble(_l1Weight));

        // Calculate perceptual loss if feature extractor is available
        T perceptualLoss = NumOps.Zero;
        if (_featureExtractor != null && _perceptualWeight > 0)
        {
            // Convert vectors to tensors for feature extraction
            var predictedTensor = VectorToTensor(predicted);
            var actualTensor = VectorToTensor(actual);

            var predictedFeatures = _featureExtractor(predictedTensor);
            var actualFeatures = _featureExtractor(actualTensor);

            // Calculate L2 distance in feature space
            perceptualLoss = CalculateFeatureLoss(predictedFeatures, actualFeatures);
        }
        T weightedPerceptual = NumOps.Multiply(perceptualLoss, NumOps.FromDouble(_perceptualWeight));

        // Note: GAN loss is computed separately during training
        // Total = L1 + Perceptual (GAN added in training loop)
        return NumOps.Add(weightedL1, weightedPerceptual);
    }

    /// <summary>
    /// Calculates the full combined loss including GAN component.
    /// </summary>
    /// <param name="predicted">The predicted (super-resolved) output.</param>
    /// <param name="actual">The ground truth high-resolution target.</param>
    /// <param name="discriminatorOutput">The discriminator's output for the predicted image.</param>
    /// <returns>The combined loss value including GAN loss.</returns>
    /// <remarks>
    /// <para>
    /// Use this method during training when you have access to the discriminator output.
    /// The GAN loss encourages the generator to produce outputs that fool the discriminator.
    /// </para>
    /// </remarks>
    public T CalculateCombinedLoss(Vector<T> predicted, Vector<T> actual, Vector<T> discriminatorOutput)
    {
        // Get reconstruction losses (L1 + Perceptual)
        T reconstructionLoss = CalculateLoss(predicted, actual);

        // Calculate GAN loss (generator wants discriminator to output 1 = "real")
        // Binary cross-entropy: -log(D(G(x)))
        T ganLoss = CalculateGeneratorGANLoss(discriminatorOutput);
        T weightedGAN = NumOps.Multiply(ganLoss, NumOps.FromDouble(_ganWeight));

        return NumOps.Add(reconstructionLoss, weightedGAN);
    }

    /// <summary>
    /// Calculates the generator's GAN loss component.
    /// </summary>
    /// <param name="discriminatorOutput">The discriminator's output for generated images.</param>
    /// <returns>The GAN loss value.</returns>
    /// <remarks>
    /// <para>
    /// The generator wants the discriminator to classify its output as real (1.0).
    /// This computes: -E[log(D(G(x)))]
    /// </para>
    /// </remarks>
    public T CalculateGeneratorGANLoss(Vector<T> discriminatorOutput)
    {
        T loss = NumOps.Zero;
        double epsilon = 1e-7;

        for (int i = 0; i < discriminatorOutput.Length; i++)
        {
            double d = Math.Max(NumOps.ToDouble(discriminatorOutput[i]), epsilon);
            loss = NumOps.Add(loss, NumOps.FromDouble(-Math.Log(d)));
        }

        return NumOps.Divide(loss, NumOps.FromDouble(discriminatorOutput.Length));
    }

    /// <summary>
    /// Calculates the discriminator loss.
    /// </summary>
    /// <param name="realOutput">Discriminator output for real images.</param>
    /// <param name="fakeOutput">Discriminator output for generated images.</param>
    /// <returns>The discriminator loss value.</returns>
    /// <remarks>
    /// <para>
    /// The discriminator wants to output 1 for real images and 0 for fake images.
    /// This computes: -E[log(D(x))] - E[log(1 - D(G(x)))]
    /// </para>
    /// </remarks>
    public T CalculateDiscriminatorLoss(Vector<T> realOutput, Vector<T> fakeOutput)
    {
        T loss = NumOps.Zero;
        double epsilon = 1e-7;

        // Real images: want D(x) = 1, so loss = -log(D(x))
        for (int i = 0; i < realOutput.Length; i++)
        {
            double d = Math.Max(NumOps.ToDouble(realOutput[i]), epsilon);
            loss = NumOps.Add(loss, NumOps.FromDouble(-Math.Log(d)));
        }

        // Fake images: want D(G(x)) = 0, so loss = -log(1 - D(G(x)))
        for (int i = 0; i < fakeOutput.Length; i++)
        {
            double d = NumOps.ToDouble(fakeOutput[i]);
            double oneMinusD = Math.Max(1.0 - d, epsilon);
            loss = NumOps.Add(loss, NumOps.FromDouble(-Math.Log(oneMinusD)));
        }

        // Average over both batches
        int totalSamples = realOutput.Length + fakeOutput.Length;
        return NumOps.Divide(loss, NumOps.FromDouble(totalSamples));
    }

    /// <summary>
    /// Calculates the derivative of the combined loss for backpropagation.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The gradient vector.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // L1 loss derivative: sign(predicted - actual) / n
        Vector<T> l1Gradient = _l1Loss.CalculateDerivative(predicted, actual);

        // Scale by L1 weight
        l1Gradient = l1Gradient.Multiply(NumOps.FromDouble(_l1Weight));

        // Note: Perceptual and GAN gradients are computed separately in the training loop
        // as they require the feature extractor network and discriminator
        return l1Gradient;
    }

    /// <summary>
    /// Calculates the L2 distance between feature tensors.
    /// </summary>
    private T CalculateFeatureLoss(Tensor<T> predictedFeatures, Tensor<T> actualFeatures)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < predictedFeatures.Length; i++)
        {
            T diff = NumOps.Subtract(predictedFeatures.Data.Span[i], actualFeatures.Data.Span[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predictedFeatures.Length));
    }

    /// <summary>
    /// Converts a vector to a tensor for feature extraction.
    /// </summary>
    private static Tensor<T> VectorToTensor(Vector<T> vector)
    {
        // Assume vector is flattened image data
        return new Tensor<T>([vector.Length], vector);
    }
}
