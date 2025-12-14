namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Wasserstein loss function used in Wasserstein Generative Adversarial Networks (WGAN).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Wasserstein loss (also known as Earth Mover's Distance loss) measures the distance between
/// two probability distributions. In the context of GANs, it provides a meaningful gradient signal
/// even when the discriminator (critic) is well-trained.
/// </para>
/// <para>
/// <b>Mathematical Formula:</b>
/// <list type="bullet">
/// <item><description>Loss = mean(predicted * label)</description></item>
/// <item><description>Where label is +1 for real samples, -1 for fake samples</description></item>
/// <item><description>The critic aims to maximize E[critic(real)] - E[critic(fake)]</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Wasserstein loss is a special way to measure how different two groups of data are.
///
/// Why use Wasserstein loss instead of regular binary cross-entropy?
/// <list type="bullet">
/// <item><description>More stable training - gradients don't vanish when the critic is confident</description></item>
/// <item><description>The loss value correlates with image quality - lower loss means better images</description></item>
/// <item><description>No mode collapse - the generator doesn't get stuck producing the same output</description></item>
/// <item><description>Can train the critic to convergence without breaking training</description></item>
/// </list>
///
/// How it works:
/// <list type="bullet">
/// <item><description>For real images, we want the critic to output high scores (label = +1)</description></item>
/// <item><description>For fake images, we want the critic to output low scores (label = -1)</description></item>
/// <item><description>The loss is simply the average of (score * label)</description></item>
/// <item><description>A well-trained critic gives positive scores to real images and negative scores to fakes</description></item>
/// </list>
///
/// Reference: Arjovsky et al., "Wasserstein GAN" (2017)
/// </para>
/// </remarks>
public class WassersteinLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Wasserstein loss between predicted critic scores and labels.
    /// </summary>
    /// <param name="predicted">The critic's output scores for each sample.</param>
    /// <param name="actual">The labels: +1 for real samples, -1 for fake samples.</param>
    /// <returns>The mean Wasserstein loss across all samples.</returns>
    /// <remarks>
    /// <para>
    /// The loss is computed as the negative mean of (predicted * actual), which means:
    /// <list type="bullet">
    /// <item><description>For real samples (label=+1): we want high predicted scores</description></item>
    /// <item><description>For fake samples (label=-1): we want low predicted scores</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This computes how well the critic is doing at telling real from fake.
    ///
    /// Example:
    /// <list type="bullet">
    /// <item><description>Real image, critic outputs +5, label is +1: contributes +5 (good!)</description></item>
    /// <item><description>Fake image, critic outputs -3, label is -1: contributes +3 (good!)</description></item>
    /// <item><description>Real image, critic outputs -2, label is +1: contributes -2 (bad!)</description></item>
    /// </list>
    ///
    /// The loss is negated so that minimizing the loss = maximizing the Wasserstein distance.
    /// </para>
    /// </remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Wasserstein loss: -mean(predicted * actual)
        // We negate because we want to minimize loss (which maximizes the Wasserstein distance)
        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(predicted[i], actual[i]));
        }

        // Return negative mean (minimizing this maximizes expected score for real, minimizes for fake)
        T mean = NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
        return NumOps.Negate(mean);
    }

    /// <summary>
    /// Calculates the derivative of the Wasserstein loss function.
    /// </summary>
    /// <param name="predicted">The critic's output scores for each sample.</param>
    /// <param name="actual">The labels: +1 for real samples, -1 for fake samples.</param>
    /// <returns>A vector containing the derivatives of the Wasserstein loss for each prediction.</returns>
    /// <remarks>
    /// <para>
    /// The derivative of the Wasserstein loss with respect to the predicted scores is simply
    /// the negative of the labels (after accounting for the mean).
    /// </para>
    /// <para>
    /// <b>Mathematical Derivation:</b>
    /// <list type="bullet">
    /// <item><description>Loss = -mean(predicted * actual) = -(1/n) * sum(predicted_i * actual_i)</description></item>
    /// <item><description>dLoss/d(predicted_i) = -actual_i / n</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells the network which direction to adjust.
    ///
    /// For a real sample (label = +1):
    /// <list type="bullet">
    /// <item><description>Derivative is negative, so increasing the score decreases the loss</description></item>
    /// <item><description>This pushes the critic to give higher scores to real images</description></item>
    /// </list>
    ///
    /// For a fake sample (label = -1):
    /// <list type="bullet">
    /// <item><description>Derivative is positive, so decreasing the score decreases the loss</description></item>
    /// <item><description>This pushes the critic to give lower scores to fake images</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // d/d(predicted) of -mean(predicted * actual) = -actual / n
        Vector<T> derivative = new Vector<T>(predicted.Length);
        T divisor = NumOps.FromDouble(predicted.Length);

        for (int i = 0; i < predicted.Length; i++)
        {
            // Derivative: -actual[i] / n
            derivative[i] = NumOps.Divide(NumOps.Negate(actual[i]), divisor);
        }

        return derivative;
    }
}
