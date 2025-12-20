

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Noise Contrastive Estimation (NCE) loss function for efficient training with large output spaces.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Noise Contrastive Estimation (NCE) is a loss function designed to efficiently train
/// models with very large output spaces, such as language models with large vocabularies.
/// 
/// Instead of computing probabilities for all possible outputs (which could be millions in language models),
/// NCE transforms the problem into a binary classification task: distinguishing the true data from noise samples.
/// 
/// The key idea is to:
/// - Sample a small number of "negative" examples from a noise distribution
/// - Train the model to distinguish between true data points and these negative samples
/// 
/// This approach is much more computationally efficient than computing full softmax probabilities
/// over all possible outputs, especially when the output space is very large.
/// 
/// NCE is commonly used in:
/// - Word embedding models like Word2Vec
/// - Neural language models with large vocabularies
/// - Any model with a very large output space
/// </para>
/// </remarks>
public class NoiseContrastiveEstimationLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The number of noise samples to use per true sample.
    /// </summary>
    private readonly int _numNoiseSamples;

    /// <summary>
    /// Initializes a new instance of the NoiseContrastiveEstimationLoss class.
    /// </summary>
    /// <param name="numNoiseSamples">The number of noise samples to use per true sample. Default is 10.</param>
    public NoiseContrastiveEstimationLoss(int numNoiseSamples = 10)
    {
        _numNoiseSamples = numNoiseSamples;
    }

    /// <summary>
    /// Calculates the NCE loss between target and noise samples.
    /// </summary>
    /// <param name="targetLogits">The logits for the true target samples.</param>
    /// <param name="noiseLogits">The logits for the noise samples.</param>
    /// <returns>The NCE loss value.</returns>
    public T Calculate(Vector<T> targetLogits, Matrix<T> noiseLogits)
    {
        // Ensure dimensions match
        if (targetLogits.Length != noiseLogits.Rows)
        {
            throw new ArgumentException("Number of target samples must match number of rows in noise samples.");
        }

        if (noiseLogits.Columns != _numNoiseSamples)
        {
            throw new ArgumentException("Number of noise samples per target must match the configured value.");
        }

        T loss = NumOps.Zero;
        for (int i = 0; i < targetLogits.Length; i++)
        {
            // P(target is real | target)
            T targetProb = Sigmoid(targetLogits[i]);

            // Log P(target is real | target)
            T targetTerm = NumericalStabilityHelper.SafeLog(targetProb, NumericalStabilityHelper.SmallEpsilon);

            // Sum of log(1 - P(noise is real | noise))
            T noiseSum = NumOps.Zero;
            for (int j = 0; j < _numNoiseSamples; j++)
            {
                T noiseProb = Sigmoid(noiseLogits[i, j]);
                T noiseTerm = NumericalStabilityHelper.SafeLog(
                    NumOps.Subtract(NumOps.One, noiseProb),
                    NumericalStabilityHelper.SmallEpsilon
                );
                noiseSum = NumOps.Add(noiseSum, noiseTerm);
            }

            // -(log P(target is real) + sum(log P(noise is noise)))
            loss = NumOps.Add(loss, NumOps.Negate(NumOps.Add(targetTerm, noiseSum)));
        }

        return NumOps.Divide(loss, NumOps.FromDouble(targetLogits.Length));
    }

    /// <summary>
    /// Calculates the gradient of the NCE loss function.
    /// </summary>
    /// <param name="targetLogits">The logits for the true target samples.</param>
    /// <param name="noiseLogits">The logits for the noise samples.</param>
    /// <returns>A tuple containing the gradients for target and noise logits.</returns>
    public (Vector<T>, Matrix<T>) CalculateDerivative(Vector<T> targetLogits, Matrix<T> noiseLogits)
    {
        // Ensure dimensions match
        if (targetLogits.Length != noiseLogits.Rows)
        {
            throw new ArgumentException("Number of target samples must match number of rows in noise samples.");
        }

        if (noiseLogits.Columns != _numNoiseSamples)
        {
            throw new ArgumentException("Number of noise samples per target must match the configured value.");
        }

        Vector<T> targetGradient = new Vector<T>(targetLogits.Length);
        Matrix<T> noiseGradient = new Matrix<T>(noiseLogits.Rows, noiseLogits.Columns);

        for (int i = 0; i < targetLogits.Length; i++)
        {
            // P(target is real | target)
            T targetProb = Sigmoid(targetLogits[i]);

            // -(1 - P(target is real | target))
            targetGradient[i] = NumOps.Negate(NumOps.Subtract(NumOps.One, targetProb));

            for (int j = 0; j < _numNoiseSamples; j++)
            {
                // P(noise is real | noise)
                T noiseProb = Sigmoid(noiseLogits[i, j]);

                // P(noise is real | noise)
                noiseGradient[i, j] = noiseProb;
            }
        }

        // Scale by batch size
        T scale = NumOps.Divide(NumOps.One, NumOps.FromDouble(targetLogits.Length));
        targetGradient = targetGradient.Transform(x => NumOps.Multiply(x, scale));
        noiseGradient = noiseGradient.Transform((x, _, __) => NumOps.Multiply(x, scale));

        return (targetGradient, noiseGradient);
    }

    /// <summary>
    /// This method is not used for NCE Loss as it requires specific input formats.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (target) values vector.</param>
    /// <returns>Throws NotSupportedException.</returns>
    /// <exception cref="NotSupportedException">Always thrown as NCE Loss requires specific input formats.</exception>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "NCE Loss requires specific input formats. " +
            "Use the Calculate(Vector<T>, Matrix<T>) method instead."
        );
    }

    /// <summary>
    /// This method is not used for NCE Loss as it requires specific input formats.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (target) values vector.</param>
    /// <returns>Throws NotSupportedException.</returns>
    /// <exception cref="NotSupportedException">Always thrown as NCE Loss requires specific input formats.</exception>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "NCE Loss requires specific input formats. " +
            "Use the CalculateDerivative(Vector<T>, Matrix<T>) method instead."
        );
    }

    /// <summary>
    /// Applies the sigmoid function to a scalar value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The sigmoid of the input, a value between 0 and 1.</returns>
    private T Sigmoid(T x)
    {
        return NumOps.Divide(
            NumOps.One,
            NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(x)))
        );
    }
}
