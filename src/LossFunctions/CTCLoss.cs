

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Connectionist Temporal Classification (CTC) loss function for sequence-to-sequence learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Connectionist Temporal Classification (CTC) is a loss function designed for 
/// sequence-to-sequence learning problems where the alignment between input and output sequences is unknown.
/// 
/// For example, in speech recognition, we have:
/// - Input: An audio waveform (long sequence of sound samples)
/// - Output: Text transcript (shorter sequence of characters)
/// 
/// The key challenge is that we don't know exactly which parts of the audio correspond to each character.
/// CTC solves this by considering all possible alignments between the input and output sequences.
/// 
/// CTC introduces a special "blank" token to handle:
/// - Repetitions of characters (e.g., "hello" vs "hheellloo")
/// - Silence or transitions between sounds
/// 
/// This loss function is commonly used in:
/// - Speech recognition
/// - Handwriting recognition
/// - Any task where input and output sequences have different lengths and unknown alignment
/// </para>
/// </remarks>
public class CTCLoss<T> : ISequenceLossFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _blankIndex;
    private readonly bool _inputsAreLogProbs;
    private readonly T _logZero;

    /// <summary>
    /// Initializes a new instance of the CTCLoss class.
    /// </summary>
    /// <param name="numericOperations">The numeric operations provider for type T.</param>
    /// <param name="blankIndex">The index of the blank symbol in the vocabulary. Default is 0.</param>
    /// <param name="inputsAreLogProbs">Whether inputs are already in log space. Default is true.</param>
    /// <exception cref="ArgumentNullException">Thrown when numericOperations is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when blankIndex is negative.</exception>
    public CTCLoss(int blankIndex = 0, bool inputsAreLogProbs = true)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        if (blankIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(blankIndex), "Blank index cannot be negative.");
        }

        _blankIndex = blankIndex;
        _inputsAreLogProbs = inputsAreLogProbs;
        _logZero = _numOps.FromDouble(-1000.0); // Effectively zero in log space
    }

    /// <summary>
    /// Calculates the CTC loss for a batch of sequences.
    /// </summary>
    /// <param name="logProbs">Log probabilities tensor [batch, time, classes].</param>
    /// <param name="targets">Target label sequences for each batch item.</param>
    /// <param name="inputLengths">Actual lengths of each input sequence.</param>
    /// <param name="targetLengths">Actual lengths of each target sequence.</param>
    /// <returns>The average CTC loss value across the batch.</returns>
    public T CalculateLoss(Tensor<T> logProbs, int[][] targets, int[] inputLengths, int[] targetLengths)
    {
        // Validate inputs
        ValidateInputs(logProbs, targets, inputLengths, targetLengths);

        int batchSize = inputLengths.Length;
        T batchLoss = _numOps.Zero;

        // Process each batch item
        for (int b = 0; b < batchSize; b++)
        {
            // Create the extended target sequence with blanks
            List<int> extendedLabels = new List<int>(targetLengths[b] * 2 + 1);
            extendedLabels.Add(_blankIndex); // Start with blank

            for (int i = 0; i < targetLengths[b]; i++)
            {
                extendedLabels.Add(targets[b][i]);
                extendedLabels.Add(_blankIndex);
            }

            int extendedLength = extendedLabels.Count;
            int sequenceLength = inputLengths[b];

            // Create forward and backward variables for dynamic programming
            T[,] alpha = new T[sequenceLength, extendedLength];
            T[,] beta = new T[sequenceLength, extendedLength];

            // Initialize with log zero probability
            for (int t = 0; t < sequenceLength; t++)
            {
                for (int s = 0; s < extendedLength; s++)
                {
                    alpha[t, s] = _logZero;
                    beta[t, s] = _logZero;
                }
            }

            // Forward pass (alpha)
            ComputeAlpha(b, alpha, extendedLabels, logProbs, sequenceLength, extendedLength);

            // Backward pass (beta)
            ComputeBeta(b, beta, extendedLabels, logProbs, sequenceLength, extendedLength);

            // Compute log probability by summing over all valid paths
            T logProb = _logZero;

            // Sum probabilities for all valid ending positions (last blank and last label)
            if (extendedLength > 1) // More than just a blank
            {
                logProb = LogSumExp(alpha[sequenceLength - 1, extendedLength - 1],
                                  alpha[sequenceLength - 1, extendedLength - 2]);
            }
            else
            {
                logProb = alpha[sequenceLength - 1, 0]; // Just the blank
            }

            // Negative log likelihood
            T loss = _numOps.Negate(logProb);
            batchLoss = _numOps.Add(batchLoss, loss);
        }

        // Return average loss
        return _numOps.Divide(batchLoss, _numOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Calculates the gradient of the CTC loss with respect to the inputs.
    /// </summary>
    /// <param name="logProbs">Log probabilities tensor [batch, time, classes].</param>
    /// <param name="targets">Target label sequences for each batch item.</param>
    /// <param name="inputLengths">Actual lengths of each input sequence.</param>
    /// <param name="targetLengths">Actual lengths of each target sequence.</param>
    /// <returns>The gradient tensor with same shape as inputs.</returns>
    public Tensor<T> CalculateGradient(Tensor<T> logProbs, int[][] targets, int[] inputLengths, int[] targetLengths)
    {
        // Validate inputs
        ValidateInputs(logProbs, targets, inputLengths, targetLengths);

        int batchSize = inputLengths.Length;
        int maxTime = logProbs.Shape[1];
        int numClasses = logProbs.Shape[2];

        // Initialize gradient tensor with same shape as input
        var gradient = new Tensor<T>(logProbs.Shape);

        // For each item in the batch
        for (int b = 0; b < batchSize; b++)
        {
            // Create the extended target sequence with blanks
            List<int> extendedLabels = new List<int>(targetLengths[b] * 2 + 1);
            extendedLabels.Add(_blankIndex); // Start with blank

            for (int i = 0; i < targetLengths[b]; i++)
            {
                extendedLabels.Add(targets[b][i]);
                extendedLabels.Add(_blankIndex);
            }

            int extendedLength = extendedLabels.Count;
            int sequenceLength = inputLengths[b];

            // Create forward and backward variables
            T[,] alpha = new T[sequenceLength, extendedLength];
            T[,] beta = new T[sequenceLength, extendedLength];

            // Initialize
            for (int t = 0; t < sequenceLength; t++)
            {
                for (int s = 0; s < extendedLength; s++)
                {
                    alpha[t, s] = _logZero;
                    beta[t, s] = _logZero;
                }
            }

            // Forward pass
            ComputeAlpha(b, alpha, extendedLabels, logProbs, sequenceLength, extendedLength);

            // Backward pass
            ComputeBeta(b, beta, extendedLabels, logProbs, sequenceLength, extendedLength);

            // Compute total log probability (denominator for gradient)
            T totalLogProb = _logZero;

            if (extendedLength > 1)
            {
                totalLogProb = LogSumExp(alpha[sequenceLength - 1, extendedLength - 1],
                                       alpha[sequenceLength - 1, extendedLength - 2]);
            }
            else
            {
                totalLogProb = alpha[sequenceLength - 1, 0];
            }

            // Compute gradients for this sequence
            for (int t = 0; t < sequenceLength; t++)
            {
                // Initialize all class gradients to zero (in log space)
                var classGradients = new T[numClasses];
                for (int c = 0; c < numClasses; c++)
                {
                    classGradients[c] = _logZero;
                }

                // For each position in the extended label sequence
                for (int s = 0; s < extendedLength; s++)
                {
                    int labelIdx = extendedLabels[s];

                    // Posterior probability of this path (alpha * beta / totalProb)
                    // In log space: alpha + beta - totalLogProb
                    T logPathProb = _numOps.Subtract(
                        _numOps.Add(alpha[t, s], beta[t, s]),
                        totalLogProb
                    );

                    // Add to the gradient for this class
                    // Convert from log space to get actual gradient
                    T expPathProb = _numOps.Exp(logPathProb);

                    // Accumulate probability for this label
                    if (labelIdx < numClasses) // Safety check
                    {
                        classGradients[labelIdx] = LogSumExp(
                            classGradients[labelIdx],
                            NumericalStabilityHelper.SafeLog(expPathProb, NumericalStabilityHelper.SmallEpsilon)
                        );
                    }
                }

                // Set gradients for this time step
                // Gradient is derivative of -log(p) which is -1/p
                for (int c = 0; c < numClasses; c++)
                {
                    // Skip updating gradient for positions outside sequence length
                    if (t >= sequenceLength)
                    {
                        break;
                    }

                    T expProb = _numOps.Exp(classGradients[c]);
                    T gradValue = _numOps.Negate(expProb);

                    // Add to batch element's gradient
                    gradient[new[] { b, t, c }] = gradValue;
                }
            }
        }

        return gradient;
    }

    /// <summary>
    /// Computes the forward variables (alpha) for the CTC algorithm.
    /// </summary>
    private void ComputeAlpha(int batchIndex, T[,] alpha, List<int> extendedLabels,
                             Tensor<T> logProbs, int sequenceLength, int extendedLength)
    {
        // Initialize first time step (t=0)
        T logProb0 = GetLogProb(logProbs, batchIndex, 0, extendedLabels[0]);
        alpha[0, 0] = logProb0;

        if (extendedLength > 1)
        {
            T logProb1 = GetLogProb(logProbs, batchIndex, 0, extendedLabels[1]);
            alpha[0, 1] = logProb1;
        }

        // Forward pass
        for (int t = 1; t < sequenceLength; t++)
        {
            for (int s = 0; s < extendedLength; s++)
            {
                int label = extendedLabels[s];

                // Initialize with the no-skip case
                T logProbSum = alpha[t - 1, s];

                // Can come from previous label (except for first blank label)
                if (s > 0)
                {
                    logProbSum = LogSumExp(logProbSum, alpha[t - 1, s - 1]);
                }

                // Special case for label pairs like a-a, where we skip the blank between repeated labels
                if (s > 1 && label != _blankIndex && extendedLabels[s - 2] == label)
                {
                    logProbSum = LogSumExp(logProbSum, alpha[t - 1, s - 2]);
                }

                // Multiply by current observation probability (add in log space)
                T logProbCurrent = GetLogProb(logProbs, batchIndex, t, label);
                alpha[t, s] = _numOps.Add(logProbSum, logProbCurrent);
            }
        }
    }

    /// <summary>
    /// Computes the backward variables (beta) for the CTC algorithm.
    /// </summary>
    private void ComputeBeta(int batchIndex, T[,] beta, List<int> extendedLabels,
                            Tensor<T> logProbs, int sequenceLength, int extendedLength)
    {
        // Initialize last time step (t=T-1)
        beta[sequenceLength - 1, extendedLength - 1] = _numOps.Zero; // log(1) = 0

        if (extendedLength > 1)
        {
            beta[sequenceLength - 1, extendedLength - 2] = _numOps.Zero; // log(1) = 0
        }

        // Backward pass
        for (int t = sequenceLength - 2; t >= 0; t--)
        {
            for (int s = 0; s < extendedLength; s++)
            {
                // Initialize with the no-skip case
                T logProbSum = _numOps.Add(
                    beta[t + 1, s],
                    GetLogProb(logProbs, batchIndex, t + 1, extendedLabels[s])
                );

                // Can go to next label
                if (s < extendedLength - 1)
                {
                    T nextLogProb = _numOps.Add(
                        beta[t + 1, s + 1],
                        GetLogProb(logProbs, batchIndex, t + 1, extendedLabels[s + 1])
                    );
                    logProbSum = LogSumExp(logProbSum, nextLogProb);
                }

                // Special case for merging paths with repeated characters
                if (s < extendedLength - 2 && extendedLabels[s] == extendedLabels[s + 2] && extendedLabels[s] != _blankIndex)
                {
                    T skipLogProb = _numOps.Add(
                        beta[t + 1, s + 2],
                        GetLogProb(logProbs, batchIndex, t + 1, extendedLabels[s + 2])
                    );
                    logProbSum = LogSumExp(logProbSum, skipLogProb);
                }

                beta[t, s] = logProbSum;
            }
        }
    }

    /// <summary>
    /// Gets the log probability for a specific batch, time, and label.
    /// </summary>
    private T GetLogProb(Tensor<T> logProbs, int batch, int time, int label)
    {
        // Check bounds to avoid index errors
        if (label >= logProbs.Shape[2])
        {
            return _logZero;
        }

        T value = logProbs[new[] { batch, time, label }];

        // If inputs are not already in log space, convert them
        if (!_inputsAreLogProbs)
        {
            value = NumericalStabilityHelper.SafeLog(value, NumericalStabilityHelper.SmallEpsilon);
        }

        return value;
    }

    /// <summary>
    /// Computes log(exp(x) + exp(y)) in a numerically stable way.
    /// </summary>
    private T LogSumExp(T x, T y)
    {
        // Handle edge cases for numerical stability
        if (_numOps.LessThan(x, _logZero))
            return y;
        if (_numOps.LessThan(y, _logZero))
            return x;

        T maxVal = MathHelper.Max(x, y);
        return _numOps.Add(
            maxVal,
            NumericalStabilityHelper.SafeLog(
                _numOps.Add(
                    _numOps.Exp(_numOps.Subtract(x, maxVal)),
                    _numOps.Exp(_numOps.Subtract(y, maxVal))
                ),
                NumericalStabilityHelper.SmallEpsilon
            )
        );
    }

    /// <summary>
    /// Validates input parameters for the CTC loss calculation.
    /// </summary>
    private void ValidateInputs(Tensor<T> logProbs, int[][] targets, int[] inputLengths, int[] targetLengths)
    {
        if (logProbs == null)
            throw new ArgumentNullException(nameof(logProbs));
        if (targets == null)
            throw new ArgumentNullException(nameof(targets));
        if (inputLengths == null)
            throw new ArgumentNullException(nameof(inputLengths));
        if (targetLengths == null)
            throw new ArgumentNullException(nameof(targetLengths));

        int batchSize = logProbs.Shape[0];

        if (targets.Length != batchSize)
            throw new ArgumentException($"Number of target sequences ({targets.Length}) doesn't match batch size ({batchSize})");
        if (inputLengths.Length != batchSize)
            throw new ArgumentException($"Number of input lengths ({inputLengths.Length}) doesn't match batch size ({batchSize})");
        if (targetLengths.Length != batchSize)
            throw new ArgumentException($"Number of target lengths ({targetLengths.Length}) doesn't match batch size ({batchSize})");

        for (int b = 0; b < batchSize; b++)
        {
            if (inputLengths[b] <= 0 || inputLengths[b] > logProbs.Shape[1])
                throw new ArgumentException($"Invalid input length at index {b}: {inputLengths[b]}");

            if (targetLengths[b] <= 0)
                throw new ArgumentException($"Invalid target length at index {b}: {targetLengths[b]}");

            if (targetLengths[b] > inputLengths[b])
                throw new ArgumentException($"Target length ({targetLengths[b]}) exceeds input length ({inputLengths[b]}) at index {b}");

            if (targets[b] == null || targets[b].Length < targetLengths[b])
                throw new ArgumentException($"Target sequence at index {b} is null or shorter than specified length");
        }
    }
}

/// <summary>
/// Provides an adapter for using CTCLoss within the LossFunctionBase framework.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CTCLossAdapter<T> : LossFunctionBase<T>
{
    private readonly CTCLoss<T> _ctcLoss;
    private readonly int _numClasses;

    /// <summary>
    /// Initializes a new instance of the CTCLossAdapter class.
    /// </summary>
    /// <param name="numericOperations">The numeric operations provider.</param>
    /// <param name="numClasses">The number of classes/vocabulary size.</param>
    /// <param name="blankIndex">The blank symbol index.</param>
    public CTCLossAdapter(int numClasses, int blankIndex = 0)
    {
        _ctcLoss = new CTCLoss<T>(blankIndex);
        _numClasses = numClasses;
    }

    /// <summary>
    /// Adapts the standard loss interface to work with CTC loss.
    /// </summary>
    /// <param name="predicted">The predicted log probabilities as a flattened vector.</param>
    /// <param name="actual">The target labels as a flattened vector.</param>
    /// <returns>The CTC loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        // Validate inputs
        if (predicted == null)
            throw new ArgumentNullException(nameof(predicted));
        if (actual == null)
            throw new ArgumentNullException(nameof(actual));

        // First element is the batch size
        int batchSize = (int)Convert.ToDouble(actual[0]);

        if (batchSize <= 0)
            throw new ArgumentException("Invalid batch size encoded in actual vector");

        // Calculate time steps per batch from predicted vector length
        int totalElements = predicted.Length;
        int timeStepsTotal = totalElements / (_numClasses * batchSize);

        // Reshape predicted into a 3D tensor
        Tensor<T> logProbs = new Tensor<T>(new[] { batchSize, timeStepsTotal, _numClasses });

        // Fill the tensor with values from predicted
        int index = 0;
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < timeStepsTotal; t++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    logProbs[new[] { b, t, c }] = predicted[index++];
                }
            }
        }

        // Parse the actual vector format
        int actualIndex = 1; // Skip batch size
        int[][] targets = new int[batchSize][];
        int[] inputLengths = new int[batchSize];
        int[] targetLengths = new int[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            // Get sequence length
            targetLengths[b] = (int)Convert.ToDouble(actual[actualIndex++]);

            // Default to full time steps
            inputLengths[b] = timeStepsTotal;

            // Get labels
            targets[b] = new int[targetLengths[b]];
            for (int i = 0; i < targetLengths[b]; i++)
            {
                targets[b][i] = (int)Convert.ToDouble(actual[actualIndex++]);
            }
        }

        // Call the CTCLoss implementation
        return _ctcLoss.CalculateLoss(logProbs, targets, inputLengths, targetLengths);
    }

    /// <summary>
    /// Calculates the derivative of the CTC loss.
    /// </summary>
    /// <param name="predicted">The predicted log probabilities as a flattened vector.</param>
    /// <param name="actual">The target labels as a flattened vector.</param>
    /// <returns>The gradient vector with respect to inputs.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        // Follow the same reconstruction process as in CalculateLoss
        if (predicted == null)
            throw new ArgumentNullException(nameof(predicted));
        if (actual == null)
            throw new ArgumentNullException(nameof(actual));

        // Reconstruct the batch structure
        int batchSize = (int)Convert.ToDouble(actual[0]);

        if (batchSize <= 0)
            throw new ArgumentException("Invalid batch size encoded in actual vector");

        // Calculate time steps per batch
        int totalElements = predicted.Length;
        int timeStepsTotal = totalElements / (_numClasses * batchSize);

        // Reshape predicted into a 3D tensor
        Tensor<T> logProbs = new Tensor<T>(new[] { batchSize, timeStepsTotal, _numClasses });

        // Fill the tensor
        int index = 0;
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < timeStepsTotal; t++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    logProbs[new[] { b, t, c }] = predicted[index++];
                }
            }
        }

        // Parse the actual vector format
        int actualIndex = 1; // Skip batch size
        int[][] targets = new int[batchSize][];
        int[] inputLengths = new int[batchSize];
        int[] targetLengths = new int[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            // Get sequence length
            targetLengths[b] = (int)Convert.ToDouble(actual[actualIndex++]);

            // Default to full time steps
            inputLengths[b] = timeStepsTotal;

            // Get labels
            targets[b] = new int[targetLengths[b]];
            for (int i = 0; i < targetLengths[b]; i++)
            {
                targets[b][i] = (int)Convert.ToDouble(actual[actualIndex++]);
            }
        }

        // Calculate gradients
        Tensor<T> gradientTensor = _ctcLoss.CalculateGradient(logProbs, targets, inputLengths, targetLengths);

        // Flatten the gradient tensor to match the input vector format
        Vector<T> gradientVector = new Vector<T>(predicted.Length);

        index = 0;
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < timeStepsTotal; t++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    gradientVector[index++] = gradientTensor[new[] { b, t, c }];
                }
            }
        }

        return gradientVector;
    }
}
