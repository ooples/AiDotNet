

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.Engines.Autodiff;

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
[LossCategory(LossCategory.Classification)]
[LossTask(LossTask.TextGeneration)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = true, ExpectedOutput = OutputType.Probabilities, ApiShape = LossApiShape.Sequence)]
public class CTCLoss<T> : LossFunctionBase<T>, ISequenceLossFunction<T>
{
    private readonly int _blankIndex;

    /// <summary>Gets the blank symbol index used by this CTC loss.</summary>
    public int BlankIndex => _blankIndex;

    /// <summary>Gets the vocabulary cardinality expected on the final prediction axis.</summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Gets whether predictions supplied to this loss must already be log probabilities.
    /// </summary>
    public bool InputsAreLogProbabilities => _inputsAreLogProbs;

    private readonly bool _inputsAreLogProbs;
    private readonly int _numClasses;

    /// <summary>
    /// Initializes a new instance of the CTCLoss class.
    /// </summary>
    /// <param name="numericOperations">The numeric operations provider for type T.</param>
    /// <param name="blankIndex">The index of the blank symbol in the vocabulary. Default is 0.</param>
    /// <param name="inputsAreLogProbs">Whether inputs are already in log space. Default is true.</param>
    /// <exception cref="ArgumentNullException">Thrown when numericOperations is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when blankIndex is negative.</exception>
    public CTCLoss(int numClasses = 29, int blankIndex = 0, bool inputsAreLogProbs = true)
    {
        if (numClasses <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses), "CTC requires at least two classes.");
        }

        if (blankIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(blankIndex), "Blank index cannot be negative.");
        }

        if (blankIndex >= numClasses)
        {
            throw new ArgumentOutOfRangeException(
                nameof(blankIndex), "Blank index must be smaller than the number of classes.");
        }

        _numClasses = numClasses;
        _blankIndex = blankIndex;
        _inputsAreLogProbs = inputsAreLogProbs;
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
        ValidateInputs(logProbs, targets, inputLengths, targetLengths);

        return ComputeMeanLoss(logProbs, targets, inputLengths, targetLengths)[0];
    }

    /// <summary>
    /// The batch-mean CTC loss as a scalar tensor. <see cref="CalculateLoss(Tensor{T}, int[][], int[], int[])"/>
    /// reads its value and <see cref="CalculateGradient(Tensor{T}, int[][], int[], int[])"/> differentiates it, so the reported loss and the
    /// gradient always describe the same function.
    /// </summary>
    private Tensor<T> ComputeMeanLoss(Tensor<T> predictions, int[][] targets, int[] inputLengths, int[] targetLengths)
    {
        var perItemLoss = ComputePerItemLoss(
            predictions, ConcatenateTargets(targets, targetLengths), inputLengths, targetLengths);
        return Engine.ReduceMean(perItemLoss, new[] { 0 }, keepDims: false);
    }

    /// <summary>
    /// The per-sequence CTC loss, -log p(target | input), for batch-first predictions [batch, time, classes].
    /// </summary>
    /// <remarks>
    /// Probability inputs are moved to log space with the safe log first. The engine's CTC primitive runs the
    /// forward (alpha) recursion over the blank-extended label lattice, reading only the first
    /// <c>inputLengths[b]</c> frames and <c>targetLengths[b]</c> labels, and records its forward-backward
    /// gradient on the active tape. Every loss path (CalculateLoss, CalculateGradient and the tape loss)
    /// goes through this method, so they all score a batch identically.
    /// </remarks>
    private Tensor<T> ComputePerItemLoss(
        Tensor<T> predictions, Tensor<int> concatenatedTargets, int[] inputLengths, int[] targetLengths)
    {
        var logSpace = _inputsAreLogProbs
            ? predictions
            : SafeLogTensor(predictions, NumericalStabilityHelper.SmallEpsilon);

        // AiDotNet models expose sequence outputs as [batch, time, classes], while the
        // tensor engine's CTC primitive follows the conventional [time, batch, classes]
        // layout. Transpose through the engine so the operation remains tape-tracked.
        var timeMajorLogProbs = Engine.TensorPermute(logSpace, new[] { 1, 0, 2 });
        return Engine.TensorCTCLoss(
            timeMajorLogProbs, concatenatedTargets, inputLengths, targetLengths, _blankIndex);
    }

    /// <summary>
    /// Concatenates the first <c>targetLengths[b]</c> labels of every target sequence, the layout the
    /// engine's CTC primitive expects.
    /// </summary>
    private static Tensor<int> ConcatenateTargets(int[][] targets, int[] targetLengths)
    {
        var concatenated = new int[targetLengths.Sum()];
        int offset = 0;
        for (int b = 0; b < targets.Length; b++)
        {
            Array.Copy(targets[b], 0, concatenated, offset, targetLengths[b]);
            offset += targetLengths[b];
        }

        return new Tensor<int>(concatenated, new[] { concatenated.Length });
    }

    /// <summary>
    /// Decodes the flat CTC target encoding <c>[batch, len_0, labels_0..., len_1, labels_1..., ...]</c>.
    /// </summary>
    /// <param name="encoded">The encoded targets.</param>
    /// <param name="batchSize">The number of sequences to decode.</param>
    /// <param name="paramName">The caller's parameter name for the encoded targets, used in exceptions.</param>
    /// <returns>The label sequences and their lengths.</returns>
    private static (int[][] Targets, int[] TargetLengths) DecodeTargets(Vector<T> encoded, int batchSize, string paramName)
    {
        // The encoding holds integer metadata (lengths and label ids). Each code is rounded to the nearest
        // integer in bulk before the int conversion, so a float code such as 1.9999999 reads as label 2.
        int[] codes = AiDotNetEngine.Current.TensorRound(encoded.AsTensor()).Cast<int>().ToArray();

        var targets = new int[batchSize][];
        var targetLengths = new int[batchSize];
        int index = 1;
        for (int b = 0; b < batchSize; b++)
        {
            if (index >= codes.Length)
                throw new ArgumentOutOfRangeException(paramName,
                    $"The encoded targets end before the length of sequence {b}.");

            int length = codes[index++];
            if (length < 0 || length > codes.Length - index)
                throw new ArgumentOutOfRangeException(paramName,
                    $"Sequence {b} has encoded length {length}, but {codes.Length - index} codes remain.");

            targetLengths[b] = length;
            targets[b] = new int[length];
            Array.Copy(codes, index, targets[b], 0, length);
            index += length;
        }

        return (targets, targetLengths);
    }

    /// <summary>
    /// Calculates the gradient of the CTC loss with respect to the inputs.
    /// </summary>
    /// <param name="logProbs">Log probabilities tensor [batch, time, classes] (probabilities when the loss was
    /// built with <c>inputsAreLogProbs: false</c>).</param>
    /// <param name="targets">Target label sequences for each batch item.</param>
    /// <param name="inputLengths">Actual lengths of each input sequence.</param>
    /// <param name="targetLengths">Actual lengths of each target sequence.</param>
    /// <returns>The gradient tensor with same shape as inputs.</returns>
    /// <remarks>
    /// This is the exact derivative of the batch-mean value <see cref="CalculateLoss(Tensor{T}, int[][], int[], int[])"/>
    /// returns: the same engine computation is recorded on a gradient tape and differentiated, so the
    /// engine's CTC forward-backward (Graves et al. 2006, skip transitions only between differing labels)
    /// supplies the per-frame posteriors, the 1/batch factor of the mean is included, probability inputs
    /// chain through the safe log, and frames past <c>inputLengths[b]</c> receive zero gradient.
    /// </remarks>
    public Tensor<T> CalculateGradient(Tensor<T> logProbs, int[][] targets, int[] inputLengths, int[] targetLengths)
    {
        ValidateInputs(logProbs, targets, inputLengths, targetLengths);

        using var tape = new GradientTape<T>();
        var meanLoss = ComputeMeanLoss(logProbs, targets, inputLengths, targetLengths);
        var gradients = tape.ComputeGradients(meanLoss, new[] { logProbs });
        if (gradients.TryGetValue(logProbs, out var gradient) && gradient is not null)
        {
            return gradient;
        }

        throw new InvalidOperationException("CTC gradient computation produced no gradient for the inputs.");
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

    // ═══════════════════════════════════════════════════════════════
    // LossFunctionBase<T> implementation — no adapter needed
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Calculates CTC loss from flattened predicted log-probs and encoded target labels.
    /// Target encoding: [batchSize, targetLen0, label0_0, ..., targetLen1, label1_0, ...]
    /// </summary>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (actual is null) throw new ArgumentNullException(nameof(actual));

        int batchSize = (int)Convert.ToDouble(actual[0]);
        if (batchSize <= 0)
            throw new ArgumentException("Invalid batch size encoded in actual vector");

        int timeStepsTotal = predicted.Length / (_numClasses * batchSize);
        int usedLength = batchSize * timeStepsTotal * _numClasses;
        var logProbs = Tensor<T>.FromVector(
            predicted.GetSubVector(0, usedLength), new[] { batchSize, timeStepsTotal, _numClasses });

        var (targets, targetLengths) = DecodeTargets(actual, batchSize, nameof(actual));
        var inputLengths = Enumerable.Repeat(timeStepsTotal, batchSize).ToArray();

        return CalculateLoss(logProbs, targets, inputLengths, targetLengths);
    }

    /// <inheritdoc />
    /// <remarks>The encoded labels are decoded into index arrays on the host (<see cref="DecodeTargets"/>).</remarks>
    internal override bool ReadsTargetOnHost => true;

    /// <inheritdoc />
    /// <remarks>
    /// Uses the tape-tracked TensorCTCLoss engine op for differentiable CTC loss.
    /// predicted: log-probabilities [batch, time, classes]
    /// target: encoded labels [batchSize, targetLen0, label0_0, ..., targetLen1, label1_0, ...]
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // The base contract is a scalar: the mean of the per-sequence losses, the value CalculateLoss reports.
        return Engine.ReduceMean(ComputePerSequenceTapeLoss(predicted, target), new[] { 0 }, keepDims: false);
    }

    /// <summary>
    /// The unreduced tape-tracked CTC loss, one entry per batch sequence ([batch]), for callers that weight the
    /// sequences themselves before reducing (for example by label length).
    /// </summary>
    /// <param name="predicted">Log-probabilities [batch, time, classes].</param>
    /// <param name="target">Encoded labels [batchSize, targetLen0, label0_0, ..., targetLen1, label1_0, ...].</param>
    /// <returns>The per-sequence losses, shape [batch].</returns>
    internal Tensor<T> ComputePerSequenceTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        int batchSize = predicted.Shape.Length == 3 ? predicted.Shape[0]
            : (int)Convert.ToDouble(NumOps.ToDouble(target[0]));
        int timeSteps = predicted.Shape.Length == 3 ? predicted.Shape[1]
            : predicted.Length / (_numClasses * batchSize);

        var (targets, targetLengths) = DecodeTargets(target.ToVector(), batchSize, nameof(target));
        var inputLengths = Enumerable.Repeat(timeSteps, batchSize).ToArray();
        var targetsTensor = ConcatenateTargets(targets, targetLengths);

        var batchFirst = predicted;
        if (predicted.Shape.Length != 3)
            batchFirst = Engine.Reshape(predicted, new[] { batchSize, timeSteps, _numClasses });

        // Same per-sequence computation as CalculateLoss, including the log-space conversion of
        // probability inputs when the loss was built with inputsAreLogProbs: false.
        return ComputePerItemLoss(batchFirst, targetsTensor, inputLengths, targetLengths);
    }
}
