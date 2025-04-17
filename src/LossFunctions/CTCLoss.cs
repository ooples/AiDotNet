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
public class CTCLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The blank symbol index used in the CTC algorithm.
    /// </summary>
    private readonly int _blankIndex;
    
    /// <summary>
    /// Small value to prevent numerical instability.
    /// </summary>
    private readonly T _epsilon;
    
    /// <summary>
    /// Initializes a new instance of the CTCLoss class.
    /// </summary>
    /// <param name="blankIndex">The index of the blank symbol in the vocabulary. Default is 0.</param>
    public CTCLoss(int blankIndex = 0)
    {
        _blankIndex = blankIndex;
        _epsilon = NumOps.FromDouble(1e-15);
    }
    
    /// <summary>
    /// Calculates the CTC loss for sequence-to-sequence learning.
    /// </summary>
    /// <param name="logProbs">The log probabilities matrix [time, classes].</param>
    /// <param name="labels">The label sequence (target).</param>
    /// <param name="inputLengths">The actual lengths of each input sequence.</param>
    /// <param name="labelLengths">The actual lengths of each label sequence.</param>
    /// <returns>The CTC loss value.</returns>
    /// <remarks>
    /// This is a simplified implementation of CTC loss. A full implementation would use the
    /// forward-backward algorithm to efficiently compute the loss and gradients.
    /// </remarks>
    public T Calculate(Matrix<T> logProbs, Vector<int> labels, Vector<int> inputLengths, Vector<int> labelLengths)
    {
        // This is a placeholder for a complete CTC implementation
        // A full implementation would use the forward-backward algorithm
        
        // For a simple case with a single sequence
        if (inputLengths.Length == 1 && labelLengths.Length == 1)
        {
            int sequenceLength = inputLengths[0];
            int labelLength = labelLengths[0];
            
            // Create the extended label sequence with blanks
            Vector<int> extendedLabels = new Vector<int>(labelLength * 2 + 1);
            extendedLabels[0] = _blankIndex;
            
            for (int i = 0; i < labelLength; i++)
            {
                extendedLabels[i * 2 + 1] = labels[i];
                extendedLabels[i * 2 + 2] = _blankIndex;
            }
            
            // Forward algorithm (simplified)
            int T = sequenceLength;
            int S = extendedLabels.Length;
            
            // Alpha is the forward variable
            Matrix<T> alpha = new Matrix<T>(T, S);
            
            // Initialize first time step
            alpha[0, 0] = logProbs[0, extendedLabels[0]];
            alpha[0, 1] = logProbs[0, extendedLabels[1]];
            
            // Forward pass
            for (int t = 1; t < T; t++)
            {
                for (int s = 0; s < S; s++)
                {
                    T sum = NumOps.Zero;
                    
                    // Current label can be reached from the previous label
                    sum = NumOps.Add(sum, alpha[t - 1, s]);
                    
                    // If not blank and not a repeat, can come from two labels back
                    if (s > 0)
                    {
                        sum = NumOps.Add(sum, alpha[t - 1, s - 1]);
                    }
                    
                    // If not repeat of non-blank, can come from two labels back
                    if (s > 1 && extendedLabels[s] != extendedLabels[s - 2])
                    {
                        sum = NumOps.Add(sum, alpha[t - 1, s - 2]);
                    }
                    
                    alpha[t, s] = NumOps.Multiply(sum, logProbs[t, extendedLabels[s]]);
                }
            }
            
            // Total probability is the sum of the last two entries
            T logLikelihood = NumOps.Add(alpha[T - 1, S - 1], alpha[T - 1, S - 2]);
            
            // CTC loss is negative log-likelihood
            return NumOps.Negate(NumOps.Log(MathHelper.Max(logLikelihood, _epsilon)));
        }
        
        // For multiple sequences, we would compute the loss for each and average
        throw new NotImplementedException("Full CTC implementation with multiple sequences is not implemented.");
    }
    
    /// <summary>
    /// This method is not used for CTC Loss as it requires specific input formats.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (target) values vector.</param>
    /// <returns>Throws NotSupportedException.</returns>
    /// <exception cref="NotSupportedException">Always thrown as CTC Loss requires specific input formats.</exception>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "CTC Loss requires specific input formats. " +
            "Use the Calculate(Matrix<T>, Vector<int>, Vector<int>, Vector<int>) method instead."
        );
    }
    
    /// <summary>
    /// This method is not used for CTC Loss as it requires specific input formats.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (target) values vector.</param>
    /// <returns>Throws NotSupportedException.</returns>
    /// <exception cref="NotSupportedException">Always thrown as CTC Loss requires specific input formats.</exception>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "CTC Loss requires specific input formats and is typically calculated using automatic differentiation."
        );
    }
}