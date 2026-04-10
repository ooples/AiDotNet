using AiDotNet.HarmonicEngine.Layers;
using AiDotNet.HarmonicEngine.Options;

namespace AiDotNet.HarmonicEngine.Models;

/// <summary>
/// Character-level sequence model built on the Harmonic Resonance Engine.
/// Designed to demonstrate the architecture's ability to capture long-range periodic patterns
/// in discrete sequences — a capability where transformers with limited context struggle.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This model predicts the next character in a sequence by treating
/// character codes as a signal and processing them through the HRE's spectral layers.
///
/// The key advantage over transformers: periodic patterns at ANY period length are captured
/// naturally by the spectral representation. A pattern repeating every 13 characters is just
/// a frequency component at f = 1/13, which the HRE represents with a single coefficient.
/// A transformer would need its context window to be at least 13 characters long and would
/// need to learn the periodicity from examples, whereas the HRE captures it structurally.
///
/// This is the "novel capability" experiment for the paper: showing that HRE maintains
/// prediction accuracy on periods longer than a transformer's context window.
/// </para>
/// </remarks>
public class HRESequenceModel<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly HREModel<T> _model;
    private readonly int _contextLength;
    private readonly int _vocabularySize;

    /// <summary>
    /// Gets the underlying HRE model.
    /// </summary>
    public HREModel<T> Model => _model;

    /// <summary>
    /// Gets the context window length.
    /// </summary>
    public int ContextLength => _contextLength;

    /// <summary>
    /// Gets the vocabulary size (number of distinct characters).
    /// </summary>
    public int VocabularySize => _vocabularySize;

    /// <summary>
    /// Creates a new HRE sequence model.
    /// </summary>
    /// <param name="contextLength">Number of past characters to consider. Must be power of 2.</param>
    /// <param name="vocabularySize">Number of distinct characters in the vocabulary.</param>
    /// <param name="options">HRE model options.</param>
    public HRESequenceModel(int contextLength = 64, int vocabularySize = 128, HREModelOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _contextLength = contextLength;
        _vocabularySize = vocabularySize;

        options ??= new HREModelOptions();
        options.InputSize = contextLength;
        options.OutputSize = vocabularySize;

        _model = new HREModel<T>(options);
    }

    /// <summary>
    /// Predicts the next character given a context of past characters.
    /// </summary>
    /// <param name="context">Character codes (integers cast to T) of length ContextLength.</param>
    /// <returns>Log-probabilities for each character in the vocabulary.</returns>
    public Vector<T> PredictNext(Vector<T> context)
    {
        if (context.Length != _contextLength)
            throw new ArgumentException(
                $"Context must have {_contextLength} elements, got {context.Length}.");

        // Normalize character codes to [-1, 1] range
        var normalized = new Tensor<T>([_contextLength]);
        var vocabScale = _numOps.FromDouble(2.0 / _vocabularySize);
        var one = _numOps.One;

        for (int i = 0; i < _contextLength; i++)
        {
            normalized[i] = _numOps.Subtract(
                _numOps.Multiply(context[i], vocabScale), one);
        }

        _model.SetTrainingMode(false);
        var output = _model.Forward(normalized);

        // Convert to log-probabilities via log-softmax
        var logProbs = new Vector<T>(_vocabularySize);
        double maxVal = double.NegativeInfinity;

        for (int i = 0; i < _vocabularySize; i++)
        {
            double val = _numOps.ToDouble(i < output.Length ? output[i] : _numOps.Zero);
            if (val > maxVal) maxVal = val;
        }

        double sumExp = 0;
        for (int i = 0; i < _vocabularySize; i++)
        {
            double val = _numOps.ToDouble(i < output.Length ? output[i] : _numOps.Zero);
            sumExp += Math.Exp(val - maxVal);
        }
        double logSumExp = maxVal + Math.Log(sumExp);

        for (int i = 0; i < _vocabularySize; i++)
        {
            double val = _numOps.ToDouble(i < output.Length ? output[i] : _numOps.Zero);
            logProbs[i] = _numOps.FromDouble(val - logSumExp);
        }

        return logProbs;
    }

    /// <summary>
    /// Predicts the most likely next character.
    /// </summary>
    /// <param name="context">Character code context.</param>
    /// <returns>The character code with highest probability.</returns>
    public int PredictNextCharacter(Vector<T> context)
    {
        var logProbs = PredictNext(context);
        int bestIdx = 0;
        double bestVal = double.NegativeInfinity;

        for (int i = 0; i < logProbs.Length; i++)
        {
            double val = _numOps.ToDouble(logProbs[i]);
            if (val > bestVal)
            {
                bestVal = val;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /// <summary>
    /// Generates a sequence of characters autoregressively.
    /// </summary>
    /// <param name="seed">Initial context characters.</param>
    /// <param name="length">Number of characters to generate.</param>
    /// <returns>The generated character codes.</returns>
    public int[] Generate(int[] seed, int length)
    {
        var generated = new int[length];
        var context = new Vector<T>(_contextLength);

        // Fill context from seed
        for (int i = 0; i < _contextLength; i++)
        {
            int seedIdx = seed.Length - _contextLength + i;
            context[i] = _numOps.FromDouble(seedIdx >= 0 ? seed[seedIdx] : 0);
        }

        for (int t = 0; t < length; t++)
        {
            int nextChar = PredictNextCharacter(context);
            generated[t] = nextChar;

            // Slide context window
            for (int i = 0; i < _contextLength - 1; i++)
            {
                context[i] = context[i + 1];
            }
            context[_contextLength - 1] = _numOps.FromDouble(nextChar);
        }

        return generated;
    }
}
