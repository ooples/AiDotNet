using AiDotNet.Helpers;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// InfoNCE (Noise Contrastive Estimation) Loss for contrastive learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> InfoNCE is the loss function used in MoCo (Momentum Contrast).
/// It's similar to NT-Xent but designed to work efficiently with a large memory queue of
/// negative samples.</para>
///
/// <para><b>Key differences from NT-Xent:</b></para>
/// <list type="bullet">
/// <item>Uses a memory queue for negatives instead of in-batch negatives</item>
/// <item>Typically uses more negatives (65536) but smaller batch sizes</item>
/// <item>Asymmetric: query from online encoder, keys from momentum encoder</item>
/// </list>
///
/// <para><b>Loss formula:</b></para>
/// <code>
/// L = -log( exp(q·k+ / τ) / (exp(q·k+ / τ) + Σ exp(q·k- / τ)) )
/// </code>
/// </remarks>
public class InfoNCELoss<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _temperature;
    private readonly bool _normalize;

    /// <summary>
    /// Gets the temperature parameter.
    /// </summary>
    public double Temperature => _temperature;

    /// <summary>
    /// Initializes a new instance of the InfoNCELoss class.
    /// </summary>
    /// <param name="temperature">Temperature scaling parameter (default: 0.07).</param>
    /// <param name="normalize">Whether to L2-normalize embeddings (default: true).</param>
    public InfoNCELoss(double temperature = 0.07, bool normalize = true)
    {
        if (temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be positive");

        _temperature = temperature;
        _normalize = normalize;
    }

    /// <summary>
    /// Computes the InfoNCE loss using queries, positive keys, and negative keys from memory bank.
    /// </summary>
    /// <param name="queries">Query embeddings from online encoder [batch_size, dim].</param>
    /// <param name="positiveKeys">Positive key embeddings from momentum encoder [batch_size, dim].</param>
    /// <param name="negativeKeys">Negative key embeddings from memory bank [num_negatives, dim].</param>
    /// <returns>The computed loss value.</returns>
    public T ComputeLoss(Tensor<T> queries, Tensor<T> positiveKeys, Tensor<T> negativeKeys)
    {
        if (queries is null) throw new ArgumentNullException(nameof(queries));
        if (positiveKeys is null) throw new ArgumentNullException(nameof(positiveKeys));
        if (negativeKeys is null) throw new ArgumentNullException(nameof(negativeKeys));

        var batchSize = queries.Shape[0];
        var dim = queries.Shape[1];
        var numNegatives = negativeKeys.Shape[0];

        // Normalize if required
        var q = _normalize ? L2Normalize(queries) : queries;
        var kPos = _normalize ? L2Normalize(positiveKeys) : positiveKeys;
        var kNeg = _normalize ? L2Normalize(negativeKeys) : negativeKeys;

        T totalLoss = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            // Compute positive logit: q_i · k+_i / τ
            T posLogit = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                posLogit = NumOps.Add(posLogit, NumOps.Multiply(q[i, d], kPos[i, d]));
            }
            posLogit = NumOps.Divide(posLogit, NumOps.FromDouble(_temperature));

            // Compute negative logits: q_i · k-_j / τ for all j
            var negLogits = new T[numNegatives];
            T maxLogit = posLogit;

            for (int j = 0; j < numNegatives; j++)
            {
                T dot = NumOps.Zero;
                for (int d = 0; d < dim; d++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(q[i, d], kNeg[j, d]));
                }
                negLogits[j] = NumOps.Divide(dot, NumOps.FromDouble(_temperature));

                if (NumOps.GreaterThan(negLogits[j], maxLogit))
                    maxLogit = negLogits[j];
            }

            // Compute log-softmax with numerical stability
            T sumExp = NumOps.Exp(NumOps.Subtract(posLogit, maxLogit));
            for (int j = 0; j < numNegatives; j++)
            {
                sumExp = NumOps.Add(sumExp, NumOps.Exp(NumOps.Subtract(negLogits[j], maxLogit)));
            }

            var logSumExp = NumOps.Add(maxLogit, NumOps.Log(sumExp));
            var loss = NumOps.Subtract(logSumExp, posLogit);
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes InfoNCE loss with in-batch negatives (SimCLR style but asymmetric).
    /// </summary>
    /// <param name="queries">Query embeddings [batch_size, dim].</param>
    /// <param name="keys">Key embeddings [batch_size, dim].</param>
    /// <returns>The computed loss value.</returns>
    public T ComputeLossInBatch(Tensor<T> queries, Tensor<T> keys)
    {
        if (queries is null) throw new ArgumentNullException(nameof(queries));
        if (keys is null) throw new ArgumentNullException(nameof(keys));

        var batchSize = queries.Shape[0];
        var dim = queries.Shape[1];

        var q = _normalize ? L2Normalize(queries) : queries;
        var k = _normalize ? L2Normalize(keys) : keys;

        T totalLoss = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            // Compute all logits: q_i · k_j / τ
            var logits = new T[batchSize];
            T maxLogit = NumOps.FromDouble(double.MinValue);

            for (int j = 0; j < batchSize; j++)
            {
                T dot = NumOps.Zero;
                for (int d = 0; d < dim; d++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(q[i, d], k[j, d]));
                }
                logits[j] = NumOps.Divide(dot, NumOps.FromDouble(_temperature));

                if (NumOps.GreaterThan(logits[j], maxLogit))
                    maxLogit = logits[j];
            }

            // Positive is at index i (diagonal)
            T sumExp = NumOps.Zero;
            for (int j = 0; j < batchSize; j++)
            {
                sumExp = NumOps.Add(sumExp, NumOps.Exp(NumOps.Subtract(logits[j], maxLogit)));
            }

            var logSumExp = NumOps.Add(maxLogit, NumOps.Log(sumExp));
            var loss = NumOps.Subtract(logSumExp, logits[i]);
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes InfoNCE loss with in-batch negatives and gradients.
    /// </summary>
    /// <param name="queries">Query embeddings [batch_size, dim].</param>
    /// <param name="keys">Key embeddings [batch_size, dim].</param>
    /// <returns>Tuple of (loss, gradQueries, gradKeys).</returns>
    public (T loss, Tensor<T> gradQueries, Tensor<T> gradKeys) ComputeLossInBatchWithGradients(
        Tensor<T> queries, Tensor<T> keys)
    {
        if (queries is null) throw new ArgumentNullException(nameof(queries));
        if (keys is null) throw new ArgumentNullException(nameof(keys));

        var batchSize = queries.Shape[0];
        var dim = queries.Shape[1];

        var q = _normalize ? L2Normalize(queries) : queries;
        var k = _normalize ? L2Normalize(keys) : keys;

        var gradQ = new T[batchSize * dim];
        var gradK = new T[batchSize * dim];
        T totalLoss = NumOps.Zero;

        var invTemp = NumOps.FromDouble(1.0 / _temperature);

        for (int i = 0; i < batchSize; i++)
        {
            // Compute all logits: q_i · k_j / τ
            var logits = new T[batchSize];
            T maxLogit = NumOps.FromDouble(double.MinValue);

            for (int j = 0; j < batchSize; j++)
            {
                T dot = NumOps.Zero;
                for (int d = 0; d < dim; d++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(q[i, d], k[j, d]));
                }
                logits[j] = NumOps.Multiply(dot, invTemp);

                if (NumOps.GreaterThan(logits[j], maxLogit))
                    maxLogit = logits[j];
            }

            // Compute softmax probabilities
            T sumExp = NumOps.Zero;
            var exps = new T[batchSize];
            for (int j = 0; j < batchSize; j++)
            {
                exps[j] = NumOps.Exp(NumOps.Subtract(logits[j], maxLogit));
                sumExp = NumOps.Add(sumExp, exps[j]);
            }

            // Positive is at index i (diagonal)
            var logSumExp = NumOps.Add(maxLogit, NumOps.Log(sumExp));
            var loss = NumOps.Subtract(logSumExp, logits[i]);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Compute gradients
            for (int j = 0; j < batchSize; j++)
            {
                var prob = NumOps.Divide(exps[j], sumExp);
                var gradScale = j == i
                    ? NumOps.Subtract(prob, NumOps.One)  // p_j - 1 for positive
                    : prob;                               // p_j for negatives

                for (int d = 0; d < dim; d++)
                {
                    // Gradient w.r.t. query_i from logit_j
                    var gradFromLogit = NumOps.Multiply(NumOps.Multiply(gradScale, invTemp), k[j, d]);
                    gradQ[i * dim + d] = NumOps.Add(gradQ[i * dim + d], gradFromLogit);

                    // Gradient w.r.t. key_j from query_i's loss
                    var gradToKey = NumOps.Multiply(NumOps.Multiply(gradScale, invTemp), q[i, d]);
                    gradK[j * dim + d] = NumOps.Add(gradK[j * dim + d], gradToKey);
                }
            }
        }

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        var scale = NumOps.FromDouble(1.0 / batchSize);

        for (int i = 0; i < gradQ.Length; i++)
        {
            gradQ[i] = NumOps.Multiply(gradQ[i], scale);
            gradK[i] = NumOps.Multiply(gradK[i], scale);
        }

        return (avgLoss, new Tensor<T>(gradQ, [batchSize, dim]), new Tensor<T>(gradK, [batchSize, dim]));
    }

    /// <summary>
    /// Computes InfoNCE loss with gradients.
    /// </summary>
    public (T loss, Tensor<T> gradQueries, Tensor<T> gradPositiveKeys) ComputeLossWithGradients(
        Tensor<T> queries, Tensor<T> positiveKeys, Tensor<T> negativeKeys)
    {
        var batchSize = queries.Shape[0];
        var dim = queries.Shape[1];
        var numNegatives = negativeKeys.Shape[0];

        var q = _normalize ? L2Normalize(queries) : queries;
        var kPos = _normalize ? L2Normalize(positiveKeys) : positiveKeys;
        var kNeg = _normalize ? L2Normalize(negativeKeys) : negativeKeys;

        var gradQ = new T[batchSize * dim];
        var gradKPos = new T[batchSize * dim];
        T totalLoss = NumOps.Zero;

        var invTemp = NumOps.FromDouble(1.0 / _temperature);

        for (int i = 0; i < batchSize; i++)
        {
            // Compute logits
            T posLogit = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                posLogit = NumOps.Add(posLogit, NumOps.Multiply(q[i, d], kPos[i, d]));
            }
            posLogit = NumOps.Multiply(posLogit, invTemp);

            var negLogits = new T[numNegatives];
            T maxLogit = posLogit;

            for (int j = 0; j < numNegatives; j++)
            {
                T dot = NumOps.Zero;
                for (int d = 0; d < dim; d++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(q[i, d], kNeg[j, d]));
                }
                negLogits[j] = NumOps.Multiply(dot, invTemp);
                if (NumOps.GreaterThan(negLogits[j], maxLogit))
                    maxLogit = negLogits[j];
            }

            // Softmax
            T posExp = NumOps.Exp(NumOps.Subtract(posLogit, maxLogit));
            T sumExp = posExp;
            var negExps = new T[numNegatives];

            for (int j = 0; j < numNegatives; j++)
            {
                negExps[j] = NumOps.Exp(NumOps.Subtract(negLogits[j], maxLogit));
                sumExp = NumOps.Add(sumExp, negExps[j]);
            }

            var logSumExp = NumOps.Add(maxLogit, NumOps.Log(sumExp));
            totalLoss = NumOps.Add(totalLoss, NumOps.Subtract(logSumExp, posLogit));

            // Gradients
            var posProb = NumOps.Divide(posExp, sumExp);
            var posProbMinusOne = NumOps.Subtract(posProb, NumOps.One);

            for (int d = 0; d < dim; d++)
            {
                // Gradient w.r.t. query
                var gradFromPos = NumOps.Multiply(NumOps.Multiply(posProbMinusOne, invTemp), kPos[i, d]);
                gradQ[i * dim + d] = NumOps.Add(gradQ[i * dim + d], gradFromPos);

                // Add gradients from negatives
                for (int j = 0; j < numNegatives; j++)
                {
                    var negProb = NumOps.Divide(negExps[j], sumExp);
                    var gradFromNeg = NumOps.Multiply(NumOps.Multiply(negProb, invTemp), kNeg[j, d]);
                    gradQ[i * dim + d] = NumOps.Add(gradQ[i * dim + d], gradFromNeg);
                }

                // Gradient w.r.t. positive key
                gradKPos[i * dim + d] = NumOps.Multiply(NumOps.Multiply(posProbMinusOne, invTemp), q[i, d]);
            }
        }

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        var scale = NumOps.FromDouble(1.0 / batchSize);

        for (int i = 0; i < gradQ.Length; i++)
        {
            gradQ[i] = NumOps.Multiply(gradQ[i], scale);
            gradKPos[i] = NumOps.Multiply(gradKPos[i], scale);
        }

        return (avgLoss, new Tensor<T>(gradQ, [batchSize, dim]), new Tensor<T>(gradKPos, [batchSize, dim]));
    }

    /// <summary>
    /// Computes accuracy of the contrastive task (useful for monitoring).
    /// </summary>
    public double ComputeAccuracy(Tensor<T> queries, Tensor<T> positiveKeys, Tensor<T> negativeKeys)
    {
        var batchSize = queries.Shape[0];
        var dim = queries.Shape[1];
        var numNegatives = negativeKeys.Shape[0];

        var q = _normalize ? L2Normalize(queries) : queries;
        var kPos = _normalize ? L2Normalize(positiveKeys) : positiveKeys;
        var kNeg = _normalize ? L2Normalize(negativeKeys) : negativeKeys;

        int correct = 0;

        for (int i = 0; i < batchSize; i++)
        {
            // Compute positive similarity
            T posSim = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                posSim = NumOps.Add(posSim, NumOps.Multiply(q[i, d], kPos[i, d]));
            }

            // Check if positive has highest similarity
            bool isCorrect = true;
            for (int j = 0; j < numNegatives && isCorrect; j++)
            {
                T negSim = NumOps.Zero;
                for (int d = 0; d < dim; d++)
                {
                    negSim = NumOps.Add(negSim, NumOps.Multiply(q[i, d], kNeg[j, d]));
                }

                if (NumOps.GreaterThan(negSim, posSim))
                    isCorrect = false;
            }

            if (isCorrect) correct++;
        }

        return (double)correct / batchSize;
    }

    private Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        var batchSize = tensor.Shape[0];
        var dim = tensor.Shape[1];
        var result = new T[batchSize * dim];

        for (int i = 0; i < batchSize; i++)
        {
            T sumSquared = NumOps.Zero;
            for (int j = 0; j < dim; j++)
            {
                var val = tensor[i, j];
                sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(val, val));
            }

            var norm = NumOps.Sqrt(NumOps.Add(sumSquared, NumOps.FromDouble(1e-8)));

            for (int j = 0; j < dim; j++)
            {
                result[i * dim + j] = NumOps.Divide(tensor[i, j], norm);
            }
        }

        return new Tensor<T>(result, [batchSize, dim]);
    }
}
