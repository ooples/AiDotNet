using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Translation-Equivariant Transformer Neural Process (TE-TNP, 2024).
/// Combines TNP's transformer attention with relative positional encoding so that
/// predictions are equivariant to translations of the input space.
/// </summary>
/// <remarks>
/// <para><b>Key Idea:</b> Standard TNP uses absolute positions in attention. TE-TNP replaces
/// position keys/queries with sinusoidal encodings of the <em>relative</em> displacement between
/// context and target points: PE(x_i - x_j). This makes the mapping equivariant to input translations.</para>
/// <para><b>Algorithm:</b>
/// <list type="number">
/// <item>Encode each context pair (x, y) into a representation r_i via the base encoder.</item>
/// <item>Compute pairwise relative position encodings between all context points.</item>
/// <item>Apply multi-head self-attention using relative positional keys to refine representations.</item>
/// <item>Aggregate refined representations and modulate model parameters.</item>
/// </list>
/// </para>
/// <para><b>Reference:</b> Gridded Transformer Neural Processes for Large Unstructured Spatio-Temporal Data (2024).</para>
/// </remarks>
public class TETNPAlgorithm<T, TInput, TOutput> : NeuralProcessBase<T, TInput, TOutput>
{
    private readonly TETNPOptions<T, TInput, TOutput> _algoOptions;
    private Vector<T> _relPosParams;
    private readonly int _numBands;
    private readonly int _numHeads;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.TETNP;

    public TETNPAlgorithm(TETNPOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
        _algoOptions = options;
        _numBands = options.NumFrequencyBands;
        _numHeads = Math.Max(1, options.NumHeads);
        // Relative position parameters: per-head scoring weights over the encoding dimension
        int relEncodingDim = 1 + 2 * _numBands; // raw delta + sin/cos pairs
        _relPosParams = InitializeParams(_numHeads * relEncodingDim);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);
            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var supportLabels = ConvertToVector(task.SupportOutput);

            // 1. Encode context pairs into per-example representations
            var contextReps = EncodeContextSet(supportFeatures, supportLabels);

            // 2. Refine representations with relative-position self-attention
            var refinedReps = ApplyRelativeSelfAttention(contextReps, supportFeatures, supportLabels);

            // 3. Aggregate and modulate
            var aggRep = AggregateRepresentations(refinedReps);
            double scale = ComputeModScale(aggRep);
            var modParams = new Vector<T>(initParams.Length);
            for (int i = 0; i < initParams.Length; i++)
                modParams[i] = NumOps.Multiply(initParams[i], NumOps.FromDouble(scale));
            MetaModel.SetParameters(modParams);

            // 4. Query loss + equivariance regularization
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            double equivReg = ComputeEquivarianceReg(contextReps, refinedReps);
            double totalLoss = NumOps.ToDouble(queryLoss) + _algoOptions.EquivarianceRegWeight * equivReg;
            losses.Add(NumOps.FromDouble(totalLoss));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
            MetaModel.SetParameters(ApplyGradients(initParams, AverageVectors(metaGradients), _algoOptions.OuterLearningRate));

        UpdateAuxiliaryParamsSPSA(taskBatch, ref EncoderParams, _algoOptions.OuterLearningRate, ComputeAuxLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _relPosParams, _algoOptions.OuterLearningRate, ComputeAuxLoss);
        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();
        var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
        var supportLabels = ConvertToVector(task.SupportOutput);

        var contextReps = EncodeContextSet(supportFeatures, supportLabels);
        var refinedReps = ApplyRelativeSelfAttention(contextReps, supportFeatures, supportLabels);
        var aggRep = AggregateRepresentations(refinedReps);
        double sc = ComputeModScale(aggRep);
        var modParams = new Vector<T>(currentParams.Length);
        for (int i = 0; i < currentParams.Length; i++)
            modParams[i] = NumOps.Multiply(currentParams[i], NumOps.FromDouble(sc));

        return new NeuralProcessModel<T, TInput, TOutput>(MetaModel, modParams, aggRep);
    }

    /// <summary>Encodes support features/labels into per-example context representations.</summary>
    private List<Vector<T>> EncodeContextSet(Vector<T>? supportFeatures, Vector<T>? supportLabels)
    {
        var contextReps = new List<Vector<T>>();
        if (supportFeatures == null || supportLabels == null || supportFeatures.Length == 0)
            return contextReps;

        int numEx = Math.Max(1, supportLabels.Length);
        int fDim = Math.Max(1, supportFeatures.Length / numEx);
        for (int i = 0; i < numEx; i++)
        {
            int fStart = i * fDim;
            int fLen = Math.Min(fDim, supportFeatures.Length - fStart);
            if (fLen <= 0) break;
            var f = new Vector<T>(fLen);
            for (int j = 0; j < fLen; j++) f[j] = supportFeatures[fStart + j];
            var l = new Vector<T>(1);
            l[0] = supportLabels[Math.Min(i, supportLabels.Length - 1)];
            contextReps.Add(EncodeContextPair(f, l));
        }
        return contextReps;
    }

    /// <summary>
    /// Applies multi-head self-attention with relative positional encoding.
    /// Attention score between i,j uses sinusoidal encoding of (x_i - x_j).
    /// </summary>
    private List<Vector<T>> ApplyRelativeSelfAttention(
        List<Vector<T>> reps, Vector<T>? supportFeatures, Vector<T>? supportLabels)
    {
        int n = reps.Count;
        if (n <= 1 || supportFeatures == null || supportLabels == null)
            return reps;

        int numEx = Math.Max(1, supportLabels.Length);
        int fDim = Math.Max(1, supportFeatures.Length / numEx);
        int relDim = 1 + 2 * _numBands;
        int headDim = Math.Max(1, RepresentationDim / _numHeads);

        // Extract scalar position for each example (mean of features)
        var positions = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            int start = i * fDim;
            int count = 0;
            for (int j = 0; j < fDim && start + j < supportFeatures.Length; j++)
            {
                sum += NumOps.ToDouble(supportFeatures[start + j]);
                count++;
            }
            positions[i] = count > 0 ? sum / count : 0;
        }

        var refined = new List<Vector<T>>(n);
        for (int i = 0; i < n; i++)
        {
            var output = new Vector<T>(RepresentationDim);
            // Copy original representation as residual
            for (int d = 0; d < RepresentationDim && d < reps[i].Length; d++)
                output[d] = reps[i][d];

            // Multi-head relative attention
            for (int h = 0; h < _numHeads; h++)
            {
                // Compute attention weights for head h
                double[] attnLogits = new double[n];
                double maxLogit = double.NegativeInfinity;

                for (int j = 0; j < n; j++)
                {
                    double delta = positions[i] - positions[j];
                    // Sinusoidal relative positional encoding
                    double score = 0;
                    int pBase = h * relDim;
                    // Raw delta term
                    score += delta * NumOps.ToDouble(_relPosParams[pBase % _relPosParams.Length]);
                    // Sinusoidal terms
                    for (int b = 0; b < _numBands; b++)
                    {
                        double freq = Math.Pow(2.0, b) * Math.PI;
                        int sinIdx = (pBase + 1 + 2 * b) % _relPosParams.Length;
                        int cosIdx = (pBase + 2 + 2 * b) % _relPosParams.Length;
                        score += Math.Sin(freq * delta) * NumOps.ToDouble(_relPosParams[sinIdx]);
                        score += Math.Cos(freq * delta) * NumOps.ToDouble(_relPosParams[cosIdx]);
                    }
                    // Content-based attention dot product for this head
                    int hStart = h * headDim;
                    for (int d = 0; d < headDim && hStart + d < RepresentationDim; d++)
                    {
                        int idx = hStart + d;
                        if (idx < reps[i].Length && idx < reps[j].Length)
                            score += NumOps.ToDouble(reps[i][idx]) * NumOps.ToDouble(reps[j][idx]);
                    }
                    score /= Math.Sqrt(headDim + relDim);
                    attnLogits[j] = score;
                    if (score > maxLogit) maxLogit = score;
                }

                // Softmax
                double sumExp = 0;
                for (int j = 0; j < n; j++)
                {
                    attnLogits[j] = Math.Exp(attnLogits[j] - maxLogit);
                    sumExp += attnLogits[j];
                }
                if (sumExp > 0)
                    for (int j = 0; j < n; j++) attnLogits[j] /= sumExp;

                // Weighted sum of values for this head → add to output (residual)
                int hOut = h * headDim;
                for (int d = 0; d < headDim && hOut + d < RepresentationDim; d++)
                {
                    double val = 0;
                    for (int j = 0; j < n; j++)
                    {
                        int idx = hOut + d;
                        if (idx < reps[j].Length)
                            val += attnLogits[j] * NumOps.ToDouble(reps[j][idx]);
                    }
                    int outIdx = hOut + d;
                    output[outIdx] = NumOps.Add(output[outIdx], NumOps.FromDouble(val * 0.5));
                }
            }
            refined.Add(output);
        }
        return refined;
    }

    /// <summary>
    /// Equivariance regularization: penalizes variance in representation norms
    /// to encourage consistent behavior under translations.
    /// </summary>
    private double ComputeEquivarianceReg(List<Vector<T>> original, List<Vector<T>> refined)
    {
        if (refined.Count <= 1) return 0;
        // Variance of L2 norms — equivariant representations should have similar norms
        double mean = 0;
        var norms = new double[refined.Count];
        for (int i = 0; i < refined.Count; i++)
        {
            double sq = 0;
            for (int d = 0; d < refined[i].Length; d++)
                sq += NumOps.ToDouble(refined[i][d]) * NumOps.ToDouble(refined[i][d]);
            norms[i] = Math.Sqrt(sq / Math.Max(refined[i].Length, 1));
            mean += norms[i];
        }
        mean /= refined.Count;
        double var = 0;
        for (int i = 0; i < refined.Count; i++)
            var += (norms[i] - mean) * (norms[i] - mean);
        return var / refined.Count;
    }

    private double ComputeModScale(Vector<T> rep)
    {
        double norm = 0;
        for (int i = 0; i < rep.Length; i++) norm += NumOps.ToDouble(rep[i]) * NumOps.ToDouble(rep[i]);
        norm = Math.Sqrt(norm / Math.Max(rep.Length, 1));
        return 0.5 + 0.5 / (1.0 + Math.Exp(-norm + 1.0));
    }

    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> tb)
    {
        var ip = MetaModel.GetParameters();
        double total = 0;
        foreach (var t in tb.Tasks)
        {
            MetaModel.SetParameters(ip);
            total += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(t.QueryInput), t.QueryOutput));
        }
        MetaModel.SetParameters(ip);
        return total / Math.Max(tb.Tasks.Length, 1);
    }
}
