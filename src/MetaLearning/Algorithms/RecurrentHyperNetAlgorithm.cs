using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Recurrent HyperNetwork for meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// A GRU-like recurrent cell processes compressed gradient information at each adaptation
/// step, maintaining hidden state that captures the optimization trajectory. The recurrent
/// output is used to compute per-parameter learning rate modulation factors, enabling
/// adaptive step sizes that evolve through the inner loop based on gradient history.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// h_0 = 0  (zero-initialized hidden state)
///
/// For each inner step t:
///   x_t = compress(grad_t)                               →  [InputDim]
///   z_t = sigmoid(W_z · [h_{t-1}, x_t] + ForgetBias)    (update gate)
///   r_t = sigmoid(W_r · [h_{t-1}, x_t])                  (reset gate)
///   h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])             (candidate)
///   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t             (new state)
///
///   Modulation: m_d = 1 + tanh(h_t[d % HiddenDim])
///   θ_d ← θ_d - η * m_d * grad_d
///
/// L_meta = L_query + CellRegWeight * Σ h_T²
/// Outer: update θ via meta-gradient, update GRU weights via SPSA
/// </code>
/// </para>
/// </remarks>
public class RecurrentHyperNetAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly RecurrentHyperNetOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>GRU weights: W_z, W_r, W_h — each (hidDim + inputDim) × hidDim.</summary>
    private Vector<T> _gruWeights;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.RecurrentHyperNet;

    public RecurrentHyperNetAlgorithm(RecurrentHyperNetOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;

        int hidDim = options.HiddenStateDim;
        int inDim = options.InputDim;
        int concatDim = hidDim + inDim;
        // 3 gates (z, r, h̃), each concatDim × hidDim
        int totalWeights = 3 * concatDim * hidDim;
        _gruWeights = new Vector<T>(totalWeights);
        double scale = 1.0 / Math.Sqrt(concatDim);
        for (int i = 0; i < totalWeights; i++)
            _gruWeights[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();
        int hidDim = _algoOptions.HiddenStateDim;

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            // Initialize hidden state
            var hidden = new double[hidDim];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Compress gradient to input dim
                var input = CompressGradient(grad);

                // GRU step
                hidden = GRUStep(hidden, input);

                // Compute per-parameter modulation from hidden state
                for (int d = 0; d < _paramDim; d++)
                {
                    int h = d % hidDim;
                    double modulation = 1.0 + Math.Tanh(hidden[h]); // [0, 2] range
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * modulation * NumOps.ToDouble(grad[d])));
                }
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Cell state regularization
            double cellNorm = 0;
            for (int h = 0; h < hidDim; h++) cellNorm += hidden[h] * hidden[h];
            var totalLoss = NumOps.Add(queryLoss,
                NumOps.FromDouble(_algoOptions.CellRegWeight * cellNorm / hidDim));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _gruWeights, _algoOptions.OuterLearningRate * 0.1, ComputeRecurrentLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        int hidDim = _algoOptions.HiddenStateDim;
        var hidden = new double[hidDim];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var input = CompressGradient(grad);
            hidden = GRUStep(hidden, input);

            for (int d = 0; d < _paramDim; d++)
            {
                int h = d % hidDim;
                double modulation = 1.0 + Math.Tanh(hidden[h]);
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * modulation * NumOps.ToDouble(grad[d])));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private double[] CompressGradient(Vector<T> grad)
    {
        int inDim = _algoOptions.InputDim;
        var result = new double[inDim];
        int bucketSize = Math.Max(1, _paramDim / inDim);
        for (int e = 0; e < inDim; e++)
        {
            double sum = 0;
            int start = e * bucketSize;
            for (int d = start; d < start + bucketSize && d < grad.Length; d++)
                sum += NumOps.ToDouble(grad[d]);
            result[e] = Math.Tanh(sum / bucketSize);
        }
        return result;
    }

    private double[] GRUStep(double[] prevHidden, double[] input)
    {
        int hidDim = _algoOptions.HiddenStateDim;
        int inDim = _algoOptions.InputDim;
        int concatDim = hidDim + inDim;

        // Concatenate [h, x]
        var concat = new double[concatDim];
        Array.Copy(prevHidden, 0, concat, 0, hidDim);
        Array.Copy(input, 0, concat, hidDim, inDim);

        int gateSize = concatDim * hidDim;

        // Update gate: z = sigmoid(W_z · [h, x] + forgetBias)
        var z = new double[hidDim];
        for (int h = 0; h < hidDim; h++)
        {
            double sum = _algoOptions.ForgetBias;
            for (int c = 0; c < concatDim; c++)
                sum += NumOps.ToDouble(_gruWeights[h * concatDim + c]) * concat[c];
            z[h] = 1.0 / (1.0 + Math.Exp(-sum));
        }

        // Reset gate: r = sigmoid(W_r · [h, x])
        var r = new double[hidDim];
        int rOffset = gateSize;
        for (int h = 0; h < hidDim; h++)
        {
            double sum = 0;
            for (int c = 0; c < concatDim; c++)
                sum += NumOps.ToDouble(_gruWeights[rOffset + h * concatDim + c]) * concat[c];
            r[h] = 1.0 / (1.0 + Math.Exp(-sum));
        }

        // Candidate: h̃ = tanh(W_h · [r ⊙ h, x])
        var resetConcat = new double[concatDim];
        for (int h = 0; h < hidDim; h++) resetConcat[h] = r[h] * prevHidden[h];
        Array.Copy(input, 0, resetConcat, hidDim, inDim);

        var candidate = new double[hidDim];
        int hOffset = 2 * gateSize;
        for (int h = 0; h < hidDim; h++)
        {
            double sum = 0;
            for (int c = 0; c < concatDim; c++)
                sum += NumOps.ToDouble(_gruWeights[hOffset + h * concatDim + c]) * resetConcat[c];
            candidate[h] = Math.Tanh(sum);
        }

        // New hidden: h_new = (1 - z) ⊙ h_prev + z ⊙ h̃
        var newHidden = new double[hidDim];
        for (int h = 0; h < hidDim; h++)
            newHidden[h] = (1.0 - z[h]) * prevHidden[h] + z[h] * candidate[h];

        return newHidden;
    }

    private double ComputeRecurrentLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        int hidDim = _algoOptions.HiddenStateDim;
        foreach (var task in taskBatch.Tasks)
        {
            var ap = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) ap[d] = initParams[d];
            var hid = new double[hidDim];
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                var inp = CompressGradient(g);
                hid = GRUStep(hid, inp);
                for (int d = 0; d < _paramDim; d++)
                {
                    double mod = 1.0 + Math.Tanh(hid[d % hidDim]);
                    ap[d] = NumOps.Subtract(ap[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * mod * NumOps.ToDouble(g[d])));
                }
            }
            MetaModel.SetParameters(ap);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
