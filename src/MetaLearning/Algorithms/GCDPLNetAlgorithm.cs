using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of GCDPLNet: Graph-based Cross-Domain Prototype Learning Network.
/// </summary>
/// <remarks>
/// <para>
/// GCDPLNet treats parameter groups as nodes in a graph and uses learned attention-based
/// message passing to propagate adaptation signals between groups. Each group computes a
/// gradient feature, then message passing allows information from related groups to influence
/// adaptation — enabling cross-domain knowledge transfer through parameter-space structure.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Parameter groups: G_1..G_N (equal-size blocks)
/// Node features: h_g = mean(|grad_g|)  (per-group gradient magnitude)
///
/// Message passing (L rounds):
///   attention: α_{g,g'} = softmax_g'(W_attn * [h_g, h_g'])
///   message: m_g = Σ_{g'} α_{g,g'} * h_g'
///   update: h_g = tanh(h_g + MessageWeight * m_g)
///
/// Adaptive LR per group: η_g = η * (1 + h_g)
///
/// Inner loop: θ_d ← θ_d - η_{group(d)} * grad_d
/// Outer loop: update θ, W_attn (via SPSA)
/// </code>
/// </para>
/// </remarks>
public class GCDPLNetAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly GCDPLNetOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _numNodes;
    private readonly int _nodeSize;

    /// <summary>Graph attention parameters: numNodes × numNodes (pairwise attention).</summary>
    private Vector<T> _graphAttention;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.GCDPLNet;

    public GCDPLNetAlgorithm(GCDPLNetOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _numNodes = Math.Max(1, options.NumGraphNodes);
        _nodeSize = (_paramDim + _numNodes - 1) / _numNodes;

        _graphAttention = new Vector<T>(_numNodes * _numNodes);
        double scale = 1.0 / Math.Sqrt(_numNodes);
        for (int i = 0; i < _graphAttention.Length; i++)
            _graphAttention[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Compute per-group features (mean absolute gradient)
                var nodeFeatures = ComputeNodeFeatures(grad);

                // Message passing
                for (int round = 0; round < _algoOptions.MessagePassingSteps; round++)
                    nodeFeatures = MessagePass(nodeFeatures);

                // Apply graph-informed adaptive learning rate
                for (int d = 0; d < _paramDim; d++)
                {
                    int g = Math.Min(d / _nodeSize, _numNodes - 1);
                    double adaptiveLR = _algoOptions.InnerLearningRate * (1.0 + nodeFeatures[g]);
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(adaptiveLR * NumOps.ToDouble(grad[d])));
                }
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _graphAttention, _algoOptions.OuterLearningRate * 0.1, ComputeGraphLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var nodeFeatures = ComputeNodeFeatures(grad);
            for (int round = 0; round < _algoOptions.MessagePassingSteps; round++)
                nodeFeatures = MessagePass(nodeFeatures);

            for (int d = 0; d < _paramDim; d++)
            {
                int g = Math.Min(d / _nodeSize, _numNodes - 1);
                double adaptiveLR = _algoOptions.InnerLearningRate * (1.0 + nodeFeatures[g]);
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(adaptiveLR * NumOps.ToDouble(grad[d])));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private double[] ComputeNodeFeatures(Vector<T> grad)
    {
        var features = new double[_numNodes];
        for (int g = 0; g < _numNodes; g++)
        {
            double sum = 0;
            int count = 0;
            int start = g * _nodeSize;
            for (int d = start; d < start + _nodeSize && d < _paramDim; d++)
            {
                sum += Math.Abs(NumOps.ToDouble(grad[d]));
                count++;
            }
            features[g] = count > 0 ? Math.Tanh(sum / count) : 0;
        }
        return features;
    }

    private double[] MessagePass(double[] features)
    {
        var updated = new double[_numNodes];
        for (int g = 0; g < _numNodes; g++)
        {
            // Compute attention weights from this node to all others
            var scores = new double[_numNodes];
            double maxScore = double.NegativeInfinity;
            for (int g2 = 0; g2 < _numNodes; g2++)
            {
                scores[g2] = NumOps.ToDouble(_graphAttention[g * _numNodes + g2])
                           * (features[g] + features[g2]);
                if (scores[g2] > maxScore) maxScore = scores[g2];
            }
            double sumExp = 0;
            for (int g2 = 0; g2 < _numNodes; g2++) { scores[g2] = Math.Exp(scores[g2] - maxScore); sumExp += scores[g2]; }
            for (int g2 = 0; g2 < _numNodes; g2++) scores[g2] /= (sumExp + 1e-10);

            // Weighted message
            double message = 0;
            for (int g2 = 0; g2 < _numNodes; g2++) message += scores[g2] * features[g2];

            updated[g] = Math.Tanh(features[g] + _algoOptions.MessageWeight * message);
        }
        return updated;
    }

    private double ComputeGraphLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var ap = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) ap[d] = initParams[d];
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                var nf = ComputeNodeFeatures(g);
                for (int r = 0; r < _algoOptions.MessagePassingSteps; r++) nf = MessagePass(nf);
                for (int d = 0; d < _paramDim; d++)
                {
                    int grp = Math.Min(d / _nodeSize, _numNodes - 1);
                    ap[d] = NumOps.Subtract(ap[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * (1.0 + nf[grp]) * NumOps.ToDouble(g[d])));
                }
            }
            MetaModel.SetParameters(ap);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
