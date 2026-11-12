using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

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
/// <para>Positions are the support INPUTS (the mean of each example's input features), so attention depends on
/// x_i - x_j and the refinement is translation equivariant by construction; the paper uses no regularizer for
/// it (<c>EquivarianceRegWeight</c> is an optional extra, 0 by default).</para>
/// <para><b>Reference:</b> Ashman et al., "Translation-Equivariant Transformer Neural Processes", ICML 2024.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Translation-Equivariant Transformer Neural Processes",
    "https://arxiv.org/abs/2406.12409",
    Year = 2024,
    Authors = "Matthew Ashman, Cristiana Diaconu, Junhyuck Kim, Lakee Sivaraya, Stratis Markou, James Requeima, " +
              "Wessel P. Bruinsma, Richard E. Turner")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class TETNPAlgorithm<T, TInput, TOutput> : NeuralProcessBase<T, TInput, TOutput>
{
    private readonly TETNPOptions<T, TInput, TOutput> _algoOptions;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _relPosParams;
    private readonly int _numBands;
    private readonly int _numHeads;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.TETNP;

    public TETNPAlgorithm(TETNPOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
        if (options.NumFrequencyBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumFrequencyBands must be positive.");

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
        var initParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            InterfaceGuard.Parameterizable(MetaModel).SetParameters(initParams);
            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var supportLabels = ConvertToVector(task.SupportOutput);

            // 1. Encode context pairs into per-example representations
            var contextReps = BuildContextRepresentations(supportFeatures, supportLabels);

            // 2. Refine representations with relative-position self-attention
            var refinedReps = ApplyRelativeSelfAttention(contextReps, task.SupportInput);

            // 3. Aggregate and modulate
            var aggRep = AggregateRepresentations(refinedReps);
            ModulateParameters(initParams, ComputeModScale(aggRep));

            // 4. Query loss (+ the optional norm-variance regularizer, off by default)
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(WithRegularizer(queryLoss, contextReps, refinedReps));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref EncoderParams, _algoOptions.OuterLearningRate, ComputeAuxLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _relPosParams, _algoOptions.OuterLearningRate, ComputeAuxLoss);
        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();
        var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
        var supportLabels = ConvertToVector(task.SupportOutput);

        var contextReps = BuildContextRepresentations(supportFeatures, supportLabels);
        var refinedReps = ApplyRelativeSelfAttention(contextReps, task.SupportInput);
        var aggRep = AggregateRepresentations(refinedReps);

        return new NeuralProcessModel<T, TInput, TOutput>(MetaModel, ScaleVector(currentParams, ComputeModScale(aggRep)), aggRep);
    }

    /// <summary>
    /// Applies multi-head self-attention with relative positional encoding.
    /// Attention score between i,j uses sinusoidal encoding of (x_i - x_j).
    /// </summary>
    /// <remarks>
    /// score_h(i, j) = (r_i^h . r_j^h + w_h . [delta, sin(2^b pi delta), cos(2^b pi delta)]_b) / sqrt(headDim + relDim)
    /// with delta = x_i - x_j; out_i = r_i + 0.5 * concat_h(sum_j softmax_j(score_h) r_j^h).
    /// </remarks>
    internal List<Vector<T>> ApplyRelativeSelfAttention(List<Vector<T>> reps, TInput supportInput)
    {
        int n = reps.Count;
        if (n <= 1) return reps;
        var positions = InputPositions(supportInput, n);
        if (positions is null) return reps;

        int dim = RepresentationDim;
        int heads = Math.Min(_numHeads, dim);
        int headDim = Math.Max(1, dim / _numHeads);
        int relDim = 1 + 2 * _numBands;

        var stacked = new Tensor<T>[n];
        for (int i = 0; i < n; i++) stacked[i] = Tensor<T>.FromVector(Fit(reps[i], dim));
        var r = Engine.TensorStack(stacked, 0);

        // delta[i, j] = x_i - x_j, then the relative encoding [delta, sin(f_b delta), cos(f_b delta)] on the last axis.
        var column = Engine.TensorBroadcastTo(positions.Reshape(new[] { n, 1 }), new[] { n, n });
        var row = Engine.TensorBroadcastTo(positions.Reshape(new[] { 1, n }), new[] { n, n });
        var delta = Engine.TensorSubtract(column, row);
        // Frequencies f_b = 2^b pi = pi exp(b ln 2); sin and cos interleave per band: [sin_0, cos_0, sin_1, ...].
        var frequencies = Engine.TensorMultiplyScalar(
            Engine.TensorExp(Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, _numBands, 1), NumOps.FromDouble(Math.Log(2.0)))),
            NumOps.FromDouble(Math.PI));
        var scaled = Engine.TensorMultiply(
            Engine.TensorBroadcastTo(delta.Reshape(new[] { n, n, 1 }), new[] { n, n, _numBands }),
            Engine.TensorBroadcastTo(frequencies.Reshape(new[] { 1, 1, _numBands }), new[] { n, n, _numBands }));
        var waves = Engine.TensorStack(new[] { Engine.TensorSin(scaled), Engine.TensorCos(scaled) }, 3)
            .Reshape(new[] { n, n, 2 * _numBands });
        var encoding = Engine.TensorConcatenate(new[] { delta.Reshape(new[] { n, n, 1 }), waves }, 2)
            .Reshape(new[] { n * n, relDim });
        var weights = Engine.TensorTranspose(Tensor<T>.FromVector(
            _relPosParams.GetSubVector(0, _numHeads * relDim), new[] { _numHeads, relDim }));
        var relative = Engine.TensorMatMul(encoding, weights);    // [n*n, numHeads]

        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(headDim + relDim));
        var headOutputs = new Tensor<T>[heads + 1];
        int used = 0;
        for (int h = 0; h < heads; h++)
        {
            int width = Math.Min(headDim, dim - h * headDim);
            if (width <= 0) break;
            var rh = Engine.TensorSlice(r, new[] { 0, h * headDim }, new[] { n, width });
            var content = Engine.TensorMatMul(rh, Engine.TensorTranspose(rh));
            var rel = Engine.TensorSlice(relative, new[] { 0, h }, new[] { n * n, 1 }).Reshape(new[] { n, n });
            var attention = Engine.TensorSoftmax(Engine.TensorMultiplyScalar(Engine.TensorAdd(content, rel), scale), 1);
            headOutputs[used++] = Engine.TensorMatMul(attention, rh);
        }

        int covered = Math.Min(dim, used * headDim);
        if (covered < dim) headOutputs[used++] = new Tensor<T>(new[] { n, dim - covered });
        var attended = Engine.TensorConcatenate(headOutputs.Take(used).ToArray(), 1);
        var output = Engine.TensorAdd(r, Engine.TensorMultiplyScalar(attended, NumOps.FromDouble(0.5)));

        var refined = new List<Vector<T>>(n);
        for (int i = 0; i < n; i++)
            refined.Add(Engine.TensorSlice(output, new[] { i, 0 }, new[] { 1, dim }).ToVector());
        return refined;
    }

    /// <summary>The position of each support example: the mean of its input features. Null if counts disagree.</summary>
    private static Tensor<T>? InputPositions(TInput supportInput, int n)
    {
        Tensor<T> x;
        try
        {
            x = ConversionsHelper.ConvertToTensor<T>(supportInput as object ?? throw new ArgumentNullException(nameof(supportInput)));
        }
        catch (InvalidOperationException)
        {
            return null;
        }

        if (x.Rank == 0 || x.Shape[0] != n) return null;
        int features = x.Length / n;
        return Engine.ReduceMean(x.Reshape(new[] { n, features }), new[] { 1 }, keepDims: false);
    }

    /// <summary>The first <paramref name="dim"/> entries, zero-padded if shorter.</summary>
    private static Vector<T> Fit(Vector<T> v, int dim)
        => v.Length == dim ? v
            : v.Length > dim ? v.GetSubVector(0, dim)
            : Vector<T>.Concatenate(new[] { v, new Vector<T>(dim - v.Length) });

    /// <summary>
    /// Optional regularizer: the variance of the representations' RMS norms. Not part of TE-TNP, which is
    /// equivariant by construction.
    /// </summary>
    private T ComputeEquivarianceReg(List<Vector<T>> original, List<Vector<T>> refined)
    {
        if (refined.Count <= 1) return NumOps.Zero;
        int dim = RepresentationDim;
        var rows = new Tensor<T>[refined.Count];
        for (int i = 0; i < refined.Count; i++) rows[i] = Tensor<T>.FromVector(Fit(refined[i], dim));
        var stacked = Engine.TensorStack(rows, 0);
        var norms = Engine.TensorSqrt(Engine.ReduceMean(Engine.TensorSquare(stacked), new[] { 1 }, keepDims: false));
        var centered = Engine.TensorSubtract(norms,
            Engine.TensorBroadcastTo(Engine.ReduceMean(norms, new[] { 0 }, keepDims: true), new[] { refined.Count }));
        return Engine.TensorMean(Engine.TensorSquare(centered));
    }

    /// <summary>queryLoss + EquivarianceRegWeight * the optional regularizer.</summary>
    private T WithRegularizer(T queryLoss, List<Vector<T>> contextReps, List<Vector<T>> refinedReps)
        => NumOps.Add(queryLoss, NumOps.Multiply(
            NumOps.FromDouble(_algoOptions.EquivarianceRegWeight), ComputeEquivarianceReg(contextReps, refinedReps)));

    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> tb)
    {
        var ip = InterfaceGuard.Parameterizable(MetaModel).GetParameters();
        var losses = new List<T>();
        foreach (var t in tb.Tasks)
        {
            InterfaceGuard.Parameterizable(MetaModel).SetParameters(ip);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(t.QueryInput), t.QueryOutput);

            // Include the optional regularizer so SPSA optimizes the same objective as training.
            var sf = ConvertToVector(MetaModel.Predict(t.SupportInput));
            var sl = ConvertToVector(t.SupportOutput);
            if (sf != null && sl != null && sf.Length > 0 && sl.Length > 0)
            {
                var contextReps = BuildContextRepresentations(sf, sl);
                var refinedReps = ApplyRelativeSelfAttention(contextReps, t.SupportInput);
                queryLoss = WithRegularizer(queryLoss, contextReps, refinedReps);
            }

            losses.Add(queryLoss);
        }
        InterfaceGuard.Parameterizable(MetaModel).SetParameters(ip);
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
