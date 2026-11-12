using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Distillation;

/// <summary>
/// FedMD — Model-agnostic federated learning via mutual distillation on a public dataset.
/// </summary>
/// <remarks>
/// <para>
/// FedMD (Li &amp; Wang, 2019) enables heterogeneous model architectures across clients.
/// Each client computes logits on a shared public dataset, the server averages the logits,
/// and clients distill the consensus logits into their local models.
/// </para>
/// <para>
/// <b>For Beginners:</b> All clients are given the same "test questions" (public dataset).
/// Each client submits their answers (predicted probabilities). The server averages all
/// answers to get a "consensus answer," then each client learns from this consensus.
/// This way, a simple model on a phone can learn from a powerful model on a server.
/// </para>
/// <para>
/// Reference: Li &amp; Wang (2019), "FedMD: Heterogeneous Federated Learning via Model Distillation".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FedMDDistillation<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedDistillationStrategy<T>
{
    private readonly double _temperature;
    private readonly double _distillationAlpha;
    private readonly int _distillationEpochs;
    private readonly double _learningRate;

    /// <summary>
    /// Creates a new FedMD distillation strategy.
    /// </summary>
    /// <param name="temperature">Softmax temperature for soft labels. Default: 3.0.</param>
    /// <param name="distillationAlpha">Weight for distillation loss vs task loss. Default: 0.5.</param>
    /// <param name="distillationEpochs">Number of local distillation epochs. Default: 5.</param>
    /// <param name="learningRate">Learning rate for distillation updates. Default: 0.01.</param>
    public FedMDDistillation(double temperature = 3.0, double distillationAlpha = 0.5,
        int distillationEpochs = 5, double learningRate = 0.01)
    {
        _temperature = temperature;
        _distillationAlpha = distillationAlpha;
        _distillationEpochs = distillationEpochs;
        _learningRate = learningRate;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Class scores z / τ with z = X Wᵀ, where W[j, k] = θ[j·C + k] for the C public-data columns while
    /// j·C + k &lt; d, and 0 beyond the parameter count.
    /// </remarks>
    public Matrix<T> ExtractKnowledge(Vector<T> localModelParameters, Matrix<T>? publicData)
    {
        Guard.NotNull(localModelParameters);
        if (publicData is null)
            throw new ArgumentException("FedMD requires a public dataset for logit extraction.", nameof(publicData));
        ValidateInputs(localModelParameters, publicData);

        // Compute logits: simple linear projection of public data through model parameters
        // In production, this would use the actual model forward pass
        return ComputeScaledLogits(localModelParameters, publicData, out _, out _).ToMatrix();
    }

    /// <inheritdoc/>
    public Matrix<T> AggregateKnowledge(Dictionary<int, Matrix<T>> clientKnowledge, Dictionary<int, double>? clientWeights)
    {
        Guard.NotNull(clientKnowledge);
        if (clientKnowledge.Count == 0)
            throw new ArgumentException("No client knowledge provided.", nameof(clientKnowledge));

        // Get dimensions from first client
        var first = clientKnowledge.Values.First();
        int n = first.Rows;
        int outputDim = first.Columns;

        double totalWeight = clientKnowledge.Keys.Sum(id => clientWeights?.GetValueOrDefault(id, 1.0) ?? 1.0);
        if (totalWeight <= 0)
            throw new ArgumentException("Client weights must sum to a positive value.", nameof(clientWeights));

        // Weighted average of all client logits (weights normalized by their total).
        var aggregated = new Tensor<T>(new[] { n, outputDim });
        foreach (var (clientId, knowledge) in clientKnowledge)
        {
            if (knowledge.Rows != n || knowledge.Columns != outputDim)
            {
                throw new ArgumentException(
                    $"Client {clientId} knowledge is {knowledge.Rows}x{knowledge.Columns}; expected {n}x{outputDim}.",
                    nameof(clientKnowledge));
            }

            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            aggregated = Engine.TensorAdd(aggregated,
                Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(knowledge), NumOps.FromDouble(w / totalWeight)));
        }

        return aggregated.ToMatrix();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The digest step: gradient descent on α/(2n) Σ ‖z/τ − consensus‖² over the public samples, so
    /// ∇W = α (z/τ − consensus)ᵀ X / (n τ), scattered into the parameters W was read from. The scores and the
    /// consensus share the constructor temperature, so <paramref name="temperature"/> is not used.
    /// </remarks>
    public Vector<T> ApplyKnowledge(Vector<T> localModelParameters, Matrix<T> globalKnowledge,
        Matrix<T>? publicData, double temperature)
    {
        Guard.NotNull(localModelParameters);
        Guard.NotNull(globalKnowledge);
        if (publicData is null)
            throw new ArgumentException("FedMD requires a public dataset.", nameof(publicData));
        ValidateInputs(localModelParameters, publicData);

        int d = localModelParameters.Length;
        int n = publicData.Rows;
        int columns = publicData.Columns;
        int outputDim = OutputDimension(d);
        if (globalKnowledge.Rows != n || globalKnowledge.Columns != outputDim)
        {
            throw new ArgumentException(
                $"Global knowledge is {globalKnowledge.Rows}x{globalKnowledge.Columns}; expected {n}x{outputDim}.",
                nameof(globalKnowledge));
        }

        var data = Tensor<T>.FromMatrix(publicData);
        var consensus = Tensor<T>.FromMatrix(globalKnowledge);
        var step = NumOps.FromDouble(_learningRate * _distillationAlpha / (n * _temperature));

        // Distillation: move the local scores toward the global consensus
        var parameters = localModelParameters.Clone();
        for (int epoch = 0; epoch < _distillationEpochs; epoch++)
        {
            var scores = ComputeScaledLogits(parameters, publicData, out var weightIndex, out var weightMask);
            var residual = Engine.TensorMultiplyScalar(Engine.TensorSubtract(scores, consensus), step);
            var gradWeights = Engine.TensorMultiply(Engine.TensorMatMul(Engine.TensorTranspose(residual), data), weightMask);
            var gradParams = Engine.TensorIndexAdd(new Tensor<T>(new[] { d }), 0, weightIndex,
                gradWeights.Reshape(new[] { outputDim * columns }));
            parameters = Engine.Subtract(parameters, gradParams.ToVector());
        }

        return parameters;
    }

    private static int OutputDimension(int parameterCount) => Math.Max(1, parameterCount / 10); // Heuristic: output dim is a fraction of params

    private static void ValidateInputs(Vector<T> parameters, Matrix<T> publicData)
    {
        if (parameters.Length == 0)
            throw new ArgumentException("Model parameters cannot be empty.", nameof(parameters));
        if (publicData.Rows == 0 || publicData.Columns == 0)
            throw new ArgumentException("Public data cannot be empty.", nameof(publicData));
    }

    private Tensor<T> ComputeScaledLogits(Vector<T> parameters, Matrix<T> publicData, out Tensor<int> weightIndex, out Tensor<T> weightMask)
    {
        int d = parameters.Length;
        int columns = publicData.Columns;
        int outputDim = OutputDimension(d);

        var index = new int[outputDim * columns];
        var mask = new int[outputDim * columns];
        for (int k = 0; k < index.Length; k++)
        {
            bool inRange = k < d;
            index[k] = inRange ? k : 0;
            mask[k] = inRange ? 1 : 0;
        }

        weightIndex = new Tensor<int>(index, new[] { index.Length });
        weightMask = new Tensor<int>(mask, new[] { outputDim, columns }).Cast<T>();
        var weights = Engine.TensorMultiply(
            Engine.TensorGather(parameters.AsTensor(), weightIndex, 0).Reshape(new[] { outputDim, columns }), weightMask);

        // Apply temperature scaling
        var logits = Engine.TensorMatMul(Tensor<T>.FromMatrix(publicData), Engine.TensorTranspose(weights));
        return Engine.TensorMultiplyScalar(logits, NumOps.FromDouble(1.0 / _temperature));
    }
}
