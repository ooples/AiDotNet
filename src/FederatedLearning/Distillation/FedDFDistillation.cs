using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Distillation;

/// <summary>
/// FedDF — Federated ensemble distillation using model averaging on unlabeled public data.
/// </summary>
/// <remarks>
/// <para>
/// FedDF (Lin et al., 2020) performs federated distillation by treating each client model
/// as an ensemble member. The server distills the ensemble's collective predictions on
/// unlabeled data into a single global model. Unlike FedMD, FedDF works with completely
/// different model architectures and does not require labeled public data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as a "committee vote." Each client model votes on
/// what the answer should be for some shared unlabeled examples. The server then trains
/// a fresh model to match the committee's averaged predictions. This means each client
/// can use a completely different type of model.
/// </para>
/// <para>
/// Reference: Lin et al. (2020), "Ensemble Distillation for Robust Model Fusion in Federated Learning".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FedDFDistillation<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedDistillationStrategy<T>
{
    private readonly double _temperature;
    private readonly int _serverDistillationEpochs;
    private readonly double _serverLearningRate;

    /// <summary>
    /// Creates a new FedDF distillation strategy.
    /// </summary>
    /// <param name="temperature">Softmax temperature for ensemble averaging. Default: 3.0.</param>
    /// <param name="serverDistillationEpochs">Number of server-side distillation epochs. Default: 10.</param>
    /// <param name="serverLearningRate">Server-side learning rate. Default: 0.01.</param>
    public FedDFDistillation(double temperature = 3.0, int serverDistillationEpochs = 10,
        double serverLearningRate = 0.01)
    {
        _temperature = temperature;
        _serverDistillationEpochs = serverDistillationEpochs;
        _serverLearningRate = serverLearningRate;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Returns the client's logits f(x̂_k, d) on the public data. Lin et al. (2020), Algorithm 1, averages the
    /// client logits and applies the temperature softmax only inside the distillation loss.
    /// </remarks>
    public Matrix<T> ExtractKnowledge(Vector<T> localModelParameters, Matrix<T>? publicData)
    {
        Guard.NotNull(localModelParameters);
        if (publicData is null)
            throw new ArgumentException("FedDF requires unlabeled public data.", nameof(publicData));
        ValidateInputs(localModelParameters, publicData);

        return ComputeLogits(localModelParameters, publicData, out _).ToMatrix();
    }

    /// <inheritdoc/>
    /// <remarks>Weighted average of the client logits (weights normalized over the participating clients).</remarks>
    public Matrix<T> AggregateKnowledge(Dictionary<int, Matrix<T>> clientKnowledge, Dictionary<int, double>? clientWeights)
    {
        Guard.NotNull(clientKnowledge);
        if (clientKnowledge.Count == 0)
            throw new ArgumentException("No client knowledge provided.", nameof(clientKnowledge));

        var first = clientKnowledge.Values.First();
        int n = first.Rows;
        int outputDim = first.Columns;

        double totalWeight = clientKnowledge.Keys.Sum(id => clientWeights?.GetValueOrDefault(id, 1.0) ?? 1.0);
        if (totalWeight <= 0)
            throw new ArgumentException("Client weights must sum to a positive value.", nameof(clientWeights));

        // Ensemble average: uniform or weighted combination of client logits.
        var ensemble = new Tensor<T>(new[] { n, outputDim });
        foreach (var (clientId, knowledge) in clientKnowledge)
        {
            if (knowledge.Rows != n || knowledge.Columns != outputDim)
            {
                throw new ArgumentException(
                    $"Client {clientId} knowledge is {knowledge.Rows}x{knowledge.Columns}; expected {n}x{outputDim}.",
                    nameof(clientKnowledge));
            }

            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            ensemble = Engine.TensorAdd(ensemble,
                Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(knowledge), NumOps.FromDouble(w / totalWeight)));
        }

        return ensemble.ToMatrix();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Runs gradient descent on KL(σ(ensemble / τ) || σ(f(x, d) / τ)) averaged over the public samples. With
    /// logits z = X Wᵀ, the gradient with respect to W is (σ(z / τ) − σ(ensemble / τ))ᵀ X / (n τ). A positive
    /// <paramref name="temperature"/> overrides the constructor temperature.
    /// </remarks>
    public Vector<T> ApplyKnowledge(Vector<T> localModelParameters, Matrix<T> globalKnowledge,
        Matrix<T>? publicData, double temperature)
    {
        Guard.NotNull(localModelParameters);
        Guard.NotNull(globalKnowledge);
        if (publicData is null)
            throw new ArgumentException("FedDF requires unlabeled public data.", nameof(publicData));
        ValidateInputs(localModelParameters, publicData);

        double tau = temperature > 0 ? temperature : _temperature;
        if (!(tau > 0) || double.IsInfinity(tau))
            throw new ArgumentOutOfRangeException(nameof(temperature), "Distillation temperature must be positive and finite.");

        int d = localModelParameters.Length;
        int n = publicData.Rows;
        int outputDim = OutputDimension(d);
        if (globalKnowledge.Rows != n || globalKnowledge.Columns != outputDim)
        {
            throw new ArgumentException(
                $"Global knowledge is {globalKnowledge.Rows}x{globalKnowledge.Columns}; expected {n}x{outputDim}.",
                nameof(globalKnowledge));
        }

        var data = Tensor<T>.FromMatrix(publicData);
        var invTau = NumOps.FromDouble(1.0 / tau);
        var target = Engine.Softmax(Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(globalKnowledge), invTau), 1);
        var step = NumOps.FromDouble(_serverLearningRate / (n * tau));

        var parameters = localModelParameters.Clone();
        for (int epoch = 0; epoch < _serverDistillationEpochs; epoch++)
        {
            var logits = ComputeLogits(parameters, publicData, out var weightIndex);
            var student = Engine.Softmax(Engine.TensorMultiplyScalar(logits, invTau), 1);

            // dKL/dz = (p - q) / τ, averaged over samples, chained through z = X Wᵀ.
            var gradLogits = Engine.TensorMultiplyScalar(Engine.TensorSubtract(student, target), step);
            var gradWeights = Engine.TensorMatMul(Engine.TensorTranspose(gradLogits), data);
            var gradParams = Engine.TensorIndexAdd(new Tensor<T>(new[] { d }), 0, weightIndex,
                gradWeights.Reshape(new[] { outputDim * publicData.Columns }));
            parameters = Engine.Subtract(parameters, gradParams.ToVector());
        }

        return parameters;
    }

    private static int OutputDimension(int parameterCount) => Math.Max(2, (int)Math.Sqrt(parameterCount));

    private static void ValidateInputs(Vector<T> parameters, Matrix<T> publicData)
    {
        if (parameters.Length == 0)
            throw new ArgumentException("Model parameters cannot be empty.", nameof(parameters));
        if (publicData.Rows == 0 || publicData.Columns == 0)
            throw new ArgumentException("Public data cannot be empty.", nameof(publicData));
    }

    /// <summary>
    /// Linear projection logits z = X Wᵀ, where W[j, k] = θ[(j·C + k) mod d] for C public-data columns.
    /// </summary>
    private Tensor<T> ComputeLogits(Vector<T> parameters, Matrix<T> publicData, out Tensor<int> weightIndex)
    {
        int d = parameters.Length;
        int columns = publicData.Columns;
        int outputDim = OutputDimension(d);

        var index = new int[outputDim * columns];
        for (int k = 0; k < index.Length; k++)
        {
            index[k] = k % d;
        }

        weightIndex = new Tensor<int>(index, new[] { index.Length });
        var weights = Engine.TensorGather(parameters.AsTensor(), weightIndex, 0).Reshape(new[] { outputDim, columns });
        return Engine.TensorMatMul(Tensor<T>.FromMatrix(publicData), Engine.TensorTranspose(weights));
    }
}
