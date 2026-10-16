using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

/// <summary>
/// One optimizer step a model declares instead of writing its own training loop: the step's data and the loss built
/// from it. The network base class packs the data, runs the step on the shared fused session (the compiled plan when
/// it applies, the shared eager tape otherwise) and restores whatever the step changed.
/// </summary>
/// <remarks>
/// <para>
/// A model declares its objectives in <c>NeuralNetworkBase.DefineTrainingObjectives</c>. A model with one network
/// and one loss returns one objective. A model that steps several networks with their own optimizers (a GAN's critic
/// and generator) returns one per network, in the order they step.
/// </para>
/// <para>
/// Everything that changes from step to step (noise, timesteps, masks, an adjacency matrix, labels) goes in
/// <see cref="Data"/>. The base class packs it into the step's input, so a compiled plan replays each step on that
/// step's own data. <see cref="Loss"/> receives views of the packed tensors in the same order and shapes; any other
/// host value it reads is frozen into the plan at trace time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class TrainingObjective<T>
{
    /// <summary>Creates an objective over the given per-step data.</summary>
    /// <param name="data">The tensors that change from step to step.</param>
    /// <param name="loss">Builds the scalar loss from views of <paramref name="data"/>, with engine operations.</param>
    public TrainingObjective(IReadOnlyList<Tensor<T>> data, Func<IReadOnlyList<Tensor<T>>, Tensor<T>> loss)
    {
        Guard.NotNull(data);
        Guard.NotNull(loss);
        if (data.Count == 0)
            throw new ArgumentException("A training objective needs at least one data tensor.", nameof(data));
        _data = data;
        Loss = loss;
    }

    /// <summary>
    /// Creates an objective whose data is drawn when its step runs: after the objectives declared before it have
    /// stepped. A GAN's critic step draws its fake samples from the generator as it stands at that step.
    /// </summary>
    /// <param name="drawData">Draws the step's data; called once, right before the step.</param>
    /// <param name="loss">Builds the scalar loss from views of the drawn data, with engine operations.</param>
    public TrainingObjective(Func<IReadOnlyList<Tensor<T>>> drawData, Func<IReadOnlyList<Tensor<T>>, Tensor<T>> loss)
    {
        Guard.NotNull(drawData);
        Guard.NotNull(loss);
        _drawData = drawData;
        Loss = loss;
    }

    private IReadOnlyList<Tensor<T>>? _data;
    private readonly Func<IReadOnlyList<Tensor<T>>>? _drawData;

    /// <summary>
    /// The tensors that change from step to step, in the order <see cref="Loss"/> receives them. Drawn on first read
    /// for an objective created with a draw function.
    /// </summary>
    public IReadOnlyList<Tensor<T>> Data
    {
        get
        {
            if (_data is null && _drawData is not null)
            {
                var drawn = _drawData() ?? throw new InvalidOperationException("A training objective's draw returned null.");
                if (drawn.Count == 0)
                    throw new InvalidOperationException("A training objective needs at least one data tensor.");
                _data = drawn;
            }

            return _data ?? Array.Empty<Tensor<T>>();
        }
    }

    /// <summary>Builds the scalar loss from views of <see cref="Data"/>.</summary>
    public Func<IReadOnlyList<Tensor<T>>, Tensor<T>> Loss { get; }

    /// <summary>The optimizer for this step, or null for the model's training optimizer.</summary>
    public IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer { get; init; }

    /// <summary>
    /// The tensors this step trains, or null for the model's whole trainable set. Read after the first forward, so
    /// lazily shaped layers are included.
    /// </summary>
    public Func<IReadOnlyList<Tensor<T>>>? Parameters { get; init; }

    /// <summary>Runs after the step, whether it succeeded or not: restores state the loss changed.</summary>
    public Action? AfterStep { get; init; }

    /// <summary>
    /// Receives the step's loss after a successful step: loss histories, and updates that follow an optimizer step
    /// (weight clipping, a moving average).
    /// </summary>
    public Action<T>? OnLoss { get; init; }

    /// <summary>
    /// The network whose fused session runs this step, or null for the model that declared it. A model that steps several
    /// networks (a GAN's critic and generator) gives each objective its own network, so each keeps its own compiled plan
    /// and optimizer state.
    /// </summary>
    public AiDotNet.NeuralNetworks.NeuralNetworkBase<T>? Network { get; init; }
}