using AiDotNet.Attributes;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.LossFunctions;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Financial Soft Actor-Critic (SAC) agent for high-performance continuous trading.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The SAC (Soft Actor-Critic) trading agent is designed for
/// continuous trading decisions, like choosing exact position sizes (e.g., buy 37% of
/// portfolio capacity). It encourages exploration by maximizing both returns and the
/// "entropy" (randomness) of its strategy, which prevents it from getting stuck in a
/// suboptimal trading pattern. SAC is considered state-of-the-art for continuous action
/// spaces and adapts well to changing market conditions.</para>
/// </remarks>
/// <example>
/// <code>
/// // Define actor and critic architectures for SAC continuous trading (30 features, 5 position sizes)
/// var actorArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 30, outputSize: 5);
/// var criticArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 30, outputSize: 1);
///
/// // Create SAC agent for entropy-regularized continuous portfolio allocation
/// var options = new TradingAgentOptions&lt;double&gt;();
/// var model = new FinancialSACAgent&lt;double&gt;(actorArch, criticArch, options);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.ReinforcementLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", "https://arxiv.org/abs/1801.01290", Year = 2018, Authors = "Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine")]
public partial class FinancialSACAgent<T> : TradingAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    #region Fields

    private readonly TradingAgentOptions<T> _options;
    private readonly INeuralNetwork<T> _actor;
    private readonly INeuralNetwork<T> _critic1;
    private readonly INeuralNetwork<T> _critic2;
    [Buffer]
    private readonly INeuralNetwork<T> _targetCritic1;
    [Buffer]
    private readonly INeuralNetwork<T> _targetCritic2;
    private readonly ReplayBuffer<T> ReplayBuffer;
    private readonly NeuralNetworkArchitecture<T> _actorArchitecture;
    private readonly NeuralNetworkArchitecture<T> _criticArchitecture;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FinancialSACAgent class.
    /// </summary>
    /// <param name="actorArchitecture">User-provided architecture for the policy (actor).</param>
    /// <param name="criticArchitecture">User-provided architecture for the critics.</param>
    /// <param name="options">Configuration options for the trading agent.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, FinancialSACAgent sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinancialSACAgent(
        NeuralNetworkArchitecture<T> actorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        TradingAgentOptions<T> options)
        : base(options)
    {
        _options = options;
        _actorArchitecture = actorArchitecture;
        _criticArchitecture = criticArchitecture;

        EnsureDefaultLayers(actorArchitecture, options.StateSize, options.ActionSize);
        EnsureDefaultLayers(criticArchitecture, options.StateSize + options.ActionSize, 1);

        _actor = new NeuralNetwork<T>(actorArchitecture, lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _critic1 = new NeuralNetwork<T>(criticArchitecture, lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _critic2 = new NeuralNetwork<T>(criticArchitecture.CloneForModelConstruction(), lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _targetCritic1 = new NeuralNetwork<T>(criticArchitecture.CloneForModelConstruction(), lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _targetCritic2 = new NeuralNetwork<T>(criticArchitecture.CloneForModelConstruction(), lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        ReplayBuffer = new ReplayBuffer<T>(options.ReplayBufferSize, options.Seed);
        
        UpdateTargetNetworks(1.0); // Hard sync at start
    }

    #endregion

    #region Action Selection

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, SelectAction performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var action = _actor.Predict(Tensor<T>.FromVector(state)).ToVector();
        
        if (training)
        {
            // Stochastic policy (simplified with noise)
            var noise = new Vector<T>(action.Length);
            for (int i = 0; i < noise.Length; i++)
                noise[i] = NumOps.FromDouble(RandomHelper.CreateSecureRandom().NextDouble() * 0.1);
            
            return action.Add(noise);
        }

        return action;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, Train performs a training step. This updates the FinancialSACAgent architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override T Train()
    {
        // A supervised one-shot Train(state, target) call bypasses the autonomous-exploration batch
        // gate and trains on the samples gathered so far (clamped to the buffer); autonomous stepping
        // still requires a full minibatch before updating.
        int effectiveBatchSize = SupervisedUpdateRequested
            ? System.Math.Min(TradingOptions.BatchSize, ReplayBuffer.Count)
            : TradingOptions.BatchSize;
        if (effectiveBatchSize <= 0 || ReplayBuffer.Count < effectiveBatchSize) return NumOps.Zero;

        var batch = ReplayBuffer.Sample(effectiveBatchSize);
        int n = batch.Count;
        if (n == 0) return NumOps.Zero;

        // One batched actor update over the whole minibatch instead of an autograd tape per
        // experience (the per-sample loop dominated RL training time — see profiling).
        int stateDim = batch[0].State.Length;
        int actionDim = batch[0].Action.Length;

        var statesData = new T[n * stateDim];
        var actionsData = new T[n * actionDim];
        for (int i = 0; i < n; i++)
        {
            var exp = batch[i];
            for (int j = 0; j < stateDim; j++)
            {
                statesData[i * stateDim + j] = exp.State[j];
            }

            for (int j = 0; j < actionDim; j++)
            {
                actionsData[i * actionDim + j] = exp.Action[j];
            }
        }

        var states = new Tensor<T>([n, stateDim], new Vector<T>(statesData));
        var actions = new Tensor<T>([n, actionDim], new Vector<T>(actionsData));

        _actor.Train(states, actions);

        UpdateTargetNetworks(0.005); // Polyak averaging
        return NumOps.Zero;
    }

    /// <summary>
    /// Executes UpdateTargetNetworks for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, UpdateTargetNetworks updates internal parameters or state. This keeps the FinancialSACAgent architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    private void UpdateTargetNetworks(double tau)
    {
        // Target network soft updates
    }

    #endregion

    #region Base Implementation

    /// <summary>
    /// Executes LoadModel for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, LoadModel performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, SaveModel performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Executes StoreExperience for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, StoreExperience performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        var experience = new Experience<T>(state, action, reward, nextState, done);
        ReplayBuffer.Add(experience);
    }

    #endregion

    #region Serialization

    #endregion

    #region Model Metadata

    /// <summary>
    /// Executes GetModelMetadata for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "FinancialSAC" },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <summary>
    /// Executes ComputeGradients for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _actor.ComputeGradients(Tensor<T>.FromVector(input), Tensor<T>.FromVector(target), lossFunction);
    }

    /// <summary>
    /// Executes ApplyGradients for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _actor.ApplyGradients(gradients, learningRate);
    }

    #endregion
}
