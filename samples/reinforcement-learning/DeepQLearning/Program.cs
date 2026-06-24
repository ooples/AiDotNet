// AiDotNet — Deep Q-Learning (DQN)
//
// Trains a DQN agent on the CartPole environment through the AiModelBuilder
// facade: ConfigureModel(agent) supplies the IRLAgent, ConfigureReinforcement
// Learning(options) supplies the environment + training schedule, and BuildAsync
// runs the episode loop. Greedy action selection flows through the returned
// AiModelResult via result.Predict(state).

using AiDotNet;
using AiDotNet.Configuration;                          // RLTrainingOptions
using AiDotNet.Models.Options;                         // DQNOptions
using AiDotNet.ReinforcementLearning.Agents.DQN;       // DQNAgent
using AiDotNet.ReinforcementLearning.Environments;     // CartPoleEnvironment
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Deep Q-Learning (DQN) ===");
Console.WriteLine("Training a DQN agent on CartPole via the AiModelBuilder facade\n");

const int stateSize = 4;    // CartPole: cart position, cart velocity, pole angle, pole angular velocity
const int actionSize = 2;   // push left / push right
const int episodes = 25;

var env = new CartPoleEnvironment<double>(maxSteps: 200, seed: 42);
var agent = new DQNAgent<double>(new DQNOptions<double>
{
    StateSize = stateSize,
    ActionSize = actionSize,
    LearningRate = 0.001,
    DiscountFactor = 0.99,
    HiddenLayers = new List<int> { 64, 64 },
    BatchSize = 32,
    WarmupSteps = 100,
    ReplayBufferSize = 10_000,
    EpsilonDecay = 0.95
});

Console.WriteLine($"Environment: CartPole (state {stateSize}, actions {actionSize})");
Console.WriteLine("Agent: DQN, hidden layers [64, 64]\n");

var rlOptions = new RLTrainingOptions<double>
{
    Environment = env,
    Episodes = episodes,
    MaxStepsPerEpisode = 200,
    WarmupSteps = 100,
    BatchSize = 32,
    Seed = 42,
    OnEpisodeComplete = m =>
    {
        if (m.Episode == 1 || m.Episode % 5 == 0 || m.Episode == episodes)
            Console.WriteLine($"  Episode {m.Episode,3}: reward={m.TotalReward,6:F1}  recent avg={m.AverageRewardRecent,6:F1}  steps={m.Steps}");
    }
};

Console.WriteLine($"Training {episodes} episodes through AiModelBuilder.ConfigureReinforcementLearning ...\n");
try
{
    // RL agents are IFullModel<T, Vector<T>, Vector<T>>, so the builder is typed
    // over Vector<T> and the agent is supplied via ConfigureModel.
    var result = await new AiModelBuilder<double, Vector<double>, Vector<double>>()
        .ConfigureModel(agent)
        .ConfigureReinforcementLearning(rlOptions)
        .BuildAsync();

    Console.WriteLine("\nTraining complete.");

    // Greedy Q-values for the initial state, through the result object.
    var state = env.Reset();
    var qValues = result.Predict(state);
    int bestAction = 0;
    for (int i = 1; i < qValues.Length; i++)
        if (qValues[i] > qValues[bestAction]) bestAction = i;
    var formatted = Enumerable.Range(0, qValues.Length).Select(i => qValues[i].ToString("F3"));
    Console.WriteLine($"  Q-values for the initial state: [{string.Join(", ", formatted)}]");
    Console.WriteLine($"  Greedy action: {bestAction}");
}
catch (Exception ex)
{
    Console.WriteLine($"  RL training reported: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");
