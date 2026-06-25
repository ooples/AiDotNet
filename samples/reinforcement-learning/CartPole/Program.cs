// AiDotNet — CartPole with PPO
//
// Trains a PPO agent on the CartPole environment through the AiModelBuilder
// facade: ConfigureModel(agent) supplies the IRLAgent, ConfigureReinforcement
// Learning(options) supplies the environment + schedule, and BuildAsync runs the
// episode loop. Greedy action selection flows through result.Predict(state).

using AiDotNet;
using AiDotNet.Configuration;                       // RLTrainingOptions
using AiDotNet.Models.Options;                      // PPOOptions
using AiDotNet.ReinforcementLearning.Agents.PPO;    // PPOAgent
using AiDotNet.ReinforcementLearning.Environments;  // CartPoleEnvironment
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet CartPole RL ===");
Console.WriteLine("Training a PPO agent to balance a pole on a cart via the facade\n");

const int stateSize = 4;    // cart position, cart velocity, pole angle, pole angular velocity
const int actionSize = 2;   // push left / push right
const int episodes = 25;

var env = new CartPoleEnvironment<double>(maxSteps: 200, seed: 42);
var agent = new PPOAgent<double>(new PPOOptions<double>
{
    StateSize = stateSize,
    ActionSize = actionSize,
    PolicyLearningRate = 3e-4,
    ValueLearningRate = 1e-3,
    DiscountFactor = 0.99,
    GaeLambda = 0.95,
    ClipEpsilon = 0.2,
    PolicyHiddenLayers = new List<int> { 64, 64 }
});

Console.WriteLine($"Environment: CartPole (state {stateSize}, actions {actionSize})");
Console.WriteLine("Agent: PPO, policy hidden layers [64, 64]\n");

var rlOptions = new RLTrainingOptions<double>
{
    Environment = env,
    Episodes = episodes,
    MaxStepsPerEpisode = 200,
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
    var result = await new AiModelBuilder<double, Vector<double>, Vector<double>>()
        .ConfigureModel(agent)
        .ConfigureReinforcementLearning(rlOptions)
        .BuildAsync();

    Console.WriteLine("\nTraining complete.");

    // Action distribution for the initial state, through the result object.
    var state = env.Reset();
    var action = result.Predict(state);
    int best = 0;
    for (int i = 1; i < action.Length; i++)
        if (action[i] > action[best]) best = i;
    Console.WriteLine($"  Policy output for the initial state: shape [{action.Length}]  ->  action {best}");
}
catch (Exception ex)
{
    // Surface failures so the samples CI catches broken RL/facade wiring instead of passing.
    Console.Error.WriteLine($"  RL sample failed: {ex.Message}");
    throw;
}

Console.WriteLine("\n=== Sample Complete ===");
