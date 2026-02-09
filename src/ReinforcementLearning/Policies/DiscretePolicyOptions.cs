using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Configuration options for discrete action space policies in reinforcement learning.
    /// Discrete policies select from a finite set of actions using categorical (softmax) distributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Discrete policies are fundamental to reinforcement learning in environments with finite action spaces,
    /// such as game playing (left/right/jump), robot arm control with discrete positions, or trading decisions
    /// (buy/sell/hold). The policy network outputs logits (unnormalized log probabilities) for each action,
    /// which are then converted to a probability distribution via softmax. Actions are sampled from this
    /// distribution during training to enable exploration, while the most probable action is typically
    /// selected during evaluation.
    /// </para>
    /// <para>
    /// This configuration class provides sensible defaults aligned with modern deep reinforcement learning
    /// best practices from libraries like Stable Baselines3 and RLlib. The default epsilon-greedy exploration
    /// strategy balances exploration (trying random actions) with exploitation (using learned policy).
    /// </para>
    /// <para><b>For Beginners:</b> Discrete policies are for situations where your AI agent must choose
    /// between specific, separate options rather than continuous values.
    ///
    /// Think of it like a video game character deciding between actions:
    /// - Move Left
    /// - Move Right
    /// - Jump
    /// - Duck
    ///
    /// The policy learns which action is best in each situation by:
    /// 1. Looking at the current state (what's on screen)
    /// 2. Calculating probabilities for each action (40% jump, 35% left, 20% right, 5% duck)
    /// 3. Choosing an action based on these probabilities
    ///
    /// During training, it sometimes picks random actions (exploration) to discover new strategies.
    /// During evaluation/playing, it picks the best action it has learned.
    ///
    /// This options class lets you configure:
    /// - How many different actions are available (ActionSize)
    /// - How complex the neural network should be (HiddenLayers)
    /// - How much random exploration to use (ExplorationStrategy)
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    public class DiscretePolicyOptions<T> : ModelOptions
    {
        /// <summary>
        /// Gets or sets the size of the observation/state space.
        /// </summary>
        /// <value>The number of input features that describe the environment state. Must be greater than 0.</value>
        /// <remarks>
        /// <para>
        /// The state size defines the dimensionality of observations from the environment. For example,
        /// in a CartPole environment this might be 4 (cart position, cart velocity, pole angle, pole velocity).
        /// In an Atari game using pixel inputs, this would be the flattened image size or number of features
        /// extracted from preprocessing.
        /// </para>
        /// <para><b>For Beginners:</b> This is how many numbers describe "what's happening" in your environment.
        ///
        /// Examples:
        /// - Simple game: 4 numbers (player X, player Y, enemy X, enemy Y)
        /// - Chess board: 64 squares × types of pieces = hundreds of features
        /// - Robot arm: 6 numbers (one for each joint angle)
        ///
        /// Set this to match your environment's observation space size.
        /// </para>
        /// </remarks>
        public int StateSize { get; set; } = 0;

        /// <summary>
        /// Gets or sets the number of discrete actions available to the agent.
        /// </summary>
        /// <value>The number of distinct actions the agent can choose from. Must be greater than 0.</value>
        /// <remarks>
        /// <para>
        /// This defines the output size of the policy network and the dimensionality of the action probability
        /// distribution. Common values range from 2 (binary decisions) to hundreds (complex action spaces like
        /// language models). The network outputs logits for each action, which are converted to probabilities
        /// via softmax.
        /// </para>
        /// <para><b>For Beginners:</b> How many different actions can your agent choose from?
        ///
        /// Examples:
        /// - Trading bot: 3 actions (buy, sell, hold)
        /// - Pac-Man: 4 actions (up, down, left, right)
        /// - Fighting game: 12 actions (punch, kick, block, move in 4 directions, etc.)
        ///
        /// More actions make learning harder because the agent has more to explore.
        /// Start simple with fewer actions when possible.
        /// </para>
        /// </remarks>
        public int ActionSize { get; set; } = 0;

        /// <summary>
        /// Gets or sets the architecture of hidden layers in the policy network.
        /// </summary>
        /// <value>An array where each element specifies the number of neurons in that hidden layer.
        /// Defaults to [128, 128] for a two-layer network with 128 neurons each.</value>
        /// <remarks>
        /// <para>
        /// The hidden layer configuration determines the network's capacity to learn complex policies.
        /// Deeper networks (more layers) can learn more complex relationships but are harder to train
        /// and slower to execute. Wider networks (more neurons per layer) increase capacity without
        /// adding depth. The default [128, 128] works well for many problems including Atari games
        /// and robotic control tasks. For simple problems (like CartPole), [64] may suffice. For
        /// complex problems (like Go or high-dimensional robotics), consider [256, 256, 256] or larger.
        /// </para>
        /// <para><b>For Beginners:</b> This controls how "smart" your neural network can be.
        ///
        /// The default [128, 128] means:
        /// - Your network has 2 hidden layers
        /// - Each layer has 128 artificial neurons
        /// - This creates a network like: Input → [128 neurons] → [128 neurons] → Output
        ///
        /// Think of layers like levels of thinking:
        /// - First layer: Recognizes basic patterns ("is enemy close?")
        /// - Second layer: Combines patterns into strategies ("enemy close + have weapon = attack")
        ///
        /// You might want more layers/neurons [256, 256, 256] if:
        /// - Your problem is very complex (chess, robot navigation)
        /// - Simple networks aren't learning well
        /// - You have lots of training data and computing power
        ///
        /// You might want fewer [64] or [64, 64] if:
        /// - Your problem is simple (tic-tac-toe, balancing a pole)
        /// - Training is too slow
        /// - You're just experimenting
        ///
        /// Good rule of thumb: Start with the default and adjust based on results.
        /// </para>
        /// </remarks>
        public int[] HiddenLayers { get; set; } = new int[] { 128, 128 };

        /// <summary>
        /// Gets or sets the loss function used to train the policy network.
        /// </summary>
        /// <value>The loss function for computing training error. Defaults to Mean Squared Error.</value>
        /// <remarks>
        /// <para>
        /// The loss function quantifies how well the policy's predictions match the target values during
        /// training. For policy gradient methods (PPO, A2C), this is typically used for value function
        /// approximation or advantage estimation. Mean Squared Error is the standard choice as it provides
        /// stable gradients and works well with continuous value predictions. Some advanced algorithms may
        /// benefit from Huber loss for robustness to outliers.
        /// </para>
        /// <para><b>For Beginners:</b> The loss function measures "how wrong" the policy is during learning.
        ///
        /// The default Mean Squared Error (MSE) works by:
        /// - Taking the difference between predicted and actual values
        /// - Squaring it (so negatives don't cancel positives)
        /// - Averaging across all examples
        ///
        /// You almost never need to change this from the default. MSE is the industry standard
        /// and works well for reinforcement learning. Only consider alternatives if you're implementing
        /// advanced research algorithms or experiencing specific training instabilities.
        /// </para>
        /// </remarks>
        public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

        /// <summary>
        /// Gets or sets the exploration strategy for balancing exploration vs exploitation during training.
        /// </summary>
        /// <value>The exploration strategy. Defaults to epsilon-greedy with decaying epsilon from 1.0 to 0.01.</value>
        /// <remarks>
        /// <para>
        /// Exploration is critical in reinforcement learning because the agent must try different actions
        /// to discover which ones lead to high rewards. Epsilon-greedy exploration randomly selects actions
        /// with probability ε (epsilon), and follows the learned policy with probability 1-ε. The epsilon
        /// typically starts high (e.g., 1.0 for 100% random) and gradually decreases (to 0.01 for 1% random)
        /// as the agent gains experience. Alternative strategies include Boltzmann (softmax) exploration,
        /// or no exploration for pure exploitation.
        /// </para>
        /// <para><b>For Beginners:</b> Exploration means trying new things instead of always doing what
        /// you think is best.
        ///
        /// The default epsilon-greedy strategy works like this:
        /// - Start of training: 100% random actions (explore everything!)
        /// - Middle of training: Mix of random and learned actions
        /// - End of training: 99% learned actions, 1% random (mostly exploit what you know)
        ///
        /// Think of learning to play a new video game:
        /// - First hour: Press random buttons to see what they do (high exploration)
        /// - After some practice: Mostly use moves you know work, occasionally try something new
        /// - Expert level: Almost always use best strategies, rarely experiment
        ///
        /// You might want different exploration if:
        /// - Your environment is very random → Keep higher exploration longer
        /// - Your environment is very predictable → Reduce exploration faster
        /// - You're fine-tuning a pre-trained model → Start with low exploration
        ///
        /// Available strategies:
        /// - EpsilonGreedyExploration (default): Simple, effective for discrete actions
        /// - BoltzmannExploration: Temperature-based, good for multi-armed bandits
        /// - NoExploration: For evaluation or when using off-policy algorithms
        /// </para>
        /// </remarks>
        public IExplorationStrategy<T> ExplorationStrategy { get; set; } = new EpsilonGreedyExploration<T>();

        /// <summary>
        /// Gets or sets the random seed for reproducible training runs.
        /// </summary>
        /// <value>Optional random seed. When null, uses a random seed. When set to a value, ensures deterministic behavior.</value>
        /// <remarks>
        /// <para>
        /// Setting a specific seed value ensures that training runs are reproducible, which is essential
        /// for debugging, comparing algorithms, and scientific research. However, in production or when
        /// seeking diverse solutions, using null (random seed) allows for variation across runs that
        /// might discover better policies. Note that reproducibility also requires deterministic environment
        /// implementations and consistent hardware/software configurations.
        /// </para>
        /// <para><b>For Beginners:</b> Random seed controls whether your training is the same every time.
        ///
        /// - Set to a number (e.g., 42): Training will be identical each time you run it
        /// - Set to null (default): Each training run will be different
        ///
        /// Use a fixed seed when:
        /// - Debugging (you want to see the exact same behavior)
        /// - Comparing algorithms (fair comparison requires same randomness)
        /// - Publishing research (others should be able to reproduce your results)
        ///
        /// Use null (random) when:
        /// - Training multiple models to pick the best one
        /// - You want variation in learned behaviors
        /// - Running in production where diversity is valuable
        ///
        /// Common practice: Use seed=42 during development, null in production.
        /// </para>
        /// </remarks>
        public new int? Seed { get; set; } = null;
    }
}
