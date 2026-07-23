namespace AiDotNet.Models.Options;

using AiDotNet.ActivationFunctions;
using AiDotNet.LossFunctions;

/// <summary>
/// Configuration options for Multilayer Perceptron (MLP), a type of feedforward artificial neural
/// network that consists of multiple layers of neurons.
/// </summary>
/// <remarks>
/// <para>
/// The Multilayer Perceptron is a versatile neural network architecture capable of learning complex
/// non-linear relationships between inputs and outputs. It consists of an input layer, one or more hidden
/// layers, and an output layer. Each neuron in a layer is connected to all neurons in the next layer,
/// forming a fully connected network. The MLP learns through a process called backpropagation, where the
/// network parameters are adjusted to minimize a loss function using gradient-based optimization techniques.
/// This class provides comprehensive configuration options for the network architecture, training process,
/// activation functions, and optimization strategy.
/// </para>
/// <para><b>For Beginners:</b> A Multilayer Perceptron (MLP) is a basic type of neural network that
/// can learn to recognize patterns and make predictions from data.
/// 
/// Think of an MLP like a system of interconnected filters that work together:
/// - The input layer receives your data (like the temperature, humidity, and pressure for weather prediction)
/// - The hidden layers process this information through a series of transformations
/// - The output layer provides the prediction (like "chance of rain: 70%")
/// 
/// As the network trains, it gradually adjusts thousands of internal settings (weights) to get better
/// at making accurate predictions. This process is similar to how a child learns to recognize animals:
/// at first they make many mistakes, but with each example, they get better at identifying the patterns
/// that distinguish a cat from a dog.
/// 
/// This class lets you configure every aspect of your neural network: how many layers it has, how it learns,
/// how quickly it adapts, and much more. The default settings provide a good starting point, but you may
/// need to adjust them based on your specific problem.
/// </para>
/// </remarks>
public class MultilayerPerceptronOptions<T, TInput, TOutput> : NonLinearRegressionOptions
{
    /// <summary>
    /// Gets or sets the sizes of each layer in the neural network, including input, hidden, and output layers.
    /// </summary>
    /// <value>A list of integers representing the number of neurons in each layer, defaulting to [1, 10, 1].</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the architecture of the neural network by specifying how many neurons are in each layer.
    /// The first element represents the input layer size (number of features), the last element represents the
    /// output layer size (number of target variables), and all elements in between represent the sizes of
    /// hidden layers. The default value creates a network with 1 input feature, 1 hidden layer with 10 neurons,
    /// and 1 output variable. The depth and width of the network should be chosen based on the complexity of
    /// the problem and the amount of available training data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the structure of your neural network -
    /// how many "neurons" are in each layer and how many layers you have.
    /// 
    /// Imagine building a factory assembly line:
    /// - The first number is how many inputs your data has (like 4 if you have height, weight, age, and blood pressure)
    /// - The middle numbers represent your "hidden layers" (the internal processing stages)
    /// - The last number is how many outputs you want (like 1 for a yes/no prediction, or 3 for classifying into three categories)
    /// 
    /// The default value [1, 10, 1] means:
    /// - 1 input feature (very simple data)
    /// - 1 hidden layer with 10 neurons (moderate processing capacity)
    /// - 1 output value (single prediction or measurement)
    /// 
    /// You should change this based on your specific data:
    /// - The first number should match the number of features in your input data
    /// - The last number should match how many values you're trying to predict
    /// - The middle numbers control the network's learning capacity:
    ///   - More/larger hidden layers = more learning capacity but requires more data and time
    ///   - Fewer/smaller hidden layers = learns faster but might miss complex patterns
    /// 
    /// For complex problems, you might use something like [50, 100, 50, 10, 3], which has 50 inputs,
    /// 3 hidden layers (with 100, 50, and 10 neurons), and 3 outputs.
    /// </para>
    /// </remarks>
    public List<int> LayerSizes { get; set; } = new List<int> { 1, 10, 1 };  // Default: 1 input, 1 hidden layer with 10 neurons, 1 output

    /// <summary>
    /// Gets or sets the maximum number of complete passes through the training dataset.
    /// </summary>
    /// <value>The maximum number of epochs, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// An epoch represents one complete pass through the entire training dataset. This parameter sets the
    /// maximum number of epochs the training process will perform. The actual training might terminate
    /// earlier if other stopping criteria are met, such as reaching a target error threshold or detecting
    /// overfitting through validation. More epochs allow the model more opportunities to learn from the
    /// training data but increase the risk of overfitting and computational cost.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many times the neural network will
    /// process your entire dataset during training.
    /// 
    /// Think of it like practicing for a music recital:
    /// - Each "epoch" is like practicing the entire piece from start to finish
    /// - More practice sessions generally lead to better performance
    /// - But too much practice might lead to memorization rather than understanding
    /// 
    /// The default value of 1000 means the algorithm will go through your entire dataset up to 1000 times.
    /// 
    /// You might want to increase this value if:
    /// - Your network is complex and learning slowly
    /// - You have a large dataset with lots of variation
    /// - You're using techniques to prevent overfitting
    /// 
    /// You might want to decrease this value if:
    /// - Your network seems to be memorizing the training data
    /// - Training is taking too long
    /// - You're doing initial experimentation
    /// 
    /// In practice, neural networks are often trained with early stopping mechanisms that monitor
    /// performance on validation data and stop training when improvement plateaus, regardless of
    /// whether this maximum has been reached.
    /// </para>
    /// </remarks>
    public int MaxEpochs { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the number of training examples used in each parameter update step.
    /// </summary>
    /// <value>The batch size, defaulting to 32.</value>
    /// <remarks>
    /// <para>
    /// The batch size determines how many training examples are processed before the model parameters
    /// are updated. When set to 1, this becomes stochastic gradient descent (updating after each example).
    /// When set to the size of the training set, this becomes batch gradient descent (updating after
    /// seeing all examples). Mini-batch training (values between these extremes) is often the most efficient
    /// approach, balancing the stability of batch updates with the speed of stochastic updates. The optimal
    /// batch size depends on the specific problem, hardware constraints, and the size of the training dataset.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many examples the network looks at before
    /// making each adjustment to its internal settings.
    /// 
    /// Imagine learning to cook a new dish:
    /// - BatchSize = 1: You taste and adjust seasoning after each ingredient (frequent but potentially erratic adjustments)
    /// - BatchSize = 32: You add 32 ingredients, then taste and adjust (more stable but less frequent adjustments)
    /// - BatchSize = [entire recipe]: You only taste and adjust after completing the whole recipe (very stable but only one chance to adjust)
    /// 
    /// The default value of 32 works well for many problems because:
    /// - It's large enough to provide somewhat stable gradient estimates
    /// - It's small enough to allow for frequent updates
    /// - It often fits well in memory for parallel processing
    /// 
    /// You might want to increase this value if:
    /// - Training seems unstable (weights jumping around too much)
    /// - You have plenty of memory and computational resources
    /// - Your dataset is very noisy
    /// 
    /// You might want to decrease this value if:
    /// - You have limited memory available
    /// - Training seems to be progressing too slowly
    /// - You want the model to adapt more quickly
    /// 
    /// Common batch sizes are powers of 2 (16, 32, 64, 128, 256) because they often optimize
    /// performance on modern hardware.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the learning rate that controls the step size in each update of the model parameters.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// The learning rate is a critical hyperparameter that determines how large of a step to take in the
    /// direction of the negative gradient during optimization. A higher learning rate allows for faster
    /// learning but risks overshooting the optimal solution or causing instability. A lower learning rate
    /// provides more stable updates but may require more iterations to converge and risks getting stuck
    /// in local minima. Note that the actual learning rate used in training may be further modified by
    /// the chosen optimizer, which may implement adaptive learning rate strategies.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how big of an adjustment the network makes
    /// to its internal settings during each update.
    /// 
    /// Think of it like turning a dial to tune a radio:
    /// - A high learning rate (like 0.1) means making big turns of the dial
    /// - A low learning rate (like 0.0001) means making tiny, precise turns
    /// 
    /// The default value of 0.001 is relatively conservative, which helps prevent:
    /// - Overshooting the optimal settings
    /// - Unstable behavior during training
    /// 
    /// You might want to increase this value if:
    /// - Training is progressing very slowly
    /// - You have a tight compute budget and need faster results
    /// - You're in early exploration phases
    /// 
    /// You might want to decrease this value if:
    /// - Training is unstable (loss is fluctuating wildly)
    /// - You're fine-tuning an already well-trained model
    /// - You want more precise final results
    /// 
    /// Note that this setting interacts with your choice of optimizer. Some optimizers (like Adam)
    /// adaptively adjust the effective learning rate, making the training less sensitive to this
    /// initial value.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets whether to display detailed progress information during training.
    /// </summary>
    /// <value>Flag indicating whether to display progress, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// When set to true, the training process will output detailed information about its progress,
    /// such as the current epoch, loss value, and potentially other metrics. This can be useful for
    /// monitoring the training process and diagnosing issues, but may slow down training slightly and
    /// generate a large amount of output for long training runs or large datasets. By default, this
    /// verbose output is disabled.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the training process will show
    /// you detailed progress updates as it runs.
    /// 
    /// Think of it like tracking a package:
    /// - When Verbose = false: You only know when the package is delivered
    /// - When Verbose = true: You get updates at each step of the delivery process
    /// 
    /// The default value of false means training will run silently without progress updates.
    /// 
    /// You might want to set this to true if:
    /// - You want to monitor how quickly the model is learning
    /// - You're debugging training issues
    /// - You want to know when to stop training early
    /// 
    /// You might want to keep it false if:
    /// - You're running many experiments and don't need the extra output
    /// - You're running training in a production environment
    /// - You're using other methods to monitor progress (like logging metrics to a file)
    /// 
    /// Enabling verbose output is especially helpful when you're new to neural networks or
    /// when you're trying to debug an underperforming model.
    /// </para>
    /// </remarks>
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Gets or sets the activation function used in the hidden layers of the network.
    /// </summary>
    /// <value>The hidden layer activation function, defaulting to ReLU.</value>
    /// <remarks>
    /// <para>
    /// Activation functions introduce non-linearity into the neural network, allowing it to learn complex
    /// patterns. This parameter sets the activation function used for all neurons in the hidden layers.
    /// The Rectified Linear Unit (ReLU) function is a popular choice for hidden layers as it helps mitigate
    /// the vanishing gradient problem and generally allows for faster training. Other common choices include
    /// sigmoid, tanh, and leaky ReLU, each with different properties that may be more suitable for specific
    /// types of problems.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the mathematical function that each
    /// "neuron" in the hidden layers uses to process its input.
    /// 
    /// Activation functions are like decision rules for neurons:
    /// - They determine how a neuron responds to different input values
    /// - They introduce non-linearity, which allows the network to learn complex patterns
    /// 
    /// The default ReLU (Rectified Linear Unit) function:
    /// - Outputs 0 for negative inputs
    /// - Outputs the input value unchanged for positive inputs
    /// - Is computationally efficient and helps networks learn faster
    /// 
    /// You might want to change this to:
    /// - Sigmoid: If outputs need to be between 0 and 1
    /// - Tanh: If outputs need to be between -1 and 1
    /// - Leaky ReLU: If you're experiencing "dead neurons" (neurons that stop learning)
    /// 
    /// For most problems, ReLU works well and is a good default choice.
    /// </para>
    /// </remarks>
    public IActivationFunction<T>? HiddenActivation { get; set; } = new ReLUActivation<T>();

    /// <summary>
    /// Gets or sets the vector-based activation function used in the hidden layers of the network.
    /// </summary>
    /// <value>The hidden layer vector activation function, defaulting to ReLU.</value>
    /// <remarks>
    /// <para>
    /// This property provides a vector-optimized implementation of the activation function for hidden layers.
    /// When set, it will be used instead of the scalar <see cref="HiddenActivation"/> property for more
    /// efficient computation on entire vectors of data. The default implementation uses ReLU activation,
    /// which is well-suited for most neural network hidden layers.
    /// </para>
    /// <para><b>For Beginners:</b> This is a more efficient version of the hidden layer activation
    /// function that works on entire groups of neurons at once.
    /// 
    /// It serves the same purpose as the regular hidden activation function, but:
    /// - It can process multiple neurons simultaneously
    /// - It's optimized for performance on modern hardware
    /// - It's particularly helpful for large networks
    /// 
    /// You typically don't need to change this unless you're implementing custom activation
    /// functions or optimizing for specific hardware.
    /// </para>
    /// </remarks>
    public IVectorActivationFunction<T>? HiddenVectorActivation { get; set; } = new ReLUActivation<T>();

    /// <summary>
    /// Gets or sets the activation function used in the output layer of the network.
    /// </summary>
    /// <value>The output layer activation function, defaulting to Linear (Identity).</value>
    /// <remarks>
    /// <para>
    /// The output activation function determines the range and type of values that the neural network
    /// can produce. The linear activation function (also called identity) is appropriate for regression
    /// problems where the output can be any real number. For classification problems, other functions like
    /// sigmoid (for binary classification) or softmax (for multi-class classification) would be more appropriate.
    /// The choice of output activation should match the nature of the target variable and the loss function.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the mathematical function that the
    /// final layer uses to produce the network's output.
    /// 
    /// The output activation function shapes your predictions:
    /// - Linear (the default): Can output any number, positive or negative
    /// - Sigmoid: Outputs values between 0 and 1, good for probabilities
    /// - Softmax: Outputs probabilities that sum to 1, good for multi-class problems
    /// 
    /// The default Linear function is appropriate for:
    /// - Regression problems (predicting continuous values like price, temperature, etc.)
    /// - Cases where you need unbounded outputs
    /// 
    /// You should change this to:
    /// - Sigmoid: For binary classification (yes/no, spam/not spam)
    /// - Softmax: For multi-class classification (cat/dog/bird)
    /// - TanH: For outputs that should be between -1 and 1
    /// 
    /// Choosing the right output activation is important - it should match both the type of
    /// problem you're solving and the loss function you're using.
    /// </para>
    /// </remarks>
    public IActivationFunction<T>? OutputActivation { get; set; } = new IdentityActivation<T>();

    /// <summary>
    /// Gets or sets the vector-based activation function used in the output layer of the network.
    /// </summary>
    /// <value>The output layer vector activation function, defaulting to Linear (Identity).</value>
    /// <remarks>
    /// <para>
    /// This property provides a vector-optimized implementation of the activation function for the output layer.
    /// When set, it will be used instead of the scalar <see cref="OutputActivation"/> property for more
    /// efficient computation on entire vectors of data. The default implementation uses the identity (linear)
    /// activation, which is appropriate for regression problems.
    /// </para>
    /// <para><b>For Beginners:</b> This is a more efficient version of the output layer activation
    /// function that works on entire groups of neurons at once.
    /// 
    /// It serves the same purpose as the regular output activation function, but:
    /// - It can process multiple output neurons simultaneously
    /// - It's optimized for performance on modern hardware
    /// - It's particularly helpful for networks with multiple outputs
    /// 
    /// For regression problems, the default linear activation is usually appropriate.
    /// For classification, you might want to use sigmoid or softmax vector activations.
    /// </para>
    /// </remarks>
    public IVectorActivationFunction<T>? OutputVectorActivation { get; set; } = new IdentityActivation<T>();

    /// <summary>
    /// Gets or sets the loss function used to calculate the error between predictions and targets.
    /// </summary>
    /// <value>The loss function, defaulting to Mean Squared Error.</value>
    /// <remarks>
    /// <para>
    /// The loss function quantifies how far the network's predictions are from the true values, providing
    /// the optimization target during training. Mean Squared Error (MSE) is commonly used for regression
    /// problems, calculating the average of the squared differences between predictions and targets. For
    /// classification problems, cross-entropy loss would be more appropriate. The choice of loss function
    /// should align with the problem type and the output activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines how the network measures its prediction errors
    /// during training.
    /// 
    /// Think of the loss function as a scorekeeper:
    /// - It calculates how far off the network's predictions are from the correct answers
    /// - The network tries to minimize this score during training
    /// - Different types of problems need different ways of keeping score
    /// 
    /// The default Mean Squared Error (MSE):
    /// - Calculates the average of the squared differences between predictions and actual values
    /// - Works well for regression problems (predicting continuous values)
    /// - Heavily penalizes large errors
    /// 
    /// You might want to change this to:
    /// - Mean Absolute Error: If you want to treat all errors equally, regardless of direction
    /// - Binary Cross-Entropy: For binary classification problems
    /// - Categorical Cross-Entropy: For multi-class classification problems
    /// 
    /// The loss function should match your problem type and output activation function. For example:
    /// - Regression ? MSE + Linear output activation
    /// - Binary classification ? Binary Cross-Entropy + Sigmoid output activation
    /// - Multi-class classification ? Categorical Cross-Entropy + Softmax output activation
    /// </para>
    /// </remarks>
    public ILossFunction<T>? LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

    /// <summary>
    /// Gets or sets the optimization algorithm used to update the network weights during training.
    /// </summary>
    /// <value>The optimizer, defaulting to Adam with specific parameters.</value>
    /// <remarks>
    /// <para>
    /// The optimizer determines how the network weights are updated based on the computed gradients during
    /// backpropagation. Adam (Adaptive Moment Estimation) is a popular choice that combines the benefits of
    /// two other optimizers: AdaGrad and RMSProp. It maintains adaptive learning rates for each parameter
    /// based on both the first moment (mean) and the second moment (variance) of the gradients. This default
    /// configuration of Adam uses standard recommended values for its hyperparameters, but different optimizers
    /// or parameters may be more effective for specific problems.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the algorithm used to adjust the network's
    /// internal parameters during training.
    /// 
    /// The optimizer is like a strategy for learning:
    /// - It decides how to use the error information to update the network
    /// - Different optimizers have different strengths and weaknesses
    /// - Some are faster, some are more stable, some work better for certain types of problems
    /// 
    /// The default Adam (Adaptive Moment Estimation) optimizer:
    /// - Adapts the learning rate differently for each parameter
    /// - Combines the benefits of several other optimization techniques
    /// - Works well across a wide range of problems
    /// - Has its own set of configurable parameters (shown in the default)
    /// 
    /// You might want to change this to:
    /// - SGD (Stochastic Gradient Descent): Simpler, sometimes more predictable
    /// - RMSProp: Good for recurrent neural networks
    /// - AdaGrad: Good when some parameters need very different learning rates
    /// 
    /// For most cases, Adam is an excellent default choice. If you're just starting with neural networks,
    /// you probably don't need to change this until you have more experience with how your specific
    /// models behave during training.
    /// </para>
    /// </remarks>
    private IOptimizer<T, TInput, TOutput>? _optimizer;

    public IOptimizer<T, TInput, TOutput> Optimizer
    {
        get
        {
            if (_optimizer == null)
            {
                var defaultModel = ModelHelper<T, TInput, TOutput>.CreateDefaultModel();
                _optimizer = new AdamOptimizer<T, TInput, TOutput>(
                    defaultModel,
                    new AdamOptimizerOptions<T, TInput, TOutput>
                    {
                        InitialLearningRate = 0.001,
                        Beta1 = 0.9,
                        Beta2 = 0.999,
                        Epsilon = 1e-8
                    });
            }
            return _optimizer;
        }
        set
        {
            _optimizer = value;
        }
    }
}
