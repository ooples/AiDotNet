namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for neural network regression models, providing fine-grained control over
/// network architecture, training parameters, activation functions, and optimization strategies.
/// </summary>
/// <remarks>
/// <para>
/// Neural network regression is a powerful approach for modeling complex nonlinear relationships between
/// input features and continuous output variables. This class encapsulates the full range of parameters
/// needed to define and train a neural network for regression tasks. It allows for customization of
/// network depth and width, training duration and batch size, activation functions, loss functions, and
/// optimization algorithms. These options collectively determine the network's capacity, learning behavior,
/// and computational requirements, making them crucial for achieving optimal predictive performance while
/// managing training time and resource usage.
/// </para>
/// <para><b>For Beginners:</b> Neural networks are AI models inspired by the human brain that can learn complex patterns.
/// 
/// Imagine building a system to predict house prices based on features like size, location, and age:
/// - Traditional methods might use simple formulas (like linear regression)
/// - Neural networks can discover complicated relationships that simple formulas miss
/// 
/// A neural network consists of layers of interconnected "neurons":
/// - Input layer: Receives your data (like house size, number of bedrooms)
/// - Hidden layers: Process the information and discover patterns
/// - Output layer: Produces the prediction (like the estimated house price)
/// 
/// The network "learns" by:
/// - Making predictions on training data
/// - Comparing those predictions to the actual values
/// - Adjusting its internal connections to reduce the errors
/// - Repeating this process many times
/// 
/// This class lets you configure every aspect of a neural network designed specifically for regression
/// (predicting continuous values like prices, temperatures, or scores), from its structure to how it learns.
/// </para>
/// </remarks>
public class NeuralNetworkRegressionOptions<T, TInput, TOutput> : NonLinearRegressionOptions
{
    /// <summary>
    /// Gets or sets the sizes of each layer in the neural network, including input, hidden, and output layers.
    /// </summary>
    /// <value>A list of integers representing the number of neurons in each layer, defaulting to [1, 10, 1].</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the architecture of the neural network by specifying the number of neurons in each layer.
    /// The first number represents the input dimension, the last number represents the output dimension, and
    /// all numbers in between represent the hidden layers. The default [1, 10, 1] creates a network with one
    /// input feature, one hidden layer containing 10 neurons, and one output. The depth (number of layers) and
    /// width (neurons per layer) of the network determine its capacity to model complex relationships. Deeper
    /// and wider networks can capture more complex patterns but require more data and computational resources
    /// to train effectively.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the structure of your neural network -
    /// how many layers it has and how many neurons are in each layer.
    /// 
    /// The default value of [1, 10, 1] means:
    /// - 1 input neuron (for a single feature like house size)
    /// - 10 neurons in one hidden layer (where the pattern recognition happens)
    /// - 1 output neuron (for a single prediction like price)
    /// 
    /// You might want more inputs if:
    /// - You have multiple features (e.g., [4, 10, 1] for size, bedrooms, bathrooms, and age)
    /// 
    /// You might want more or larger hidden layers if:
    /// - Your problem is complex (e.g., [4, 20, 20, 1] adds a second hidden layer with 20 neurons each)
    /// - You have lots of training data to support a bigger network
    /// 
    /// You might want more outputs if:
    /// - You're predicting multiple values simultaneously (e.g., [4, 20, 3] to predict price, maintenance cost, and energy efficiency)
    /// 
    /// Larger networks can learn more complex patterns but need more data and computing power.
    /// It's often best to start small and increase size if needed.
    /// </para>
    /// </remarks>
    public List<int> LayerSizes { get; set; } = new List<int> { 1, 10, 1 };  // Default: 1 input, 1 hidden layer with 10 neurons, 1 output

    /// <summary>
    /// Gets or sets the number of complete passes through the training dataset during model training.
    /// </summary>
    /// <value>The number of training epochs, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// An epoch represents one complete pass through the entire training dataset. This parameter determines
    /// how many times the neural network will see each training example during the learning process. More
    /// epochs generally allow for better learning but increase training time and may lead to overfitting if
    /// the number is too high. The optimal number of epochs varies widely depending on dataset size, problem
    /// complexity, and other factors such as learning rate and batch size. Early stopping based on validation
    /// performance is often used in practice to determine the actual number of epochs.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many times the neural network
    /// will process the entire training dataset during learning.
    /// 
    /// The default value of 1000 means:
    /// - The network will see each training example 1000 times
    /// - It gets 1000 opportunities to refine its understanding
    /// 
    /// Think of it like studying for an exam:
    /// - Each epoch is like reviewing all your study materials once
    /// - More reviews generally lead to better understanding, up to a point
    /// 
    /// You might want more epochs (like 2000) if:
    /// - Your network is still improving at the end of training
    /// - You have a complex problem requiring more learning time
    /// - You're using a small learning rate
    /// 
    /// You might want fewer epochs (like 500) if:
    /// - Your network starts overfitting (performing worse on new data)
    /// - Training takes too long
    /// - You're using a large learning rate
    /// 
    /// In practice, it's common to use "early stopping" - monitoring performance on validation data
    /// and stopping training when it stops improving, regardless of the maximum epochs setting.
    /// </para>
    /// </remarks>
    public int Epochs { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the number of training examples used in one iteration of model training.
    /// </summary>
    /// <value>The batch size, defaulting to 32.</value>
    /// <remarks>
    /// <para>
    /// Batch size determines how many training examples the network processes before updating its weights.
    /// Smaller batch sizes provide more frequent weight updates and often lead to faster initial convergence,
    /// while larger batch sizes offer better gradient estimates and can utilize hardware acceleration more
    /// efficiently. The choice of batch size impacts both training speed and the quality of the learned model.
    /// Extremely small batch sizes may lead to noisy updates and slow convergence, while extremely large batch
    /// sizes may cause the network to converge to suboptimal solutions and require more memory.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many training examples
    /// the network processes before making adjustments to its internal connections.
    /// 
    /// The default value of 32 means:
    /// - The network looks at 32 examples (like houses in our price prediction example)
    /// - Then updates its understanding based on the average error across those examples
    /// 
    /// Think of it like a teacher who might:
    /// - Grade all 32 assignments at once
    /// - Look for patterns in the mistakes
    /// - Then adjust their teaching approach based on those patterns
    /// 
    /// You might want a larger batch size (like 64 or 128) if:
    /// - You have lots of training data
    /// - You want more stable (but slower) learning
    /// - You want to utilize GPU acceleration
    /// 
    /// You might want a smaller batch size (like 8 or 16) if:
    /// - You have limited memory
    /// - You want the network to learn more quickly (though more noisily)
    /// - You have a small dataset
    /// 
    /// The best batch size often requires experimentation, as it affects both
    /// learning quality and training speed.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the step size used for updating model weights during gradient descent.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// The learning rate is a critical hyperparameter that controls how much the model's weights are adjusted
    /// in response to the estimated error each time they are updated. A higher learning rate means larger steps
    /// in the weight space, which can lead to faster convergence but might overshoot the optimal solution or
    /// cause instability. A lower learning rate means smaller, more cautious steps, which can lead to more
    /// precise convergence but might require many more iterations or get stuck in local minima. The ideal
    /// learning rate often varies by problem, network architecture, and optimizer choice.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how big of adjustments the network
    /// makes when it's learning from its mistakes.
    /// 
    /// The default value of 0.01 means:
    /// - Each time the network updates its connections, it makes relatively small adjustments
    /// - These adjustments are proportional to how wrong its predictions were
    /// 
    /// Think of it like turning a steering wheel:
    /// - A high learning rate is like making big turns - you'll change direction quickly but might overshoot
    /// - A low learning rate is like making tiny adjustments - more precise but slower to respond
    /// 
    /// You might want a higher learning rate (like 0.1) if:
    /// - Training is progressing too slowly
    /// - You're early in the training process
    /// - You're using an adaptive optimizer that handles large rates well
    /// 
    /// You might want a lower learning rate (like 0.001) if:
    /// - Training is unstable with predictions jumping around
    /// - You're fine-tuning an already trained network
    /// - You need very precise final predictions
    /// 
    /// Finding the right learning rate is one of the most important parts of training
    /// neural networks. Too high and training becomes unstable; too low and training
    /// becomes impractically slow.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the activation function applied to the outputs of hidden layer neurons.
    /// </summary>
    /// <value>The hidden layer activation function, defaulting to ReLU (Rectified Linear Unit).</value>
    /// <remarks>
    /// <para>
    /// The activation function introduces non-linearity into the network, enabling it to learn complex patterns.
    /// The default ReLU function (f(x) = max(0, x)) is widely used because it combats the vanishing gradient
    /// problem and promotes sparse activation, leading to efficient learning in deep networks. Other common
    /// choices include sigmoid, tanh, and leaky ReLU, each with different properties regarding the range of
    /// outputs, gradient behavior, and computational efficiency. The choice of activation function can
    /// significantly impact learning dynamics and the network's ability to model certain types of relationships.
    /// </para>
    /// <para><b>For Production Scenarios:</b> ReLU is generally recommended for hidden layers in regression tasks because:
    /// 
    /// - It's computationally efficient, making training and inference faster
    /// - It helps mitigate the vanishing gradient problem in deeper networks
    /// - It tends to converge faster than sigmoid or tanh
    /// - It works well with a wide range of regression problems
    /// 
    /// However, if you encounter "dying ReLU" issues (where neurons become inactive and stop learning),
    /// consider using Leaky ReLU instead, which allows a small gradient when the input is negative.
    /// For very deep networks, variants like ELU (Exponential Linear Unit) or SELU (Scaled Exponential
    /// Linear Unit) may provide better performance due to their self-normalizing properties.
    /// </para>
    /// </remarks>
    public IActivationFunction<T>? HiddenActivationFunction { get; set; } = new ReLUActivation<T>();

    /// <summary>
    /// Gets or sets the activation function applied to the outputs of the final layer neurons.
    /// </summary>
    /// <value>The output layer activation function, defaulting to the identity function (f(x) = x).</value>
    /// <remarks>
    /// <para>
    /// For regression tasks, the linear (identity) activation function is typically used in the output layer
    /// to allow the network to predict any numerical value within the range of the representable numbers.
    /// This contrasts with classification tasks, which might use sigmoid or softmax activations to produce
    /// probabilities. The linear activation is appropriate when the target variable is unbounded and can take
    /// any real value, which is common in many regression problems such as price prediction, time series
    /// forecasting, and physical measurements.
    /// </para>
    /// <para><b>For Production Scenarios:</b> The identity function (linear activation) is strongly recommended
    /// for the output layer in regression tasks because:
    /// 
    /// - It allows the network to predict any value in the real number range
    /// - It doesn't constrain the output, which is usually necessary for regression
    /// - It works well with common regression loss functions like MSE
    /// 
    /// However, if your regression target has known constraints, consider these alternatives:
    /// 
    /// - For strictly positive outputs (e.g., prices, counts): ReLU or Softplus
    /// - For outputs bounded between 0 and 1 (e.g., percentages): Sigmoid
    /// - For outputs bounded between -1 and 1: Tanh
    /// 
    /// Matching your output activation to the natural range of your target variable can improve
    /// model performance and convergence speed.
    /// </para>
    /// </remarks>
    public IActivationFunction<T> OutputActivationFunction { get; set; } = new IdentityActivation<T>();

    /// <summary>
    /// Gets or sets the vector activation function applied to the outputs of hidden layer neurons.
    /// </summary>
    /// <value>The hidden layer vector activation function, defaulting to null.</value>
    /// <remarks>
    /// <para>
    /// This property allows you to specify a vector-based activation function for hidden layers, which can
    /// operate directly on vectors or tensors rather than individual scalar values. Vector activation functions
    /// can be more computationally efficient as they can leverage optimized vector operations. If both this
    /// property and HiddenActivationFunction are set, this vector implementation takes precedence.
    /// </para>
    /// <para><b>For Beginners:</b> This is an advanced setting that provides a more efficient way
    /// to apply activation functions to entire layers at once.
    /// 
    /// The default value of null means:
    /// - The system will use the scalar HiddenActivationFunction instead
    /// - Each neuron's output will be processed individually
    /// 
    /// You might want to set this if:
    /// - You're working with very large networks where performance is critical
    /// - You have a custom vector-optimized activation function
    /// - You're using hardware acceleration that benefits from vectorized operations
    /// 
    /// In most cases, you can leave this as null and just use the HiddenActivationFunction property,
    /// which is more intuitive and works for all scenarios.
    /// </para>
    /// </remarks>
    public IVectorActivationFunction<T>? HiddenVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function applied to the outputs of the final layer neurons.
    /// </summary>
    /// <value>The output layer vector activation function, defaulting to null.</value>
    /// <remarks>
    /// <para>
    /// This property allows you to specify a vector-based activation function for the output layer, which can
    /// operate directly on vectors or tensors rather than individual scalar values. Vector activation functions
    /// can be more computationally efficient as they can leverage optimized vector operations. If both this
    /// property and OutputActivationFunction are set, this vector implementation takes precedence.
    /// </para>
    /// <para><b>For Beginners:</b> This is an advanced setting that provides a more efficient way
    /// to apply activation functions to the output layer all at once.
    /// 
    /// The default value of null means:
    /// - The system will use the scalar OutputActivationFunction instead
    /// - Each output neuron will be processed individually
    /// 
    /// You might want to set this if:
    /// - You're working with very large networks where performance is critical
    /// - You have a custom vector-optimized activation function
    /// - You're using hardware acceleration that benefits from vectorized operations
    /// 
    /// In most cases, you can leave this as null and just use the OutputActivationFunction property,
    /// which is more intuitive and works for all scenarios.
    /// </para>
    /// </remarks>
    public IVectorActivationFunction<T>? OutputVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the function used to calculate the error between predicted and actual values.
    /// </summary>
    /// <value>The loss function, defaulting to Mean Squared Error (MSE).</value>
    /// <remarks>
    /// <para>
    /// The loss function quantifies the difference between the network's predictions and the actual target values,
    /// providing a measure of model performance that is minimized during training. Mean Squared Error (MSE) is
    /// the default choice for regression tasks as it heavily penalizes large errors and has favorable mathematical
    /// properties for optimization. Alternative loss functions include Mean Absolute Error (less sensitive to
    /// outliers), Huber loss (combines MSE and MAE properties), and custom domain-specific loss functions. The
    /// choice of loss function should align with what constitutes a "good" prediction in the specific application.
    /// </para>
    /// <para><b>For Production Scenarios:</b> The choice of loss function should be guided by your specific
    /// requirements and data characteristics:
    /// 
    /// - Mean Squared Error (MSE): Best general-purpose choice when:
    ///   - Large errors are significantly more important to minimize than small errors
    ///   - Your data doesn't contain extreme outliers
    ///   - You want faster convergence due to larger gradients for larger errors
    /// 
    /// - Mean Absolute Error (MAE): Consider when:
    ///   - Your data contains outliers that would disproportionately influence MSE
    ///   - All error magnitudes should be treated more equally
    ///   - You care about the median prediction rather than the mean
    /// 
    /// - Huber Loss: Excellent compromise when:
    ///   - You want MSE's sensitivity to error for smaller errors
    ///   - You want MAE's robustness to outliers for larger errors
    ///   - You need a differentiable loss function with controlled sensitivity
    /// 
    /// - Root Mean Squared Error (RMSE): Consider when:
    ///   - You want the error in the same units as your target variable
    ///   - You still want to penalize larger errors more heavily
    /// 
    /// - Mean Squared Logarithmic Error (MSLE): Useful when:
    ///   - Your target values have a wide range (e.g., house prices)
    ///   - Relative errors are more important than absolute errors
    ///   - Underprediction should be penalized more than overprediction
    /// 
    /// For most production regression tasks, MSE or Huber Loss provide the best balance of
    /// mathematical properties and practical performance.
    /// </para>
    /// </remarks>
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

    /// <summary>
    /// Gets or sets the optimization algorithm used to update the network weights during training.
    /// </summary>
    /// <value>The optimizer instance, defaulting to null (in which case a default optimizer will be used).</value>
    /// <remarks>
    /// <para>
    /// The optimizer determines how the network's weights are updated based on the calculated gradients during
    /// backpropagation. Different optimization algorithms have various properties regarding convergence speed,
    /// stability, and ability to navigate complex error surfaces. Common choices include Stochastic Gradient
    /// Descent (SGD), Adam, RMSProp, and AdaGrad. If this property is left null, a default optimizer
    /// (typically SGD with momentum) will be used. Advanced optimizers often adapt learning rates per-parameter
    /// or incorporate momentum to accelerate training and help escape local minima.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the algorithm used to update the
    /// network's internal connections during training.
    /// 
    /// The default value of null means:
    /// - The system will choose a standard optimizer for you
    /// - This is usually basic Stochastic Gradient Descent (SGD) or SGD with momentum
    /// 
    /// Think of the optimizer like a navigation strategy:
    /// - Basic SGD is like always walking directly downhill
    /// - Advanced optimizers (like Adam) are like having a smart GPS that considers terrain, momentum, and history
    /// 
    /// You might want to specify an optimizer if:
    /// - Training is slow or unstable with the default
    /// - You're working with a challenging problem
    /// - You have specific requirements for training speed or final accuracy
    /// 
    /// Popular optimizer options include:
    /// - Adam: Generally good performance across many problems
    /// - RMSProp: Good for non-stationary problems
    /// - SGD with momentum: Simple but effective with proper tuning
    /// - Nesterov Accelerated Gradient: Helps avoid overshooting minima
    /// 
    /// If you're new to neural networks, you can start with the default (null) and
    /// explore different optimizers as you gain experience.
    /// </para>
    /// </remarks>
    public IOptimizer<T, TInput, TOutput>? Optimizer { get; set; }
}
