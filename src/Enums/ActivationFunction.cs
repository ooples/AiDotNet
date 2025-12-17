namespace AiDotNet.Enums;

/// <summary>
/// Represents different activation functions used in neural networks and deep learning.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Activation functions are mathematical operations that determine whether a neuron in a 
/// neural network should be "activated" (output a signal) or not.
/// 
/// Think of a neuron as a decision-maker that:
/// 1. Receives multiple inputs
/// 2. Calculates a weighted sum of these inputs
/// 3. Applies an activation function to decide what value to output
/// 
/// Without activation functions, neural networks would just be linear models (like basic regression), 
/// unable to learn complex patterns. Activation functions add non-linearity, allowing networks to learn 
/// complicated relationships in data.
/// 
/// Different activation functions have different properties that make them suitable for different tasks. 
/// Choosing the right activation function can significantly impact how well your neural network learns 
/// and performs.
/// </para>
/// </remarks>
public enum ActivationFunction
{
    /// <summary>
    /// Rectified Linear Unit - returns 0 for negative inputs and the input value for positive inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ReLU (Rectified Linear Unit) is the most commonly used activation function in 
    /// modern neural networks.
    /// 
    /// How it works:
    /// - If the input is negative, ReLU outputs 0
    /// - If the input is positive, ReLU outputs the input value unchanged
    /// 
    /// Formula: f(x) = max(0, x)
    /// 
    /// Advantages:
    /// - Simple and fast to compute
    /// - Helps networks learn faster than older functions like Sigmoid
    /// - Reduces the "vanishing gradient problem" (where networks stop learning)
    /// - Works well in hidden layers of deep networks
    /// 
    /// Limitations:
    /// - Can cause "dying ReLU" problem where neurons permanently stop learning
    /// - Not zero-centered, which can make training slightly less efficient
    /// 
    /// ReLU is typically the default choice for hidden layers in most neural networks.
    /// </para>
    /// </remarks>
    ReLU,

    /// <summary>
    /// Sigmoid function - maps any input to a value between 0 and 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Sigmoid function squeezes any input value into an output between 0 and 1, 
    /// creating an S-shaped curve.
    /// 
    /// How it works:
    /// - Large negative inputs approach 0
    /// - Large positive inputs approach 1
    /// - Input of 0 gives output of 0.5
    /// 
    /// Formula: f(x) = 1 / (1 + e^(-x))
    /// 
    /// Advantages:
    /// - Smooth and bounded output (always between 0 and 1)
    /// - Provides a clear probability interpretation
    /// - Good for binary classification output layers
    /// - Historically important in neural networks
    /// 
    /// Limitations:
    /// - Suffers from vanishing gradient problem for extreme inputs
    /// - Outputs are not zero-centered
    /// - Computationally more expensive than ReLU
    /// 
    /// Sigmoid is now mostly used in output layers for binary classification or in specific 
    /// architectures like LSTMs, but rarely in hidden layers of deep networks.
    /// </para>
    /// </remarks>
    Sigmoid,

    /// <summary>
    /// Hyperbolic Tangent - maps any input to a value between -1 and 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Tanh (hyperbolic tangent) function is similar to Sigmoid but maps inputs 
    /// to values between -1 and 1 instead of 0 and 1.
    /// 
    /// How it works:
    /// - Large negative inputs approach -1
    /// - Large positive inputs approach 1
    /// - Input of 0 gives output of 0
    /// 
    /// Formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    /// 
    /// Advantages:
    /// - Zero-centered outputs (centered around 0), which helps with learning
    /// - Bounded output (always between -1 and 1)
    /// - Often performs better than Sigmoid in hidden layers
    /// - Works well in recurrent neural networks (RNNs)
    /// 
    /// Limitations:
    /// - Still suffers from vanishing gradient problem for extreme inputs
    /// - Computationally more expensive than ReLU
    /// 
    /// Tanh is commonly used in recurrent neural networks and sometimes in hidden layers when 
    /// zero-centered outputs are important.
    /// </para>
    /// </remarks>
    Tanh,

    /// <summary>
    /// Linear activation - simply returns the input value unchanged.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Linear activation function simply returns the input without any change.
    /// 
    /// How it works:
    /// - Output equals input: f(x) = x
    /// 
    /// Advantages:
    /// - Simplest possible activation function
    /// - No computation cost
    /// - Useful for regression problems in the output layer
    /// - Allows unbounded outputs
    /// 
    /// Limitations:
    /// - Provides no non-linearity, so networks with only linear activations can't learn complex patterns
    /// - Essentially reduces the network to a linear model regardless of depth
    /// 
    /// Linear activation is typically only used in the output layer of regression networks 
    /// (when predicting continuous values like prices, temperatures, etc.).
    /// </para>
    /// </remarks>
    Linear,

    /// <summary>
    /// Leaky Rectified Linear Unit - similar to ReLU but allows a small gradient for negative inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LeakyReLU is a variation of ReLU that allows a small, non-zero output for 
    /// negative inputs.
    /// 
    /// How it works:
    /// - If input is positive, output equals input (same as ReLU)
    /// - If input is negative, output equals input multiplied by a small factor (typically 0.01)
    /// 
    /// Formula: f(x) = max(0.01x, x)
    /// 
    /// Advantages:
    /// - Prevents the "dying ReLU" problem by allowing small gradients for negative inputs
    /// - Almost as computationally efficient as ReLU
    /// - All the benefits of ReLU with added robustness
    /// 
    /// Limitations:
    /// - The leakage parameter (0.01) is another hyperparameter to tune
    /// - Still not zero-centered
    /// 
    /// LeakyReLU is a good alternative to ReLU when you're concerned about neurons "dying" during training.
    /// </para>
    /// </remarks>
    LeakyReLU,

    /// <summary>
    /// Exponential Linear Unit - smooth version of ReLU that can output negative values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ELU (Exponential Linear Unit) is an activation function that combines the benefits 
    /// of ReLU while addressing some of its limitations.
    /// 
    /// How it works:
    /// - For positive inputs: same as ReLU, output equals input
    /// - For negative inputs: output equals a * (e^x - 1), where a is typically 1.0
    ///   This creates a smooth curve that approaches -a for large negative inputs
    /// 
    /// Formula: f(x) = x if x > 0, a * (e^x - 1) if x = 0
    /// 
    /// Advantages:
    /// - Smooth function including for negative values (helps with learning)
    /// - Reduces the "dying neuron" problem of ReLU
    /// - Can produce negative outputs, pushing mean activations closer to zero
    /// - Often leads to faster learning
    /// 
    /// Limitations:
    /// - More computationally expensive than ReLU due to exponential operation
    /// - Has an extra hyperparameter a to tune
    /// 
    /// ELU is a good choice when you want better performance than ReLU and can afford the 
    /// slightly higher computational cost.
    /// </para>
    /// </remarks>
    ELU,

    /// <summary>
    /// Scaled Exponential Linear Unit - self-normalizing version of ELU.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SELU (Scaled Exponential Linear Unit) is a special activation function designed 
    /// to make neural networks "self-normalizing."
    /// 
    /// How it works:
    /// - Similar to ELU but with carefully chosen scaling parameters
    /// - For positive inputs: output equals input multiplied by a scale factor ?
    /// - For negative inputs: output equals ? * a * (e^x - 1)
    ///   where ? ≈ 1.0507 and a ≈ 1.6733 are specific constants
    /// 
    /// Formula: f(x) = ? * x if x > 0, ? * a * (e^x - 1) if x = 0
    /// 
    /// Advantages:
    /// - Self-normalizing property helps maintain stable activations across many layers
    /// - Can eliminate the need for techniques like batch normalization
    /// - Helps prevent vanishing and exploding gradients
    /// - Works particularly well for deep networks
    /// 
    /// Limitations:
    /// - Requires specific initialization (LeCun normal) to work properly
    /// - Benefits are most apparent in fully-connected networks
    /// - More computationally expensive than ReLU
    /// 
    /// SELU is particularly useful for deep fully-connected networks where maintaining normalized 
    /// activations is important.
    /// </para>
    /// </remarks>
    SELU,

    /// <summary>
    /// Softmax function - converts a vector of values to a probability distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Softmax is special because it works on a group of neurons together, not just one at a time.
    /// It converts a set of numbers into a probability distribution that sums to 1.
    /// 
    /// How it works:
    /// - Takes a vector of numbers (e.g., scores for different classes)
    /// - Applies exponential function (e^x) to each number
    /// - Divides each result by the sum of all exponentials
    /// 
    /// Formula: softmax(x_i) = e^x_i / S(e^x_j) for all j
    /// 
    /// Advantages:
    /// - Outputs are between 0 and 1 and sum to exactly 1 (perfect for probabilities)
    /// - Emphasizes the largest values while suppressing lower values
    /// - Ideal for multi-class classification problems
    /// - Differentiable, so works well with gradient-based learning
    /// 
    /// Limitations:
    /// - Only meaningful when applied to multiple neurons together (output layer)
    /// - Can be numerically unstable (requires special implementation techniques)
    /// - Not suitable for hidden layers
    /// 
    /// Softmax is almost exclusively used in the output layer of classification networks when you need 
    /// to predict probabilities across multiple classes.
    /// </para>
    /// </remarks>
    Softmax,

    /// <summary>
    /// Softplus function - a smooth approximation of the ReLU function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Softplus is a smooth version of ReLU that has a more gradual transition 
    /// from 0 to positive values.
    /// 
    /// How it works:
    /// - For large negative inputs, output approaches 0
    /// - For large positive inputs, output approaches the input value
    /// - The transition is smooth and differentiable everywhere
    /// 
    /// Formula: f(x) = ln(1 + e^x)
    /// 
    /// Advantages:
    /// - Smooth everywhere (no sharp corner like ReLU)
    /// - Always has a non-zero gradient, avoiding the "dying neuron" problem
    /// - Outputs are always positive
    /// 
    /// Limitations:
    /// - More computationally expensive than ReLU
    /// - Can still suffer from vanishing gradient for very negative inputs
    /// - Not zero-centered
    /// 
    /// Softplus is sometimes used as an alternative to ReLU when a smoother activation function is desired.
    /// </para>
    /// </remarks>
    Softplus,

    /// <summary>
    /// SoftSign function - maps inputs to values between -1 and 1 with a smoother approach to the asymptotes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SoftSign is similar to Tanh but approaches its asymptotes more slowly, 
    /// which can help with learning in some cases.
    /// 
    /// How it works:
    /// - Maps inputs to a range between -1 and 1
    /// - For large negative inputs, output approaches -1
    /// - For large positive inputs, output approaches 1
    /// - Has a gentler slope than Tanh for extreme values
    /// 
    /// Formula: f(x) = x / (1 + |x|)
    /// 
    /// Advantages:
    /// - Zero-centered like Tanh
    /// - Bounded output between -1 and 1
    /// - Approaches asymptotes more gradually than Tanh
    /// - Less prone to saturation (getting "stuck" at the extremes)
    /// 
    /// Limitations:
    /// - Not as widely used or studied as other activation functions
    /// - Computationally more expensive than ReLU
    /// 
    /// SoftSign can be used as an alternative to Tanh when you want a function that saturates more slowly.
    /// </para>
    /// </remarks>
    SoftSign,

    /// <summary>
    /// Swish function - a self-gated activation function developed by researchers at Google.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Swish is a newer activation function that was discovered through automated search 
    /// techniques and often outperforms ReLU in deep networks.
    /// 
    /// How it works:
    /// - Multiplies the input by a sigmoid of the input
    /// - For positive inputs, behaves similarly to ReLU
    /// - For negative inputs, allows small negative values instead of zeroing them out
    /// - Has a slight dip below zero for some negative inputs
    /// 
    /// Formula: f(x) = x * sigmoid(x) or f(x) = x / (1 + e^(-x))
    /// 
    /// Advantages:
    /// - Often achieves better accuracy than ReLU in deep networks
    /// - Smooth function with non-zero gradients everywhere
    /// - Allows negative outputs for some inputs, which can be beneficial
    /// - Works well with normalization techniques
    /// 
    /// Limitations:
    /// - More computationally expensive than ReLU
    /// - Relatively new, so less extensively tested in all scenarios
    /// - May require more careful initialization
    /// 
    /// Swish is a good alternative to try when ReLU isn't giving optimal results, especially in deep networks.
    /// </para>
    /// </remarks>
    Swish,

    /// <summary>
    /// Gaussian Error Linear Unit - a smooth activation function that performs well in transformers and language models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GELU (Gaussian Error Linear Unit) is an activation function that has become popular 
    /// in modern language models like BERT and GPT.
    /// 
    /// How it works:
    /// - Multiplies the input by the cumulative distribution function of the standard normal distribution
    /// - For positive inputs, behaves similarly to ReLU but with a smooth curve
    /// - For negative inputs, allows small negative values with a smooth transition
    /// 
    /// Formula: f(x) = 0.5 * x * (1 + tanh(sqrt(2/p) * (x + 0.044715 * x^3)))
    /// (This is an approximation of the actual formula for computational efficiency)
    /// 
    /// Advantages:
    /// - Performs exceptionally well in transformer architectures
    /// - Smooth function with non-zero gradients for most inputs
    /// - Combines benefits of ReLU, ELU, and Swish
    /// - State-of-the-art results in many language models
    /// 
    /// Limitations:
    /// - More computationally expensive than simpler functions like ReLU
    /// - Relatively complex mathematical formulation
    /// - Best suited for specific architectures like transformers
    /// 
    /// GELU is particularly recommended for transformer-based models and when working with natural language processing tasks.
    /// </para>
    /// </remarks>
    GELU,

    /// <summary>
    /// Identity function - returns the input value unchanged, providing a direct pass-through.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Identity activation function simply passes the input through unchanged - 
    /// whatever value goes in is exactly what comes out.
    /// 
    /// How it works:
    /// - Output equals input: f(x) = x
    /// - No transformation is applied at all
    /// 
    /// Formula: f(x) = x
    /// 
    /// Advantages:
    /// - Zero computational cost (fastest possible activation)
    /// - Preserves the full range of input values
    /// - Useful for testing or when you want a layer to pass values through unchanged
    /// - Can be used in the final layer of some regression networks
    /// 
    /// Limitations:
    /// - Provides no non-linearity whatsoever
    /// - A network using only Identity activations reduces to a simple linear model
    /// - Cannot help the network learn complex patterns
    /// 
    /// The Identity function is primarily used in specific scenarios like skip connections in residual networks,
    /// or when you want to debug a network by temporarily removing non-linearities.
    /// </para>
    /// </remarks>
    Identity,

    /// <summary>
    /// Linearly Scaled Hyperbolic Tangent - a self-regularized activation function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LiSHT (Linearly Scaled Hyperbolic Tangent) is an activation function
    /// that combines the benefits of linear and tanh functions.
    ///
    /// How it works:
    /// - Multiplies the input by its own tanh: f(x) = x * tanh(x)
    /// - For positive inputs, behaves similarly to the input itself
    /// - For negative inputs, output is negative but bounded
    ///
    /// Formula: f(x) = x * tanh(x)
    ///
    /// Advantages:
    /// - Non-monotonic function that can help with learning complex patterns
    /// - Smooth and differentiable everywhere
    /// - Self-regularized, helping prevent overfitting
    /// - Has bounded gradient properties
    ///
    /// Limitations:
    /// - More computationally expensive than ReLU
    /// - Relatively new, so less extensively tested
    ///
    /// LiSHT is useful when you need a self-regularizing activation function with good gradient properties.
    /// </para>
    /// </remarks>
    LiSHT
}
