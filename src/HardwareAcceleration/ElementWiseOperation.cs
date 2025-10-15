namespace AiDotNet.HardwareAcceleration
{
    /// <summary>
    /// Element-wise operations supported by hardware accelerators
    /// </summary>
    public enum ElementWiseOperation
    {
        /// <summary>
        /// Addition operation (a + b)
        /// </summary>
        Add,

        /// <summary>
        /// Subtraction operation (a - b)
        /// </summary>
        Subtract,

        /// <summary>
        /// Multiplication operation (a * b)
        /// </summary>
        Multiply,

        /// <summary>
        /// Division operation (a / b)
        /// </summary>
        Divide,

        /// <summary>
        /// Power operation (a ^ b)
        /// </summary>
        Power,

        /// <summary>
        /// Maximum operation (max(a, b))
        /// </summary>
        Maximum,

        /// <summary>
        /// Minimum operation (min(a, b))
        /// </summary>
        Minimum,

        /// <summary>
        /// Sigmoid activation function
        /// </summary>
        Sigmoid,

        /// <summary>
        /// Hyperbolic tangent activation function
        /// </summary>
        Tanh,

        /// <summary>
        /// Rectified Linear Unit activation function
        /// </summary>
        ReLU,

        /// <summary>
        /// Leaky ReLU activation function
        /// </summary>
        LeakyReLU,

        /// <summary>
        /// Exponential function
        /// </summary>
        Exp,

        /// <summary>
        /// Natural logarithm function
        /// </summary>
        Log,

        /// <summary>
        /// Square root function
        /// </summary>
        Sqrt,

        /// <summary>
        /// Absolute value function
        /// </summary>
        Abs
    }
}