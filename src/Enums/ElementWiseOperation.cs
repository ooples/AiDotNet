namespace AiDotNet.Enums
{
    /// <summary>
    /// Specifies the type of element-wise operation to perform on tensors
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
        /// Maximum operation (max(a, b))
        /// </summary>
        Maximum,

        /// <summary>
        /// Minimum operation (min(a, b))
        /// </summary>
        Minimum,

        /// <summary>
        /// Power operation (a ^ b)
        /// </summary>
        Power,

        /// <summary>
        /// Modulo operation (a % b)
        /// </summary>
        Modulo,

        /// <summary>
        /// Absolute difference operation (|a - b|)
        /// </summary>
        AbsoluteDifference,

        /// <summary>
        /// Square root of sum of squares (sqrt(a^2 + b^2))
        /// </summary>
        Hypot
    }
}