namespace AiDotNet.Helpers;

/// <summary>
/// Provides helper methods for neural network operations including activation functions and loss functions.
/// </summary>
/// <typeparam name="T">The numeric type used in calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// For Beginners: Neural networks are computing systems inspired by the human brain. They process information
/// through interconnected nodes (neurons) that transform input data using mathematical functions.
/// This helper class provides those mathematical functions needed to build neural networks.
/// </para>
/// </remarks>
public static class NeuralNetworkHelper<T>
{
    /// <summary>
    /// Provides operations for the numeric type T.
    /// </summary>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Applies the Rectified Linear Unit (ReLU) activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The maximum of x and 0.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: ReLU is one of the most popular activation functions in neural networks.
    /// It simply outputs the input directly if it's positive, otherwise, it outputs zero.
    /// Think of it as a function that "turns on" only for positive values.
    /// </para>
    /// </remarks>
    public static T ReLU(T x) => MathHelper.Max(x, _numOps.Zero);

    /// <summary>
    /// Calculates the derivative of the ReLU activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>1 if x is greater than 0, otherwise 0.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The derivative tells us how the function changes at a specific point.
    /// For ReLU, the derivative is 1 for positive inputs (the function increases at a constant rate)
    /// and 0 for negative inputs (the function doesn't change).
    /// </para>
    /// </remarks>
    public static T ReLUDerivative(T x) => _numOps.GreaterThan(x, _numOps.Zero) ? _numOps.One : _numOps.Zero;

    /// <summary>
    /// Applies the Sigmoid activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>A value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The Sigmoid function transforms any input into a value between 0 and 1.
    /// It's shaped like an S-curve and is useful when you need to predict probabilities.
    /// Large negative inputs become close to 0, and large positive inputs become close to 1.
    /// </para>
    /// </remarks>
    public static T Sigmoid(T x) => _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, _numOps.Exp(_numOps.Negate(x))));

    /// <summary>
    /// Calculates the derivative of the Sigmoid activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The derivative value.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The derivative of the Sigmoid function is highest at x=0 and approaches zero
    /// as x gets very large or very small. This means the function changes most rapidly near x=0.
    /// </para>
    /// </remarks>
    public static T SigmoidDerivative(T x)
    {
        T sigmoid = Sigmoid(x);
        return _numOps.Multiply(sigmoid, _numOps.Subtract(_numOps.One, sigmoid));
    }

    /// <summary>
    /// Applies the Hyperbolic Tangent (TanH) activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>A value between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: TanH is similar to Sigmoid but outputs values between -1 and 1 instead of 0 and 1.
    /// It's often preferred over Sigmoid because it's zero-centered, which can help with learning.
    /// </para>
    /// </remarks>
    public static T TanH(T x)
    {
        T exp2x = _numOps.Exp(_numOps.Multiply(x, _numOps.FromDouble(2)));
        return _numOps.Divide(_numOps.Subtract(exp2x, _numOps.One), _numOps.Add(exp2x, _numOps.One));
    }

    /// <summary>
    /// Calculates the derivative of the TanH activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The derivative value.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The derivative of TanH is highest at x=0 and approaches zero
    /// as x gets very large or very small, similar to Sigmoid's derivative.
    /// </para>
    /// </remarks>
    public static T TanHDerivative(T x)
    {
        T tanh = TanH(x);
        return _numOps.Subtract(_numOps.One, _numOps.Multiply(tanh, tanh));
    }

    /// <summary>
    /// Applies the Softmax activation function to a vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A vector where all elements sum to 1.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: Softmax converts a vector of numbers into a vector of probabilities that sum to 1.
    /// It's commonly used in the output layer of neural networks for multi-class classification problems.
    /// Larger input values result in larger probabilities relative to other values.
    /// </para>
    /// </remarks>
    public static Vector<T> Softmax(Vector<T> x)
    {
        Vector<T> expValues = x.Transform(_numOps.Exp);
        T sum = expValues.Sum();

        return expValues.Transform(v => _numOps.Divide(v, sum));
    }

    /// <summary>
    /// Calculates the Jacobian matrix of the Softmax function.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>The Jacobian matrix.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The Jacobian matrix represents how each output of the Softmax function
    /// changes with respect to each input. This is more complex than a simple derivative because
    /// Softmax outputs multiple values that are all interdependent.
    /// </para>
    /// <para>
    /// A Jacobian matrix is a matrix of partial derivatives that shows how each output
    /// changes when each input changes slightly.
    /// </para>
    /// </remarks>
    public static Matrix<T> SoftmaxDerivative(Vector<T> x)
    {
        Vector<T> s = Softmax(x);
        int n = s.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = _numOps.Multiply(s[i], _numOps.Subtract(_numOps.One, s[i]));
                }
                else
                {
                    jacobian[i, j] = _numOps.Negate(_numOps.Multiply(s[i], s[j]));
                }
            }
        }

        return jacobian;
    }

    /// <summary>
    /// Applies the Linear activation function (identity function).
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The input value unchanged.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The Linear activation function simply returns the input without any transformation.
    /// It's useful in regression problems or when you want the raw output of a neuron.
    /// </para>
    /// </remarks>
    public static T Linear(T x) => x;

    /// <summary>
    /// Calculates the derivative of the Linear activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>Always returns 1.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The derivative of a linear function is constant.
    /// Since the Linear activation function has a slope of 1 everywhere, its derivative is always 1.
    /// </para>
    /// </remarks>
    public static T LinearDerivative(T x) => _numOps.One;

    /// <summary>
    /// Applies the Leaky ReLU activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <param name="alpha">The slope for negative inputs (typically a small value like 0.01).</param>
    /// <returns>x if x > 0, otherwise alpha * x.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: Leaky ReLU is an improved version of ReLU that allows a small gradient
    /// when the input is negative. This helps prevent "dying ReLU" problem where neurons can
    /// get stuck during training. Instead of outputting zero for negative inputs, it outputs
    /// a small negative value.
    /// </para>
    /// </remarks>
    public static T LeakyReLU(T x, T alpha) => _numOps.GreaterThan(x, _numOps.Zero) ? x : _numOps.Multiply(alpha, x);

    /// <summary>
    /// Calculates the derivative of the Leaky ReLU activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <param name="alpha">The slope for negative inputs.</param>
    /// <returns>1 if x > 0, otherwise alpha.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The derivative of Leaky ReLU is 1 for positive inputs and alpha for negative inputs.
    /// This means the function changes at a constant rate in both regions, but at different rates.
    /// </para>
    /// </remarks>
    public static T LeakyReLUDerivative(T x, T alpha) => _numOps.GreaterThan(x, _numOps.Zero) ? _numOps.One : alpha;

    /// <summary>
    /// Applies the Exponential Linear Unit (ELU) activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <param name="alpha">The alpha parameter that controls the value when x is negative.</param>
    /// <returns>x if x > 0, otherwise alpha * (e^x - 1).</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: ELU is an activation function that, like ReLU, returns the input directly for positive values.
    /// For negative inputs, it returns a smooth curve that approaches -alpha as the input becomes more negative.
    /// This helps prevent "dying neurons" (a problem with ReLU) while still providing the benefits of non-linearity.
    /// </para>
    /// </remarks>
    public static T ELU(T x, T alpha) => _numOps.GreaterThan(x, _numOps.Zero) ? x : _numOps.Multiply(alpha, _numOps.Subtract(_numOps.Exp(x), _numOps.One));

    /// <summary>
    /// Calculates the derivative of the ELU activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <param name="alpha">The alpha parameter that controls the value when x is negative.</param>
    /// <returns>1 if x > 0, otherwise ELU(x, alpha) + alpha.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The derivative tells us how much the ELU function changes when its input changes slightly.
    /// This is essential for training neural networks through backpropagation.
    /// </para>
    /// </remarks>
    public static T ELUDerivative(T x, T alpha) => _numOps.GreaterThan(x, _numOps.Zero) ? _numOps.One : _numOps.Add(ELU(x, alpha), alpha);

    /// <summary>
    /// Applies the Scaled Exponential Linear Unit (SELU) activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The SELU activation of x.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: SELU is a variant of ELU that includes specific scaling factors (alpha and scale).
    /// These values are carefully chosen to ensure that the neural network self-normalizes during training,
    /// which helps with training stability and can eliminate the need for batch normalization.
    /// </para>
    /// </remarks>
    public static T SELU(T x)
    {
        T _alpha = _numOps.FromDouble(1.6732632423543772848170429916717);
        T _scale = _numOps.FromDouble(1.0507009873554804934193349852946);

        return _numOps.Multiply(_scale, ELU(x, _alpha));
    }

    /// <summary>
    /// Calculates the derivative of the SELU activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The derivative of SELU at point x.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: This calculates how much the SELU function changes when its input changes slightly.
    /// The specific values of alpha and scale are chosen to maintain self-normalization properties.
    /// </para>
    /// </remarks>
    public static T SELUDerivative(T x)
    {
        T _alpha = _numOps.FromDouble(1.6732632423543772848170429916717);
        T _scale = _numOps.FromDouble(1.0507009873554804934193349852946);

        return _numOps.Multiply(_scale, ELUDerivative(x, _alpha));
    }

    /// <summary>
    /// Applies the Softplus activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>ln(1 + e^x)</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: Softplus is a smooth approximation of the ReLU function.
    /// Unlike ReLU which has a sharp corner at x=0, Softplus transitions smoothly,
    /// which can be beneficial for some neural network applications.
    /// </para>
    /// </remarks>
    public static T Softplus(T x) => _numOps.Log(_numOps.Add(_numOps.One, _numOps.Exp(x)));

    /// <summary>
    /// Calculates the derivative of the Softplus activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The derivative of Softplus at point x.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The derivative of Softplus is actually the sigmoid function,
    /// which means it smoothly transitions from 0 to 1 as x increases.
    /// </para>
    /// </remarks>
    public static T SoftplusDerivative(T x) => _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, _numOps.Exp(_numOps.Negate(x))));

    /// <summary>
    /// Applies the SoftSign activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>x / (1 + |x|)</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: SoftSign is similar to the tanh function but approaches its asymptotes more slowly.
    /// This can help with the vanishing gradient problem in deep networks.
    /// The function maps inputs to values between -1 and 1.
    /// </para>
    /// </remarks>
    public static T SoftSign(T x) => _numOps.Divide(x, _numOps.Add(_numOps.One, _numOps.Abs(x)));

    /// <summary>
    /// Calculates the derivative of the SoftSign activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The derivative of SoftSign at point x.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: This calculates how much the SoftSign function changes when its input changes slightly.
    /// The derivative approaches zero as |x| gets larger, but more slowly than tanh.
    /// </para>
    /// </remarks>
    public static T SoftSignDerivative(T x)
    {
        T _denominator = _numOps.Add(_numOps.One, _numOps.Abs(x));
        return _numOps.Divide(_numOps.One, _numOps.Multiply(_denominator, _denominator));
    }

    /// <summary>
    /// Applies the Swish activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>x * sigmoid(x)</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: Swish is a newer activation function developed by researchers at Google.
    /// It's similar to ReLU but smoother and has been shown to outperform ReLU in some deep networks.
    /// It multiplies the input by the sigmoid of the input.
    /// </para>
    /// </remarks>
    public static T Swish(T x) => _numOps.Multiply(x, Sigmoid(x));

    /// <summary>
    /// Calculates the derivative of the Swish activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The derivative of Swish at point x.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: This calculates how much the Swish function changes when its input changes slightly.
    /// The formula looks complex but is essential for training neural networks that use Swish activation.
    /// </para>
    /// </remarks>
    public static T SwishDerivative(T x)
    {
        T _sigX = Sigmoid(x);
        return _numOps.Add(_numOps.Multiply(x, _numOps.Multiply(_sigX, _numOps.Subtract(_numOps.One, _sigX))), _sigX);
    }

    /// <summary>
    /// Applies the Gaussian Error Linear Unit (GELU) activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The GELU activation of x.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: GELU is an activation function used in modern transformer models like BERT.
    /// It can be thought of as a smoother version of ReLU that weights inputs by their value.
    /// Inputs near zero are suppressed, while larger positive inputs are passed through almost unchanged.
    /// </para>
    /// </remarks>
    public static T GELU(T x)
    {
        T _sqrt2OverPi = _numOps.FromDouble(0.7978845608028654);
        T _half = _numOps.FromDouble(0.5);
        T _tanh = MathHelper.Tanh(_numOps.Multiply(_sqrt2OverPi, _numOps.Add(x, _numOps.Multiply(_numOps.FromDouble(0.044715), _numOps.Power(x, _numOps.FromDouble(3))))));

        return _numOps.Multiply(_half, _numOps.Multiply(x, _numOps.Add(_numOps.One, _tanh)));
    }

    /// <summary>
    /// Calculates the derivative of the GELU activation function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The derivative of GELU at point x.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: This calculates how much the GELU function changes when its input changes slightly.
    /// The formula is complex but necessary for training neural networks that use GELU activation.
    /// </para>
    /// </remarks>
    public static T GELUDerivative(T x)
    {
        T _sqrt2OverPi = _numOps.FromDouble(0.7978845608028654);
        T _half = _numOps.FromDouble(0.5);
        T _tanh = MathHelper.Tanh(_numOps.Multiply(_sqrt2OverPi, _numOps.Add(x, _numOps.Multiply(_numOps.FromDouble(0.044715), _numOps.Power(x, _numOps.FromDouble(3))))));
        T _sech2 = _numOps.Subtract(_numOps.One, _numOps.Multiply(_tanh, _tanh));

        return _numOps.Add(_half, _numOps.Multiply(_half, _numOps.Multiply(_tanh, _numOps.Add(_numOps.One, _numOps.Multiply(x, _numOps.Multiply(_sqrt2OverPi, _sech2))))));
    }

    /// <summary>
    /// Calculates the Mean Squared Error (MSE) loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The mean squared error.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: Mean Squared Error is one of the most common loss functions used in regression problems.
    /// It calculates the average of the squared differences between predicted and actual values.
    /// Squaring the differences ensures that larger errors have a proportionally larger effect on the total loss.
    /// </para>
    /// </remarks>
    public static T MeanSquaredError(Vector<T> predicted, Vector<T> actual)
    {
        return StatisticsHelper<T>.CalculateMeanSquaredError(predicted, actual);
    }
    /// <summary>
    /// Calculates the derivative of the Mean Squared Error loss function.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivative of MSE for each element.</returns>
    /// <exception cref="ArgumentException">Thrown when predicted and actual vectors have different lengths.</exception>
    /// <remarks>
    /// <para>For Beginners: The derivative tells us how to adjust our model to reduce the error.
    /// For Mean Squared Error, the derivative is 2*(predicted-actual)/n, where n is the number of elements.
    /// This helps the model learn by showing which direction to move to reduce the error.</para>
    /// </remarks>
    public static Vector<T> MeanSquaredErrorDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        return predicted.Subtract(actual).Transform(x => _numOps.Multiply(_numOps.FromDouble(2), x)).Divide(_numOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the Mean Absolute Error between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The mean absolute error value.</returns>
    /// <remarks>
    /// <para>For Beginners: Mean Absolute Error (MAE) measures the average magnitude of errors between 
    /// predicted and actual values, without considering their direction. It's simply the average of 
    /// absolute differences between predictions and actual values.</para>
    /// </remarks>
    public static T MeanAbsoluteError(Vector<T> predicted, Vector<T> actual)
    {
        return StatisticsHelper<T>.CalculateMeanAbsoluteError(predicted, actual);
    }

    /// <summary>
    /// Calculates the derivative of the Mean Absolute Error loss function.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivative of MAE for each element.</returns>
    /// <exception cref="ArgumentException">Thrown when predicted and actual vectors have different lengths.</exception>
    /// <remarks>
    /// <para>For Beginners: The derivative of MAE is +1 when the prediction is greater than the actual value,
    /// and -1 when the prediction is less than the actual value. This helps the model understand which 
    /// direction to adjust its predictions.</para>
    /// </remarks>
    public static Vector<T> MeanAbsoluteErrorDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        return predicted.Subtract(actual).Transform(x => _numOps.GreaterThan(x, _numOps.Zero) ? _numOps.One : _numOps.Negate(_numOps.One)).Divide(_numOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the Binary Cross Entropy loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities between 0 and 1).</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <returns>The binary cross entropy loss value.</returns>
    /// <exception cref="ArgumentException">Thrown when predicted and actual vectors have different lengths.</exception>
    /// <remarks>
    /// <para>For Beginners: Binary Cross Entropy is commonly used for binary classification problems
    /// (where outputs are either 0 or 1). It measures how well the predicted probabilities match the 
    /// actual binary outcomes. Lower values indicate better predictions.</para>
    /// <para>The function includes a small epsilon value (1e-15) to prevent log(0) errors.</para>
    /// </remarks>
    public static T BinaryCrossEntropy(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        T sum = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T p = MathHelper.Clamp(predicted[i], _numOps.FromDouble(1e-15), _numOps.FromDouble(1 - 1e-15));
            sum = _numOps.Add(sum, _numOps.Add(
                _numOps.Multiply(actual[i], _numOps.Log(p)),
                _numOps.Multiply(_numOps.Subtract(_numOps.One, actual[i]), _numOps.Log(_numOps.Subtract(_numOps.One, p)))
            ));
        }

        return _numOps.Negate(_numOps.Divide(sum, _numOps.FromDouble(predicted.Length)));
    }

    /// <summary>
    /// Calculates the derivative of the Binary Cross Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities between 0 and 1).</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <returns>A vector containing the derivative of BCE for each element.</returns>
    /// <exception cref="ArgumentException">Thrown when predicted and actual vectors have different lengths.</exception>
    /// <remarks>
    /// <para>For Beginners: This derivative helps the model learn by indicating how to adjust predictions
    /// to better match the actual values. For binary cross entropy, the derivative is (p-y)/(p*(1-p)),
    /// where p is the predicted probability and y is the actual value.</para>
    /// </remarks>
    public static Vector<T> BinaryCrossEntropyDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T p = MathHelper.Clamp(predicted[i], _numOps.FromDouble(1e-15), _numOps.FromDouble(1 - 1e-15));
            derivative[i] = _numOps.Divide(
                _numOps.Subtract(p, actual[i]),
                _numOps.Multiply(p, _numOps.Subtract(_numOps.One, p))
            );
        }
        return derivative.Divide(_numOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the Categorical Cross Entropy loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities that sum to 1 across categories).</param>
    /// <param name="actual">The actual (target) values (typically one-hot encoded).</param>
    /// <returns>The categorical cross entropy loss value.</returns>
    /// <exception cref="ArgumentException">Thrown when predicted and actual vectors have different lengths.</exception>
    /// <remarks>
    /// <para>For Beginners: Categorical Cross Entropy is used for multi-class classification problems.
    /// It measures how well the predicted probability distribution matches the actual distribution.
    /// Lower values indicate better predictions.</para>
    /// <para>The function clamps values to prevent log(0) errors.</para>
    /// </remarks>
    public static T CategoricalCrossEntropy(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        T sum = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T p = MathHelper.Clamp(predicted[i], _numOps.FromDouble(1e-15), _numOps.FromDouble(1 - 1e-15));
            sum = _numOps.Add(sum, _numOps.Multiply(actual[i], _numOps.Log(p)));
        }
        return _numOps.Negate(sum);
    }

    /// <summary>
    /// Calculates the derivative of the Categorical Cross Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities that sum to 1 across categories).</param>
    /// <param name="actual">The actual (target) values (typically one-hot encoded).</param>
    /// <returns>A vector containing the derivative of CCE for each element.</returns>
    /// <exception cref="ArgumentException">Thrown when predicted and actual vectors have different lengths.</exception>
    /// <remarks>
    /// <para>For Beginners: This derivative helps the model learn by indicating how to adjust predictions
    /// to better match the actual class distribution. For categorical cross entropy with softmax outputs,
    /// the derivative simplifies to (predicted-actual)/n, where n is the number of samples.</para>
    /// </remarks>
    public static Vector<T> CategoricalCrossEntropyDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        return predicted.Subtract(actual).Divide(_numOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the Huber loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <param name="delta">The threshold parameter that determines the transition between quadratic and linear loss.</param>
    /// <returns>The Huber loss value.</returns>
    /// <exception cref="ArgumentException">Thrown when predicted and actual vectors have different lengths.</exception>
    /// <remarks>
    /// <para>For Beginners: Huber loss combines the best properties of MSE and MAE. For small errors
    /// (less than delta), it behaves like MSE, which is smooth and easily optimized. For large errors,
    /// it behaves like MAE, which is less sensitive to outliers.</para>
    /// <para>This makes Huber loss robust to outliers while still providing smooth gradients for small errors.</para>
    /// </remarks>
    public static T HuberLoss(Vector<T> predicted, Vector<T> actual, T? delta = default)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        delta = delta ?? _numOps.FromDouble(1.0);

        T sum = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = _numOps.Abs(_numOps.Subtract(predicted[i], actual[i]));
            if (_numOps.LessThanOrEquals(diff, delta))
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Multiply(diff, diff)));
            }
            else
            {
                sum = _numOps.Add(sum, _numOps.Subtract(
                    _numOps.Multiply(delta, diff),
                    _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Multiply(delta, delta))
                ));
            }
        }

        return _numOps.Divide(sum, _numOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Huber loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <param name="delta">The threshold parameter that determines the transition between quadratic and linear loss. Defaults to 1.0.</param>
    /// <returns>A vector containing the derivative of Huber loss for each element.</returns>
    /// <exception cref="ArgumentException">Thrown when predicted and actual vectors have different lengths.</exception>
    /// <remarks>
    /// <para>For Beginners: The Huber loss derivative helps the model learn by showing how to adjust predictions.
    /// It behaves like Mean Squared Error for small errors (less than delta) and like Mean Absolute Error for large errors.
    /// This makes it less sensitive to outliers than Mean Squared Error while still providing smooth gradients near zero.</para>
    /// </remarks>
    public static Vector<T> HuberLossDerivative(Vector<T> predicted, Vector<T> actual, T? delta = default)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        delta = delta ?? _numOps.FromDouble(1.0);

        Vector<T> derivative = new(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = _numOps.Subtract(predicted[i], actual[i]);
            if (_numOps.LessThanOrEquals(_numOps.Abs(diff), delta))
            {
                derivative[i] = diff;
            }
            else
            {
                derivative[i] = _numOps.Multiply(delta, _numOps.GreaterThan(diff, _numOps.Zero) ? _numOps.One : _numOps.Negate(_numOps.One));
            }
        }

        return derivative.Divide(_numOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the Log-Cosh loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The Log-Cosh loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Log-Cosh loss is a smooth approximation of the Mean Absolute Error.
    /// It calculates the logarithm of the hyperbolic cosine of the difference between predictions and actual values.
    /// This loss function is useful because it's smooth everywhere (unlike Huber loss) and less affected by outliers
    /// than Mean Squared Error.</para>
    /// </remarks>
    public static T LogCoshLoss(Vector<T> predicted, Vector<T> actual)
    {
        var sum = _numOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            var diff = _numOps.Subtract(predicted[i], actual[i]);
            var logCosh = _numOps.Log(_numOps.Add(_numOps.Exp(diff), _numOps.Exp(_numOps.Negate(diff))));
            sum = _numOps.Add(sum, logCosh);
        }

        return _numOps.Divide(sum, _numOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Log-Cosh loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivative of Log-Cosh loss for each element.</returns>
    /// <remarks>
    /// <para>For Beginners: The derivative of Log-Cosh loss is simply the hyperbolic tangent (tanh) of the difference
    /// between predicted and actual values. This derivative is always between -1 and 1, which helps prevent
    /// exploding gradients during training.</para>
    /// </remarks>
    public static Vector<T> LogCoshLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length);

        for (int i = 0; i < predicted.Length; i++)
        {
            var diff = _numOps.Subtract(predicted[i], actual[i]);
            derivative[i] = MathHelper.Tanh(diff);
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Quantile loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <param name="quantile">The quantile value between 0 and 1 to calculate the loss for.</param>
    /// <returns>The Quantile loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Quantile loss helps predict specific percentiles of data rather than just the average.
    /// For example, with quantile=0.5, it predicts the median value. With quantile=0.9, it predicts the 90th percentile.
    /// This is useful when you care more about certain parts of the distribution, like predicting worst-case scenarios
    /// or ensuring predictions don't fall below a certain threshold.</para>
    /// </remarks>
    public static T QuantileLoss(Vector<T> predicted, Vector<T> actual, T quantile)
    {
        var sum = _numOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            var diff = _numOps.Subtract(actual[i], predicted[i]);
            var loss = _numOps.GreaterThan(diff, _numOps.Zero)
                ? _numOps.Multiply(quantile, diff)
                : _numOps.Multiply(_numOps.Subtract(_numOps.One, quantile), _numOps.Negate(diff));
            sum = _numOps.Add(sum, loss);
        }

        return _numOps.Divide(sum, _numOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Quantile loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <param name="quantile">The quantile value between 0 and 1.</param>
    /// <returns>A vector containing the derivative of Quantile loss for each element.</returns>
    /// <remarks>
    /// <para>For Beginners: The derivative of Quantile loss tells the model how to adjust its predictions
    /// to better match the desired quantile. When the actual value is greater than predicted, the derivative
    /// is -quantile; otherwise, it's (1-quantile). This asymmetric gradient helps the model focus on
    /// predicting the specific quantile rather than just the average.</para>
    /// </remarks>
    public static Vector<T> QuantileLossDerivative(Vector<T> predicted, Vector<T> actual, T quantile)
    {
        var derivative = new Vector<T>(predicted.Length);

        for (int i = 0; i < predicted.Length; i++)
        {
            var diff = _numOps.Subtract(actual[i], predicted[i]);
            derivative[i] = _numOps.GreaterThan(diff, _numOps.Zero)
                ? _numOps.Negate(quantile)
                : _numOps.Subtract(_numOps.One, quantile);
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Cross-Entropy loss between predicted and actual probability distributions.
    /// </summary>
    /// <param name="predicted">The predicted probability distribution.</param>
    /// <param name="actual">The actual (target) probability distribution.</param>
    /// <returns>The Cross-Entropy loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Cross-Entropy loss measures how different two probability distributions are.
    /// It's commonly used for classification problems where the model outputs probabilities.
    /// Lower values indicate that the predicted distribution is closer to the actual distribution.
    /// This loss function encourages the model to be confident about correct predictions and
    /// penalizes it heavily for being confident about incorrect predictions.</para>
    /// </remarks>
    public static T CrossEntropyLoss(Vector<T> predicted, Vector<T> actual)
    {
        var sum = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = _numOps.Add(sum, _numOps.Multiply(actual[i], _numOps.Log(predicted[i])));
        }

        return _numOps.Negate(_numOps.Divide(sum, _numOps.FromDouble(predicted.Length)));
    }

    /// <summary>
    /// Calculates the derivative of the Cross-Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted probability distribution.</param>
    /// <param name="actual">The actual (target) probability distribution.</param>
    /// <returns>A vector containing the derivative of Cross-Entropy loss for each element.</returns>
    /// <remarks>
    /// <para>For Beginners: The derivative of Cross-Entropy loss shows how to adjust the predicted probabilities
    /// to make them closer to the actual probabilities. For each class, the derivative is -actual/predicted,
    /// which means the model will adjust more aggressively when the predicted probability is low but the
    /// actual probability is high.</para>
    /// </remarks>
    public static Vector<T> CrossEntropyLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = _numOps.Divide(_numOps.Negate(actual[i]), predicted[i]);
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Binary Cross-Entropy loss between predicted and actual binary values.
    /// </summary>
    /// <param name="predicted">The predicted probabilities (values between 0 and 1).</param>
    /// <param name="actual">The actual (target) binary values (0 or 1).</param>
    /// <returns>The Binary Cross-Entropy loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Binary Cross-Entropy loss is specifically designed for binary classification problems
    /// (where there are only two possible outcomes, like yes/no or true/false). It measures how well the model's
    /// predicted probabilities match the actual binary outcomes. Lower values indicate better predictions.
    /// This loss function is ideal when your model needs to output a probability between 0 and 1.</para>
    /// </remarks>
    public static T BinaryCrossEntropyLoss(Vector<T> predicted, Vector<T> actual)
    {
        var sum = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = _numOps.Add(sum, 
                _numOps.Add(
                    _numOps.Multiply(actual[i], _numOps.Log(predicted[i])),
                    _numOps.Multiply(_numOps.Subtract(_numOps.One, actual[i]), _numOps.Log(_numOps.Subtract(_numOps.One, predicted[i])))
                )
            );
        }

        return _numOps.Negate(_numOps.Divide(sum, _numOps.FromDouble(predicted.Length)));
    }

    /// <summary>
    /// Calculates the derivative of the Binary Cross Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model (probabilities between 0 and 1).</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <returns>A vector containing the gradient of the loss with respect to each prediction.</returns>
    /// <remarks>
    /// <para>For Beginners: The derivative tells the model how to adjust its predictions to get closer to the actual values.
    /// Binary Cross Entropy is commonly used for binary classification problems where we're predicting
    /// if something belongs to one of two categories.</para>
    /// </remarks>
    public static Vector<T> BinaryCrossEntropyLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = _numOps.Divide(
                _numOps.Subtract(predicted[i], actual[i]),
                _numOps.Multiply(predicted[i], _numOps.Subtract(_numOps.One, predicted[i]))
            );
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Categorical Cross Entropy loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted probability distributions (each row should sum to 1).</param>
    /// <param name="actual">The actual (target) probability distributions or one-hot encoded vectors.</param>
    /// <returns>The average Categorical Cross Entropy loss across all samples.</returns>
    /// <remarks>
    /// <para>For Beginners: This loss function measures how well your model is predicting multiple categories.
    /// It's commonly used when classifying data into more than two categories (like recognizing digits 0-9).
    /// Lower values indicate better predictions.</para>
    /// </remarks>
    public static T CategoricalCrossEntropyLoss(Matrix<T> predicted, Matrix<T> actual)
    {
        var sum = _numOps.Zero;
        for (int i = 0; i < predicted.Rows; i++)
        {
            for (int j = 0; j < predicted.Columns; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(actual[i, j], _numOps.Log(predicted[i, j])));
            }
        }

        return _numOps.Negate(_numOps.Divide(sum, _numOps.FromDouble(predicted.Rows)));
    }

    /// <summary>
    /// Calculates the derivative of the Categorical Cross Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted probability distributions (each row should sum to 1).</param>
    /// <param name="actual">The actual (target) probability distributions or one-hot encoded vectors.</param>
    /// <returns>A matrix containing the gradient of the loss with respect to each prediction.</returns>
    /// <remarks>
    /// <para>For Beginners: This derivative helps the model learn by showing how to adjust its predictions
    /// for multi-category classification problems. The gradient points in the direction that would
    /// increase the loss, so the model updates in the opposite direction.</para>
    /// </remarks>
    public static Matrix<T> CategoricalCrossEntropyLossDerivative(Matrix<T> predicted, Matrix<T> actual)
    {
        var derivative = new Matrix<T>(predicted.Rows, predicted.Columns);
        for (int i = 0; i < predicted.Rows; i++)
        {
            for (int j = 0; j < predicted.Columns; j++)
            {
                derivative[i, j] = _numOps.Divide(_numOps.Negate(actual[i, j]), predicted[i, j]);
            }
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Hinge loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values, typically -1 or 1.</param>
    /// <returns>The average Hinge loss across all samples.</returns>
    /// <remarks>
    /// <para>For Beginners: Hinge loss is commonly used in support vector machines (SVMs) for classification.
    /// It penalizes predictions that are both incorrect and not confident enough. The actual values
    /// should be -1 or 1, representing the two classes. Lower values indicate better predictions.</para>
    /// </remarks>
    public static T HingeLoss(Vector<T> predicted, Vector<T> actual)
    {
        var sum = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = _numOps.Add(sum, MathHelper.Max(_numOps.Zero, _numOps.Subtract(_numOps.One, _numOps.Multiply(actual[i], predicted[i]))));
        }

        return _numOps.Divide(sum, _numOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Hinge loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values, typically -1 or 1.</param>
    /// <returns>A vector containing the gradient of the loss with respect to each prediction.</returns>
    /// <remarks>
    /// <para>For Beginners: This derivative helps the model learn by showing how to adjust its predictions
    /// for binary classification problems using the hinge loss. If the prediction is correct and confident
    /// enough, the gradient is zero (no change needed). Otherwise, it provides direction for improvement.</para>
    /// </remarks>
    public static Vector<T> HingeLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = _numOps.LessThan(_numOps.Multiply(actual[i], predicted[i]), _numOps.One) ? _numOps.Negate(actual[i]) : _numOps.Zero;
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Kullback-Leibler Divergence between predicted and actual probability distributions.
    /// </summary>
    /// <param name="predicted">The predicted probability distribution.</param>
    /// <param name="actual">The actual (target) probability distribution.</param>
    /// <returns>The KL divergence value.</returns>
    /// <remarks>
    /// <para>For Beginners: KL Divergence measures how one probability distribution differs from another.
    /// It's not a true distance metric (it's asymmetric) but helps quantify the difference between
    /// two distributions. It's often used in variational autoencoders and reinforcement learning.
    /// Lower values indicate the distributions are more similar.</para>
    /// </remarks>
    public static T KullbackLeiblerDivergence(Vector<T> predicted, Vector<T> actual)
    {
        var sum = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = _numOps.Add(sum, _numOps.Multiply(actual[i], _numOps.Log(_numOps.Divide(actual[i], predicted[i]))));
        }

        return sum;
    }

    /// <summary>
    /// Calculates the derivative of the Kullback-Leibler Divergence.
    /// </summary>
    /// <param name="predicted">The predicted probability distribution.</param>
    /// <param name="actual">The actual (target) probability distribution.</param>
    /// <returns>A vector containing the gradient of the KL divergence with respect to each prediction.</returns>
    /// <remarks>
    /// <para>For Beginners: This derivative helps the model learn by showing how to adjust its predicted
    /// probability distribution to make it more similar to the actual distribution. The gradient
    /// indicates the direction of steepest increase in the divergence.</para>
    /// </remarks>
    public static Vector<T> KullbackLeiblerDivergenceDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = _numOps.Negate(_numOps.Divide(actual[i], predicted[i]));
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Poisson loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model (should be positive).</param>
    /// <param name="actual">The actual (target) values (typically counts or rates).</param>
    /// <returns>The average Poisson loss across all samples.</returns>
    /// <remarks>
    /// <para>For Beginners: Poisson loss is used when predicting count data (like number of events in a time period).
    /// It's based on the Poisson distribution, which models the probability of a given number of events
    /// occurring in a fixed interval of time or space. Lower values indicate better predictions.</para>
    /// </remarks>
    public static T PoissonLoss(Vector<T> predicted, Vector<T> actual)
    {
        var sum = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = _numOps.Add(sum, _numOps.Subtract(predicted[i], _numOps.Multiply(actual[i], _numOps.Log(predicted[i]))));
        }

        return _numOps.Divide(sum, _numOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Poisson loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model (should be positive).</param>
    /// <param name="actual">The actual (target) values (typically counts or rates).</param>
    /// <returns>A vector containing the gradient of the loss with respect to each prediction.</returns>
    /// <remarks>
    /// <para>For Beginners: This derivative helps the model learn by showing how to adjust its predictions
    /// for count data. It indicates the direction of steepest increase in the Poisson loss.</para>
    /// </remarks>
    public static Vector<T> PoissonLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = _numOps.Subtract(_numOps.One, _numOps.Divide(actual[i], predicted[i]));
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Cosine Similarity Loss between two vectors.
    /// </summary>
    /// <param name="predicted">The predicted vector from the model.</param>
    /// <param name="actual">The actual (target) vector.</param>
    /// <returns>A scalar value representing the cosine similarity loss.</returns>
    /// <remarks>
    /// <para>For Beginners: Cosine similarity measures how similar two vectors are in terms of their direction,
    /// regardless of their magnitude (size). The loss is calculated as 1 minus the cosine similarity,
    /// so a value of 0 means the vectors are perfectly aligned (very similar), while a value of 1 means
    /// they are completely different.</para>
    /// </remarks>
    public static T CosineSimilarityLoss(Vector<T> predicted, Vector<T> actual)
    {
        var dotProduct = _numOps.Zero;
        var normPredicted = _numOps.Zero;
        var normActual = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(predicted[i], actual[i]));
            normPredicted = _numOps.Add(normPredicted, _numOps.Multiply(predicted[i], predicted[i]));
            normActual = _numOps.Add(normActual, _numOps.Multiply(actual[i], actual[i]));
        }

        return _numOps.Subtract(_numOps.One, _numOps.Divide(dotProduct, _numOps.Multiply(_numOps.Sqrt(normPredicted), _numOps.Sqrt(normActual))));
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the Cosine Similarity Loss with respect to the predicted values.
    /// </summary>
    /// <param name="predicted">The predicted vector from the model.</param>
    /// <param name="actual">The actual (target) vector.</param>
    /// <returns>A vector containing the gradient of the loss with respect to each prediction.</returns>
    /// <remarks>
    /// <para>For Beginners: The derivative tells the model how to adjust its predictions to make them more similar
    /// to the actual values in terms of direction. This helps the model learn to align its predictions with the
    /// target values during training.</para>
    /// </remarks>
    public static Vector<T> CosineSimilarityLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var dotProduct = _numOps.Zero;
        var normPredicted = _numOps.Zero;
        var normActual = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(predicted[i], actual[i]));
            normPredicted = _numOps.Add(normPredicted, _numOps.Multiply(predicted[i], predicted[i]));
            normActual = _numOps.Add(normActual, _numOps.Multiply(actual[i], actual[i]));
        }
    
        var derivative = new Vector<T>(predicted.Length);
        var normProduct = _numOps.Multiply(_numOps.Sqrt(normPredicted), _numOps.Sqrt(normActual));
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = _numOps.Divide(
                _numOps.Subtract(
                    _numOps.Multiply(actual[i], normPredicted),
                    _numOps.Multiply(predicted[i], dotProduct)
                ),
                _numOps.Multiply(normProduct, normPredicted)
            );
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Euclidean distance between two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>A scalar value representing the Euclidean distance.</returns>
    /// <remarks>
    /// <para>For Beginners: Euclidean distance is the straight-line distance between two points in space.
    /// Think of it as measuring the length of a ruler placed between two points. This is used in many
    /// machine learning algorithms to measure how different two data points are.</para>
    /// </remarks>
    private static T EuclideanDistance(Vector<T> v1, Vector<T> v2)
    {
        var sum = _numOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            var diff = _numOps.Subtract(v1[i], v2[i]);
            sum = _numOps.Add(sum, _numOps.Multiply(diff, diff));
        }

        return _numOps.Sqrt(sum);
    }

    /// <summary>
    /// Calculates the Focal Loss, which is a modified version of cross-entropy loss that gives more weight to hard-to-classify examples.
    /// </summary>
    /// <param name="predicted">The predicted probabilities from the model.</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <param name="gamma">The focusing parameter that adjusts how much to focus on hard-to-classify examples.</param>
    /// <param name="alpha">The weighting factor for the positive class.</param>
    /// <returns>A scalar value representing the focal loss.</returns>
    /// <remarks>
    /// <para>For Beginners: Focal Loss is designed to help when your data is imbalanced (one class appears much more than others).
    /// It gives more importance to examples that are difficult to classify correctly, helping the model focus on learning from
    /// its mistakes rather than getting better at what it already does well. The gamma parameter controls this focus - higher
    /// values mean more focus on hard examples. The alpha parameter helps balance between positive and negative examples.</para>
    /// </remarks>
    public static T FocalLoss(Vector<T> predicted, Vector<T> actual, T gamma, T alpha)
    {
        T loss = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T pt = _numOps.Equals(actual[i], _numOps.One) ? predicted[i] : _numOps.Subtract(_numOps.One, predicted[i]);
            T alphaT = _numOps.Equals(actual[i], _numOps.One) ? alpha : _numOps.Subtract(_numOps.One, alpha);
            loss = _numOps.Add(loss, _numOps.Multiply(_numOps.Negate(alphaT), 
                _numOps.Multiply(_numOps.Power(_numOps.Subtract(_numOps.One, pt), gamma), _numOps.Log(pt))));
        }

        return _numOps.Divide(loss, _numOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the Focal Loss with respect to the predicted values.
    /// </summary>
    /// <param name="predicted">The predicted probabilities from the model.</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <param name="gamma">The focusing parameter that adjusts how much to focus on hard-to-classify examples.</param>
    /// <param name="alpha">The weighting factor for the positive class.</param>
    /// <returns>A vector containing the gradient of the loss with respect to each prediction.</returns>
    /// <remarks>
    /// <para>For Beginners: This derivative tells the model how to adjust its predictions to reduce the Focal Loss.
    /// It gives stronger signals for hard-to-classify examples, helping the model improve where it's struggling the most.</para>
    /// </remarks>
    public static Vector<T> FocalLossDerivative(Vector<T> predicted, Vector<T> actual, T gamma, T alpha)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T pt = _numOps.Equals(actual[i], _numOps.One) ? predicted[i] : _numOps.Subtract(_numOps.One, predicted[i]);
            T alphaT = _numOps.Equals(actual[i], _numOps.One) ? alpha : _numOps.Subtract(_numOps.One, alpha);
            T term1 = _numOps.Multiply(_numOps.Negate(alphaT), _numOps.Power(_numOps.Subtract(_numOps.One, pt), _numOps.Subtract(gamma, _numOps.One)));
            T term2 = _numOps.Subtract(_numOps.Multiply(gamma, _numOps.Subtract(_numOps.One, pt)), pt);
            derivative[i] = _numOps.Multiply(term1, term2);
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Triplet Loss, which is used for learning embeddings where similar items should be close together and dissimilar items far apart.
    /// </summary>
    /// <param name="anchor">The anchor samples (reference points).</param>
    /// <param name="positive">The positive samples (similar to anchors).</param>
    /// <param name="negative">The negative samples (dissimilar to anchors).</param>
    /// <param name="margin">The minimum desired difference between positive and negative distances.</param>
    /// <returns>A scalar value representing the triplet loss.</returns>
    /// <remarks>
    /// <para>For Beginners: Triplet Loss helps create embeddings (numerical representations) where similar items are close together
    /// and different items are far apart. It works with triplets of data: an anchor (reference point), a positive example (similar to anchor),
    /// and a negative example (different from anchor). The loss encourages the model to make the distance between the anchor and positive
    /// smaller than the distance between the anchor and negative by at least the margin amount.</para>
    /// <para>This is commonly used in face recognition, recommendation systems, and other applications where you need to measure similarity.</para>
    /// </remarks>
    public static T TripletLoss(Matrix<T> anchor, Matrix<T> positive, Matrix<T> negative, T margin)
    {
        var batchSize = anchor.Rows;
        var totalLoss = _numOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            var anchorSample = anchor.GetRow(i);
            var positiveSample = positive.GetRow(i);
            var negativeSample = negative.GetRow(i);

            var positiveDistance = EuclideanDistance(anchorSample, positiveSample);
            var negativeDistance = EuclideanDistance(anchorSample, negativeSample);

            var loss = MathHelper.Max(_numOps.Zero, _numOps.Add(_numOps.Subtract(positiveDistance, negativeDistance), margin));
            totalLoss = _numOps.Add(totalLoss, loss);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Calculates the gradient of the triplet loss function for anchor, positive, and negative samples.
    /// </summary>
    /// <param name="anchor">The anchor samples matrix.</param>
    /// <param name="positive">The positive samples matrix (similar to anchor).</param>
    /// <param name="negative">The negative samples matrix (dissimilar to anchor).</param>
    /// <param name="margin">The margin value that enforces a minimum distance between positive and negative pairs.</param>
    /// <returns>A tuple containing the gradients for anchor, positive, and negative samples.</returns>
    /// <remarks>
    /// <para>For Beginners: Triplet loss helps a model learn to place similar items close together and 
    /// dissimilar items far apart in a feature space. The "anchor" is your reference point, the "positive" 
    /// is something similar to your anchor, and the "negative" is something different. This function 
    /// calculates how to adjust each of these points to improve the model's understanding of similarity.</para>
    /// </remarks>
    public static (Matrix<T> AnchorGradient, Matrix<T> PositiveGradient, Matrix<T> NegativeGradient) TripletLossDerivative(Matrix<T> anchor, Matrix<T> positive, Matrix<T> negative, T margin)
    {
        var batchSize = anchor.Rows;
        var featureCount = anchor.Columns;

        var anchorGradient = new Matrix<T>(batchSize, featureCount);
        var positiveGradient = new Matrix<T>(batchSize, featureCount);
        var negativeGradient = new Matrix<T>(batchSize, featureCount);

        for (int i = 0; i < batchSize; i++)
        {
            var anchorSample = anchor.GetRow(i);
            var positiveSample = positive.GetRow(i);
            var negativeSample = negative.GetRow(i);

            var positiveDistance = EuclideanDistance(anchorSample, positiveSample);
            var negativeDistance = EuclideanDistance(anchorSample, negativeSample);

            var loss = _numOps.Subtract(_numOps.Add(positiveDistance, margin), negativeDistance);

            if (_numOps.GreaterThan(loss, _numOps.Zero))
            {
                for (int j = 0; j < featureCount; j++)
                {
                    var anchorPositiveDiff = _numOps.Subtract(anchorSample[j], positiveSample[j]);
                    var anchorNegativeDiff = _numOps.Subtract(anchorSample[j], negativeSample[j]);

                    anchorGradient[i, j] = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Subtract(anchorPositiveDiff, anchorNegativeDiff));
                    positiveGradient[i, j] = _numOps.Multiply(_numOps.FromDouble(-2), anchorPositiveDiff);
                    negativeGradient[i, j] = _numOps.Multiply(_numOps.FromDouble(2), anchorNegativeDiff);
                }
            }
            else
            {
                // If the triplet loss is zero or negative, the gradients are zero
                for (int j = 0; j < featureCount; j++)
                {
                    anchorGradient[i, j] = _numOps.Zero;
                    positiveGradient[i, j] = _numOps.Zero;
                    negativeGradient[i, j] = _numOps.Zero;
                }
            }
        }

        return (anchorGradient, positiveGradient, negativeGradient);
    }

    /// <summary>
    /// Calculates the contrastive loss between two output vectors based on their similarity.
    /// </summary>
    /// <param name="output1">The first output vector.</param>
    /// <param name="output2">The second output vector.</param>
    /// <param name="similarityLabel">A value of 1 indicates similar pairs, 0 indicates dissimilar pairs.</param>
    /// <param name="margin">The margin value for dissimilar pairs.</param>
    /// <returns>The contrastive loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Contrastive loss helps the model learn to group similar items together and 
    /// push dissimilar items apart. When two items are labeled as similar (similarityLabel = 1), 
    /// the loss increases as their distance increases. When they're labeled as dissimilar (similarityLabel = 0), 
    /// the loss increases only if they're closer than the specified margin.</para>
    /// </remarks>
    public static T ContrastiveLoss(Vector<T> output1, Vector<T> output2, T similarityLabel, T margin)
    {
        T distance = EuclideanDistance(output1, output2);
        T similarTerm = _numOps.Multiply(similarityLabel, _numOps.Power(distance, _numOps.FromDouble(2)));
        T dissimilarTerm = _numOps.Multiply(_numOps.Subtract(_numOps.One, similarityLabel), 
            _numOps.Power(MathHelper.Max(_numOps.Zero, _numOps.Subtract(margin, distance)), _numOps.FromDouble(2)));

        return _numOps.Add(similarTerm, dissimilarTerm);
    }

    /// <summary>
    /// Calculates the gradient of the contrastive loss function for both output vectors.
    /// </summary>
    /// <param name="output1">The first output vector.</param>
    /// <param name="output2">The second output vector.</param>
    /// <param name="similarityLabel">A value of 1 indicates similar pairs, 0 indicates dissimilar pairs.</param>
    /// <param name="margin">The margin value for dissimilar pairs.</param>
    /// <returns>A tuple containing the gradients for both output vectors.</returns>
    /// <remarks>
    /// <para>For Beginners: This function calculates how to adjust both vectors to minimize the contrastive loss.
    /// For similar pairs, it pushes the vectors closer together. For dissimilar pairs that are closer than
    /// the margin, it pushes them apart.</para>
    /// </remarks>
    public static (Vector<T>, Vector<T>) ContrastiveLossDerivative(Vector<T> output1, Vector<T> output2, T similarityLabel, T margin)
    {
        T distance = EuclideanDistance(output1, output2);
        Vector<T> grad1 = new Vector<T>(output1.Length);
        Vector<T> grad2 = new Vector<T>(output2.Length);

        for (int i = 0; i < output1.Length; i++)
        {
            T diff = _numOps.Subtract(output1[i], output2[i]);
            if (_numOps.Equals(similarityLabel, _numOps.One))
            {
                grad1[i] = _numOps.Multiply(_numOps.FromDouble(2), diff);
                grad2[i] = _numOps.Multiply(_numOps.FromDouble(-2), diff);
            }
            else
            {
                if (_numOps.LessThan(distance, margin))
                {
                    grad1[i] = _numOps.Multiply(_numOps.FromDouble(-2), _numOps.Multiply(_numOps.Subtract(margin, distance), diff));
                    grad2[i] = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Multiply(_numOps.Subtract(margin, distance), diff));
                }
            }
        }

        return (grad1, grad2);
    }

    /// <summary>
    /// Calculates the Dice loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The Dice loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Dice loss measures the overlap between the predicted and actual values.
    /// It's commonly used in image segmentation tasks where you need to identify specific regions in an image.
    /// A value of 0 means perfect overlap, while 1 means no overlap at all.</para>
    /// <para>The formula is: 1 - (2 * intersection) / (sum of predicted + sum of actual)</para>
    /// </remarks>
    public static T DiceLoss(Vector<T> predicted, Vector<T> actual)
    {
        T intersection = _numOps.Zero;
        T sumPredicted = _numOps.Zero;
        T sumActual = _numOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = _numOps.Add(intersection, _numOps.Multiply(predicted[i], actual[i]));
            sumPredicted = _numOps.Add(sumPredicted, predicted[i]);
            sumActual = _numOps.Add(sumActual, actual[i]);
        }

        T diceCoefficient = _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2), intersection), 
            _numOps.Add(sumPredicted, sumActual));

        return _numOps.Subtract(_numOps.One, diceCoefficient);
    }

    /// <summary>
    /// Calculates the gradient of the Dice loss function.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The gradient vector for the Dice loss.</returns>
    /// <remarks>
    /// <para>For Beginners: This function calculates how to adjust the predicted values to minimize
    /// the Dice loss. It helps the model learn to increase the overlap between predictions and actual values.</para>
    /// </remarks>
    public static Vector<T> DiceLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        T intersection = _numOps.Zero;
        T sumPredicted = _numOps.Zero;
        T sumActual = _numOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = _numOps.Add(intersection, _numOps.Multiply(predicted[i], actual[i]));
            sumPredicted = _numOps.Add(sumPredicted, predicted[i]);
            sumActual = _numOps.Add(sumActual, actual[i]);
        }

        T denominator = _numOps.Power(_numOps.Add(sumPredicted, sumActual), _numOps.FromDouble(2));

        for (int i = 0; i < predicted.Length; i++)
        {
            T numerator = _numOps.Subtract(
                _numOps.Multiply(_numOps.FromDouble(2), _numOps.Multiply(actual[i], _numOps.Add(sumPredicted, sumActual))),
                _numOps.Multiply(_numOps.FromDouble(2), _numOps.Multiply(intersection, _numOps.FromDouble(2)))
            );
            derivative[i] = _numOps.Divide(numerator, denominator);
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Jaccard loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>The Jaccard loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Jaccard loss measures how dissimilar two sets are. It's calculated as 1 minus 
    /// the size of the intersection divided by the size of the union. In simpler terms, it measures how 
    /// much the predicted and actual values don't overlap. A value of 0 means perfect overlap (identical), 
    /// while 1 means no overlap at all.</para>
    /// </remarks>
    public static T JaccardLoss(Vector<T> predicted, Vector<T> actual)
    {
        T intersection = _numOps.Zero;
        T union = _numOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = _numOps.Add(intersection, MathHelper.Min(predicted[i], actual[i]));
            union = _numOps.Add(union, MathHelper.Max(predicted[i], actual[i]));
        }

        return _numOps.Subtract(_numOps.One, _numOps.Divide(intersection, union));
    }

    /// <summary>
    /// Calculates the derivative of the Jaccard loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>A vector containing the derivatives of the Jaccard loss with respect to each predicted value.</returns>
    /// <remarks>
    /// <para>For Beginners: The derivative tells us how to adjust our predictions to minimize the Jaccard loss.
    /// It shows how much the loss would change if we slightly changed each prediction. This helps the model
    /// learn in the right direction during training.</para>
    /// </remarks>
    public static Vector<T> JaccardLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        T intersection = _numOps.Zero;
        T union = _numOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = _numOps.Add(intersection, MathHelper.Min(predicted[i], actual[i]));
            union = _numOps.Add(union, MathHelper.Max(predicted[i], actual[i]));
        }

        for (int i = 0; i < predicted.Length; i++)
        {
            if (_numOps.GreaterThan(predicted[i], actual[i]))
            {
                derivative[i] = _numOps.Divide(_numOps.Subtract(union, intersection), _numOps.Power(union, _numOps.FromDouble(2)));
            }
            else if (_numOps.LessThan(predicted[i], actual[i]))
            {
                derivative[i] = _numOps.Divide(_numOps.Negate(_numOps.Subtract(union, intersection)), _numOps.Power(union, _numOps.FromDouble(2)));
            }
            else
            {
                derivative[i] = _numOps.Zero;
            }
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the weighted cross-entropy loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <param name="weights">The weights vector for each sample.</param>
    /// <returns>The weighted cross-entropy loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Cross-entropy loss measures how well your model's predictions match the actual values,
    /// especially for classification problems. The "weighted" part means that some mistakes are considered more 
    /// important than others. This is useful when some classes are more important or when you have imbalanced data 
    /// (where some classes appear much more frequently than others).</para>
    /// </remarks>
    public static T WeightedCrossEntropyLoss(Vector<T> predicted, Vector<T> actual, Vector<T> weights)
    {
        T loss = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            loss = _numOps.Add(loss, _numOps.Multiply(weights[i], 
                _numOps.Multiply(actual[i], _numOps.Log(predicted[i]))));
        }

        return _numOps.Negate(loss);
    }

    /// <summary>
    /// Calculates the derivative of the weighted cross-entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <param name="weights">The weights vector for each sample.</param>
    /// <returns>A vector containing the derivatives of the weighted cross-entropy loss with respect to each predicted value.</returns>
    /// <remarks>
    /// <para>For Beginners: This derivative helps the model understand how to adjust its predictions to reduce the weighted 
    /// cross-entropy loss. The weights make certain samples more influential in guiding the model's learning process.</para>
    /// </remarks>
    public static Vector<T> WeightedCrossEntropyLossDerivative(Vector<T> predicted, Vector<T> actual, Vector<T> weights)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = _numOps.Multiply(
                weights[i],
                _numOps.Divide(
                    _numOps.Subtract(predicted[i], actual[i]),
                    _numOps.Multiply(predicted[i], _numOps.Subtract(_numOps.One, predicted[i]))
                )
            );
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the ordinal regression loss for predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <param name="numClasses">The number of classes or categories in the ordinal scale.</param>
    /// <returns>The ordinal regression loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Ordinal regression is used when your categories have a meaningful order (like ratings: poor, fair, good, excellent).
    /// Unlike regular classification, this loss function takes into account that being off by one category is better than being off by multiple categories.
    /// For example, predicting "good" when the actual value is "fair" is a smaller error than predicting "excellent".</para>
    /// </remarks>
    public static T OrdinalRegressionLoss(Vector<T> predicted, Vector<T> actual, int numClasses)
    {
        T loss = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            for (int j = 0; j < numClasses - 1; j++)
            {
                T indicator = _numOps.GreaterThan(actual[i], _numOps.FromDouble(j)) ? _numOps.One : _numOps.Zero;
                loss = _numOps.Add(loss, _numOps.Log(_numOps.Add(_numOps.One, _numOps.Exp(_numOps.Negate(_numOps.Multiply(indicator, predicted[i]))))));
            }
        }

        return loss;
    }

    /// <summary>
    /// Calculates the derivative of the ordinal regression loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <param name="numClasses">The number of classes or categories in the ordinal scale.</param>
    /// <returns>A vector containing the derivatives of the ordinal regression loss with respect to each predicted value.</returns>
    /// <remarks>
    /// <para>For Beginners: This derivative guides the model in learning how to better predict ordered categories.
    /// It helps the model understand not just whether a prediction is right or wrong, but how far off it is in the
    /// ordered sequence of categories.</para>
    /// </remarks>
    public static Vector<T> OrdinalRegressionLossDerivative(Vector<T> predicted, Vector<T> actual, int numClasses)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < numClasses - 1; j++)
            {
                T indicator = _numOps.GreaterThan(actual[i], _numOps.FromDouble(j)) ? _numOps.One : _numOps.Zero;
                T expTerm = _numOps.Exp(_numOps.Negate(_numOps.Multiply(indicator, predicted[i])));
                sum = _numOps.Add(sum, _numOps.Divide(_numOps.Negate(_numOps.Multiply(indicator, expTerm)), _numOps.Add(_numOps.One, expTerm)));
            }
            derivative[i] = sum;
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the exponential loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>The exponential loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Exponential loss heavily penalizes incorrect predictions, especially those that are far off.
    /// It's commonly used in boosting algorithms like AdaBoost. The exponential function makes the penalty grow very quickly
    /// as the error increases, which helps the model focus on the most difficult examples.</para>
    /// </remarks>
    public static T ExponentialLoss(Vector<T> predicted, Vector<T> actual)
    {
        T loss = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            loss = _numOps.Add(loss, _numOps.Exp(_numOps.Negate(_numOps.Multiply(actual[i], predicted[i]))));
        }

        return loss;
    }

    /// <summary>
    /// Calculates the derivative of the exponential loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>A vector containing the derivatives of the exponential loss with respect to each predicted value.</returns>
    /// <remarks>
    /// <para>For Beginners: This derivative helps the model understand how to adjust its predictions to reduce the exponential loss.
    /// Because exponential loss grows rapidly with error size, this derivative will push the model to quickly correct large mistakes.</para>
    /// </remarks>
    public static Vector<T> ExponentialLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = _numOps.Multiply(
                _numOps.Negate(actual[i]),
                _numOps.Exp(_numOps.Negate(_numOps.Multiply(actual[i], predicted[i])))
            );
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Squared Hinge Loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>The squared hinge loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Squared Hinge Loss is commonly used in classification problems, especially for 
    /// Support Vector Machines (SVMs). It measures how well your model separates different classes. 
    /// The "margin" represents how confidently and correctly your model classifies each example. 
    /// When predictions are correct and confident, the loss is zero. When they're incorrect or not confident 
    /// enough, the loss increases quadratically (squared), which heavily penalizes large mistakes.</para>
    /// </remarks>
    public static T SquaredHingeLoss(Vector<T> predicted, Vector<T> actual)
    {
        T loss = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T margin = _numOps.Subtract(_numOps.One, _numOps.Multiply(actual[i], predicted[i]));
            loss = _numOps.Add(loss, _numOps.Power(MathHelper.Max(_numOps.Zero, margin), _numOps.FromDouble(2)));
        }

        return loss;
    }

    /// <summary>
    /// Calculates the derivative of the Squared Hinge Loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>A vector containing the derivatives of the squared hinge loss with respect to each predicted value.</returns>
    /// <remarks>
    /// <para>For Beginners: This derivative helps the model learn by showing how to adjust predictions to reduce the loss.
    /// When the margin is positive (meaning the prediction isn't confident enough or is incorrect), the derivative provides
    /// guidance on how to improve. When the margin is negative or zero (meaning the prediction is already good), 
    /// the derivative is zero, indicating no changes are needed for that prediction.</para>
    /// </remarks>
    public static Vector<T> SquaredHingeLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T margin = _numOps.Subtract(_numOps.One, _numOps.Multiply(actual[i], predicted[i]));
            if (_numOps.GreaterThan(margin, _numOps.Zero))
            {
                derivative[i] = _numOps.Multiply(_numOps.FromDouble(-2), _numOps.Multiply(actual[i], margin));
            }
            else
            {
                derivative[i] = _numOps.Zero;
            }
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Modified Huber Loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>The modified huber loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Modified Huber Loss is a smoother version of the hinge loss that's less sensitive to outliers.
    /// It combines quadratic behavior near zero with linear behavior for large negative values. This makes it more robust
    /// when dealing with noisy data or outliers. It's particularly useful in classification problems where you want to
    /// balance between being sensitive to errors but not overly influenced by extreme mistakes.</para>
    /// </remarks>
    public static T ModifiedHuberLoss(Vector<T> predicted, Vector<T> actual)
    {
        T loss = _numOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T z = _numOps.Multiply(actual[i], predicted[i]);
            if (_numOps.GreaterThanOrEquals(z, _numOps.FromDouble(-1)))
            {
                loss = _numOps.Add(loss, _numOps.Power(MathHelper.Max(_numOps.Zero, _numOps.Subtract(_numOps.One, z)), _numOps.FromDouble(2)));
            }
            else
            {
                loss = _numOps.Add(loss, _numOps.Negate(_numOps.Multiply(_numOps.FromDouble(4), z)));
            }
        }

        return loss;
    }

    /// <summary>
    /// Calculates the derivative of the Modified Huber Loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>A vector containing the derivatives of the modified huber loss with respect to each predicted value.</returns>
    /// <remarks>
    /// <para>For Beginners: This derivative guides the model's learning process by indicating how to adjust predictions.
    /// It has different behaviors depending on how far off the prediction is:
    /// - For severely wrong predictions (z &lt; -1), it provides a constant gradient to move in the right direction
    /// - For moderately wrong predictions (-1 ≤ z &lt; 1), it provides a gradient proportional to the error
    /// - For correct predictions (z ≥ 1), it provides no gradient (zero) since no adjustment is needed</para>
    /// </remarks>
    public static Vector<T> ModifiedHuberLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T z = _numOps.Multiply(actual[i], predicted[i]);
            if (_numOps.GreaterThanOrEquals(z, _numOps.FromDouble(-1)))
            {
                if (_numOps.LessThan(z, _numOps.One))
                {
                    derivative[i] = _numOps.Multiply(_numOps.FromDouble(-2), _numOps.Multiply(actual[i], _numOps.Subtract(_numOps.One, z)));
                }
                else
                {
                    derivative[i] = _numOps.Zero;
                }
            }
            else
            {
                derivative[i] = _numOps.Multiply(_numOps.FromDouble(-4), actual[i]);
            }
        }

        return derivative;
    }

    /// <summary>
    /// Calculates the Elastic Net Loss between predicted and actual values with L1 and L2 regularization.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <param name="l1Ratio">The mixing parameter between L1 and L2 regularization (0 to 1).</param>
    /// <param name="alpha">The regularization strength parameter.</param>
    /// <returns>The elastic net loss value.</returns>
    /// <remarks>
    /// <para>For Beginners: Elastic Net Loss combines the Mean Squared Error (which measures prediction accuracy) 
    /// with two types of regularization (which prevent overfitting):
    /// - L1 regularization (also called Lasso): Helps select only the most important features by pushing some weights to zero
    /// - L2 regularization (also called Ridge): Prevents any single weight from becoming too large
    /// 
    /// The l1Ratio parameter (between 0 and 1) controls the balance between these two types of regularization:
    /// - When l1Ratio = 1: Only L1 regularization is used (Lasso)
    /// - When l1Ratio = 0: Only L2 regularization is used (Ridge)
    /// - Values in between: A mix of both (Elastic Net)
    /// 
    /// The alpha parameter controls the overall strength of regularization - higher values mean stronger regularization.</para>
    /// </remarks>
    public static T ElasticNetLoss(Vector<T> predicted, Vector<T> actual, T l1Ratio, T alpha)
    {
        T mseLoss = MeanSquaredError(predicted, actual);
        T l1Regularization = _numOps.Zero;
        T l2Regularization = _numOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            l1Regularization = _numOps.Add(l1Regularization, _numOps.Abs(predicted[i]));
            l2Regularization = _numOps.Add(l2Regularization, _numOps.Power(predicted[i], _numOps.FromDouble(2)));
        }

        T l1Term = _numOps.Multiply(_numOps.Multiply(alpha, l1Ratio), l1Regularization);
        T l2Term = _numOps.Multiply(_numOps.Multiply(_numOps.Multiply(alpha, _numOps.Subtract(_numOps.One, l1Ratio)), _numOps.FromDouble(0.5)), l2Regularization);

        return _numOps.Add(_numOps.Add(mseLoss, l1Term), l2Term);
    }

    /// <summary>
    /// Calculates the derivative of the Elastic Net Loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <param name="l1Ratio">The mixing parameter between L1 and L2 regularization (0 to 1).</param>
    /// <param name="alpha">The regularization strength parameter.</param>
    /// <returns>A vector containing the derivatives of the elastic net loss with respect to each predicted value.</returns>
    /// <remarks>
    /// <para>For Beginners: This derivative guides the model's learning by showing how to adjust predictions to minimize both
    /// prediction errors and overfitting. It combines three components:
    /// 
    /// 1. MSE Gradient: Pushes predictions toward actual values to reduce prediction errors
    /// 2. L1 Gradient: Pushes weights toward zero, but maintains their sign (positive/negative)
    /// 3. L2 Gradient: Reduces the magnitude of weights proportionally to their size
    /// 
    /// Together, these components help create a model that makes accurate predictions while staying as simple as possible.
    /// The l1Ratio and alpha parameters control the relative importance of these different objectives.</para>
    /// </remarks>
    public static Vector<T> ElasticNetLossDerivative(Vector<T> predicted, Vector<T> actual, T l1Ratio, T alpha)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T mseGradient = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Subtract(predicted[i], actual[i]));
            T l1Gradient = _numOps.Multiply(alpha, _numOps.Multiply(l1Ratio, _numOps.SignOrZero(predicted[i])));
            T l2Gradient = _numOps.Multiply(_numOps.Multiply(alpha, _numOps.Subtract(_numOps.One, l1Ratio)), predicted[i]);
        
            derivative[i] = _numOps.Add(_numOps.Add(mseGradient, l1Gradient), l2Gradient);
        }

        return derivative;
    }
}