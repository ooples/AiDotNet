namespace AiDotNet.Regression;

/// <summary>
/// Implements Support Vector Regression (SVR), which creates a regression model by finding
/// a hyperplane that lies within a specified margin (epsilon) of the training data.
/// This approach is effective for both linear and nonlinear regression problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Support Vector Regression (SVR) works by:
/// - Transforming data into a higher-dimensional space using kernel functions
/// - Finding the optimal hyperplane that fits within an epsilon-width tube around the data
/// - Using only a subset of the training examples (support vectors) to make predictions
/// - Balancing model complexity and training error through the C parameter
/// 
/// Unlike traditional regression methods that minimize squared errors, SVR aims to find
/// a function that deviates from training data by no more than epsilon while remaining as flat as possible.
/// </para>
/// <para><b>For Beginners:</b> Support Vector Regression is like creating a tunnel through your data.
/// 
/// Think of it like this:
/// - You want to draw a line (or curve) through your data points
/// - Instead of drawing the line directly through the points, you create a tunnel of a certain width
/// - You try to include as many points as possible inside this tunnel
/// - Points outside the tunnel are called "support vectors" and help define its shape
/// 
/// For example, when predicting house prices, SVR would create a tunnel through the data
/// that captures the general trend while allowing some houses to fall outside the tunnel
/// if they're unusually priced for their features.
/// </para>
/// </remarks>
public class SupportVectorRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Configuration options for the Support Vector Regression model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control key aspects of the SVR algorithm, including:
    /// - Epsilon: The width of the tube within which no penalty is given to errors
    /// - C: The regularization parameter that balances fitting the data vs. keeping the model simple
    /// - MaxIterations: The maximum number of optimization iterations to perform
    /// - KernelType: The type of kernel function to use for transforming the data
    /// </para>
    /// <para><b>For Beginners:</b> These settings control how your model learns from data:
    /// 
    /// - Epsilon: How wide your tunnel is (wider = more tolerant of errors)
    /// - C: How strictly you want to fit your data (higher = follows data more closely)
    /// - MaxIterations: How many attempts the model makes to improve itself
    /// - KernelType: What kind of shape your tunnel can take (straight, curved, etc.)
    /// 
    /// These parameters let you balance between a model that fits your training data very closely
    /// (which might not generalize well) and one that's too simple to capture important patterns.
    /// </para>
    /// </remarks>
    private readonly SupportVectorRegressionOptions _options;

    /// <summary>
    /// Creates a new Support Vector Regression model.
    /// </summary>
    /// <param name="options">
    /// Optional configuration settings for the SVR model. These settings control aspects like:
    /// - The width of the epsilon-insensitive tube (epsilon)
    /// - The regularization parameter (C)
    /// - The type of kernel function to use
    /// If not provided, default options will be used.
    /// </param>
    /// <param name="regularization">
    /// Optional regularization method to prevent overfitting.
    /// If not provided, no additional regularization will be applied beyond the built-in regularization of SVR.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Support Vector Regression model with the specified configuration options
    /// and regularization method. If options are not provided, default values are used.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up a new SVR model before training.
    /// 
    /// Think of it like configuring a new tool:
    /// - You can use the default settings (by not specifying options)
    /// - Or you can customize how it works (by providing specific options)
    /// - The regularization parameter provides extra protection against overfitting
    /// 
    /// After setting up the model with these options, you'll need to train it on your data
    /// to find the support vectors and coefficients that best describe your data pattern.
    /// </para>
    /// </remarks>
    public SupportVectorRegression(SupportVectorRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new SupportVectorRegressionOptions();
    }

    /// <summary>
    /// Optimizes the SVR model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input feature matrix, where rows represent observations and columns represent features.</param>
    /// <param name="y">The target values vector containing the actual output values to predict.</param>
    /// <remarks>
    /// <para>
    /// This method implements the core optimization for SVR. It:
    /// 1. Applies regularization to the input matrix
    /// 2. Uses the Sequential Minimal Optimization (SMO) algorithm to find the optimal parameters
    /// 3. Applies regularization to the resulting alpha coefficients
    /// 
    /// The SMO algorithm is an efficient way to solve the quadratic programming problem
    /// that arises when training an SVR model.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the best tunnel through your data points.
    /// 
    /// The process works like this:
    /// 
    /// 1. The model prepares your data with regularization (like smoothing it out)
    /// 2. It uses a special algorithm (SMO) to find the optimal tunnel shape
    /// 3. It identifies which points (support vectors) are most important for defining this tunnel
    /// 4. It calculates how much influence each support vector should have on predictions
    /// 
    /// After optimization, the model knows exactly how to predict new values based on
    /// the patterns it found in your training data.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Note: SVR regularization is controlled through the C parameter (penalty),
        // not through data transformation
        SequentialMinimalOptimization(x, y);
    }

    /// <summary>
    /// Predicts target values for a matrix of input features.
    /// </summary>
    /// <param name="input">The input feature matrix for which to make predictions.</param>
    /// <returns>A vector of predicted values, one for each row in the input matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method makes predictions for multiple input samples. It:
    /// 1. Applies regularization to the input data
    /// 2. Predicts each sample individually using the PredictSingle method
    /// 3. Returns a vector of all predictions
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your trained model to predict values for new data.
    /// 
    /// The prediction process works like this:
    /// 1. The model first prepares your input data (regularization)
    /// 2. For each row of data (like each house you want to price):
    ///    - It calculates how similar this house is to each support vector
    ///    - It combines these similarities using the trained weights (alphas)
    ///    - It adds the bias term (B) to get the final prediction
    /// 3. It returns all the predictions together
    /// 
    /// This lets you predict many values at once, like estimating prices for multiple houses simultaneously.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // Note: SVR doesn't transform input data - predictions use kernel similarity
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
    }

    /// <summary>
    /// Predicts a target value for a single input feature vector.
    /// </summary>
    /// <param name="input">The input feature vector for which to make a prediction.</param>
    /// <returns>The predicted value for the input vector.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the SVR prediction function for a single input sample:
    /// f(x) = b + sum(alpha_i * K(support_vector_i, x))
    /// 
    /// where:
    /// - b is the bias term (B)
    /// - alpha_i are the Lagrange multipliers (Alphas)
    /// - K is the kernel function
    /// - support_vector_i are the training examples that define the model
    /// </para>
    /// <para><b>For Beginners:</b> This method predicts a value for a single data point.
    /// 
    /// Think of it like this:
    /// 1. Start with a base value (the bias term B)
    /// 2. For each support vector (key data points from training):
    ///    - Calculate how similar your input is to this support vector using the kernel function
    ///    - Multiply this similarity by the importance (alpha) of that support vector
    ///    - Add this contribution to your prediction
    /// 3. The final sum is your predicted value
    /// 
    /// For example, predicting a house price might involve comparing the house to several key 
    /// houses (support vectors) from your training data, with each comparison weighted by how
    /// important that house is for making predictions.
    /// </para>
    /// </remarks>
    protected override T PredictSingle(Vector<T> input)
    {
        T result = B;
        for (int i = 0; i < SupportVectors.Rows; i++)
        {
            result = NumOps.Add(result, NumOps.Multiply(Alphas[i],
                KernelFunction(SupportVectors.GetRow(i), input)));
        }

        return result;
    }

    /// <summary>
    /// Implements the Sequential Minimal Optimization (SMO) algorithm for SVR.
    /// </summary>
    /// <param name="x">The input feature matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// This method implements the Sequential Minimal Optimization algorithm for training SVR models.
    /// SMO breaks down the complex quadratic programming problem into a series of smallest possible
    /// sub-problems that can be solved analytically. It:
    /// 
    /// 1. Initializes alpha coefficients to zero
    /// 2. Iteratively selects pairs of coefficients to optimize
    /// 3. Updates the bias term (B) after each optimization step
    /// 4. Continues until convergence or maximum iterations are reached
    /// 5. Extracts the support vectors (data points with non-zero alphas)
    /// </para>
    /// <para><b>For Beginners:</b> This method is like a coach training athletes one pair at a time.
    /// 
    /// The training process works like this:
    /// 
    /// 1. Start with all influence values (alphas) set to zero
    /// 2. For multiple rounds (iterations):
    ///    - Pick a data point that violates the current model's predictions
    ///    - Find another data point to pair it with
    ///    - Adjust both points' influence values to improve the model
    ///    - Update the base prediction value (bias) accordingly
    /// 3. Keep the data points that ended up with non-zero influence (support vectors)
    /// 
    /// This approach is efficient because it focuses on the most problematic data points
    /// and only keeps the ones that are truly important for defining the model.
    /// </para>
    /// </remarks>
    private void SequentialMinimalOptimization(Matrix<T> x, Vector<T> y)
    {
        int m = x.Rows;
        Alphas = new Vector<T>(m);

        // Precompute kernel matrix for efficiency
        var K = new Matrix<T>(m, m);
        for (int ki = 0; ki < m; ki++)
            for (int kj = 0; kj <= ki; kj++)
            {
                K[ki, kj] = KernelFunction(x.GetRow(ki), x.GetRow(kj));
                K[kj, ki] = K[ki, kj]; // Kernel matrix is symmetric
            }

        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T C = NumOps.FromDouble(_options.C);
        T negC = NumOps.Negate(C);

        // Initialize bias to mean of y
        T sumY = NumOps.Zero;
        for (int i = 0; i < m; i++)
            sumY = NumOps.Add(sumY, y[i]);
        B = NumOps.Divide(sumY, NumOps.FromDouble(m));

        // SMO-style optimization
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            int numChangedAlphas = 0;

            for (int i = 0; i < m; i++)
            {
                // Compute prediction and error for sample i
                T fi = B;
                for (int idx = 0; idx < m; idx++)
                    fi = NumOps.Add(fi, NumOps.Multiply(Alphas[idx], K[i, idx]));
                T Ei = NumOps.Subtract(fi, y[i]);

                // Check KKT conditions
                T yMinusEps = NumOps.Subtract(y[i], epsilon);
                T yPlusEps = NumOps.Add(y[i], epsilon);
                bool violatesKKT = (NumOps.LessThan(fi, yMinusEps) && NumOps.LessThan(Alphas[i], C)) ||
                                   (NumOps.GreaterThan(fi, yPlusEps) && NumOps.GreaterThan(Alphas[i], negC));

                if (!violatesKKT) continue;

                // Select second alpha (j != i with largest |Ei - Ej|)
                int j = (i + 1) % m;
                T maxDiff = NumOps.Zero;
                for (int k = 0; k < m; k++)
                {
                    if (k == i) continue;
                    T fk = B;
                    for (int l = 0; l < m; l++)
                        fk = NumOps.Add(fk, NumOps.Multiply(Alphas[l], K[k, l]));
                    T Ek = NumOps.Subtract(fk, y[k]);
                    T diff = NumOps.Abs(NumOps.Subtract(Ei, Ek));
                    if (NumOps.GreaterThan(diff, maxDiff))
                    {
                        maxDiff = diff;
                        j = k;
                    }
                }

                T Ej = NumOps.Subtract(
                    NumOps.Add(B, K.GetRow(j).Zip(Alphas, (k, a) => NumOps.Multiply(k, a)).Aggregate(NumOps.Zero, (acc, v) => NumOps.Add(acc, v))),
                    y[j]);

                T oldAi = Alphas[i];
                T oldAj = Alphas[j];

                // Compute eta = 2*K(i,j) - K(i,i) - K(j,j)
                T eta = NumOps.Subtract(
                    NumOps.Subtract(NumOps.Multiply(NumOps.FromDouble(2), K[i, j]), K[i, i]),
                    K[j, j]);

                if (NumOps.GreaterThanOrEquals(eta, NumOps.Zero)) continue;

                // Compute new alpha_j: for SVR, use error-based update
                // alpha_j_new = alpha_j - (Ei - Ej) / eta
                Alphas[j] = NumOps.Subtract(Alphas[j],
                    NumOps.Divide(NumOps.Subtract(Ei, Ej), eta));

                // Compute bounds for alpha_j
                T sum = NumOps.Add(oldAi, oldAj);
                T L = MathHelper.Max(negC, NumOps.Subtract(sum, C));
                T H = MathHelper.Min(C, NumOps.Add(sum, C));
                Alphas[j] = Clip(Alphas[j], L, H);

                if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(Alphas[j], oldAj)), NumOps.FromDouble(1e-5)))
                    continue;

                // Update alpha_i to maintain sum constraint
                Alphas[i] = NumOps.Add(oldAi, NumOps.Subtract(oldAj, Alphas[j]));
                Alphas[i] = Clip(Alphas[i], negC, C);

                // Update bias
                T deltaAi = NumOps.Subtract(Alphas[i], oldAi);
                T deltaAj = NumOps.Subtract(Alphas[j], oldAj);

                T b1 = NumOps.Subtract(B, NumOps.Add(Ei,
                    NumOps.Add(NumOps.Multiply(deltaAi, K[i, i]), NumOps.Multiply(deltaAj, K[i, j]))));
                T b2 = NumOps.Subtract(B, NumOps.Add(Ej,
                    NumOps.Add(NumOps.Multiply(deltaAi, K[i, j]), NumOps.Multiply(deltaAj, K[j, j]))));

                if (NumOps.GreaterThan(Alphas[i], negC) && NumOps.LessThan(Alphas[i], C))
                    B = b1;
                else if (NumOps.GreaterThan(Alphas[j], negC) && NumOps.LessThan(Alphas[j], C))
                    B = b2;
                else
                    B = NumOps.Divide(NumOps.Add(b1, b2), NumOps.FromDouble(2));

                numChangedAlphas++;
            }

            if (numChangedAlphas == 0)
                break;
        }

        // Store all training data as support vectors
        SupportVectors = x;
    }

    /// <summary>
    /// Selects a second alpha coefficient to optimize along with the first one.
    /// </summary>
    /// <param name="i">The index of the first alpha coefficient.</param>
    /// <param name="m">The total number of training examples.</param>
    /// <returns>The index of the second alpha coefficient.</returns>
    /// <remarks>
    /// <para>
    /// This method randomly selects a second alpha coefficient that is different from the first one.
    /// In more advanced implementations, heuristics can be used to select a second alpha that
    /// maximizes the optimization step.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds a partner for training.
    /// 
    /// It's like choosing a different student to pair with the first student for a group project:
    /// - You need two different students to work together
    /// - The selection is random in this implementation
    /// - More advanced versions could choose partners more strategically
    /// 
    /// This pairing approach is fundamental to the SMO algorithm, which always
    /// optimizes two coefficients at a time.
    /// </para>
    /// </remarks>
    private readonly Random _random = RandomHelper.CreateSecureRandom();

    private int SelectSecondAlpha(int i, int m)
    {
        int j;
        do
        {
            j = _random.Next(m);
        } while (j == i);

        return j;
    }

    /// <summary>
    /// Computes the bounds for alpha coefficient optimization.
    /// </summary>
    /// <param name="yi">The target value for the first example.</param>
    /// <param name="yj">The target value for the second example.</param>
    /// <param name="ai">The current alpha value for the first example.</param>
    /// <param name="aj">The current alpha value for the second example.</param>
    /// <returns>A tuple containing the lower (L) and upper (H) bounds for the second alpha.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the bounds within which the second alpha coefficient must lie
    /// to satisfy the constraints of the SVR optimization problem. The bounds depend on:
    /// - Whether the target values of the two examples are equal
    /// - The current values of both alpha coefficients
    /// - The regularization parameter C
    /// </para>
    /// <para><b>For Beginners:</b> This method sets limits for how much a value can change.
    /// 
    /// Think of it like setting boundaries for a negotiation:
    /// - You need to keep the solution within certain limits
    /// - These limits depend on the current values and the relationship between the two examples
    /// - The C parameter sets the maximum possible value
    /// - The calculations ensure the solution remains valid
    /// 
    /// These bounds help the algorithm make controlled adjustments that maintain
    /// the mathematical constraints of the problem.
    /// </para>
    /// </remarks>
    private (T L, T H) ComputeBounds(T yi, T yj, T ai, T aj)
    {
        // For SVR, alphas can be in [-C, C] to represent (alpha - alpha*)
        // Maintain constraint: ai + aj = constant (sum of alphas)
        T sum = NumOps.Add(ai, aj);
        T C = NumOps.FromDouble(_options.C);
        T negC = NumOps.Negate(C);

        // L = max(-C, sum - C), H = min(C, sum + C)
        T L = MathHelper.Max(negC, NumOps.Subtract(sum, C));
        T H = MathHelper.Min(C, NumOps.Add(sum, C));

        return (L, H);
    }

    /// <summary>
    /// Gets metadata about the SVR model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the SVR model, including:
    /// - Base metadata from the parent class
    /// - The epsilon parameter (width of the insensitive tube)
    /// - The C parameter (regularization strength)
    /// - The type of regularization used
    /// 
    /// This metadata is useful for model inspection, logging, and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This method shares details about your model's configuration.
    /// 
    /// It's like getting a summary sheet about your model:
    /// - It includes the basic information about your model
    /// - It adds SVR-specific settings like epsilon and C
    /// - It notes what type of regularization was used
    /// 
    /// This information is helpful for comparing different models or documenting
    /// which settings worked best for your problem.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Epsilon"] = _options.Epsilon;
        metadata.AdditionalInfo["C"] = _options.C;
        metadata.AdditionalInfo["RegularizationType"] = Regularization.GetType().Name;

        return metadata;
    }

    /// <summary>
    /// Returns the type identifier for this regression model.
    /// </summary>
    /// <returns>
    /// The model type identifier for support vector regression.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns the enum value that identifies this model as a support vector regression model.
    /// This is used for model identification in serialization/deserialization and for logging purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply tells the system what kind of model this is.
    /// 
    /// It's like a name tag for the model that says "I am a support vector regression model."
    /// This is useful when:
    /// - Saving the model to a file
    /// - Loading a model from a file
    /// - Logging information about the model
    /// 
    /// You generally won't need to call this method directly in your code.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.SupportVectorRegression;
    }

    /// <summary>
    /// Serializes the support vector regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the model, including its coefficients, support vectors, and configuration options, into a 
    /// byte array. This enables the model to be saved to a file, stored in a database, or transmitted over a network.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the model to computer memory so you can use it later.
    /// 
    /// Think of it like taking a snapshot of the model:
    /// - It captures all the important values, settings, and support vectors
    /// - It converts them into a format that can be easily stored
    /// - The resulting byte array can be saved to a file or database
    /// 
    /// This is useful when you want to:
    /// - Train the model once and use it many times
    /// - Share the model with others
    /// - Use the model in a different application
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize SVR specific data
        writer.Write(_options.Epsilon);
        writer.Write(_options.C);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the support vector regression model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model from a byte array created by the Serialize method. It restores 
    /// the model's coefficients, support vectors, and configuration options, allowing a previously saved model 
    /// to be loaded and used for predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a saved model from computer memory.
    /// 
    /// Think of it like opening a saved document:
    /// - It takes the byte array created by the Serialize method
    /// - It rebuilds all the settings, support vectors, and coefficients
    /// - The model is then ready to use for making predictions
    /// 
    /// This allows you to:
    /// - Use a previously trained model without having to train it again
    /// - Load models that others have shared with you
    /// - Use the same model across different applications
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize SVR specific data
        _options.Epsilon = reader.ReadDouble();
        _options.C = reader.ReadDouble();
    }

    /// <summary>
    /// Creates a new instance of the Support Vector Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Support Vector Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Support Vector Regression model, including its options,
    /// support vectors, alpha coefficients, bias term, and regularization settings. The new instance is completely 
    /// independent of the original, allowing modifications without affecting the original model.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect duplicate of your tunnel:
    /// - It copies all the configuration settings (like epsilon, C, and kernel type)
    /// - It preserves the support vectors (the key data points that define your tunnel)
    /// - It maintains the alpha coefficients (how important each support vector is)
    /// - It keeps the bias term (B) which affects the overall position of your tunnel
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before further modifying the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        var newModel = new SupportVectorRegression<T>(_options, Regularization);

        // Copy support vectors if they exist
        if (SupportVectors != null)
        {
            newModel.SupportVectors = SupportVectors.Clone();
        }

        // Copy alpha coefficients if they exist
        if (Alphas != null)
        {
            newModel.Alphas = Alphas.Clone();
        }

        // Copy the bias term
        newModel.B = B;

        return newModel;
    }
}
