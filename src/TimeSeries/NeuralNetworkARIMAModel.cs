global using AiDotNet.ActivationFunctions;
global using AiDotNet.NeuralNetworks;
using AiDotNet.Autodiff;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Represents a Neural Network ARIMA (Autoregressive Integrated Moving Average) model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class combines traditional ARIMA modeling with neural networks to create a hybrid model for time series forecasting.
/// It incorporates both linear (ARIMA) and non-linear (neural network) components to capture complex patterns in the data.
/// </para>
/// <para><b>For Beginners:</b> This model is like a super-powered crystal ball for predicting future values in a sequence of data.
/// 
/// Imagine you're trying to predict tomorrow's temperature:
/// - The ARIMA part looks at recent temperatures and how they've been changing.
/// - The Neural Network part can spot complex patterns, like how weekends or holidays might affect temperature.
/// 
/// By combining these two approaches, this model can make more accurate predictions than either method alone.
/// It's especially useful for data that changes over time, like stock prices, weather patterns, or sales figures.
/// </para>
/// </remarks>
public class NeuralNetworkARIMAModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Configuration options for the Neural Network ARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stores all configuration parameters for the model, including the AR order, MA order, 
    /// differencing order, and neural network specifications.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the blueprint or recipe for your model.
    /// 
    /// It contains all the settings that define how your model works:
    /// - How many past values to consider (AR order)
    /// - How many past errors to consider (MA order)
    /// - How to prepare the data (differencing)
    /// - How to configure the neural network
    /// 
    /// Think of it as the instruction manual that tells the model how it should be built and operate.
    /// </para>
    /// </remarks>
    private readonly NeuralNetworkARIMAOptions<T> _nnarimaOptions;

    /// <summary>
    /// Gets whether parameter optimization succeeded during the most recent training run.
    /// </summary>
    /// <remarks>
    /// When <see cref="NeuralNetworkARIMAOptions{T}.OptimizeParameters"/> is disabled or optimization fails,
    /// this value is <see langword="false"/>.
    /// </remarks>
    public bool IsOptimized { get; private set; }

    /// <summary>
    /// Coefficients for the Autoregressive (AR) component of the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These parameters determine how much influence each past value has on the prediction.
    /// They are optimized during the training process to minimize prediction errors.
    /// </para>
    /// <para><b>For Beginners:</b> These are weights that determine how much each past value matters.
    /// 
    /// Imagine you're predicting tomorrow's temperature:
    /// - If yesterday's temperature is very important, its AR parameter will be high
    /// - If temperatures from a week ago don't matter much, their AR parameters will be low
    /// 
    /// These values are learned from the data during training, so the model figures out 
    /// which past values are most important for making accurate predictions.
    /// </para>
    /// </remarks>
    private Vector<T> _arParameters;

    /// <summary>
    /// Coefficients for the Moving Average (MA) component of the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These parameters determine how much influence each past prediction error has on the current prediction.
    /// They help the model correct for systematic errors in its forecasts.
    /// </para>
    /// <para><b>For Beginners:</b> These are weights that determine how much each past mistake matters.
    /// 
    /// If your model has been consistently underestimating temperatures:
    /// - The MA parameters help recognize this pattern
    /// - They make adjustments to correct for these systematic errors
    /// 
    /// For example, if the model has been off by +2 degrees for the past three days,
    /// the MA component might suggest adding some correction to today's prediction.
    /// </para>
    /// </remarks>
    private Vector<T> _maParameters;

    /// <summary>
    /// The prediction errors (residuals) for each training example.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the differences between the actual values and the model's predictions during training.
    /// They are used in the MA component of the model and help assess prediction quality.
    /// </para>
    /// <para><b>For Beginners:</b> These are your model's mistakes during training.
    /// 
    /// For each data point in your training set:
    /// - The residual is how far off your prediction was
    /// - Positive values mean you predicted too low
    /// - Negative values mean you predicted too high
    /// 
    /// These errors are important because:
    /// 1. They help measure how good your model is
    /// 2. The Moving Average part uses them to improve future predictions
    /// 
    /// It's like keeping a record of how much you overshot or undershot the target
    /// so you can adjust your aim next time.
    /// </para>
    /// </remarks>
    private Vector<T> _residuals;

    /// <summary>
    /// The predicted values for the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the values predicted by the model for each input in the training dataset.
    /// They are compared to the actual values to calculate residuals and evaluate model performance.
    /// </para>
    /// <para><b>For Beginners:</b> These are the predictions your model made during training.
    /// 
    /// After training, this contains what your model would have predicted for each point in your training data.
    /// By comparing these to the actual values:
    /// - You can see how well your model learned from the data
    /// - You can calculate how big the errors were
    /// - You can identify where your model struggles most
    /// 
    /// It's like looking back at your practice test results to see which questions you got right and wrong.
    /// </para>
    /// </remarks>
    private Vector<T> _fitted;

    /// <summary>
    /// The optimization algorithm used to find the best parameter values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This component searches for the optimal values of the model parameters during training.
    /// It tries to minimize the prediction error by adjusting the parameters systematically.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "learner" that finds the best settings for your model.
    /// 
    /// Imagine you're adjusting the volume and bass on a speaker to get the best sound:
    /// - The optimizer tries different combinations of settings
    /// - It measures how good each combination is
    /// - It gradually homes in on the best possible settings
    /// 
    /// Instead of you having to try thousands of different combinations manually,
    /// the optimizer does this automatically and efficiently finds the best parameters.
    /// </para>
    /// </remarks>
    private readonly IOptimizer<T, Matrix<T>, Vector<T>> _optimizer;

    /// <summary>
    /// The target values used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the actual values that the model is trained to predict.
    /// They are compared to the model's predictions to calculate errors and update parameters.
    /// </para>
    /// <para><b>For Beginners:</b> These are the correct answers that your model is trying to learn.
    /// 
    /// If you're predicting daily temperatures:
    /// - This would contain the actual recorded temperatures
    /// - Your model tries to predict these values
    /// - The difference between predictions and these values shows how well your model is doing
    /// 
    /// Think of it as the answer key for a test that your model is taking.
    /// </para>
    /// </remarks>
    private Vector<T> _y;

    /// <summary>
    /// The neural network component of the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the non-linear component of the Neural Network ARIMA model.
    /// It captures complex patterns in the data that the linear ARIMA components cannot model.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "brain" of your model that finds complex patterns.
    /// 
    /// While the ARIMA parts (AR and MA) are good at finding simple trends and patterns:
    /// - The neural network can detect complex relationships
    /// - It can learn patterns that don't follow simple rules
    /// - It can combine multiple factors in sophisticated ways
    /// 
    /// For example, if temperature depends on a complex interaction between humidity, wind speed,
    /// and the day of the week, the neural network can capture these relationships.
    /// 
    /// Think of it as having a creative expert working alongside a methodical analyst - together
    /// they can solve problems neither could handle alone.
    /// </para>
    /// </remarks>
    private readonly INeuralNetwork<T> _neuralNetwork;

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralNetworkARIMAModel{T}"/> class.
    /// </summary>
    /// <param name="options">The options for configuring the Neural Network ARIMA model. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Neural Network ARIMA model with the specified options or default values.
    /// It initializes the optimizer, neural network, and other necessary components for the model.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your prediction tool before you start using it.
    /// 
    /// You can customize how the model works by providing options, or let it use default settings:
    /// - It prepares the mathematical tools (optimizer) that help the model learn from data.
    /// - It sets up the neural network, which is like the "brain" of the model.
    /// - It initializes empty containers to store important information as the model learns.
    /// 
    /// Think of it as assembling and configuring your crystal ball before you start making predictions.
    /// </para>
    /// </remarks>
    public NeuralNetworkARIMAModel(NeuralNetworkARIMAOptions<T>? options = null) : base(options ?? new())
    {
        _nnarimaOptions = options ?? new NeuralNetworkARIMAOptions<T>();
        _optimizer = _nnarimaOptions.Optimizer ?? new LBFGSOptimizer<T, Matrix<T>, Vector<T>>(this);
        _arParameters = Vector<T>.Empty();
        _maParameters = Vector<T>.Empty();
        _residuals = Vector<T>.Empty();
        _fitted = Vector<T>.Empty();
        _y = Vector<T>.Empty();
        _neuralNetwork = _nnarimaOptions.NeuralNetwork ?? CreateDefaultNeuralNetwork();
    }

    /// <summary>
    /// Creates a default neural network architecture for the model.
    /// </summary>
    /// <returns>A neural network with a default architecture.</returns>
    /// <remarks>
    /// <para>
    /// This method sets up a simple neural network with one hidden layer when no custom network is provided.
    /// The network is configured for regression tasks and uses ReLU activation in the hidden layer and linear activation in the output layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a basic "brain" for your model if you haven't provided a custom one.
    /// 
    /// The neural network it creates:
    /// - Has an input layer that receives data
    /// - Has a hidden layer that processes the data (using ReLU, which helps capture non-linear patterns)
    /// - Has an output layer that produces the final prediction
    /// 
    /// It's like setting up a simple assembly line in a factory:
    /// 1. Raw materials come in (input layer)
    /// 2. Workers process the materials (hidden layer)
    /// 3. Final product comes out (output layer)
    /// 
    /// This basic setup can handle many prediction tasks, but you can provide a more complex network if needed.
    /// </para>
    /// </remarks>
    private NeuralNetwork<T> CreateDefaultNeuralNetwork()
    {
        // Define input and output dimensions
        int inputSize = Math.Max(1, _nnarimaOptions.LaggedPredictions + _nnarimaOptions.ExogenousVariables + 1);
        int hiddenSize = 10;
        int outputSize = 1;

        // Create layers directly
        var inputLayer = new InputLayer<T>(inputSize);

        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> linearActivation = new IdentityActivation<T>();

        var hiddenLayer = new DenseLayer<T>(inputSize, hiddenSize, activationFunction: reluActivation);
        var outputLayer = new DenseLayer<T>(hiddenSize, outputSize, activationFunction: linearActivation);

        var defaultArchitecture = new NeuralNetworkArchitecture<T>(
            InputType.OneDimensional,           // Input type
            NeuralNetworkTaskType.Regression,   // Task type
            NetworkComplexity.Simple,           // Network complexity,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: new List<ILayer<T>>         // Default layers
            {
                inputLayer,
                hiddenLayer,
                outputLayer
            }
        );

        return new NeuralNetwork<T>(architecture: defaultArchitecture);
    }

    /// <summary>
    /// Initializes the AR and MA parameters of the model with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up initial values for the Autoregressive (AR) and Moving Average (MA) parameters.
    /// These parameters are initialized with small random values to provide a starting point for optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like giving your model a starting guess before it begins learning.
    /// 
    /// AR parameters: Help the model use past values to predict future ones.
    /// MA parameters: Help the model use past prediction errors to improve future predictions.
    /// 
    /// By starting with small random values:
    /// - The model doesn't start with any preconceived notions
    /// - It can more easily find the best values during training
    /// 
    /// It's similar to starting a game of "Hot and Cold" in a neutral position, 
    /// allowing you to move in any direction to find the hidden object.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        int p = _nnarimaOptions.AROrder;
        int q = _nnarimaOptions.MAOrder;

        _arParameters = new Vector<T>(p);
        _maParameters = new Vector<T>(q);

        // Initialize with small random values
        var rand = RandomHelper.CreateSecureRandom();
        for (int i = 0; i < p; i++) _arParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
        for (int i = 0; i < q; i++) _maParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);

        SyncModelParametersFromState();
    }

    private void SyncModelParametersFromState()
    {
        var nnParameters = _neuralNetwork.GetParameters();
        var combined = new Vector<T>(_arParameters.Length + _maParameters.Length + nnParameters.Length);

        int index = 0;
        for (int i = 0; i < _arParameters.Length; i++)
        {
            combined[index++] = _arParameters[i];
        }

        for (int i = 0; i < _maParameters.Length; i++)
        {
            combined[index++] = _maParameters[i];
        }

        for (int i = 0; i < nnParameters.Length; i++)
        {
            combined[index++] = nnParameters[i];
        }

        base.ApplyParameters(combined);
    }

    /// <summary>
    /// Optimizes the model parameters using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input matrix containing features or past values.</param>
    /// <param name="y">The target vector containing the values to be predicted.</param>
    /// <remarks>
    /// <para>
    /// This method uses the configured optimizer to find the best values for the model parameters.
    /// It prepares the input data for optimization, runs the optimization process, and updates the model parameters with the best solution found.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model fine-tunes its predictions to match the data as closely as possible.
    /// 
    /// Imagine you're adjusting the dials on a radio to get the clearest signal:
    /// 1. You start with some initial dial positions (your current parameters)
    /// 2. You listen to the signal quality (how well the model predicts)
    /// 3. You adjust the dials slightly (change parameters)
    /// 4. You repeat steps 2-3 until you get the best signal possible
    /// 
    /// The optimizer automates this process, trying many different "dial positions" 
    /// to find the ones that make the model's predictions most accurate.
    /// </para>
    /// </remarks>
    private void OptimizeParameters(Matrix<T> x, Vector<T> y)
    {
        var inputData = new OptimizationInputData<T, Matrix<T>, Vector<T>>
        {
            XTrain = x,
            YTrain = y
        };

        OptimizationResult<T, Matrix<T>, Vector<T>> result = _optimizer.Optimize(inputData);
        UpdateModelParameters(result.BestSolution?.GetParameters() ?? Vector<T>.Empty());
    }

    /// <summary>
    /// Updates the model parameters with the optimized values.
    /// </summary>
    /// <param name="optimizedParameters">A vector containing the optimized parameter values.</param>
    /// <remarks>
    /// <para>
    /// This method takes the optimized parameters and updates the AR parameters, MA parameters, and neural network parameters of the model.
    /// It ensures that each component of the model is updated with its corresponding optimized values.
    /// </para>
    /// <para><b>For Beginners:</b> This is like updating your model with the best settings it has found.
    /// 
    /// After the optimization process:
    /// 1. The AR parameters are updated (like adjusting how much past values influence predictions)
    /// 2. The MA parameters are updated (like adjusting how much past errors influence predictions)
    /// 3. The neural network's internal values are updated
    /// 
    /// It's similar to a chef adjusting their recipe based on taste tests to make the perfect dish.
    /// </para>
    /// </remarks>
    private void UpdateModelParameters(Vector<T> optimizedParameters)
    {
        if (optimizedParameters.Length == 0)
        {
            return;
        }

        int paramIndex = 0;

        // Update AR parameters
        for (int i = 0; i < _arParameters.Length; i++)
        {
            _arParameters[i] = optimizedParameters[paramIndex++];
        }

        // Update MA parameters
        for (int i = 0; i < _maParameters.Length; i++)
        {
            _maParameters[i] = optimizedParameters[paramIndex++];
        }

        // Calculate the length of neural network parameters
        int nnParamsLength = optimizedParameters.Length - paramIndex;

        // Update neural network parameters
        _neuralNetwork.UpdateParameters(optimizedParameters.Slice(paramIndex, nnParamsLength));

        SyncModelParametersFromState();
    }

    /// <summary>
    /// Computes the residuals (errors) of the model predictions.
    /// </summary>
    /// <param name="x">The input matrix containing features or past values.</param>
    /// <param name="y">The actual target vector.</param>
    /// <remarks>
    /// <para>
    /// This method calculates the difference between the actual values and the model's predictions.
    /// These residuals are used in the MA component of the model and help assess the model's performance.
    /// </para>
    /// <para><b>For Beginners:</b> This is like checking how far off your predictions were.
    /// 
    /// For each data point:
    /// 1. The model makes a prediction
    /// 2. We compare this prediction to the actual value
    /// 3. The difference (error) is stored
    /// 
    /// These errors help the model improve future predictions and are used in the Moving Average part of ARIMA.
    /// It's like keeping a record of how much you over or underestimated in past guesses to improve future guesses.
    /// </para>
    /// </remarks>
    private void ComputeResiduals(Matrix<T> x, Vector<T> y)
    {
        _fitted = Predict(x);
        _residuals = (Vector<T>)Engine.Subtract(y, _fitted);
    }

    /// <summary>
    /// Predicts values for the given input data.
    /// </summary>
    /// <param name="input">The input matrix containing features or past values.</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method generates predictions for each row in the input matrix.
    /// It uses both the ARIMA components and the neural network to make predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This is where your model makes its predictions.
    /// 
    /// For each set of input data:
    /// 1. The model looks at the input (like today's weather for predicting tomorrow's)
    /// 2. It uses its learned patterns (from ARIMA and the neural network) to make a guess
    /// 3. It adds this guess to a list of predictions
    /// 
    /// At the end, you get a list of predictions, one for each input set you provided.
    /// It's like a weather forecaster making predictions for each day of the coming week.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        int n = input.Rows;
        Vector<T> predictions = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(predictions, input.GetRow(i), i);
        }

        return predictions;
    }

    /// <summary>
    /// Predicts a single value based on previous predictions and current input.
    /// </summary>
    /// <param name="predictions">Vector of previous predictions.</param>
    /// <param name="inputRow">Current input vector.</param>
    /// <param name="index">Index of the current prediction.</param>
    /// <returns>A single predicted value.</returns>
    /// <remarks>
    /// <para>
    /// This method combines AR terms, MA terms, and neural network predictions to generate a single forecast.
    /// It uses past predictions, past errors, and current input to make its prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the model makes a single prediction.
    /// 
    /// The prediction process:
    /// 1. Uses past predictions (AR part - like considering recent temperature trends)
    /// 2. Uses past errors (MA part - like adjusting for recent over/under estimations)
    /// 3. Uses the neural network (for capturing complex patterns)
    /// 4. Combines all these to make a final prediction
    /// 
    /// It's like a weather forecaster considering recent temperatures, how accurate their recent forecasts were,
    /// and any complex weather patterns they've learned about to predict tomorrow's temperature.
    /// </para>
    /// </remarks>
    private T PredictSingle(Vector<T> predictions, Vector<T> inputRow, int index)
    {
        T prediction = NumOps.Zero;

        // Add AR terms
        for (int i = 0; i < _arParameters.Length; i++)
        {
            if (index - i - 1 >= 0)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_arParameters[i], predictions[index - i - 1]));
            }
        }

        // Add MA terms
        for (int i = 0; i < _maParameters.Length; i++)
        {
            if (index - i - 1 >= 0 && _residuals != null)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_maParameters[i], _residuals[index - i - 1]));
            }
        }

        // Add neural network prediction
        var nnInput = CreateNeuralNetworkInput(predictions, inputRow, index);
        T nnPrediction = _neuralNetwork.Predict(Tensor<T>.FromVector(nnInput))[0];
        prediction = NumOps.Add(prediction, nnPrediction);

        return prediction;
    }

    /// <summary>
    /// Creates the input vector for the neural network component of the model.
    /// </summary>
    /// <param name="predictions">The vector of predictions made so far.</param>
    /// <param name="inputRow">The input vector for the current time step.</param>
    /// <param name="index">The index of the current time step.</param>
    /// <returns>A vector to be used as input for the neural network.</returns>
    /// <remarks>
    /// <para>
    /// This method prepares the input for the neural network by combining lagged predictions and exogenous variables.
    /// It ensures that the neural network receives appropriate historical and current data for making its prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This method prepares the information that the neural network part of the model will use.
    /// 
    /// It's like gathering ingredients for a recipe:
    /// 1. It takes some recent predictions (like looking at recent weather)
    /// 2. It adds current information (like today's temperature, humidity, etc.)
    /// 
    /// For example, if you're predicting tomorrow's temperature:
    /// - You might use the last 3 days' temperatures (lagged predictions)
    /// - Plus today's humidity and wind speed (exogenous variables)
    /// 
    /// The neural network uses this combined information to make its part of the prediction.
    /// </para>
    /// </remarks>
    private Vector<T> CreateNeuralNetworkInput(Vector<T> predictions, Vector<T> inputRow, int index)
    {
        List<T> nnInputList = new List<T>();

        // Add lagged predictions
        for (int i = 1; i <= _nnarimaOptions.LaggedPredictions; i++)
        {
            if (index - i >= 0)
            {
                nnInputList.Add(predictions[index - i]);
            }
            else
            {
                nnInputList.Add(NumOps.Zero);
            }
        }

        // Add exogenous variables
        nnInputList.AddRange(inputRow);

        return new Vector<T>(nnInputList);
    }

    /// <summary>
    /// Evaluates the model's performance using various metrics.
    /// </summary>
    /// <param name="xTest">The input matrix for testing.</param>
    /// <param name="yTest">The actual output vector for testing.</param>
    /// <returns>A dictionary containing various performance metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates several common metrics to assess the model's prediction accuracy:
    /// - Mean Absolute Error (MAE)
    /// - Mean Squared Error (MSE)
    /// - Root Mean Squared Error (RMSE)
    /// - R-squared (R2)
    /// These metrics provide different perspectives on how well the model's predictions match the actual values.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like giving your model a report card.
    /// 
    /// It calculates different ways to measure how accurate your predictions are:
    /// - MAE: On average, how far off are your predictions? (Like missing the bullseye by 2 inches on average)
    /// - MSE: Similar to MAE, but punishes big mistakes more (Like squaring the distance from the bullseye)
    /// - RMSE: The square root of MSE, making it easier to interpret (Back to inches from the bullseye)
    /// - R2: How much of the data's variation does your model explain? (Perfect is 1, terrible is 0)
    /// 
    /// These numbers help you understand if your model is doing a good job and compare it to other models.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>();

        // Mean Absolute Error (MAE)
        metrics["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions);

        // Mean Squared Error (MSE)
        metrics["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions);

        // Root Mean Squared Error (RMSE)
        metrics["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions);

        // R-squared (R2)
        metrics["R2"] = StatisticsHelper<T>.CalculateR2(yTest, predictions);

        return metrics;
    }

    /// <summary>
    /// Serializes the core components of the model to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write the serialized data to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the essential parts of the model:
    /// - AR and MA parameters
    /// - Neural network parameters
    /// - Model configuration options
    /// This allows the model to be saved and later reconstructed exactly as it was.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like packing up your model for storage or travel.
    /// 
    /// Imagine you've built a complex LEGO structure and want to store it:
    /// 1. You write down the exact pieces you used (AR and MA parameters)
    /// 2. You save a picture of how it's put together (neural network parameters)
    /// 3. You note any special instructions for rebuilding it (configuration options)
    /// 
    /// This way, you or someone else can rebuild the exact same model later, even on a different computer.
    /// </para>
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write model parameters
        SerializationHelper<T>.SerializeVector(writer, _arParameters);
        SerializationHelper<T>.SerializeVector(writer, _maParameters);

        // Write neural network parameters
        var serializedModel = _neuralNetwork.Serialize();
        writer.Write(serializedModel.Length);
        writer.Write(serializedModel);

        // Write options
        writer.Write(_nnarimaOptions.AROrder);
        writer.Write(_nnarimaOptions.MAOrder);
        writer.Write(_nnarimaOptions.LaggedPredictions);
    }

    /// <summary>
    /// Deserializes the core components of the model from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized data from.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model from saved data:
    /// - Reads AR and MA parameters
    /// - Reconstructs the neural network
    /// - Restores model configuration options
    /// After this process, the model is ready to make predictions, just as it was before serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like unpacking and rebuilding your model from storage.
    /// 
    /// Continuing the LEGO analogy:
    /// 1. You read the list of pieces you used (AR and MA parameters)
    /// 2. You look at the saved picture to rebuild the structure (neural network parameters)
    /// 3. You follow the special instructions to set it up correctly (configuration options)
    /// 
    /// After this process, your model is back to its original state, ready to make predictions again.
    /// </para>
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read model parameters
        _arParameters = SerializationHelper<T>.DeserializeVector(reader);
        _maParameters = SerializationHelper<T>.DeserializeVector(reader);

        // Read neural network parameters
        var serializedModelLength = reader.ReadInt32();
        var serializedModel = reader.ReadBytes(serializedModelLength);

        // Read options
        _nnarimaOptions.AROrder = reader.ReadInt32();
        _nnarimaOptions.MAOrder = reader.ReadInt32();
        _nnarimaOptions.LaggedPredictions = reader.ReadInt32();
    }

    /// <summary>
    /// Core implementation of the training process for the Neural Network ARIMA model.
    /// </summary>
    /// <param name="x">The input matrix containing features or past values.</param>
    /// <param name="y">The target vector containing the values to be predicted.</param>
    /// <remarks>
    /// <para>
    /// This is the internal implementation of the model training process, handling the details of
    /// parameter initialization, optimization, and residual computation.
    /// </para>
    /// <para><b>For Beginners:</b> This is the engine room of the training process.
    /// 
    /// While the public Train method handles validation and setup, this method does the actual work:
    /// 1. It saves the target values for later use
    /// 2. It sets up the initial parameters
    /// 3. It runs the optimization process to find the best parameters
    /// 4. It computes how much error the model still has after training
    /// 
    /// It's like the detailed instruction manual for building a complex model, while the public Train
    /// method is the simplified overview.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        _y = y;

        InitializeParameters();

        IsOptimized = false;
        if (_nnarimaOptions.OptimizeParameters)
        {
            try
            {
                OptimizeParameters(x, _y);
                IsOptimized = true;
            }
            catch (InvalidOperationException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"[NeuralNetworkARIMAModel] Parameter optimization failed; using initial estimates. {ex}");
            }
            catch (ArgumentException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"[NeuralNetworkARIMAModel] Parameter optimization failed; using initial estimates. {ex}");
            }
            catch (ArithmeticException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"[NeuralNetworkARIMAModel] Parameter optimization failed; using initial estimates. {ex}");
            }
        }

        ComputeResiduals(x, _y);
    }

    protected override void ApplyParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters), "Parameters vector cannot be null.");
        }

        int p = _nnarimaOptions.AROrder;
        int q = _nnarimaOptions.MAOrder;
        int nnParamCount = _neuralNetwork.ParameterCount;
        int expectedLength = p + q + nnParamCount;

        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, but got {parameters.Length}.", nameof(parameters));
        }

        if (_arParameters.Length != p)
        {
            _arParameters = new Vector<T>(p);
        }

        if (_maParameters.Length != q)
        {
            _maParameters = new Vector<T>(q);
        }

        int paramIndex = 0;

        for (int i = 0; i < p; i++)
        {
            _arParameters[i] = parameters[paramIndex++];
        }

        for (int i = 0; i < q; i++)
        {
            _maParameters[i] = parameters[paramIndex++];
        }

        _neuralNetwork.UpdateParameters(parameters.Slice(paramIndex, nnParamCount));

        base.ApplyParameters(parameters);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        ApplyParameters(parameters);
    }

    /// <summary>
    /// Predicts a single value for the given input vector.
    /// </summary>
    /// <param name="input">The input vector containing features or past values.</param>
    /// <returns>A single predicted value.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a single prediction based on the input vector.
    /// It uses both the ARIMA components and the neural network to make the prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This is a simplified way to get just one prediction.
    /// 
    /// Instead of providing a whole table of inputs and getting many predictions back,
    /// you provide just one row of inputs and get back a single prediction.
    /// 
    /// It's like asking the weather forecaster for just tomorrow's forecast
    /// instead of the whole week's weather outlook.
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Create a temporary matrix with a single row
        Matrix<T> singleRowMatrix = new Matrix<T>(1, input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            singleRowMatrix[0, i] = input[i];
        }

        // Use the existing Predict method and return the first (and only) prediction
        Vector<T> prediction = Predict(singleRowMatrix);
        return prediction[0];
    }

    /// <summary>
    /// Gets metadata about the model, including type, parameters, and configuration.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method provides information about the model's configuration and state.
    /// It includes details such as the model type, parameters, and training status.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a summary or "fact sheet" about your model.
    /// 
    /// It tells you important information about your model, such as:
    /// - What type of model it is
    /// - What parameters it's using
    /// - Whether it's been trained yet
    /// - How it's configured
    /// 
    /// It's like getting a specification sheet for a car, telling you its make, model,
    /// engine size, and features.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metaData = new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetworkARIMA,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AR Order", _nnarimaOptions.AROrder },
                { "MA Order", _nnarimaOptions.MAOrder },
                { "Differencing", _nnarimaOptions.DifferencingOrder },
                { "Lagged Predictions", _nnarimaOptions.LaggedPredictions },
                { "Exogenous Variables", _nnarimaOptions.ExogenousVariables },
                { "AR Parameters", _arParameters },
                { "MA Parameters", _maParameters }
            },
            ModelData = this.Serialize()
        };

        return metaData;
    }

    /// <summary>
    /// Creates a new instance of the Neural Network ARIMA model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Neural Network ARIMA model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a clone of the model with the same configuration but without any trained parameters.
    /// It's useful for creating multiple models with the same structure for ensembling or cross-validation.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a fresh copy of your model.
    /// 
    /// Think of it like making a photocopy of a blank form:
    /// - It has the same structure and fields as your original
    /// - But it doesn't have any of the information filled in
    /// 
    /// This is useful when you want to train several similar models with different data,
    /// or when you want to start over without changing your model's basic setup.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a copy of the options
        var optionsCopy = new NeuralNetworkARIMAOptions<T>
        {
            AROrder = _nnarimaOptions.AROrder,
            MAOrder = _nnarimaOptions.MAOrder,
            DifferencingOrder = _nnarimaOptions.DifferencingOrder,
            LaggedPredictions = _nnarimaOptions.LaggedPredictions,
            ExogenousVariables = _nnarimaOptions.ExogenousVariables,
            Optimizer = _nnarimaOptions.Optimizer,
            // Clone the neural network if possible, otherwise use null to let constructor create a default
            NeuralNetwork = _nnarimaOptions.NeuralNetwork?.Clone() as INeuralNetwork<T>
        };

        // Return a new instance with the copied options
        return new NeuralNetworkARIMAModel<T>(optionsCopy);
    }

    /// <summary>
    /// Gets whether this model supports JIT compilation.
    /// </summary>
    /// <value>
    /// Returns <c>true</c> when the model has valid AR/MA parameters.
    /// JIT compilation combines ARIMA linear terms with average neural network contribution.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This hybrid model can be JIT compiled by:
    /// 1. Representing ARIMA as a linear combination (weights @ lags)
    /// 2. Adding the average neural network contribution
    /// The approximation is suitable for inference speedup.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => _arParameters != null && _arParameters.Length > 0;

    /// <summary>
    /// Exports the Neural Network ARIMA model as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">A list to which input nodes will be added.</param>
    /// <returns>The output computation node representing the forecast.</returns>
    /// <remarks>
    /// <para>
    /// The computation graph represents:
    /// forecast = AR_weights @ lags + MA_weights @ residuals + avg_nn_contribution
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
        {
            throw new ArgumentNullException(nameof(inputNodes), "Input nodes list cannot be null.");
        }

        if (_arParameters == null || _arParameters.Length == 0)
        {
            throw new InvalidOperationException("Cannot export computation graph: Model has not been trained.");
        }

        // Create input node for lag values (AR terms)
        var lagInputShape = new int[] { _nnarimaOptions.AROrder };
        var lagInputTensor = new Tensor<T>(lagInputShape);
        var lagInputNode = TensorOperations<T>.Variable(lagInputTensor, "lag_input", requiresGradient: false);
        inputNodes.Add(lagInputNode);

        // Create AR weights tensor
        var arWeightsData = new T[_arParameters.Length];
        for (int i = 0; i < _arParameters.Length; i++)
        {
            arWeightsData[i] = _arParameters[i];
        }
        var arWeightsTensor = new Tensor<T>(new[] { 1, _arParameters.Length }, new Vector<T>(arWeightsData));
        var arWeightsNode = TensorOperations<T>.Constant(arWeightsTensor, "ar_weights");

        // AR contribution = weights @ lags
        var resultNode = TensorOperations<T>.MatrixVectorMultiply(arWeightsNode, lagInputNode);

        // Add constant for average MA contribution (approximation)
        if (_maParameters != null && _maParameters.Length > 0)
        {
            T avgMaContribution = NumOps.Zero;
            for (int i = 0; i < _maParameters.Length; i++)
            {
                // Approximate MA contribution assuming small residuals
                avgMaContribution = NumOps.Add(avgMaContribution, NumOps.Multiply(_maParameters[i], NumOps.FromDouble(0.01)));
            }
            var maTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { avgMaContribution }));
            var maNode = TensorOperations<T>.Constant(maTensor, "ma_contribution");
            resultNode = TensorOperations<T>.Add(resultNode, maNode);
        }

        // Add average neural network contribution (estimated during training)
        // This is an approximation - the actual NN output varies with input
        T avgNnContribution = ComputeAverageNNContribution();
        var nnTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { avgNnContribution }));
        var nnNode = TensorOperations<T>.Constant(nnTensor, "nn_contribution");
        resultNode = TensorOperations<T>.Add(resultNode, nnNode);

        return resultNode;
    }

    /// <summary>
    /// Computes an average neural network contribution for JIT approximation.
    /// </summary>
    private T ComputeAverageNNContribution()
    {
        // Use historical fitted values to estimate average NN contribution
        if (_fitted == null || _fitted.Length == 0 || _y == null || _y.Length == 0)
        {
            return NumOps.Zero;
        }

        // Average difference between fitted and pure ARIMA prediction
        T avgContribution = NumOps.Zero;
        int count = Math.Min(_fitted.Length, 10); // Use last 10 samples
        for (int i = _fitted.Length - count; i < _fitted.Length; i++)
        {
            // This is an approximation - actual contribution varies
            avgContribution = NumOps.Add(avgContribution, NumOps.Subtract(_fitted[i], _y[i]));
        }
        return count > 0 ? NumOps.Divide(avgContribution, NumOps.FromDouble(count)) : NumOps.Zero;
    }
}
