global using AiDotNet.NeuralNetworks;
global using AiDotNet.ActivationFunctions;

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
    private readonly NeuralNetworkARIMAOptions<T> _nnarimaOptions;
    private Vector<T> _arParameters;
    private Vector<T> _maParameters;
    private Vector<T> _residuals;
    private Vector<T> _fitted;
    private readonly IOptimizer<T> _optimizer;
    private Vector<T> _y;
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
        _optimizer = _nnarimaOptions.Optimizer ?? new LBFGSOptimizer<T>();
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
        int inputSize = _nnarimaOptions.LaggedPredictions + _nnarimaOptions.ExogenousVariables;
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
    /// Trains the Neural Network ARIMA model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input matrix containing features or past values.</param>
    /// <param name="y">The target vector containing the values to be predicted.</param>
    /// <exception cref="ArgumentException">Thrown when the number of rows in x doesn't match the length of y.</exception>
    /// <remarks>
    /// <para>
    /// This method performs the following steps:
    /// 1. Validates the input data
    /// 2. Initializes model parameters
    /// 3. Optimizes the parameters using the provided optimizer
    /// 4. Computes residuals to assess model performance
    /// </para>
    /// <para><b>For Beginners:</b> This is where your model learns from the data you provide.
    /// 
    /// Imagine teaching a student:
    /// 1. You give them study materials (x) and test answers (y)
    /// 2. They start with some basic knowledge (initializing parameters)
    /// 3. They study and improve their understanding (optimizing parameters)
    /// 4. They take a practice test to see how well they've learned (computing residuals)
    /// 
    /// After this process, your model is ready to make predictions on new data.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Input matrix rows must match output vector length.");
        }

        _y = y;

        InitializeParameters();
        OptimizeParameters(x, _y);
        ComputeResiduals(x, _y);
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
        Random rand = new();
        for (int i = 0; i < p; i++) _arParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
        for (int i = 0; i < q; i++) _maParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
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
        var inputData = new OptimizationInputData<T>
        {
            XTrain = x,
            YTrain = y
        };

        OptimizationResult<T> result = _optimizer.Optimize(inputData);
        UpdateModelParameters(result.BestSolution.Coefficients);
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
        _residuals = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            _residuals[i] = NumOps.Subtract(y[i], _fitted[i]);
        }
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
        Vector<T> nnInput = CreateNeuralNetworkInput(predictions, inputRow, index);
        T nnPrediction = _neuralNetwork.Predict(nnInput)[0];
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
        _neuralNetwork.Serialize(writer);

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
        _neuralNetwork.Deserialize(reader);

        // Read options
        _nnarimaOptions.AROrder = reader.ReadInt32();
        _nnarimaOptions.MAOrder = reader.ReadInt32();
        _nnarimaOptions.LaggedPredictions = reader.ReadInt32();
    }
}