global using AiDotNet.Attributes;

namespace AiDotNet.Enums;

/// <summary>
/// Defines the types of machine learning models available in the AiDotNet library.
/// </summary>
/// <remarks>
/// <para>
/// This enum provides a comprehensive list of all supported model types, organized by category
/// and annotated with metadata that describes each model's purpose, appropriate use cases,
/// and compatible evaluation metrics.
/// </para>
/// <para>
/// <b>For Beginners:</b> This enum lists all the different AI models you can use in this library,
/// with helpful metadata attached to each one. Think of these as different tools in your AI toolbox -
/// each one works best for specific types of problems. The annotations help you understand what
/// each model does and how to properly evaluate its performance.
/// </para>
/// </remarks>
public enum ModelType
{
    /// <summary>
    /// Unknown or unspecified model type.
    /// </summary>
    [ModelInfo(ModelCategory.None, new[] { MetricGroups.General }, "Unknown model type")]
    Unknown,
    
    /// <summary>
    /// Represents no model selection.
    /// </summary>
    [ModelInfo(ModelCategory.None, new[] { MetricGroups.General }, "No model selection")]
    None,

    //
    // Regression Models
    //

    /// <summary>
    /// A general linear regression model that finds the relationship between any number of input variables and an output variable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Linear Regression is a fundamental model that finds the best-fitting straight line (or hyperplane) 
    /// through your data. It assumes that outputs change proportionally with inputs - for example, if increasing temperature 
    /// by 1� typically increases ice cream sales by $20, then increasing it by 2� should increase sales by $40. 
    /// 
    /// Linear Regression models are:
    /// - Easy to understand and interpret
    /// - Computationally efficient to train
    /// - Suitable as a first approach for many problems
    /// - Useful for understanding the relationship between inputs and outputs
    /// 
    /// While simple, they provide the foundation for many more complex models and remain widely used in practice.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "General linear regression model finding relationships between inputs and outputs")]
    LinearRegression,

    /// <summary>
    /// A basic model that finds the relationship between a single input variable and an output variable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Simple Regression is like drawing a straight line through your data points.
    /// It helps you understand how one thing affects another - for example, how temperature affects
    /// ice cream sales. It's the easiest model to understand and a great starting point.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Simple linear regression with one independent variable")]
    SimpleRegression,

    /// <summary>
    /// A model that finds the relationship between multiple input variables and a single output variable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiple Regression is like Simple Regression but with more factors.
    /// Instead of just using temperature to predict ice cream sales, you might also consider
    /// day of the week, local events, and season. This gives you a more complete picture.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Linear regression with multiple independent variables")]
    MultipleRegression,

    /// <summary>
    /// A model that predicts multiple output variables based on multiple input variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multivariate Regression predicts several things at once. For example,
    /// instead of just predicting ice cream sales, you might predict sales of ice cream,
    /// cold drinks, and sunscreen all from the same set of inputs like temperature and season.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression that predicts multiple dependent variables simultaneously")]
    MultivariateRegression,

    /// <summary>
    /// A model that captures non-linear relationships using polynomial functions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Polynomial Regression uses curved lines instead of straight ones.
    /// This is useful when the relationship isn't a straight line - for example, plant growth
    /// might increase with water up to a point, then decrease if there's too much water.
    /// A curved line can capture this pattern better than a straight line.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression using polynomial functions to model non-linear relationships")]
    PolynomialRegression,

    /// <summary>
    /// A regression model that gives different importance to different data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Weighted Regression gives some data points more importance than others.
    /// For example, if you're predicting house prices, you might give more weight to recent sales
    /// and less weight to sales from many years ago, since the market changes over time.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression where some observations have more influence than others")]
    WeightedRegression,

    /// <summary>
    /// A model that captures complex, non-linear relationships in data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Non-Linear Regression captures complex relationships that can't be represented
    /// by straight lines. For example, population growth might follow an S-curve (slow at first,
    /// then rapid, then slowing down again). Non-linear models can capture these more complex patterns,
    /// but they can be harder to interpret and require more data to train effectively.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression that captures complex non-linear relationships in data")]
    NonLinearRegression,

    /// <summary>
    /// A powerful algorithm that finds patterns by mapping data to higher-dimensional spaces.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Support Vector<double> Regression finds patterns by transforming your data into a different
    /// space where the pattern becomes simpler. Imagine trying to separate mixed red and blue marbles on a table.
    /// It might be hard in 2D, but if you could lift some marbles up (adding a 3rd dimension), the separation
    /// might become easier. SVR uses a similar concept mathematically, making it powerful for complex patterns,
    /// though it can be slower and harder to tune than simpler models.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression that uses kernel methods to handle non-linear relationships")]
    SupportVectorRegression,

    /// <summary>
    /// Combines ridge regression with kernel methods to handle non-linear relationships.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kernel Ridge Regression is like Support Vector<double> Regression but with a different
    /// mathematical approach. It uses "kernels" (special mathematical functions) to transform your data
    /// into a space where complex relationships become simpler. This allows it to capture non-linear patterns
    /// while still maintaining some of the simplicity and efficiency of linear models.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Combines ridge regression with kernel methods for non-linear modeling")]
    KernelRidgeRegression,

    /// <summary>
    /// A probabilistic model that provides uncertainty estimates along with predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gaussian Process Regression doesn't just predict a value - it gives you a range
    /// of possible values with their probabilities. It's like a weather forecast that says "80% chance
    /// of rain" instead of just "it will rain." This is very useful when understanding the uncertainty
    /// of predictions is important, such as in scientific research or risk assessment.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Probabilistic regression that provides uncertainty estimates with predictions")]
    GaussianProcessRegression,

    //
    // Classification Models
    //

    /// <summary>
    /// A model for predicting binary outcomes (yes/no, true/false, 0/1).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Logistic Regression predicts the probability of something being in a particular category. 
    /// Despite its name, it's used for classification, not regression. For example, it can predict whether 
    /// an email is spam (1) or not spam (0), or the probability a customer will make a purchase. It works by 
    /// transforming a linear model into probability values between 0 and 1 using a special S-shaped curve 
    /// called the logistic function.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Classification,
              new[] { MetricGroups.BinaryClassification, MetricGroups.General },
              "Classification model for binary (yes/no) outcomes")]
    LogisticRegression,

    /// <summary>
    /// A model for predicting categorical outcomes with more than two possible values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multinomial Logistic Regression predicts which category something belongs to when 
    /// there are multiple possibilities. For example, predicting if a customer will choose small, medium, 
    /// or large size, or if an email is spam, personal, or work-related. It gives you the probability for 
    /// each possible outcome, allowing you to see not just the most likely category but also how confident 
    /// the model is.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Classification,
              new[] { MetricGroups.MulticlassClassification, MetricGroups.General },
              "Classification model for outcomes with multiple categories")]
    MultinomialLogisticRegression,

    /// <summary>
    /// A classification model that finds the optimal hyperplane to separate different classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Support Vector Machine (SVM) is a powerful classification algorithm that finds the best
    /// boundary to separate different categories in your data. Imagine drawing a line (or in higher dimensions, a plane)
    /// between two groups of points - SVM finds the line that maximizes the distance to the nearest points from both
    /// groups. It can handle complex, non-linear boundaries using "kernel tricks" and works well even with
    /// high-dimensional data.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Classification,
              new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Classification using support vectors to find optimal decision boundaries")]
    SupportVectorMachine,

    /// <summary>
    /// A tree model that uses linear regression at its leaf nodes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> M5 Model Tree combines decision trees with linear regression. Instead of making
    /// a single prediction at each leaf (end point) of the tree, it fits a small linear regression model.
    /// This is like first sorting your data into groups using yes/no questions, then finding the best
    /// straight-line fit for each group separately. This approach often works well for problems with
    /// numeric outputs.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Decision tree with linear regression models at the leaf nodes")]
    M5ModelTree,

    //
    // Tree-based Models
    //

    /// <summary>
    /// A tree-based model that makes decisions by splitting data based on feature values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Decision Tree works like a flowchart of yes/no questions. For example,
    /// to predict if someone will buy ice cream: "Is temperature > 75�F? If yes, is it a weekend?
    /// If no, is there a special event?" and so on. It's easy to understand but can be less
    /// accurate than more complex models.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Model that makes decisions through a series of binary splits")]
    DecisionTree,

    /// <summary>
    /// An ensemble model that combines multiple decision trees to improve prediction accuracy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Random Forest combines many decision trees (simple flowchart-like models)
    /// and lets them vote on the final prediction. It's like asking a large group of people
    /// instead of just one person - the combined wisdom often gives better results. This model
    /// is powerful, handles complex data well, and is less likely to overfit (memorize) your data.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Ensemble of decision trees, each trained on random subsets of data and features")]
    RandomForest,

    /// <summary>
    /// An ensemble technique that builds models sequentially, with each new model correcting errors from previous ones.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gradient Boosting builds a series of models, where each new model focuses on
    /// fixing the mistakes of previous models. It's like having a team where each member specializes
    /// in handling the cases that the rest of the team struggles with. This approach often produces
    /// very accurate predictions but requires careful tuning to avoid overfitting.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Ensemble technique that builds models sequentially to correct previous models' errors")]
    GradientBoosting,

    //
    // Time Series Models
    //

    /// <summary>
    /// A comprehensive time series model that handles non-stationary data with trends and seasonality.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ARIMA Models combine three powerful techniques for time series forecasting: 
    /// they use past values (AR), past forecast errors (MA), and differences between consecutive 
    /// observations (I for "Integrated") to handle data with trends. ARIMA is like a Swiss Army knife 
    /// for time series - versatile and effective for many forecasting problems. The model is specified 
    /// as ARIMA(p,d,q), where p, d, and q control how many past values, differences, and errors to use.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Auto-Regressive Integrated Moving Average model for time series forecasting")]
    ARIMAModel,

    /// <summary>
    /// A time series model where future values depend on past values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Autoregressive (AR) Models predict future values based on past values in the
    /// time series. They assume that what happens next depends on what happened before. For example, tomorrow's
    /// temperature is likely related to today's temperature. The "order" of an AR model (like AR(3)) tells you
    /// how many past time periods it considers - AR(3) looks at the three most recent values. These models are
    /// relatively simple but effective for many time series where recent history influences the future.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Forecasts future values based on past values in a time series")]
    ARModel,

    /// <summary>
    /// A time series model where future values depend on past forecast errors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Moving Average (MA) Models predict future values based on past forecast errors 
    /// rather than past values themselves. They assume that if recent forecasts were too high or too low, 
    /// that information can help improve future forecasts. It's like adjusting your aim in a game based 
    /// on whether your previous shots were too high or too low. MA models are particularly good at 
    /// handling short-term, irregular fluctuations in time series data.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Forecasts based on past forecast errors rather than past values")]
    MAModel,

    /// <summary>
    /// An extension of ARIMA that also captures seasonal patterns in data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Seasonal ARIMA Models extend ARIMA to handle regular patterns that repeat at 
    /// fixed intervals, like higher retail sales during holidays or increased ice cream consumption in 
    /// summer. SARIMA adds seasonal components to the standard ARIMA model, allowing it to capture both 
    /// short-term dependencies and longer seasonal patterns. It's specified with additional parameters 
    /// that control the seasonal aspects of the model, making it powerful for data with clear seasonal effects.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Extension of ARIMA that handles seasonal patterns in time series data")]
    SARIMAModel,

    /// <summary>
    /// A time series forecasting method that gives more weight to recent observations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Exponential Smoothing Models forecast by averaging past values, but giving more 
    /// weight to recent observations. Imagine predicting tomorrow's temperature by looking at the past week, 
    /// but considering yesterday's temperature more important than last week's. These models can capture 
    /// trends (values consistently increasing or decreasing) and seasonality (regular patterns like higher 
    /// sales during holidays). They're intuitive, computationally simple, and work well for many business 
    /// forecasting problems.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Forecasting that gives more weight to recent observations")]
    ExponentialSmoothingModel,

    //
    // Neural Network Models
    //

    /// <summary>
    /// A flexible model inspired by the human brain's structure that can capture complex patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural Network Regression uses interconnected "neurons" organized in layers to learn 
    /// patterns in data. Think of it as a complex web of calculations that can discover and represent 
    /// relationships that simpler models miss. Neural networks can automatically learn features from data 
    /// without being explicitly programmed, making them powerful for complex problems like image recognition, 
    /// language processing, and predicting outcomes with many interacting factors.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Neural network configured for regression tasks")]
    NeuralNetworkRegression,

    /// <summary>
    /// A specific type of neural network with multiple layers of neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multilayer Perceptron Regression is a specific type of neural network with at least 
    /// three layers: an input layer, one or more hidden layers, and an output layer. Each neuron connects 
    /// to all neurons in the next layer, creating a densely connected network. This structure allows the 
    /// model to learn complex patterns by transforming the data through successive layers, with each layer 
    /// learning increasingly abstract features. It's like having a team of specialists who each focus on 
    /// different aspects of a problem before combining their insights.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Fully connected feed-forward neural network with multiple layers")]
    MultilayerPerceptronRegression,

    /// <summary>
    /// A neural network architecture that excels at processing sequential data with long-term dependencies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Long Short-Term Memory networks are a special kind of Recurrent Neural Network 
    /// capable of learning long-term dependencies. They're like a more sophisticated form of memory, able to 
    /// remember information for long periods of time. This makes them particularly good at tasks like language 
    /// translation, speech recognition, or any task where you need to consider context from much earlier in a sequence.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Recurrent neural network architecture specialized for sequential data")]
    LSTMNeuralNetwork,

    /// <summary>
    /// A hybrid model combining neural networks with ARIMA for time series forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural Network ARIMA combines the pattern-recognition power of neural networks 
    /// with the time series expertise of ARIMA models. ARIMA is good at capturing linear relationships and 
    /// temporal dependencies, while neural networks excel at learning complex, non-linear patterns. By 
    /// combining them, this hybrid approach can capture both types of patterns in your data. It's like 
    /// having both a statistics expert and a pattern recognition expert working together on your forecasting 
    /// problem, often resulting in more accurate predictions than either approach alone.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Hybrid model combining neural networks with traditional ARIMA for time series forecasting")]
    NeuralNetworkARIMA,

    /// <summary>
    /// A neural network model optimized through quantization to reduce model size and improve inference speed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantized Neural Networks are neural networks that have been optimized to use less memory
    /// and run faster by reducing the precision of their calculations. Instead of using high-precision numbers,
    /// they use lower-precision representations (like using whole numbers instead of decimals). This makes them
    /// ideal for deployment on mobile devices or edge computing where resources are limited, with minimal impact
    /// on accuracy.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Neural network optimized through quantization for efficient deployment")]
    QuantizedNeuralNetwork,

    /// <summary>
    /// A boosting algorithm specifically designed for regression problems.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AdaBoost-R2 is a specialized version of boosting for predicting numeric values.
    /// It works by giving more attention to the data points that are hardest to predict correctly.
    /// With each round of training, it adjusts to focus more on the difficult cases, similar to
    /// how a teacher might give extra attention to students who are struggling with certain concepts.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Boosting algorithm specifically designed for regression problems")]
    AdaBoostR2,

    /// <summary>
    /// A variation of Random Forest that introduces more randomness in how trees are built.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Extremely Randomized Trees is similar to Random Forest but adds even more
    /// randomness when building each tree. This extra randomness can help prevent overfitting
    /// (when a model learns the training data too specifically and performs poorly on new data).
    /// Think of it as deliberately introducing some "noise" to make the model more flexible.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Random Forest variant with additional randomness in tree construction")]
    ExtremelyRandomizedTrees,

    /// <summary>
    /// An extension of Random Forest that predicts a range of possible values rather than a single value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantile Regression Forests don't just predict a single value (like "this house
    /// will sell for $300,000") but instead predict a range ("this house will sell for between $280,000
    /// and $320,000 with 90% confidence"). This gives you a better sense of the uncertainty in the prediction,
    /// which can be very valuable for decision-making.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Random Forest extension that predicts value distributions rather than point estimates")]
    QuantileRegressionForests,

    /// <summary>
    /// A decision tree that uses statistical tests to make splitting decisions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Conditional Inference Tree is a type of decision tree that uses statistics to
    /// decide how to split the data. This makes it less biased toward features with many possible values.
    /// For example, a regular decision tree might favor using "zip code" over "temperature" just because
    /// there are more possible zip codes, even if temperature is actually more important. Conditional
    /// Inference Trees help avoid this problem.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Decision tree based on statistical hypothesis testing")]
    ConditionalInferenceTree,

    /// <summary>
    /// A non-parametric regression technique that preserves the order of data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Isotonic Regression finds a non-decreasing (always flat or going up) or non-increasing
    /// (always flat or going down) line that best fits your data. It's useful when you know your output
    /// should never decrease as your input increases (or vice versa). For example, the risk of heart disease
    /// generally increases with age, so a model that sometimes shows risk decreasing with age would be illogical.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Non-parametric regression that preserves monotonic relationships")]
    IsotonicRegression,

    /// <summary>
    /// Predicts specific percentiles of the output distribution rather than just the mean.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantile Regression predicts different percentiles of your data. Instead of just
    /// predicting the average house price in an area, it might predict the median price (50th percentile),
    /// the luxury price (90th percentile), and the budget price (10th percentile). This gives you a more
    /// complete picture of the range of possible values.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression that predicts specific percentiles of the output distribution")]
    QuantileRegression,

    /// <summary>
    /// A model that uses radial basis functions to approximate complex patterns in data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Radial Basis Function Regression uses special functions that respond strongly 
    /// to data points close to their center and weakly to distant points. Imagine dropping a pebble 
    /// in water - the ripples are strongest near where the pebble fell and fade as they move outward. 
    /// By combining many of these "ripple patterns" centered at different points, the model can 
    /// approximate complex relationships in your data.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression using radial basis functions to approximate complex patterns")]
    RadialBasisFunctionRegression,

    /// <summary>
    /// A model that gives more weight to nearby data points when making predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Locally Weighted Regression makes predictions by focusing more on data points 
    /// that are similar to what you're trying to predict. It's like asking for restaurant recommendations 
    /// and giving more weight to opinions from people with similar taste to yours. This approach is 
    /// flexible and can capture complex patterns without requiring a specific mathematical formula.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression that prioritizes nearby data points when making predictions")]
    LocallyWeightedRegression,

    /// <summary>
    /// A model that uses piecewise polynomial functions to create smooth curves through data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spline Regression connects data points with smooth curves rather than straight lines. 
    /// Imagine drawing a smooth curve through points on a graph by hand - you naturally create gentle 
    /// curves rather than sharp angles. Splines work similarly, creating a series of connected curves 
    /// that flow smoothly through your data points, which is useful for capturing complex patterns.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression using smooth piecewise polynomial functions")]
    SplineRegression,

    /// <summary>
    /// A model that predicts based on the average of the k closest data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> K-Nearest Neighbors makes predictions by looking at the most similar examples 
    /// in your data. To predict a house price, it might find the 5 most similar houses (in terms of 
    /// size, location, etc.) and average their prices. It's like asking "What happened in similar 
    /// situations?" rather than trying to find a mathematical formula. Simple to understand but can 
    /// be slow with large datasets.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Model that predicts based on similar training examples")]
    KNearestNeighbors,

    /// <summary>
    /// A model that discovers mathematical formulas that best describe relationships in data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Symbolic Regression tries to find an actual mathematical formula that explains 
    /// your data. Instead of just fitting parameters to a pre-defined equation, it searches for the 
    /// equation itself. For example, it might discover that your data follows "y = x� + 3x - 2" rather 
    /// than just giving you numbers. This provides insights into the underlying relationships and can 
    /// be more interpretable than other complex models.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression that discovers mathematical formulas to explain relationships")]
    SymbolicRegression,

    /// <summary>
    /// A probabilistic approach that updates predictions as new data becomes available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bayesian Regression starts with an initial guess about relationships in your data, 
    /// then updates this guess as it sees more data. It's like having a theory about something, then 
    /// gradually refining your theory as you gather more evidence. A key advantage is that it doesn't 
    /// just give predictions but also tells you how confident it is in those predictions.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Probabilistic regression that updates predictions with new data")]
    BayesianRegression,

    /// <summary>
    /// A tree-like structure that represents mathematical expressions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Expression Tree represents mathematical formulas as tree structures that can be 
    /// manipulated and evolved. Each branch and leaf represents part of a formula (like addition, 
    /// multiplication, or variables). This approach is often used in genetic programming to "evolve" 
    /// mathematical formulas that fit your data, similar to how natural selection works in nature.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Tree-structured representation of mathematical expressions")]
    ExpressionTree,

    /// <summary>
    /// A model that uses principles inspired by natural evolution to find optimal solutions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Genetic Algorithm Regression mimics evolution to find the best model. It starts 
    /// with many random models, keeps the best ones ("survival of the fittest"), combines them to create 
    /// "offspring" models, and occasionally introduces random changes ("mutations"). Over many generations, 
    /// this process tends to discover increasingly better models, even for complex problems where traditional 
    /// approaches might struggle.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression that uses evolutionary principles to find optimal solutions")]
    GeneticAlgorithmRegression,

    /// <summary>
    /// A regression technique that accounts for errors in both input and output variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Orthogonal Regression accounts for measurement errors in both your input and output 
    /// variables. Standard regression assumes your input measurements are perfect and only the outputs 
    /// have errors. But in reality, you might have uncertainty in both. For example, when studying how 
    /// height affects weight, both measurements might have errors. Orthogonal regression handles this 
    /// situation better than standard approaches.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression that accounts for errors in both input and output variables")]
    OrthogonalRegression,

    /// <summary>
    /// A regression method that is less sensitive to outliers in the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Robust Regression is designed to handle outliers - unusual data points that don't 
    /// follow the general pattern. Standard regression can be heavily influenced by even a single extreme 
    /// value (like one house that sold for 10x the normal price). Robust regression reduces the influence 
    /// of these outliers, giving you a model that better represents the typical patterns in your data.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression method less influenced by outliers in the data")]
    RobustRegression,

    /// <summary>
    /// A model specifically designed for data that changes over time.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Time Series Regression is specialized for data collected over time, like daily 
    /// temperatures, stock prices, or monthly sales. It accounts for special patterns in time data, 
    /// such as seasonal effects (sales increasing during holidays), trends (gradual increase over years), 
    /// and the fact that recent values often influence future values. This makes it much better for 
    /// forecasting future values in time-based data.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Regression specifically optimized for time-based data")]
    TimeSeriesRegression,

    /// <summary>
    /// A flexible model that combines multiple simple functions to capture complex patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Generalized Additive Model Regression builds complex relationships by adding together 
    /// simpler ones. Instead of forcing a specific shape (like a straight line or curve), it lets each 
    /// input variable affect the output in its own way. For example, temperature might have a curved 
    /// relationship with ice cream sales, while day of week has a different pattern. GAMs combine these 
    /// separate patterns to create a flexible overall model.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Flexible model that combines multiple simple functions")]
    GeneralizedAdditiveModelRegression,

    /// <summary>
    /// A technique that handles correlated input variables by projecting them onto new dimensions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Partial Least Squares Regression works well when your input variables are related 
    /// to each other. For example, a person's height and weight are correlated - taller people tend to 
    /// weigh more. This correlation can confuse standard regression. PLS creates new combined variables 
    /// that capture the most important patterns while avoiding the problems caused by these correlations.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression that handles correlated input variables effectively")]
    PartialLeastSquaresRegression,

    /// <summary>
    /// A dimension reduction technique combined with regression to handle many correlated variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Principal Component Regression first simplifies your data by identifying the most 
    /// important patterns (principal components), then builds a regression model using these patterns. 
    /// It's like summarizing a 50-page document into 5 key points, then working with those summaries. 
    /// This approach works well when you have many related input variables and helps avoid overfitting 
    /// (when a model learns noise rather than true patterns).
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Combines dimension reduction with regression for correlated variables")]
    PrincipalComponentRegression,

    /// <summary>
    /// A method that automatically selects the most important variables for prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stepwise Regression automatically chooses which input variables to include in your 
    /// model. It starts with either no variables or all variables, then adds or removes them one by one 
    /// based on how much they improve predictions. This helps create simpler models by focusing only on 
    /// the most important factors, making the model easier to interpret and often more accurate on new data.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression that automatically selects the most important variables")]
    StepwiseRegression,

    /// <summary>
    /// A regression model for count data (non-negative integers).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Poisson Regression is designed for predicting counts - like the number of customer 
    /// calls per hour, website visits per day, or accidents per month. Unlike standard regression, it 
    /// ensures predictions are always non-negative (you can't have -3 customer calls) and works with the 
    /// special statistical properties of count data, where variance often increases with the mean.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression specifically designed for count data")]
    PoissonRegression,

    /// <summary>
    /// A model for count data with extra variation (overdispersion).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Negative Binomial Regression is designed for counting events (like number of customer 
    /// complaints or product defects) when there's more variability in the data than expected. While Poisson 
    /// regression assumes the average and variance are equal, real-world count data often has higher variance. 
    /// Negative Binomial Regression handles this extra variability, making it more realistic for many 
    /// real-world counting problems.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Regression for count data with extra variability")]
    NegativeBinomialRegression,

    /// <summary>
    /// A generative stochastic neural network that can learn a probability distribution over its inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deep Boltzmann Machine is a type of neural network that learns to recognize patterns 
    /// in data by trying to recreate the input data from scratch. Imagine it as an artist trying to paint a 
    /// picture after only glancing at it briefly. By doing this repeatedly, the network learns the important 
    /// features and relationships in the data. This makes it useful for tasks like feature detection, 
    /// dimensionality reduction, and generating new data similar to the training set.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Generative neural network that models probability distributions")]
    DeepBoltzmannMachine,

    /// <summary>
    /// A type of neural network particularly effective for processing grid-like data such as images.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Convolutional Neural Networks are specialized for processing data with a grid-like 
    /// structure, such as images. They use a mathematical operation called convolution to scan over the input, 
    /// detecting features like edges, textures, and shapes. This is similar to how our eyes focus on different 
    /// parts of an image. CNNs are highly effective for tasks like image classification, object detection, 
    /// and even some types of time series analysis.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Neural network specialized for grid-like data like images")]
    ConvolutionalNeuralNetwork,

    /// <summary>
    /// A neural network architecture designed to work with sequential data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Recurrent Neural Networks are designed to work with sequences of data, where the 
    /// order matters. They have a form of memory, allowing information to persist. This makes them ideal for 
    /// tasks involving time series, language, or any data where context from previous inputs is important. 
    /// Think of it like reading a book - your understanding of each word is influenced by the words that came before it.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Neural network for sequential data processing")]
    RecurrentNeuralNetwork,

    /// <summary>
    /// A type of neural network that learns to encode data into a compressed representation and then decode it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Autoencoders are neural networks that try to copy their input to their output. This 
    /// might seem pointless, but there's a catch: they have to go through a narrow "bottleneck" in the middle. 
    /// This forces the network to learn a compressed representation of the data. It's like learning to describe 
    /// a movie in just a few words, then trying to recreate the whole movie from those words. This can be used 
    /// for dimensionality reduction, feature learning, and generating new data.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network that learns compressed data representations")]
    Autoencoder,

    /// <summary>
    /// A neural network architecture that excels at processing sequential data with long-term dependencies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Long Short-Term Memory networks are a special kind of Recurrent Neural Network 
    /// capable of learning long-term dependencies. They're like a more sophisticated form of memory, able to 
    /// remember information for long periods of time. This makes them particularly good at tasks like language 
    /// translation, speech recognition, or any task where you need to consider context from much earlier in a sequence.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Advanced recurrent neural network with improved memory capabilities")]
    LongShortTermMemory,

    /// <summary>
    /// A neural network architecture that uses attention mechanisms to weigh the importance of different parts of the input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transformer models use a mechanism called "attention" to weigh the importance of 
    /// different parts of the input when producing an output. It's like being able to focus on the most relevant 
    /// words in a sentence to understand its meaning. This allows them to handle long-range dependencies in data 
    /// very effectively. Transformers have revolutionized natural language processing tasks and are also being 
    /// applied to other domains like computer vision.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Neural network architecture using attention mechanisms")]
    Transformer,

    /// <summary>
    /// A neural network architecture that uses attention mechanisms to focus on relevant parts of the input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Attention Networks are designed to focus on the most important parts of the input 
    /// when making predictions. Imagine reading a long document and highlighting the key phrases - that's 
    /// similar to what an attention network does. It learns which parts of the input are most relevant for 
    /// the task at hand. This makes them particularly effective for tasks where some input elements are more 
    /// important than others, such as in language translation, image captioning, or analyzing time series data 
    /// where certain time steps might be more crucial than others.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Neural network that focuses on relevant parts of the input")]
    AttentionNetwork,

    /// <summary>
    /// A probabilistic generative model composed of multiple layers of stochastic latent variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deep Belief Networks are like a tower of pattern recognizers stacked on top of each other. 
    /// Each layer learns to identify patterns in the output of the layer below it. The network starts by learning 
    /// simple patterns in the raw data, then progressively learns more complex and abstract patterns in higher layers. 
    /// </para>
    /// <para>
    /// For example, if analyzing images:
    /// - The bottom layer might learn to detect edges and simple shapes
    /// - Middle layers might recognize more complex features like eyes, noses, or wheels
    /// - Top layers might identify complete objects or scenes
    /// 
    /// This layer-by-layer approach allows Deep Belief Networks to learn meaningful representations of data 
    /// even when you don't have a lot of labeled examples, making them powerful for tasks like feature learning, 
    /// dimensionality reduction, and generating new data similar to the training set.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Hierarchical neural network that learns features layer by layer")]
    DeepBeliefNetwork,

    /// <summary>
    /// A neural network architecture that uses skip connections to allow training of very deep networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Residual Neural Networks solve a problem that occurs when neural networks get too deep 
    /// (have too many layers). Normally, adding more layers should make a network smarter, but in practice, very 
    /// deep networks become harder to train. ResNets use "shortcut connections" that let information skip ahead, 
    /// allowing the network to learn which information should take the shortcut and which needs more processing. 
    /// This is like having both an express lane and a local lane on a highway - some information can take the 
    /// fast route while other information needs to stop at every exit.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Deep neural network with skip connections for improved training")]
    ResidualNeuralNetwork,

    // Many more models omitted for brevity...

    /// <summary>
    /// An extension of ARIMA that includes external variables in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ARIMAX Models extend ARIMA by including external factors that might influence
    /// your time series. For example, when forecasting ice cream sales, an ARIMA model might only look at
    /// past sales patterns, but an ARIMAX model could also consider temperature, holidays, or marketing
    /// campaigns. This often improves predictions by incorporating known influences. It's like forecasting
    /// traffic not just based on historical patterns but also considering if there's a major concert or
    /// sporting event happening.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "ARIMA with external variables for improved forecasting")]
    ARIMAXModel,

    /// <summary>
    /// A time series model that combines autoregressive and moving average components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ARMA Models combine two approaches for time series forecasting: they look at
    /// how past values of the series influence future values (the AR part) and how past prediction errors
    /// affect future values (the MA part). This combination often works better than either approach alone.
    /// It's like predicting tomorrow's weather based both on weather patterns from recent days and on how
    /// accurate your previous forecasts were. ARMA models work best for stationary time series, where the
    /// statistical properties don't change over time.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Time series model combining autoregressive and moving average components")]
    ARMAModel,

    /// <summary>
    /// A decomposable time series forecasting model developed by Facebook.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Prophet is a forecasting model designed by Facebook to handle time series with 
    /// strong seasonal effects and multiple seasons. It's particularly good with data that has holidays, 
    /// seasonal patterns, and trend changes. Prophet automatically decomposes your data into trend, seasonality, 
    /// and holiday effects, making it easier to understand what's driving changes in your time series. It's 
    /// designed to be robust to missing data and outliers, and requires minimal manual tuning, making it 
    /// accessible for non-experts.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Decomposable forecasting model for time series with multiple seasonal patterns")]
    ProphetModel,

    /// <summary>
    /// A type of neural network that learns to encode data into a compressed representation and then decode it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Variational Autoencoders learn to compress data (like images) into a compact form and 
    /// then reconstruct it. What makes them special is they learn a smooth, continuous representation where 
    /// similar inputs are close together. This allows you to generate new data by sampling from this space. 
    /// For example, a VAE trained on faces could generate new, realistic faces that don't belong to any real 
    /// person, or you could blend characteristics from different faces by moving through this learned space.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network that learns probabilistic latent representations")]
    VariationalAutoencoder,

    /// <summary>
    /// A neural network architecture that better preserves spatial hierarchies between features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Capsule Networks improve on traditional neural networks by preserving information about 
    /// position and orientation of features. Regular networks might recognize a face if it has eyes, nose, and mouth, 
    /// regardless of their arrangement. Capsule Networks also care about the spatial relationships - eyes above nose, 
    /// nose above mouth. This makes them better at understanding 3D objects from different viewpoints and less likely 
    /// to be fooled by images where the parts are present but arranged incorrectly.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Neural network that preserves spatial relationships between features")]
    CapsuleNetwork,

    /// <summary>
    /// A neural network that leverages quantum computing principles for processing information.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantum Neural Networks combine principles from quantum computing with neural networks. 
    /// While traditional computers use bits (0 or 1), quantum computers use qubits that can exist in multiple states 
    /// simultaneously. This allows quantum neural networks to explore many possible solutions at once, potentially 
    /// solving certain complex problems much faster than classical approaches. This is an emerging field at the 
    /// intersection of quantum physics and machine learning.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network based on quantum computing principles")]
    QuantumNeuralNetwork,

    /// <summary>
    /// A type of unsupervised neural network that organizes data based on similarity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Self-Organizing Maps learn to arrange data so that similar items are placed close together. 
    /// Imagine organizing books on shelves so that similar topics are near each other - that's what SOMs do with data. 
    /// They create a "map" where each location represents a particular pattern, and similar patterns are placed nearby. 
    /// This makes them useful for visualizing high-dimensional data, discovering clusters, and finding relationships 
    /// between different data points.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network that creates topological mappings of data similarities")]
    SelfOrganizingMap,

    /// <summary>
    /// A neural network model inspired by how neurons process information in a liquid.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Liquid State Machines are inspired by how signals ripple through a liquid. When you drop 
    /// a stone in water, it creates ripples that spread and interact in complex ways. Similarly, these networks 
    /// process information by letting it "ripple" through a collection of randomly connected neurons. This approach 
    /// is particularly good at processing time-varying inputs like speech or video, where the timing and sequence 
    /// of information matters.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.TimeSeries, MetricGroups.General },
              "Neural network inspired by dynamics of liquid-state systems")]
    LiquidStateMachine,

    /// <summary>
    /// A recurrent neural network that can store and retrieve patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hopfield Networks are designed to store and recall patterns, similar to how human memory 
    /// works. You can train the network to "remember" certain patterns, and then when given a partial or noisy version 
    /// of a pattern, it can recover the complete original pattern. It's like recognizing a song from just hearing a few 
    /// notes, or recognizing a friend's face even when they're partially obscured. These networks are useful for pattern 
    /// completion, noise reduction, and content-addressable memory.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network for pattern storage and retrieval")]
    HopfieldNetwork,

    /// <summary>
    /// A neural network designed to work with graph-structured data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Graph Neural Networks are specialized for data that has connections between elements, like 
    /// social networks, molecules, or road systems. While traditional neural networks work well with grid-like data 
    /// (images) or sequences (text), GNNs can understand and process the relationships between entities in a graph. 
    /// They learn by passing information along the connections in the graph, allowing each node to gather information 
    /// from its neighbors. This makes them powerful for tasks like predicting interactions between proteins, 
    /// recommending friends on social media, or analyzing traffic patterns.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network for processing graph-structured data")]
    GraphNeuralNetwork,

    /// <summary>
    /// A neural network with a fast, one-pass learning algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Extreme Learning Machines offer a shortcut to training neural networks. Traditional networks 
    /// require many iterations to learn, which can be time-consuming. ELMs randomly assign the connections in the hidden 
    /// layer and only train the output layer, which can be done in one quick step. This is like building a team where you 
    /// randomly assign roles to most members but carefully select the team leaders. While this approach sacrifices some 
    /// accuracy, it can be thousands of times faster to train, making it useful for applications where speed is critical 
    /// or for providing quick initial solutions.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Fast-training neural network with random hidden layer weights")]
    ExtremeLearningMachine,

    /// <summary>
    /// A neural network architecture that combines neural networks with external memory systems.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Differentiable Neural Computers combine a neural network with a memory matrix that it can 
    /// read from and write to. This is similar to how your computer has both a processor (CPU) and memory (RAM). The 
    /// neural network learns how to store information in the memory and how to retrieve it when needed. This gives the 
    /// system the ability to remember facts and use them later, making it particularly good at tasks that require 
    /// reasoning with stored information, like answering questions about a story or navigating complex environments.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network with external, differentiable memory system")]
    DifferentiableNeuralComputer,

    /// <summary>
    /// A recurrent neural network where the internal connections are fixed and only output connections are trained.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Echo State Networks use a clever shortcut to make recurrent neural networks easier to train. 
    /// They create a large, randomly connected "reservoir" of neurons that transforms inputs in complex ways. Only the 
    /// connections from this reservoir to the output are trained, which is much simpler than training all connections. 
    /// It's like having a complex but fixed system of pipes and valves (the reservoir) that processes water flow in 
    /// intricate ways, and you only need to learn how to read the resulting patterns. This approach works well for 
    /// time series prediction, speech recognition, and other sequential data problems.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Recurrent neural network with fixed internal connections")]
    EchoStateNetwork,

    /// <summary>
    /// A neural network architecture used in reinforcement learning to learn optimal action policies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deep Q-Networks learn to make sequences of decisions to maximize a reward. They're used in 
    /// reinforcement learning, where an agent learns by interacting with an environment. The network predicts the 
    /// "value" of taking each possible action in a given situation, allowing the agent to choose the best action. 
    /// It's like learning to play chess by estimating how good each possible move is. DQNs have been used to master 
    /// video games, control robots, and optimize complex systems like data center cooling.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network for reinforcement learning applications")]
    DeepQNetwork,

    /// <summary>
    /// A framework consisting of two neural networks that compete with each other to generate realistic data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Generative Adversarial Networks consist of two neural networks competing against each other: 
    /// a generator that creates fake data (like images) and a discriminator that tries to distinguish real data from 
    /// fakes. As they train, the generator gets better at creating convincing fakes, and the discriminator gets better 
    /// at spotting them. It's like a counterfeiter and detective constantly improving to outdo each other. This 
    /// competition drives both to improve until the generator creates data so realistic that it's hard to distinguish 
    /// from real data. GANs have created remarkably realistic images, videos, and even music.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Competitive neural networks for generating realistic data")]
    GenerativeAdversarialNetwork,

    /// <summary>
    /// A neural network architecture that combines neural networks with external memory access mechanisms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural Turing Machines combine neural networks with an external memory that they can read 
    /// from and write to in a precise way. This is inspired by how computers use memory and how humans can store and 
    /// retrieve information. The network learns to control where to store information, what to store, and how to retrieve 
    /// it later. This gives it the ability to learn algorithms - step-by-step procedures for solving problems - rather 
    /// than just pattern recognition. NTMs can learn tasks like copying sequences, sorting numbers, or even simple 
    /// programming-like tasks.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network with controlled external memory access")]
    NeuralTuringMachine,

    /// <summary>
    /// A method for evolving neural networks through genetic algorithms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> NEAT (NeuroEvolution of Augmenting Topologies) uses principles from evolution to design 
    /// neural networks. Instead of manually designing the network structure, NEAT starts with simple networks and 
    /// gradually makes them more complex through a process similar to natural selection. Networks that perform better 
    /// are more likely to "reproduce" and pass on their characteristics. Over many generations, this can discover 
    /// innovative network designs that human engineers might not think of. NEAT is particularly useful for reinforcement 
    /// learning problems like game playing or robot control.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Evolutionary approach to neural network design")]
    NEAT,

    /// <summary>
    /// A neural network architecture designed to store and retrieve information from an external memory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Memory Networks are designed to improve how neural networks handle information that needs
    /// to be remembered over time. They combine neural networks with an external memory component that can be read
    /// from and written to. This allows the network to store facts, context, or other information and retrieve it
    /// when needed. Memory Networks are particularly useful for tasks like answering questions about a story, where
    /// the network needs to remember details from earlier parts of the text to answer questions correctly.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network with explicit memory storage and retrieval")]
    MemoryNetwork,

    /// <summary>
    /// A neural network inspired by the structure and function of the neocortex in the human brain.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hierarchical Temporal Memory Networks are inspired by how the human neocortex processes
    /// information. They learn patterns in a hierarchical way, with higher levels learning more abstract patterns
    /// based on the patterns detected at lower levels. HTM Networks are particularly good at finding patterns in
    /// time-based data and can learn continuously without forgetting previous patterns. They're designed to
    /// recognize anomalies, make predictions, and understand sequences in data, similar to how our brains process
    /// sensory information.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.TimeSeries, MetricGroups.General },
              "Brain-inspired network for temporal pattern recognition")]
    HTMNetwork,

    /// <summary>
    /// A general-purpose computational model inspired by the structure and function of biological neural networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural Networks are computing systems inspired by the human brain. They consist of
    /// interconnected "neurons" that process information and learn patterns from data. Each connection has a
    /// "weight" that strengthens or weakens the signal between neurons, and these weights are adjusted during
    /// training to improve performance. Neural networks can learn to recognize patterns, classify data, make
    /// predictions, and solve complex problems without being explicitly programmed with rules. They're the
    /// foundation of many modern AI systems.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "General-purpose brain-inspired computational model")]
    NeuralNetwork,

    /// <summary>
    /// A neural network architecture for 3D scene understanding and representation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Occupancy Networks represent 3D objects or scenes by learning which points in space
    /// are inside or outside an object. This is like having a detailed 3D model where you can check any point
    /// in space to see if it's part of the object. Unlike traditional 3D representations that use fixed grids
    /// or point clouds, Occupancy Networks can represent shapes at any resolution. They're useful for 3D
    /// reconstruction, shape generation, and scene understanding in applications like robotics, virtual reality,
    /// and computer-aided design.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network for 3D object and scene representation")]
    OccupancyNetwork,

    /// <summary>
    /// A stochastic neural network that can learn a probability distribution over its inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Restricted Boltzmann Machines learn patterns in data by trying to recreate the input
    /// from scratch. They consist of two layers of neurons - visible and hidden - with connections between layers
    /// but not within layers (that's the "restricted" part). RBMs can learn to recognize patterns and extract
    /// features without supervision. They're like a simplified version of Deep Boltzmann Machines and are often
    /// used as building blocks for deeper networks. RBMs have been applied to problems like recommendation systems,
    /// feature extraction, and dimensionality reduction.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Two-layer stochastic neural network for unsupervised learning")]
    RestrictedBoltzmannMachine,

    /// <summary>
    /// A neural network model that mimics the behavior of biological neurons more closely than traditional artificial neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spiking Neural Networks more closely mimic how real neurons in the brain communicate.
    /// While traditional neural networks use continuous values, spiking networks use discrete "spikes" or pulses
    /// of activity. Neurons accumulate input until they reach a threshold, then fire a spike. This approach is
    /// more biologically realistic and potentially more energy-efficient. SNNs are particularly interesting for
    /// neuromorphic computing (brain-inspired hardware), real-time processing of sensory data, and understanding
    /// how the brain processes information.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Neural network using discrete spike-based communication")]
    SpikingNeuralNetwork,

    /// <summary>
    /// A basic neural network architecture where information flows in one direction from input to output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Feed Forward Networks are the simplest type of neural network, where information
    /// travels in only one direction - from input to output, with no loops or cycles. Data passes through
    /// multiple layers of neurons, with each layer processing the information and passing it to the next.
    /// This straightforward architecture makes them easier to understand and train compared to more complex
    /// networks. They're effective for many tasks like classification, regression, and pattern recognition
    /// where the input is fixed and doesn't depend on previous inputs or states.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Basic neural network with unidirectional information flow")]
    FeedForwardNetwork,

    /// <summary>
    /// A model that represents relationships as straight lines.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Linear models represent relationships as straight lines (or their higher-dimensional
    /// equivalents). They assume that the output changes proportionally with changes in the input. For example,
    /// if doubling an input doubles the output, that's a linear relationship. These models are simple,
    /// interpretable, and computationally efficient. While they can't capture complex curved relationships on
    /// their own, they form the foundation of many more sophisticated models and can be surprisingly effective
    /// for many real-world problems.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Model that represents relationships as straight lines")]
    Linear,

    /// <summary>
    /// A model that uses polynomial functions to represent non-linear relationships.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Polynomial models use curved lines (like parabolas) instead of straight lines to
    /// represent relationships in data. They can capture more complex patterns where the rate of change varies.
    /// For example, a plant's growth might accelerate then slow down over time, forming an S-curve. Polynomial
    /// models can represent these curved relationships by including squared terms (x�), cubed terms (x�), and
    /// so on. They're more flexible than linear models but need to be used carefully to avoid overfitting.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Model using polynomial functions for curved relationships")]
    Polynomial,

    /// <summary>
    /// A model that discovers mathematical expressions to describe relationships in data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Symbolic models try to find actual mathematical formulas that explain your data.
    /// Instead of just fitting parameters to a pre-defined equation, they search for the equation itself.
    /// For example, they might discover that your data follows "y = sin(x) + x�" rather than just giving
    /// you numbers. This provides insights into the underlying relationships and can be more interpretable
    /// than black-box models. Symbolic models are particularly useful when you want to understand the
    /// mathematical laws governing a system.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Model that discovers mathematical expressions from data")]
    Symbolic,

    /// <summary>
    /// A model that uses decision trees or tree-based ensembles for prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tree-Based models make predictions by following a series of yes/no questions,
    /// similar to a flowchart. They split the data based on features, creating a tree-like structure.
    /// For example, to predict house prices: "Is the area > 2000 sq ft? If yes, is it in neighborhood A?
    /// If no, is it older than 20 years?" and so on. Tree-based models include single decision trees and
    /// powerful ensembles like Random Forests and Gradient Boosted Trees. They're flexible, handle different
    /// types of data well, and can capture complex interactions between features.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Model based on decision trees or tree ensembles")]
    TreeBased,

    /// <summary>
    /// A flexible Bayesian time series model that combines structural components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bayesian Structural Time Series Models combine several components to model time series data. 
    /// They can include trends, seasonality, and the effects of external factors. What makes them special is their 
    /// Bayesian approach, which provides uncertainty estimates and allows you to incorporate prior knowledge. 
    /// These models are particularly useful for forecasting with limited data, understanding which factors 
    /// influence your time series, and measuring the impact of interventions like marketing campaigns.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Bayesian time series model with structural components")]
    BayesianStructuralTimeSeriesModel,

    /// <summary>
    /// A hybrid model combining regression with ARIMA error correction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dynamic Regression with ARIMA Errors combines two powerful approaches. The regression 
    /// part models relationships between your time series and external factors (like how temperature affects 
    /// ice cream sales). The ARIMA part then models the errors from this regression, capturing patterns that 
    /// the regression missed. It's like having two specialists working together - one focusing on known 
    /// relationships and the other catching any remaining patterns. This hybrid approach often outperforms 
    /// either method used alone.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Regression model with ARIMA-modeled errors")]
    DynamicRegressionWithARIMAErrors,

    /// <summary>
    /// A model designed to capture volatility clustering in financial time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GARCH Models are specialized for financial data where volatility (the amount of 
    /// fluctuation) tends to cluster - periods of high volatility are followed by more high volatility, 
    /// and calm periods tend to persist as well. Think of stock markets having "nervous" periods with 
    /// big swings up and down, followed by "calm" periods with smaller changes. GARCH models capture 
    /// this pattern, making them valuable for risk management, option pricing, and financial forecasting.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Time series model for volatility clustering in financial data")]
    GARCHModel,

    /// <summary>
    /// A recurrent neural network architecture with gating mechanisms for efficient sequence processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GRU Neural Networks are a type of recurrent neural network designed for sequential 
    /// data like text or time series. They're similar to LSTM networks but with a simpler structure that 
    /// makes them faster to train while still capturing long-term patterns. GRUs use "gates" that control 
    /// what information to keep or forget, allowing them to learn which parts of the sequence are important. 
    /// They're widely used for language processing, speech recognition, and time series forecasting.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Efficient recurrent neural network with gating mechanisms")]
    GRUNeuralNetwork,

    /// <summary>
    /// A neural network architecture designed to compare and find similarities between inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Siamese Networks are specialized for comparing two inputs to determine how similar 
    /// they are. They process both inputs through identical sub-networks (hence the name "Siamese," like 
    /// Siamese twins) and then measure the distance between the outputs. This is useful for tasks like 
    /// facial recognition (is this the same person?), signature verification, or finding similar products. 
    /// They're particularly valuable when you have many categories with few examples of each.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.BinaryClassification, MetricGroups.General },
              "Neural network for comparing similarities between inputs")]
    SiameseNetwork,

    /// <summary>
    /// A time series model that assesses the impact of specific events or interventions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Intervention Analysis Models measure how specific events affect a time series. 
    /// For example, how did a new policy, marketing campaign, or competitor's action impact your sales? 
    /// These models separate the normal pattern of your time series from the effects of the intervention, 
    /// allowing you to quantify the impact. It's like measuring how much a medicine improves health by 
    /// comparing patient outcomes before and after treatment, while accounting for other factors.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Time series model for measuring effects of specific interventions")]
    InterventionAnalysisModel,

    /// <summary>
    /// A flexible forecasting model that handles multiple seasonal patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TBATS Models are designed for time series with complex seasonal patterns. While 
    /// simple models might handle yearly patterns (like holiday sales spikes), TBATS can simultaneously 
    /// model multiple seasonal patterns of different lengths. For example, retail data might show weekly 
    /// patterns (busier on weekends), monthly patterns (busier after payday), and yearly patterns (holiday 
    /// seasons). TBATS can capture all these overlapping cycles, making it powerful for complex forecasting 
    /// problems.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Forecasting model for complex, multiple seasonal patterns")]
    TBATSModel,

    /// <summary>
    /// A method that breaks down a time series into trend, seasonal, and remainder components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> STL Decomposition breaks a time series into three parts: the overall trend 
    /// (long-term direction), seasonal patterns (regular cycles), and remainder (what's left after 
    /// removing trend and seasonality). It's like separating a music recording into bass, melody, and 
    /// percussion. This decomposition helps you understand what's driving your time series and can 
    /// improve forecasting by modeling each component separately. It's particularly useful for 
    /// visualizing and interpreting time series data.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.General },
              "Method for decomposing time series into component parts")]
    STLDecomposition,

    /// <summary>
    /// A model that captures relationships between multiple related time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Vector<double> Autoregression Models analyze multiple time series that influence each 
    /// other. For example, how prices, advertising, and competitor actions all affect sales. Unlike 
    /// simpler models that look at each series separately, VAR models capture how each series affects 
    /// the others. It's like modeling an ecosystem where changes in one species affect others. This 
    /// makes VAR models powerful for understanding complex systems and forecasting interrelated variables.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Model for analyzing multiple interrelated time series")]
    VARModel,

    /// <summary>
    /// A flexible time series model that represents observed data in terms of unobserved components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Unobserved Components Models break down a time series into underlying components 
    /// that aren't directly observable, such as trend, cycle, seasonal patterns, and irregular fluctuations. 
    /// It's like a detective inferring underlying motives from observed behaviors. These models are 
    /// particularly useful for economic data, where concepts like "potential GDP" or "natural rate of 
    /// unemployment" can't be directly measured but can be estimated from observable data.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Time series model based on underlying latent components")]
    UnobservedComponentsModel,

    /// <summary>
    /// A model that describes how input changes propagate through a system over time.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transfer Function Models describe how changes in an input variable affect an 
    /// output variable over time. For example, how a change in advertising budget affects sales in the 
    /// following weeks. These models capture both the magnitude of the effect (how much sales increase) 
    /// and its timing (immediate boost, gradual increase, delayed response, etc.). They're particularly 
    /// useful for understanding cause-and-effect relationships in time series data.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Model for input-output relationships in dynamic systems")]
    TransferFunctionModel,

    /// <summary>
    /// A flexible framework for modeling time series data using hidden states.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> State Space Models represent a time series using hidden "states" that evolve 
    /// over time. Think of it like tracking a car's position when you can only see its speed - the 
    /// position is the hidden state that you infer from observable data. These models are very flexible 
    /// and can represent many time series patterns. They're particularly useful for tracking systems 
    /// that change over time, filtering noisy data, and forecasting complex time series.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.Regression, MetricGroups.General },
              "Flexible model using hidden states to track system dynamics")]
    StateSpaceModel,

    /// <summary>
    /// A frequency-domain approach to analyzing time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spectral Analysis Models examine time series data in terms of cycles and frequencies 
    /// rather than time. It's like breaking down a musical chord into individual notes - the model identifies 
    /// which cycles (daily, weekly, monthly, etc.) are present in your data and how strong each one is. This 
    /// approach is particularly useful for finding hidden periodicities in complex data, understanding cyclical 
    /// behavior, and filtering out noise. It's widely used in fields like signal processing, economics, and 
    /// climate science.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.TimeSeries,
              new[] { MetricGroups.TimeSeries, MetricGroups.General },
              "Time series analysis using frequency and cyclical components")]
    SpectralAnalysisModel,

    //
    // Reinforcement Learning Models
    //

    /// <summary>
    /// A Deep Q-Network model for value-based reinforcement learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deep Q-Network (DQN) is a breakthrough algorithm that combines neural networks 
    /// with Q-learning to solve complex decision-making problems. It learns to estimate the value of taking 
    /// each action in different states, like learning which chess moves are good in which positions by 
    /// playing many games and remembering which moves led to wins or losses. DQN uses a neural network 
    /// to approximate the Q-function, which predicts the expected future rewards for each action. It also 
    /// uses experience replay (storing and randomly sampling past experiences) and target networks (a 
    /// separate network for stable targets) to improve learning stability. DQN is particularly effective 
    /// for problems with discrete actions and has been used to master Atari games, trading strategies, 
    /// and robotic control tasks.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Deep Q-Network for value-based reinforcement learning")]
    DQNModel,

    /// <summary>
    /// A Proximal Policy Optimization model for policy-based reinforcement learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Proximal Policy Optimization (PPO) is a state-of-the-art policy gradient method 
    /// that learns a policy (strategy) directly, deciding which actions to take in different situations. 
    /// Unlike value-based methods that learn which actions are good, PPO learns the probability of taking 
    /// each action. What makes PPO special is its "proximal" constraint - it prevents the policy from 
    /// changing too much in a single update, making training much more stable. This stability, combined 
    /// with its effectiveness, has made PPO the go-to algorithm for many applications. It's widely used 
    /// for training AI agents in games (like OpenAI's Dota 2 bot), robotics (for learning complex 
    /// movements), and trading applications (for portfolio management). PPO works well with both discrete 
    /// and continuous actions, making it versatile for different types of problems.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Proximal Policy Optimization for stable policy learning")]
    PPOModel,

    /// <summary>
    /// A REINFORCE policy gradient model for reinforcement learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> REINFORCE is one of the classic and most straightforward policy gradient methods 
    /// that learns by trial and error. It works by completing entire episodes (sequences of actions), then 
    /// adjusting the probability of actions based on whether they led to good or bad outcomes. Think of it 
    /// like a student taking a test, seeing the final grade, and then figuring out which answers to change 
    /// for next time. REINFORCE uses the total reward from an episode to update its policy - if an episode 
    /// resulted in high rewards, it increases the probability of the actions taken; if rewards were low, 
    /// it decreases them. While conceptually simple and easy to understand, REINFORCE can have high variance 
    /// in its learning (results can be inconsistent), which is why more advanced methods like PPO were 
    /// developed. However, it remains valuable for its simplicity and theoretical importance.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Classic policy gradient algorithm using Monte Carlo returns")]
    REINFORCEModel,

    /// <summary>
    /// A Soft Actor-Critic model for continuous control reinforcement learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Soft Actor-Critic (SAC) is an advanced algorithm designed specifically for 
    /// continuous actions (like steering angles, motor torques, or trading amounts). What makes SAC unique 
    /// is its focus on "entropy" - it not only tries to maximize rewards but also maintains randomness 
    /// in its actions. This built-in exploration means SAC naturally balances trying new things with 
    /// exploiting what it knows works. Think of it like a chef who not only makes dishes they know are 
    /// good but also experiments with new recipes to potentially discover even better ones. SAC is 
    /// particularly robust and requires less hyperparameter tuning than many other algorithms. It's become 
    /// extremely popular for robotics (controlling robot arms and legs), autonomous driving (smooth steering 
    /// and acceleration), and algorithmic trading (determining optimal buy/sell amounts). Its stability and 
    /// effectiveness with continuous actions make it a top choice for real-world control problems.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Soft Actor-Critic for robust continuous control")]
    SACModel,

    /// <summary>
    /// A Deep Deterministic Policy Gradient model for continuous control.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deep Deterministic Policy Gradient (DDPG) combines ideas from DQN with 
    /// policy gradient methods to handle continuous actions. Unlike stochastic policies that output 
    /// probabilities, DDPG learns deterministic policies - meaning it always outputs the same action 
    /// for the same state, like a precise formula. Think of it as learning the exact angle to turn 
    /// a steering wheel rather than a probability distribution over angles. DDPG uses an actor-critic 
    /// architecture: the actor decides what action to take, and the critic evaluates how good that 
    /// action is. To explore, it adds noise to the actions during training. DDPG was one of the first 
    /// successful algorithms for continuous control in deep RL and paved the way for more advanced 
    /// methods like TD3 and SAC. It's particularly effective for robotic control tasks, physics 
    /// simulations, and any problem requiring precise, continuous outputs.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Deep Deterministic Policy Gradient for continuous control")]
    DDPGModel,

    /// <summary>
    /// A Twin Delayed Deep Deterministic policy gradient model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Twin Delayed Deep Deterministic Policy Gradient (TD3) is an improved version 
    /// of DDPG that addresses its main weakness: overestimation bias. TD3 uses three key tricks to make 
    /// learning more stable and reliable. First, it uses twin critic networks (hence the "twin" in the name) 
    /// and takes the minimum of their estimates to avoid overoptimistic value estimates. Second, it delays 
    /// policy updates, updating the actor less frequently than the critics to ensure the critics have 
    /// accurate estimates before the policy changes. Third, it adds noise to the target actions to smooth 
    /// out the value estimates. These improvements make TD3 much more stable and reliable than DDPG, often 
    /// matching or exceeding the performance of SAC. It's particularly effective for robotic control, 
    /// continuous action games, and any application where stable, reliable learning is crucial. TD3 has 
    /// become a standard baseline for continuous control problems.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Twin Delayed DDPG with improved stability")]
    TD3Model,

    /// <summary>
    /// A Model-Based Policy Optimization algorithm that combines model-free and model-based approaches.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Model-Based Policy Optimization (MBPO) represents a breakthrough in sample 
    /// efficiency for reinforcement learning. It learns a model of the environment (predicting what will 
    /// happen next given current state and action) and uses this model to generate synthetic experiences 
    /// for training. Think of it like a chess player who can imagine future moves without actually playing 
    /// them out. MBPO carefully balances using the learned model with real environment interactions to 
    /// avoid the pitfalls of pure model-based methods (which can fail if the model is inaccurate). By 
    /// generating many simulated experiences from each real interaction, MBPO can learn effective policies 
    /// with 10-100x fewer real environment steps than model-free methods. This dramatic improvement in 
    /// sample efficiency is crucial for real-world applications where data collection is expensive or 
    /// risky, such as robotics, autonomous driving, or financial trading. MBPO combines the best of both 
    /// worlds: the sample efficiency of model-based methods with the asymptotic performance of model-free methods.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Model-Based Policy Optimization for sample-efficient learning")]
    MBPOModel,

    /// <summary>
    /// A Quantile Regression Deep Q-Network for distributional reinforcement learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantile Regression DQN (QR-DQN) is a powerful extension of DQN that doesn't 
    /// just predict expected returns but the full distribution of possible returns. Instead of learning 
    /// "this action is worth 10 points on average," QR-DQN learns "this action has a 10% chance of -5 points, 
    /// 60% chance of 8-12 points, and 30% chance of 15+ points." This distributional approach provides 
    /// much richer information about uncertainty and risk. QR-DQN estimates multiple quantiles of the 
    /// return distribution, giving you a complete picture of possible outcomes. This is particularly 
    /// valuable in scenarios where understanding risk is crucial, such as financial trading (where you 
    /// need to know potential losses, not just average returns) or safety-critical robotics (where you 
    /// need to avoid catastrophic failures). The algorithm can also be configured for risk-sensitive 
    /// decision-making, choosing actions based on worst-case scenarios (risk-averse) or best-case 
    /// scenarios (risk-seeking) rather than just averages.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Quantile Regression DQN for distributional RL")]
    QRDQNModel,

    /// <summary>
    /// A Hierarchical Reinforcement Learning model using Advantage Actor-Critic.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hierarchical Reinforcement Learning with Advantage Actor-Critic (HRARL) tackles 
    /// complex tasks by breaking them down into simpler subtasks, learning both high-level strategies and 
    /// low-level actions hierarchically. Think of it like a manager and workers: the high-level policy 
    /// (manager) decides what task to do (e.g., "go to the kitchen"), while low-level policies (workers) 
    /// figure out how to do it (the specific steps to walk there). This hierarchical approach makes it 
    /// much easier to learn complex, long-horizon tasks that would be difficult for flat RL algorithms. 
    /// HRARL uses the Advantage Actor-Critic (A2C) algorithm at each level, providing stable and efficient 
    /// learning. It's particularly effective for tasks with natural hierarchical structure, such as 
    /// navigation (room-to-room then within-room movement), manipulation (reaching then grasping), or 
    /// strategic games (choosing strategy then executing tactics). The hierarchy also improves transfer 
    /// learning - low-level skills learned for one task can often be reused for other tasks.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Hierarchical RL with Advantage Actor-Critic")]
    HRARLModel,

    /// <summary>
    /// A Decision Transformer model that treats reinforcement learning as sequence modeling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Decision Transformer represents a paradigm shift in reinforcement learning by 
    /// treating it as a sequence modeling problem rather than value estimation or policy optimization. 
    /// Using the powerful Transformer architecture (the same technology behind GPT and BERT), it learns 
    /// from sequences of states, actions, and rewards. Instead of learning "what action is best here," 
    /// it learns "given that I want a high return, what sequence of actions have led to high returns 
    /// in the past?" This approach is particularly powerful for offline RL, where you learn from a 
    /// fixed dataset without interacting with the environment. Decision Transformer can leverage patterns 
    /// in historical data to generate good action sequences, similar to how language models generate text. 
    /// It's especially effective when you have diverse demonstration data and want to extract different 
    /// behaviors based on the desired outcome. This makes it valuable for learning from human demonstrations, 
    /// historical trading data, or any scenario where you have logs of past behavior and outcomes.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Transformer-based offline reinforcement learning")]
    DecisionTransformerModel,

    /// <summary>
    /// A Multi-Agent Transformer model for coordinated multi-agent reinforcement learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multi-Agent Transformer extends the Transformer architecture to handle multiple 
    /// AI agents that need to coordinate their actions. Using attention mechanisms, each agent can "pay 
    /// attention" to what other agents are doing and planning, enabling sophisticated team coordination. 
    /// Think of it like a sports team where each player needs to be aware of their teammates' positions 
    /// and intentions to execute complex plays. The transformer's attention mechanism naturally handles 
    /// the variable number of agents and their interactions. This is particularly powerful for scenarios 
    /// like multi-robot coordination (multiple robots working together in a warehouse), team-based games 
    /// (coordinating different units in strategy games), traffic optimization (multiple autonomous vehicles 
    /// coordinating at intersections), or financial markets (multiple trading agents avoiding conflicts). 
    /// The model learns both individual policies for each agent and how they should coordinate, handling 
    /// the complex dynamics of cooperation, competition, and communication that arise in multi-agent systems.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Transformer-based multi-agent coordination")]
    MultiAgentTransformerModel,

    /// <summary>
    /// An Actor-Critic model combining value estimation with policy learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Actor-Critic is a fundamental architecture that combines the strengths of 
    /// value-based and policy-based reinforcement learning methods. It uses two components: an "actor" 
    /// that decides which actions to take (the policy), and a "critic" that evaluates how good those 
    /// actions are (the value function). Think of it like learning to cook with a mentor - the actor 
    /// is you trying different cooking techniques, while the critic is your mentor providing feedback 
    /// on whether each technique improved the dish. This combination addresses key limitations of each 
    /// approach used alone: pure policy methods can have high variance (inconsistent learning), while 
    /// pure value methods can struggle with continuous actions. Actor-Critic provides more stable and 
    /// efficient learning by using the critic's value estimates to reduce the variance of policy updates. 
    /// Many modern RL algorithms (A2C, A3C, PPO, SAC, TD3) are built on the actor-critic foundation, 
    /// making it one of the most important concepts in reinforcement learning.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Fundamental actor-critic architecture for RL")]
    ActorCriticModel,

    /// <summary>
    /// A Rainbow DQN model combining multiple DQN improvements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Rainbow DQN combines seven different improvements to the basic DQN algorithm
    /// into one powerful model. It includes double Q-learning, prioritized replay, dueling networks,
    /// multi-step learning, distributional RL, and noisy networks. Think of it as a "best of all worlds"
    /// approach that takes the most effective enhancements to DQN and uses them together. This makes it
    /// one of the most powerful value-based reinforcement learning algorithms, particularly effective
    /// for complex decision-making tasks.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Rainbow DQN combining multiple DQN improvements")]
    RainbowDQNModel,

    //
    // Reasoning Models
    //

    /// <summary>
    /// A chain-of-thought reasoning model for multi-step logical problem solving.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Chain-of-Thought (CoT) reasoning models are designed to solve complex problems
    /// by breaking them down into a series of intermediate reasoning steps, similar to how humans approach
    /// difficult tasks. Instead of jumping directly to an answer, these models "think out loud" by generating
    /// explicit reasoning chains. For example, when solving a math word problem, the model might first identify
    /// what's being asked, then extract relevant information, set up equations, solve step by step, and finally
    /// check the answer. This approach dramatically improves performance on tasks requiring logical reasoning,
    /// arithmetic, commonsense reasoning, and symbolic manipulation. The transparency of the reasoning process
    /// also makes these models more interpretable and debuggable.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Chain-of-thought reasoning for complex problem solving")]
    ChainOfThoughtModel,

    /// <summary>
    /// A self-consistency reasoning model that explores multiple reasoning paths.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Self-Consistency reasoning improves upon chain-of-thought by generating multiple
    /// independent reasoning paths for the same problem and then selecting the most consistent answer. Think
    /// of it like asking several experts to solve a problem independently and then taking the answer that
    /// most of them agree on. This approach is particularly powerful for problems where there might be multiple
    /// valid reasoning approaches. By exploring different paths, the model becomes more robust to errors in
    /// any single reasoning chain and can identify when it's uncertain about an answer (when the paths disagree).
    /// This makes self-consistency ideal for high-stakes applications where reliability is crucial.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Self-consistency reasoning with multiple paths")]
    SelfConsistencyModel,

    /// <summary>
    /// A tree-of-thought reasoning model for systematic exploration of reasoning paths.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tree-of-Thought (ToT) reasoning extends chain-of-thought by maintaining a tree
    /// of possible reasoning paths instead of a single chain. At each step, the model considers multiple
    /// possible next steps, evaluates them, and may pursue several promising directions in parallel. This is
    /// like solving a puzzle where you might try different approaches, backtrack when you hit dead ends, and
    /// explore alternative solutions. The model can use various search strategies (breadth-first, depth-first,
    /// beam search) to navigate the reasoning tree efficiently. This systematic exploration makes ToT particularly
    /// effective for tasks like puzzle solving, planning, and creative problem solving where the solution space
    /// is large and complex.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Tree-of-thought systematic reasoning exploration")]
    TreeOfThoughtModel,

    /// <summary>
    /// A reasoning model with iterative refinement capabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Iterative Refinement reasoning models improve their answers through multiple
    /// rounds of self-reflection and correction. After generating an initial answer, the model critically
    /// examines its own reasoning, identifies potential errors or gaps, and produces an improved version.
    /// This process can repeat several times, with each iteration building on insights from previous ones.
    /// It's similar to writing an essay where you create a first draft, then revise it multiple times to
    /// improve clarity, fix errors, and strengthen arguments. This approach is particularly valuable for
    /// complex tasks where perfection on the first attempt is unlikely, such as code generation, mathematical
    /// proofs, or detailed analysis tasks.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Iterative refinement reasoning model")]
    IterativeRefinementModel,

    // Ensemble Model Types
    
    /// <summary>
    /// A custom ensemble model with flexible configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A custom ensemble allows you to combine any types of models (neural networks,
    /// regression models, time series models, etc.) using various strategies. You have full control over
    /// which models to include, how to train them, and how to combine their predictions. This flexibility
    /// lets you create ensembles tailored to your specific problem, potentially achieving better results
    /// than any single model or pre-defined ensemble type.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.General },
              "Custom ensemble with flexible model combination")]
    CustomEnsemble,
    
    /// <summary>
    /// An ensemble that combines predictions through voting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A voting ensemble is like asking multiple experts for their opinion and going
    /// with the majority vote (hard voting) or averaging their confidence levels (soft voting). Each model
    /// in the ensemble makes its prediction independently, and these predictions are combined through voting.
    /// This simple but effective approach often outperforms individual models because different models may
    /// capture different patterns in the data, and their errors tend to cancel out when combined.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.General },
              "Voting-based ensemble combination")]
    VotingEnsemble,
    
    /// <summary>
    /// An ensemble using stacking (stacked generalization).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stacking uses a two-level approach: first, multiple "base" models make predictions,
    /// then a "meta-learner" model learns how to best combine these predictions. It's like having a team of
    /// specialists and a manager who knows each specialist's strengths and weaknesses. The meta-learner can
    /// discover complex relationships between base model predictions, often achieving better performance than
    /// simple averaging or voting. This is particularly effective when base models have complementary strengths.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.General },
              "Stacking ensemble with meta-learner")]
    StackingEnsemble,
    
    /// <summary>
    /// An ensemble using blending for prediction combination.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Blending is similar to stacking but simpler. Instead of using cross-validation
    /// to generate base model predictions for training the meta-learner, blending uses a fixed holdout
    /// validation set. Base models are trained on the training data, make predictions on the validation set,
    /// and these predictions are used to train a blender model that learns optimal weights. While potentially
    /// less robust than stacking, blending is faster and easier to implement.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.General },
              "Blending ensemble with learned weights")]
    BlendingEnsemble,
    
    /// <summary>
    /// An ensemble using dynamic model selection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dynamic selection ensembles choose different models for different inputs based
    /// on their expected performance. Instead of always using all models or fixed weights, this approach
    /// analyzes each input and selects the model(s) most likely to perform well for that specific case.
    /// It's like having different experts for different types of problems and choosing the right expert
    /// based on the question. This can be more efficient and accurate than using all models for every prediction.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.General },
              "Dynamic model selection ensemble")]
    DynamicSelectionEnsemble,
    
    /// <summary>
    /// A Bayesian model averaging ensemble.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bayesian Model Averaging (BMA) combines models by considering the probability
    /// that each model is the "true" model given the data. Instead of picking one best model or using
    /// fixed weights, BMA accounts for model uncertainty by weighting predictions based on how likely
    /// each model is to be correct. This provides not just predictions but also measures of uncertainty,
    /// making it valuable when you need to know how confident to be in the predictions.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.General },
              "Bayesian model averaging ensemble")]
    BayesianAverageEnsemble,
    
    /// <summary>
    /// A mixture of experts ensemble model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mixture of Experts divides the problem space into regions and trains different
    /// "expert" models to specialize in different regions. A gating network learns to route each input to
    /// the most appropriate expert(s). It's like having specialists for different areas - a heart doctor,
    /// a brain doctor, etc. - and a triage nurse who decides which specialist to consult based on the
    /// symptoms. This allows the ensemble to handle complex problems with different characteristics in
    /// different parts of the input space.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.General },
              "Mixture of experts with gating network")]
    MixtureOfExpertsEnsemble,
    
    //
    // Online Learning Models
    //
    
    /// <summary>
    /// Online Perceptron for incremental binary classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Online Perceptron is one of the simplest machine learning algorithms that
    /// learns from one example at a time. It's like a student who updates their understanding after
    /// each question rather than studying all questions at once. It works well for data that arrives
    /// in a stream and when you can't store all data in memory. Perfect for simple classification
    /// tasks where data patterns don't change much over time.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Classification,
              new[] { MetricGroups.BinaryClassification, MetricGroups.General },
              "Simple online linear classifier")]
    OnlinePerceptron,
    
    /// <summary>
    /// Passive-Aggressive algorithm for online learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Passive-Aggressive algorithms are "passive" when they make correct predictions
    /// (no update needed) but "aggressive" when wrong (large updates to fix the error). It's like a
    /// student who doesn't change their approach when getting answers right, but makes significant
    /// adjustments when wrong. This makes it effective for online learning where you want to adapt
    /// quickly to mistakes while being stable when performing well.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.General },
              "Online learning with aggressive updates on errors")]
    PassiveAggressive,
    
    /// <summary>
    /// Online Stochastic Gradient Descent for various loss functions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Online SGD updates the model after each example using gradient descent.
    /// Think of it like walking downhill in the fog - you can only see one step at a time, but by
    /// always stepping in the downward direction, you eventually reach the bottom. It's versatile,
    /// works with many loss functions, and can adapt to changing data patterns. Great for large-scale
    /// learning where data arrives continuously.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Versatile online optimization algorithm")]
    OnlineSGD,
    
    /// <summary>
    /// Adaptive online learning model with drift detection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adaptive Online models can detect when data patterns change (concept drift)
    /// and adjust accordingly. Imagine learning about weather patterns, but climate change slowly alters
    /// them - these models notice the change and adapt. They're essential for real-world applications
    /// where the relationships in data evolve over time, like user preferences, market conditions, or
    /// sensor readings that degrade.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Regression,
              new[] { MetricGroups.Regression, MetricGroups.General },
              "Online learning with automatic drift adaptation")]
    AdaptiveOnline,
    
    /// <summary>
    /// Online Support Vector<double> Machine for streaming data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Online SVM adapts the powerful Support Vector<double> Machine algorithm for streaming
    /// data. It learns a decision boundary by finding support vectors - the most important examples that
    /// define the separation between classes. Like a security system that remembers only the most suspicious
    /// and most trustworthy cases to make future decisions. Can use kernel functions for non-linear boundaries.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Classification,
              new[] { MetricGroups.BinaryClassification, MetricGroups.General },
              "Online maximum margin classifier")]
    OnlineSVM,
    
    /// <summary>
    /// Adaptive Regularization of Weights (AROW) for online learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AROW maintains confidence in each feature weight - it's more cautious about
    /// updating weights it's uncertain about and more aggressive with weights it's confident in. Like a
    /// student who studies harder on topics they're unsure about while maintaining what they know well.
    /// Particularly effective when features have different scales or importance.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Classification,
              new[] { MetricGroups.BinaryClassification, MetricGroups.General },
              "Confidence-aware online classifier")]
    AROW,
    
    /// <summary>
    /// Confidence-Weighted learning for online classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Confidence-Weighted learning maintains a probability distribution over possible
    /// model parameters, updating more aggressively when predictions are uncertain. It's like a weather
    /// forecaster who adjusts predictions more when they're less sure about the forecast. This leads to
    /// faster learning on easy examples while being cautious on difficult ones.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Classification,
              new[] { MetricGroups.BinaryClassification, MetricGroups.General },
              "Probabilistic online classifier with confidence bounds")]
    ConfidenceWeighted,
    
    /// <summary>
    /// Online Random Forest for streaming data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Online Random Forest adapts the powerful Random Forest algorithm for streaming
    /// data. It maintains a forest of decision trees that can grow and adapt as new data arrives. Some
    /// trees might be replaced when they become outdated. It's like having a council of advisors where
    /// you can replace members who give outdated advice with new ones who understand current trends.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.General },
              "Ensemble of online decision trees")]
    OnlineRandomForest,
    
    /// <summary>
    /// Hoeffding Tree (Very Fast Decision Tree) for streaming data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hoeffding Trees build decision trees incrementally using statistical bounds
    /// to decide when enough data has been seen to make a split. It's like building a flowchart one
    /// decision at a time, but only adding a new decision when you're statistically confident it's the
    /// right choice. This ensures the tree built from streaming data is very similar to one built from
    /// all data at once.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Classification,
              new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Incremental decision tree with statistical guarantees")]
    HoeffdingTree,
    
    /// <summary>
    /// Online Bagging (Bootstrap Aggregating) for streaming data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Online Bagging creates multiple copies of a learning algorithm and trains
    /// each one on slightly different versions of the data stream. It's like having multiple students
    /// learn from the same teacher, but each student randomly decides when to pay attention. When making
    /// predictions, all students vote, making the final answer more reliable than any single student.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Ensemble,
              new[] { MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Online bootstrap aggregating ensemble")]
    OnlineBagging,
    
    /// <summary>
    /// Follow-The-Regularized-Leader (FTRL) for online learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FTRL is like a smart investor who learns from past mistakes while being
    /// careful not to overreact to any single piece of information. It's particularly good at handling
    /// sparse data (lots of zeros) and is widely used in online advertising to predict click-through rates.
    /// FTRL automatically identifies which features are important and ignores the rest, making it efficient
    /// for high-dimensional problems.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Classification,
              new[] { MetricGroups.BinaryClassification, MetricGroups.General },
              "Sparse online learning with per-coordinate learning rates")]
    FTRL,
    
    /// <summary>
    /// Online Naive Bayes classifier for streaming data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Online Naive Bayes is like a detective who assumes all clues are independent.
    /// Despite this "naive" assumption, it works surprisingly well. It learns incrementally, updating its
    /// beliefs about each feature as new examples arrive. It can handle both continuous data (like temperatures)
    /// using Gaussian distributions, or discrete data (like word counts) using multinomial distributions.
    /// It's especially good for text classification and spam filtering.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.Classification,
              new[] { MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification, MetricGroups.General },
              "Probabilistic classifier with feature independence assumption")]
    OnlineNaiveBayes,

    //
    // Multimodal Models
    //

    /// <summary>
    /// A model that can process and integrate multiple types of input data (text, image, audio, etc.).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multimodal models are like humans who can process information from multiple senses
    /// at once - seeing, hearing, and reading simultaneously. These models can handle different types of data
    /// (modalities) such as text, images, audio, or video, and combine them to make better predictions or
    /// understand content more deeply. For example, a multimodal model could analyze both the images and captions
    /// in a social media post to better understand its meaning, or combine audio and visual information from a
    /// video to generate accurate subtitles. These models use various fusion strategies (early, late, or
    /// cross-attention) to effectively combine information from different modalities.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General, MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
              "Model that processes and fuses multiple input modalities")]
    MultimodalModel,

    //
    // AutoML Models
    //

    /// <summary>
    /// Automated Machine Learning (AutoML) that automatically searches for the best model and hyperparameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML is like having an expert data scientist who automatically tries different
    /// models and settings to find the best solution for your problem. Instead of manually testing various
    /// algorithms and tuning their parameters, AutoML does this work for you. It explores different model
    /// types (like decision trees, neural networks, etc.) and their configurations to find what works best
    /// for your specific data. This is particularly useful when you're not sure which model to use or
    /// don't have time to manually optimize everything.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.AutoML,
              new[] { MetricGroups.General, MetricGroups.Regression, MetricGroups.BinaryClassification, MetricGroups.MulticlassClassification },
              "Automated model selection and hyperparameter optimization")]
    AutoML,

    //
    // Vision Models
    //

    /// <summary>
    /// Vision Transformer (ViT) for image classification and feature extraction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Vision Transformer applies the transformer architecture (originally designed for text)
    /// to images. It works by dividing an image into small patches (like 16x16 pixels), treating each patch as
    /// a "word" in a sentence. These patches are then processed using self-attention mechanisms that allow the
    /// model to understand relationships between different parts of the image. ViT has shown that pure transformer
    /// architectures can match or exceed traditional convolutional neural networks on image tasks, especially when
    /// trained on large datasets.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.MulticlassClassification, MetricGroups.General },
              "Transformer-based model for image classification")]
    VisionTransformer,

    //
    // Generative Models
    //

    /// <summary>
    /// Diffusion Model for high-quality image generation through iterative denoising.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Diffusion models generate images by learning to reverse a gradual noising process.
    /// Imagine taking a clear image and slowly adding static until it becomes pure noise. Diffusion models learn
    /// to reverse this process - starting from random noise and gradually removing it to create a clear image.
    /// This approach has proven incredibly effective for generating high-quality, diverse images and is the
    /// foundation of many modern AI art generators like DALL-E 2 and Stable Diffusion.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Generative model using iterative denoising process")]
    DiffusionModel,

    /// <summary>
    /// Denoising Diffusion Implicit Models (DDIM) for faster sampling in diffusion models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DDIM is an improved version of diffusion models that can generate images much faster.
    /// While standard diffusion models might need 1000 steps to generate an image, DDIM can produce similar quality
    /// with just 50-100 steps. It achieves this by using a deterministic sampling process instead of the random
    /// process used in standard diffusion models, making generation 10-50x faster while maintaining quality.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Fast deterministic sampling for diffusion models")]
    DDIMModel,

    /// <summary>
    /// Latent Diffusion Model that operates in a compressed latent space for efficiency.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Latent Diffusion Models work in a compressed representation of images rather than
    /// directly on pixels. First, an encoder compresses the image into a smaller "latent" representation (like
    /// a compressed file). The diffusion process then happens in this compressed space, which is much more
    /// efficient. Finally, a decoder expands the result back to a full image. This approach, used in Stable
    /// Diffusion, makes it possible to generate high-resolution images on consumer hardware.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Efficient diffusion model operating in latent space")]
    LatentDiffusionModel,

    /// <summary>
    /// Score-based Stochastic Differential Equation model for continuous-time diffusion.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Score-based SDEs provide a mathematical framework for diffusion models using
    /// continuous-time processes. Instead of discrete steps, they model the noising/denoising process as a
    /// smooth, continuous flow. This theoretical foundation enables more flexible sampling strategies and
    /// better understanding of how diffusion models work. While more complex mathematically, they offer
    /// advantages in terms of flexibility and theoretical guarantees.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Continuous-time diffusion using stochastic differential equations")]
    ScoreSDE,

    /// <summary>
    /// Consistency Model for single-step generation with consistency constraints.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Consistency Models are designed to generate high-quality images in just one or
    /// very few steps, unlike diffusion models that need many steps. They achieve this by learning to map
    /// any noisy image directly to the clean image it would become after full denoising. This makes them
    /// extremely fast - potentially real-time generation - while maintaining quality comparable to diffusion
    /// models. They represent a significant advancement in making AI image generation practical for
    /// real-time applications.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Fast single-step generation with consistency constraints")]
    ConsistencyModel,

    /// <summary>
    /// Flow Matching Model using optimal transport for generative modeling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Flow Matching models generate data by learning smooth transformations from simple
    /// distributions (like Gaussian noise) to complex data distributions (like images). They use concepts from
    /// optimal transport theory to find the most efficient path between noise and data. Think of it as finding
    /// the smoothest way to morph random noise into meaningful images. This approach can be more stable to
    /// train than GANs and more efficient than standard diffusion models.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Generative model using optimal transport flow matching")]
    FlowMatchingModel,

    /// <summary>
    /// Foundation Model - large pre-trained language or multimodal model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Foundation Models are large AI models pre-trained on vast amounts of data that can be
    /// adapted for many different tasks. Examples include GPT, BERT, and CLIP. These models have learned general
    /// knowledge and can be fine-tuned for specific applications like classification, regression, or generation.
    /// They represent a paradigm shift in AI where instead of training models from scratch for each task, we start
    /// with a powerful pre-trained model and adapt it. This approach often achieves better results with less data
    /// and training time.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General, MetricGroups.NeuralNetwork },
              "Large pre-trained model adaptable to various tasks")]
    FoundationModel,

    /// <summary>
    /// Interpretable Model Wrapper - adds interpretability and explainability features to any model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Interpretable Model Wrappers add transparency to AI models by providing explanations
    /// for their predictions. They implement techniques like SHAP (which shows how each input feature contributes
    /// to the prediction), LIME (which explains individual predictions), and feature importance analysis. This is
    /// crucial when you need to understand why a model made a specific decision - for example, why a loan was
    /// approved or denied, or which factors most influenced a medical diagnosis. The wrapper doesn't change how
    /// the underlying model works; it adds tools to peek inside and understand the model's reasoning.
    /// </para>
    /// </remarks>
    [ModelInfo(ModelCategory.NeuralNetwork,
              new[] { MetricGroups.General },
              "Wrapper that adds interpretability features to any model")]
    InterpretableWrapper
}