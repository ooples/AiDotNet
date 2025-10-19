namespace AiDotNet.Enums;

/// <summary>
/// Defines the types of machine learning models available in the AiDotNet library.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This enum lists all the different AI models you can use in this library.
/// Think of these as different tools in your AI toolbox - each one works best for specific
/// types of problems. Some are simple and easy to understand, while others are more powerful
/// but complex. Start with simpler models like SimpleRegression before moving to more advanced ones.
/// </para>
/// </remarks>
public enum ModelType
{
    /// <summary>
    /// Represents no model selection.
    /// </summary>
    None,

    /// <summary>
    /// An automated machine learning model that automatically selects and trains the best model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML (Automated Machine Learning) automates the process of selecting and
    /// configuring the best machine learning model for your data. Instead of manually trying different
    /// models, AutoML experiments with various algorithms, evaluates them, and chooses the one that
    /// performs best. It's like having an AI assistant that tests different approaches and picks the
    /// winner for you, saving you time and expertise.
    /// </para>
    /// </remarks>
    AutoML,
    
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
    WeightedRegression,
    
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
    RandomForest,
    
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
    DecisionTree,
    
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
    GradientBoosting,
    
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
    ConditionalInferenceTree,
    
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
    M5ModelTree,
    
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
    NonLinearRegression,
    
    /// <summary>
    /// A powerful algorithm that finds patterns by mapping data to higher-dimensional spaces.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Support Vector Regression finds patterns by transforming your data into a different
    /// space where the pattern becomes simpler. Imagine trying to separate mixed red and blue marbles on a table.
    /// It might be hard in 2D, but if you could lift some marbles up (adding a 3rd dimension), the separation
    /// might become easier. SVR uses a similar concept mathematically, making it powerful for complex patterns,
    /// though it can be slower and harder to tune than simpler models.
    /// </para>
    /// </remarks>
    SupportVectorRegression,
    
    /// <summary>
    /// Combines ridge regression with kernel methods to handle non-linear relationships.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kernel Ridge Regression is like Support Vector Regression but with a different
    /// mathematical approach. It uses "kernels" (special mathematical functions) to transform your data
    /// into a space where complex relationships become simpler. This allows it to capture non-linear patterns
    /// while still maintaining some of the simplicity and efficiency of linear models.
    /// </para>
    /// </remarks>
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
    GaussianProcessRegression,
    
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
    ExpressionTree,
    
    /// <summary>
    /// A mathematical representation of data as points in multi-dimensional space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Vector models represent your data as points in space with multiple dimensions. 
    /// Each feature becomes a dimension - so if you have height and weight, each data point is plotted 
    /// in 2D space. With more features, you get more dimensions (though these become hard to visualize). 
    /// This representation allows mathematical operations that can reveal patterns and relationships 
    /// in your data.
    /// </para>
    /// </remarks>
    Vector,
    
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
    PoissonRegression,
    
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
    MultinomialLogisticRegression,

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
    LogisticRegression,
    
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
    NegativeBinomialRegression,
    
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
    MultilayerPerceptronRegression,

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
    DeepBeliefNetwork,

    ResidualNeuralNetwork,

    VariationalAutoencoder,

    CapsuleNetwork,

    QuantumNeuralNetwork,

    SelfOrganizingMap,

    LiquidStateMachine,

    HopfieldNetwork,

    GraphNeuralNetwork,

    ExtremeLearningMachine,

    DifferentiableNeuralComputer,

    EchoStateNetwork,

    DeepQNetwork,

    GenerativeAdversarialNetwork,

    NeuralTuringMachine,

    NEAT,

    MemoryNetwork,

    LSTMNeuralNetwork,

    HTMNetwork,

    NeuralNetwork,

    OccupancyNetwork,

    RestrictedBoltzmannMachine,

    SpikingNeuralNetwork,

    FeedForwardNetwork,

    Linear,
    Polynomial,
    Symbolic,
    TreeBased,

    ARIMAXModel,

    ARMAModel,

    ARModel,

    BayesianStructuralTimeSeriesModel,

    DynamicRegressionWithARIMAErrors,

    ExponentialSmoothingModel,

    GARCHModel,

    GRUNeuralNetwork,

    SiameseNetwork,

    InterventionAnalysisModel,

    TBATSModel,

    STLDecomposition,

    VARModel,

    UnobservedComponentsModel,

    TransferFunctionModel,

    StateSpaceModel,

    ARIMAModel,

    MAModel,

    SARIMAModel,

    SpectralAnalysisModel,

    ProphetModel,

    NeuralNetworkARIMA
}