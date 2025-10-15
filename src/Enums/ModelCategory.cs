namespace AiDotNet.Enums;

/// <summary>
/// Categorizes machine learning models by their primary function or approach.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Machine learning models can be grouped into different categories based on what they do and how they work.
/// These categories help organize models by their primary purpose, such as predicting numeric values (Regression),
/// assigning labels (Classification), or analyzing data that changes over time (TimeSeries).
/// </para>
/// </remarks>
public enum ModelCategory
{
    /// <summary>
    /// Represents an unspecified or default model category.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a default value used when no specific category has been assigned.
    /// It typically indicates that a category hasn't been set yet or isn't applicable in a particular context.
    /// </para>
    /// </remarks>
    None,

    /// <summary>
    /// Models that predict continuous numeric values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Regression models predict numbers on a continuous scale. For example, predicting house prices,
    /// temperature, or a person's age. These models try to find relationships between input features and a numeric output,
    /// often by fitting a line or curve to your data. Common examples include Linear Regression, which fits a straight line,
    /// and Polynomial Regression, which fits more complex curves.
    /// </para>
    /// </remarks>
    Regression,

    /// <summary>
    /// Models that categorize data into discrete classes or labels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Classification models predict which category or class something belongs to. For example,
    /// determining if an email is spam or not, identifying the species of a flower, or diagnosing a medical condition.
    /// These models learn the boundaries between different classes from training data. Common examples include Logistic Regression
    /// (despite its name, it's used for classification), Decision Trees, and Support Vector<double> Machines.
    /// </para>
    /// </remarks>
    Classification,

    /// <summary>
    /// Models specialized for data that changes over time.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Time Series models analyze and predict data that's collected over time at regular intervals.
    /// Examples include stock prices, weather measurements, or monthly sales figures. These models account for the fact that
    /// recent values often influence future values and look for patterns like trends (general direction) and seasonality
    /// (regular cycles). Common examples include ARIMA, Exponential Smoothing, and Prophet.
    /// </para>
    /// </remarks>
    TimeSeries,

    /// <summary>
    /// Models that group similar data points together.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Clustering models find natural groupings in data without being told in advance what the groups should be.
    /// They identify which data points are similar to each other and different from others. For example, grouping customers with
    /// similar purchasing behaviors, finding regions with similar climate patterns, or segmenting images into different objects.
    /// Common examples include K-Means, Hierarchical Clustering, and DBSCAN.
    /// </para>
    /// </remarks>
    Clustering,

    /// <summary>
    /// Models that reduce the number of variables while preserving important information.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dimensionality Reduction models simplify data by reducing the number of features while keeping
    /// the most important information. It's like summarizing a long document into key points. This makes the data easier to
    /// visualize, faster to process, and can improve model performance by removing noise. Common examples include Principal
    /// Component Analysis (PCA), t-SNE, and autoencoders.
    /// </para>
    /// </remarks>
    DimensionalityReduction,

    /// <summary>
    /// Models that learn through trial and error by interacting with an environment.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reinforcement Learning models learn by taking actions in an environment and receiving rewards
    /// or penalties. They try to find the best strategy (policy) to maximize total rewards over time. It's similar to how
    /// we learn through trial and error. Examples include teaching a computer to play games, robots learning to navigate,
    /// or systems that optimize resource allocation. Common approaches include Q-Learning, Deep Q Networks, and Policy Gradient methods.
    /// </para>
    /// </remarks>
    Reinforcement,

    /// <summary>
    /// Models that combine multiple models to improve performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensemble models combine predictions from multiple models to get better results than any single model.
    /// It's like asking a group of experts instead of just one person. These methods can average out errors, reduce overfitting,
    /// and improve accuracy. Common approaches include Random Forests (which combine many decision trees), Gradient Boosting,
    /// and simple averaging of different models' predictions.
    /// </para>
    /// </remarks>
    Ensemble,

    /// <summary>
    /// Models inspired by the structure and function of the human brain.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural Network models are inspired by how neurons work in the human brain. They consist of
    /// interconnected "neurons" organized in layers that process information. Each connection has a weight that strengthens
    /// or weakens signals between neurons, and these weights are adjusted during training. Neural networks can learn complex
    /// patterns and are behind many recent AI breakthroughs. Types include Convolutional Neural Networks (for images),
    /// Recurrent Neural Networks (for sequences), and Transformers (for language).
    /// </para>
    /// </remarks>
    NeuralNetwork,

    /// <summary>
    /// Models that use probability theory to represent uncertainty.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Probabilistic models explicitly account for uncertainty in their predictions.
    /// Rather than just giving a single answer, they often provide a range of possible outcomes
    /// with associated probabilities, similar to a weather forecast saying "70% chance of rain."
    /// </para>
    /// </remarks>
    Probabilistic,

    /// <summary>
    /// Models that evaluate and optimize the ordering of items.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ranking models focus on getting the order of items right, rather than predicting
    /// specific values or categories. They're commonly used in search engines, recommendation systems,
    /// and information retrieval applications. Instead of asking "What is this?" or "How much?", ranking
    /// models answer questions like "Which items should appear first?" or "What's the best order to present these options?"
    /// 
    /// Examples include:
    /// - Search engines ranking web pages by relevance to a query
    /// - E-commerce sites ranking products a customer might want to buy
    /// - Streaming services ranking movies or shows you might want to watch
    /// - Social media ranking posts in your feed
    /// 
    /// Common approaches include Learning to Rank (LTR), PageRank, and various listwise, pairwise, and 
    /// pointwise ranking methods. Evaluation metrics typically include Mean Average Precision (MAP),
    /// Normalized Discounted Cumulative Gain (NDCG), and Mean Reciprocal Rank (MRR).
    /// </para>
    /// </remarks>
    Ranking,

    /// <summary>
    /// Models that automatically search for optimal configurations and hyperparameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML (Automated Machine Learning) models automatically find the best
    /// machine learning model and settings for your data. They handle the complex process of trying
    /// different algorithms, tuning their parameters, and evaluating performance. It's like having
    /// an experienced data scientist who systematically tests various approaches to find what works
    /// best. AutoML can save significant time and often finds better solutions than manual tuning.
    /// These models typically combine techniques like grid search, random search, or Bayesian optimization
    /// to efficiently explore the space of possible models and configurations.
    /// </para>
    /// </remarks>
    AutoML,

    /// <summary>
    /// Models that create new data samples similar to their training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Generative models learn to create new data that resembles their training data.
    /// Instead of just analyzing or classifying existing data, these models can generate entirely new examples.
    /// For instance, they can create realistic images, write text, compose music, or generate synthetic data
    /// for testing. Examples include Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs),
    /// Diffusion Models, and Large Language Models (LLMs) like GPT. These models learn the underlying patterns
    /// and distributions in their training data, then use that knowledge to produce novel outputs.
    /// </para>
    /// </remarks>
    Generative,
}