namespace AiDotNet.Enums;

/// <summary>
/// Defines the positions where components can be placed in a machine learning pipeline.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A machine learning pipeline is like an assembly line where data flows through 
/// different processing steps. This enum specifies where each step can be placed in that line.
/// </para>
/// </remarks>
public enum PipelinePosition
{
    /// <summary>
    /// Component runs at the very beginning of the pipeline.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Like the entrance to a factory - this is where raw data first enters 
    /// the pipeline before any processing.
    /// </remarks>
    Start,

    /// <summary>
    /// Alias for Start - component runs at the very beginning of the pipeline.
    /// </summary>
    Beginning = Start,

    /// <summary>
    /// Component runs during data preprocessing phase.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Where data is cleaned and prepared - like washing vegetables before cooking.
    /// Handles tasks like removing missing values or normalizing data.
    /// </remarks>
    Preprocessing,

    /// <summary>
    /// Component runs before normalization in the pipeline.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Runs just before data normalization - like preparing ingredients
    /// before measuring and standardizing them.
    /// </remarks>
    BeforeNormalization,

    /// <summary>
    /// Component runs after normalization in the pipeline.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Runs after data has been normalized - processes or validates
    /// the normalized data.
    /// </remarks>
    AfterNormalization,

    /// <summary>
    /// Component runs during feature engineering phase.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Where new features are created from existing data - like combining
    /// ingredients to make a sauce. Creates more informative inputs for the model.
    /// </remarks>
    FeatureEngineering,

    /// <summary>
    /// Component runs before feature selection phase.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Runs just before selecting which features to keep - preparing data
    /// for the selection process.
    /// </remarks>
    BeforeFeatureSelection,

    /// <summary>
    /// Component runs during feature selection phase.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Where we choose which features to keep - like selecting the best
    /// ingredients for a recipe. Removes irrelevant or redundant features.
    /// </remarks>
    FeatureSelection,

    /// <summary>
    /// Component runs after feature selection phase.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Runs after features have been selected - processes or validates
    /// the selected features.
    /// </remarks>
    AfterFeatureSelection,

    /// <summary>
    /// Component runs before outlier removal in the pipeline.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Runs just before removing outliers - prepares data for
    /// outlier detection and removal.
    /// </remarks>
    BeforeOutlierRemoval,

    /// <summary>
    /// Component runs after outlier removal in the pipeline.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Runs after outliers have been removed - processes or validates
    /// the cleaned data.
    /// </remarks>
    AfterOutlierRemoval,

    /// <summary>
    /// Component runs before model training phase.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Final preparation step before training begins - ensures
    /// everything is ready for the model to learn.
    /// </remarks>
    BeforeTraining,

    /// <summary>
    /// Component runs during model training phase.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The main learning phase where the model discovers patterns in the data -
    /// like a student studying for an exam.
    /// </remarks>
    Training,

    /// <summary>
    /// Component runs during model validation phase.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Where we test how well the model learned - like giving a practice test 
    /// to see if the student is ready for the real exam.
    /// </remarks>
    Validation,

    /// <summary>
    /// Component runs after model predictions.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Processes the model's raw outputs - like translating test scores into 
    /// letter grades. Can adjust or refine predictions.
    /// </remarks>
    PostProcessing,

    /// <summary>
    /// Component runs at the very end of the pipeline.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The final stage where results are prepared for output - like packaging 
    /// finished products for delivery.
    /// </remarks>
    End,

    /// <summary>
    /// Component can run at any position in the pipeline.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A flexible component that can be placed anywhere it's needed - like a 
    /// quality checker that can work at any stage of production.
    /// </remarks>
    Any,

    /// <summary>
    /// Component runs in parallel branches.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Runs alongside the main pipeline - like having multiple assembly lines 
    /// working on different parts that will be combined later.
    /// </remarks>
    Parallel,

    /// <summary>
    /// Component runs conditionally based on previous results.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Only runs when certain conditions are met - like a special treatment 
    /// that's only applied if the previous step detected a problem.
    /// </remarks>
    Conditional,

    /// <summary>
    /// Custom position defined by user.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Allows you to define your own custom position in the pipeline for 
    /// special requirements.
    /// </remarks>
    Custom
}