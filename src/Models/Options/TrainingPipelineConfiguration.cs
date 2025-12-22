using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for a multi-step training pipeline with customizable stages.
/// </summary>
/// <remarks>
/// <para>
/// A training pipeline defines a sequence of training stages that are executed in order.
/// Each stage can have its own training method, optimizer, learning rate, dataset, and
/// evaluation criteria. This enables advanced training workflows like:
/// </para>
/// <list type="bullet">
/// <item><description>Pre-training → SFT → DPO → Evaluation</description></item>
/// <item><description>SFT → RLHF with multiple reward model iterations</description></item>
/// <item><description>Curriculum learning with progressively harder data</description></item>
/// <item><description>Multi-task training with stage-specific objectives</description></item>
/// </list>
/// <para><b>For Beginners:</b> Think of this as a recipe with multiple cooking steps.
/// Just like you might marinate, then sear, then bake - training can have multiple
/// phases where each phase teaches the model something different.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class TrainingPipelineConfiguration<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the ordered list of training stages in the pipeline.
    /// </summary>
    /// <remarks>
    /// Stages are executed sequentially. The output model from each stage
    /// becomes the input model for the next stage.
    /// </remarks>
    public List<TrainingStage<T, TInput, TOutput>> Stages { get; set; } = new();

    /// <summary>
    /// Gets or sets whether to use automatic pipeline selection when no stages are defined.
    /// </summary>
    /// <remarks>
    /// When true and Stages is empty, the system analyzes available data and
    /// automatically constructs an appropriate training pipeline.
    /// </remarks>
    public bool EnableAutoSelection { get; set; } = true;

    /// <summary>
    /// Gets or sets the global random seed for reproducibility across all stages.
    /// </summary>
    public int? GlobalSeed { get; set; }

    /// <summary>
    /// Gets or sets whether to save checkpoints between stages.
    /// </summary>
    public bool SaveIntermediateCheckpoints { get; set; } = true;

    /// <summary>
    /// Gets or sets the directory for intermediate checkpoints.
    /// </summary>
    public string? CheckpointDirectory { get; set; }

    /// <summary>
    /// Gets or sets whether to run evaluation after each stage.
    /// </summary>
    public bool EvaluateAfterEachStage { get; set; } = true;

    /// <summary>
    /// Gets or sets the global early stopping configuration applied across stages.
    /// </summary>
    public GlobalEarlyStoppingConfig? GlobalEarlyStopping { get; set; }

    /// <summary>
    /// Gets or sets callback actions to execute between stages.
    /// </summary>
    public List<Action<TrainingStageResult<T, TInput, TOutput>>>? InterStageCallbacks { get; set; }

    /// <summary>
    /// Adds a training stage to the pipeline.
    /// </summary>
    /// <param name="stage">The stage to add.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddStage(TrainingStage<T, TInput, TOutput> stage)
    {
        Stages.Add(stage ?? throw new ArgumentNullException(nameof(stage)));
        return this;
    }

    /// <summary>
    /// Adds a supervised fine-tuning (SFT) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddSFTStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Supervised Fine-Tuning",
            StageType = TrainingStageType.SupervisedFineTuning,
            FineTuningMethod = FineTuningMethodType.SFT
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a preference optimization stage (DPO, IPO, etc.).
    /// </summary>
    /// <param name="method">The preference optimization method to use.</param>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddPreferenceStage(
        FineTuningMethodType method = FineTuningMethodType.DPO,
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = $"Preference Optimization ({method})",
            StageType = TrainingStageType.PreferenceOptimization,
            FineTuningMethod = method
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a reinforcement learning stage (RLHF, GRPO, etc.).
    /// </summary>
    /// <param name="method">The RL method to use.</param>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddRLStage(
        FineTuningMethodType method = FineTuningMethodType.RLHF,
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = $"Reinforcement Learning ({method})",
            StageType = TrainingStageType.ReinforcementLearning,
            FineTuningMethod = method
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds an evaluation-only stage (no training, just metrics).
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddEvaluationStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Evaluation",
            StageType = TrainingStageType.Evaluation,
            IsEvaluationOnly = true
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a custom training stage with user-defined logic.
    /// </summary>
    /// <param name="name">Name of the custom stage.</param>
    /// <param name="trainFunc">The custom training function.</param>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddCustomStage(
        string name,
        Func<IFullModel<T, TInput, TOutput>, FineTuningData<T, TInput, TOutput>, CancellationToken, Task<IFullModel<T, TInput, TOutput>>> trainFunc,
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = name,
            StageType = TrainingStageType.Custom,
            CustomTrainingFunction = trainFunc
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Creates a standard SFT → DPO pipeline (most common alignment workflow).
    /// </summary>
    /// <param name="sftData">Training data for the SFT stage.</param>
    /// <param name="preferenceData">Preference data for the DPO stage.</param>
    /// <returns>A configured training pipeline.</returns>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> StandardAlignment(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? preferenceData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            EnableAutoSelection = false
        };

        pipeline.AddSFTStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Options = new FineTuningOptions<T> { Epochs = 3, BatchSize = 8 };
        });

        pipeline.AddPreferenceStage(FineTuningMethodType.DPO, stage =>
        {
            stage.TrainingData = preferenceData;
            stage.Options = new FineTuningOptions<T> { Epochs = 1, BatchSize = 4 };
        });

        return pipeline;
    }

    /// <summary>
    /// Creates a full RLHF pipeline (SFT → Reward Model → PPO).
    /// </summary>
    /// <param name="sftData">Training data for the SFT stage.</param>
    /// <param name="rlData">RL data with rewards for the PPO stage.</param>
    /// <returns>A configured training pipeline.</returns>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> FullRLHF(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? rlData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            EnableAutoSelection = false
        };

        pipeline.AddSFTStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Options = new FineTuningOptions<T> { Epochs = 3, BatchSize = 8 };
        });

        pipeline.AddRLStage(FineTuningMethodType.RLHF, stage =>
        {
            stage.TrainingData = rlData;
            stage.Options = new FineTuningOptions<T> { Epochs = 1, BatchSize = 4 };
        });

        return pipeline;
    }

    /// <summary>
    /// Creates a Constitutional AI pipeline (SFT → CAI critique/revision → preference).
    /// </summary>
    /// <param name="sftData">Training data for the SFT stage.</param>
    /// <param name="principles">Constitutional principles for CAI.</param>
    /// <returns>A configured training pipeline.</returns>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> ConstitutionalAI(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        string[]? principles = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            EnableAutoSelection = false
        };

        pipeline.AddSFTStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Options = new FineTuningOptions<T> { Epochs = 2, BatchSize = 8 };
        });

        pipeline.AddStage(new TrainingStage<T, TInput, TOutput>
        {
            Name = "Constitutional AI",
            StageType = TrainingStageType.Constitutional,
            FineTuningMethod = FineTuningMethodType.ConstitutionalAI,
            Options = new FineTuningOptions<T>
            {
                ConstitutionalPrinciples = principles ?? Array.Empty<string>(),
                Epochs = 2
            }
        });

        return pipeline;
    }

    /// <summary>
    /// Creates a curriculum learning pipeline with progressively harder stages.
    /// </summary>
    /// <param name="curriculumStages">Ordered list of datasets from easy to hard.</param>
    /// <returns>A configured training pipeline.</returns>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> CurriculumLearning(
        params (string Name, FineTuningData<T, TInput, TOutput> Data)[] curriculumStages)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            EnableAutoSelection = false
        };

        for (int i = 0; i < curriculumStages.Length; i++)
        {
            var (name, data) = curriculumStages[i];
            pipeline.AddSFTStage(stage =>
            {
                stage.Name = $"Curriculum Stage {i + 1}: {name}";
                stage.TrainingData = data;
                // Reduce learning rate as difficulty increases
                stage.Options = new FineTuningOptions<T>
                {
                    Epochs = 2,
                    BatchSize = 8
                };
            });
        }

        return pipeline;
    }

    /// <summary>
    /// Creates an iterative refinement pipeline that runs multiple DPO rounds.
    /// </summary>
    /// <param name="iterations">Number of refinement iterations.</param>
    /// <param name="sftData">Initial SFT data.</param>
    /// <param name="preferenceData">Preference data for each iteration.</param>
    /// <returns>A configured training pipeline.</returns>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> IterativeRefinement(
        int iterations,
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? preferenceData = null)
    {
        if (iterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(iterations), "Must have at least 1 iteration.");
        }

        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            EnableAutoSelection = false
        };

        // Initial SFT
        pipeline.AddSFTStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Options = new FineTuningOptions<T> { Epochs = 3, BatchSize = 8 };
        });

        // Multiple DPO iterations
        for (int i = 0; i < iterations; i++)
        {
            pipeline.AddPreferenceStage(FineTuningMethodType.DPO, stage =>
            {
                stage.Name = $"DPO Iteration {i + 1}";
                stage.TrainingData = preferenceData;
                stage.Options = new FineTuningOptions<T>
                {
                    Epochs = 1,
                    BatchSize = 4,
                    // Reduce beta over iterations for more aggressive optimization
                    Beta = 0.1 / (1 + i * 0.1)
                };
            });

            // Add evaluation between iterations
            pipeline.AddEvaluationStage(stage =>
            {
                stage.Name = $"Evaluation after DPO {i + 1}";
            });
        }

        return pipeline;
    }

    /// <summary>
    /// Automatically selects an appropriate pipeline based on available data.
    /// </summary>
    /// <param name="availableData">The data available for training.</param>
    /// <returns>An automatically configured training pipeline.</returns>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> Auto(
        FineTuningData<T, TInput, TOutput> availableData)
    {
        if (availableData == null)
        {
            throw new ArgumentNullException(nameof(availableData));
        }

        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            EnableAutoSelection = true
        };

        // Build pipeline based on data characteristics

        // Stage 1: Always start with SFT if we have supervised data
        if (availableData.HasSFTData)
        {
            pipeline.AddSFTStage(stage =>
            {
                stage.TrainingData = availableData;
                stage.Options = new FineTuningOptions<T> { Epochs = 3, BatchSize = 8 };
            });
        }

        // Stage 2: Add preference optimization if we have preference data
        if (availableData.HasPairwisePreferenceData)
        {
            pipeline.AddPreferenceStage(FineTuningMethodType.DPO, stage =>
            {
                stage.TrainingData = availableData;
                stage.Options = new FineTuningOptions<T> { Epochs = 1, BatchSize = 4 };
            });
        }
        else if (availableData.HasUnpairedPreferenceData)
        {
            pipeline.AddPreferenceStage(FineTuningMethodType.KTO, stage =>
            {
                stage.TrainingData = availableData;
                stage.Options = new FineTuningOptions<T> { Epochs = 1, BatchSize = 4 };
            });
        }

        // Stage 3: Add RL if we have reward data
        if (availableData.HasRLData)
        {
            pipeline.AddRLStage(FineTuningMethodType.GRPO, stage =>
            {
                stage.TrainingData = availableData;
                stage.Options = new FineTuningOptions<T> { Epochs = 1, BatchSize = 4 };
            });
        }

        // Stage 4: Constitutional AI if we have critique data
        if (availableData.CritiqueRevisions.Length > 0)
        {
            pipeline.AddStage(new TrainingStage<T, TInput, TOutput>
            {
                Name = "Constitutional AI Refinement",
                StageType = TrainingStageType.Constitutional,
                FineTuningMethod = FineTuningMethodType.ConstitutionalAI,
                TrainingData = availableData
            });
        }

        // Always end with evaluation
        pipeline.AddEvaluationStage();

        return pipeline;
    }
}
