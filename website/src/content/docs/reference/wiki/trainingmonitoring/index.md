---
title: "Training Monitoring"
description: "All 47 public types in the AiDotNet.trainingmonitoring namespace, organized by kind."
section: "API Reference"
---

**47** public types in this namespace, organized by kind.

## Models & Types (34)

| Type | Summary |
|:-----|:--------|
| [`ConfusionMatrixDataPoint`](/docs/reference/wiki/trainingmonitoring/confusionmatrixdatapoint/) | Represents a confusion matrix data point. |
| [`ConsoleDashboard`](/docs/reference/wiki/trainingmonitoring/consoledashboard/) | Console-based training dashboard with ASCII charts. |
| [`CurveDataPoint`](/docs/reference/wiki/trainingmonitoring/curvedatapoint/) | Represents a PR/ROC curve data point. |
| [`DatasetLineage`](/docs/reference/wiki/trainingmonitoring/datasetlineage/) | Dataset lineage information. |
| [`DeploymentInfo`](/docs/reference/wiki/trainingmonitoring/deploymentinfo/) | Information about a model deployment. |
| [`EmailNotificationService`](/docs/reference/wiki/trainingmonitoring/emailnotificationservice/) | Email notification service using SMTP. |
| [`EnvironmentInfo`](/docs/reference/wiki/trainingmonitoring/environmentinfo/) | Environment information for reproducibility. |
| [`ExperimentInfo`](/docs/reference/wiki/trainingmonitoring/experimentinfo/) | Information about an experiment. |
| [`ExperimentTracker`](/docs/reference/wiki/trainingmonitoring/experimenttracker/) | Local file-based experiment tracker providing MLflow-compatible functionality. |
| [`GpuInfo`](/docs/reference/wiki/trainingmonitoring/gpuinfo/) | GPU information. |
| [`HardwareInfo`](/docs/reference/wiki/trainingmonitoring/hardwareinfo/) | Hardware information. |
| [`HistogramDataPoint`](/docs/reference/wiki/trainingmonitoring/histogramdatapoint/) | Represents a histogram data point. |
| [`HtmlDashboard`](/docs/reference/wiki/trainingmonitoring/htmldashboard/) | Generates interactive HTML dashboards for training visualization. |
| [`ImageDataPoint`](/docs/reference/wiki/trainingmonitoring/imagedatapoint/) | Represents an image data point. |
| [`LiveDashboard`](/docs/reference/wiki/trainingmonitoring/livedashboard/) | Provides a real-time training dashboard via embedded web server. |
| [`MetricValue`](/docs/reference/wiki/trainingmonitoring/metricvalue/) | A metric value with step information. |
| [`ModelLineage`](/docs/reference/wiki/trainingmonitoring/modellineage/) | Model lineage information tracking how the model was created. |
| [`ModelRegistry`](/docs/reference/wiki/trainingmonitoring/modelregistry/) | Local file-based model registry for managing model versions and deployments. |
| [`ModelVersion`](/docs/reference/wiki/trainingmonitoring/modelversion/) | A specific version of a registered model. |
| [`NotificationEventArgs`](/docs/reference/wiki/trainingmonitoring/notificationeventargs/) | Event arguments for notification events. |
| [`NotificationManager`](/docs/reference/wiki/trainingmonitoring/notificationmanager/) | Manages multiple notification services and provides unified notification delivery. |
| [`RegisteredModel`](/docs/reference/wiki/trainingmonitoring/registeredmodel/) | A registered model in the registry. |
| [`ResourceMonitor`](/docs/reference/wiki/trainingmonitoring/resourcemonitor/) | Monitors system resources (CPU, memory, GPU) during training. |
| [`ResourceSnapshot`](/docs/reference/wiki/trainingmonitoring/resourcesnapshot/) | A snapshot of resource usage at a point in time. |
| [`ResourceThresholds`](/docs/reference/wiki/trainingmonitoring/resourcethresholds/) | Resource warning thresholds. |
| [`ResourceWarningEventArgs`](/docs/reference/wiki/trainingmonitoring/resourcewarningeventargs/) | Event arguments for resource warnings. |
| [`RunComparison`](/docs/reference/wiki/trainingmonitoring/runcomparison/) | Comparison of multiple runs. |
| [`RunInfo`](/docs/reference/wiki/trainingmonitoring/runinfo/) | Information about a run. |
| [`ScalarDataPoint`](/docs/reference/wiki/trainingmonitoring/scalardatapoint/) | Represents a scalar data point. |
| [`SlackNotificationService`](/docs/reference/wiki/trainingmonitoring/slacknotificationservice/) | Slack notification service using webhooks or bot tokens. |
| [`SourceInfo`](/docs/reference/wiki/trainingmonitoring/sourceinfo/) | Source information for a run. |
| [`TextDataPoint`](/docs/reference/wiki/trainingmonitoring/textdatapoint/) | Represents a text data point. |
| [`TrainingMonitor<T>`](/docs/reference/wiki/trainingmonitoring/trainingmonitor/) | Implementation of training monitoring system for tracking model training progress. |
| [`TrainingNotification`](/docs/reference/wiki/trainingmonitoring/trainingnotification/) | Represents a training notification. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`TrainingMonitorBase<T>`](/docs/reference/wiki/trainingmonitoring/trainingmonitorbase/) | Base class for training monitoring implementations. |

## Interfaces (4)

| Type | Summary |
|:-----|:--------|
| [`IExperimentTracker`](/docs/reference/wiki/trainingmonitoring/iexperimenttracker/) | Interface for experiment tracking systems. |
| [`IModelRegistry`](/docs/reference/wiki/trainingmonitoring/imodelregistry/) | Interface for a model registry that manages model versions and deployments. |
| [`INotificationService`](/docs/reference/wiki/trainingmonitoring/inotificationservice/) | Interface for notification services. |
| [`ITrainingDashboard`](/docs/reference/wiki/trainingmonitoring/itrainingdashboard/) | Interface for training dashboards that visualize metrics and training progress. |

## Enums (6)

| Type | Summary |
|:-----|:--------|
| [`DeploymentStatus`](/docs/reference/wiki/trainingmonitoring/deploymentstatus/) | Deployment status. |
| [`ModelStage`](/docs/reference/wiki/trainingmonitoring/modelstage/) | Model version stage in the deployment lifecycle. |
| [`ModelVersionStatus`](/docs/reference/wiki/trainingmonitoring/modelversionstatus/) | Status of a model version. |
| [`NotificationSeverity`](/docs/reference/wiki/trainingmonitoring/notificationseverity/) | Severity level for training notifications. |
| [`NotificationType`](/docs/reference/wiki/trainingmonitoring/notificationtype/) | Type of training notification. |
| [`RunStatus`](/docs/reference/wiki/trainingmonitoring/runstatus/) | Status of an experiment run. |

## Options & Configuration (2)

| Type | Summary |
|:-----|:--------|
| [`EmailConfiguration`](/docs/reference/wiki/trainingmonitoring/emailconfiguration/) | Configuration for email notification service. |
| [`SlackConfiguration`](/docs/reference/wiki/trainingmonitoring/slackconfiguration/) | Configuration for Slack notification service. |

