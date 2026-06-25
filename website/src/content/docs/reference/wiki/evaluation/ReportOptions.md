---
title: "ReportOptions"
description: "Configuration options for evaluation report generation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for evaluation report generation.

## For Beginners

After evaluating your model, you need to communicate results.
These options control:

- Format (Markdown for docs, JSON for APIs, HTML for dashboards)
- Detail level (summary for stakeholders, full for data scientists)
- What to include (metrics, plots, recommendations)

## How It Works

Controls how evaluation results are formatted and presented, including output format,
detail level, and what sections to include.

## Properties

| Property | Summary |
|:-----|:--------|
| `AppendToFile` | Whether to append to existing file. |
| `Author` | Author name for the report. |
| `CustomCss` | Custom CSS for HTML reports. |
| `DecimalPlaces` | Number of decimal places for metrics. |
| `DetailLevel` | Detail level for the report. |
| `EmbedImagesInHtml` | Whether to embed images in HTML (base64). |
| `Format` | Output format for the report. |
| `IncludeBeginnerExplanations` | Whether to include "For Beginners" explanations. |
| `IncludeComputationTimes` | Whether to include computation time statistics. |
| `IncludeConfidenceIntervals` | Whether to include confidence intervals in tables. |
| `IncludeExecutiveSummary` | Whether to include executive summary. |
| `IncludeInterpretations` | Whether to include interpretation/explanation for each metric. |
| `IncludeMetricTables` | Whether to include metric tables. |
| `IncludeRawData` | Whether to include raw data exports. |
| `IncludeRecommendations` | Whether to include recommendations for improvement. |
| `IncludeSectionNumbers` | Whether to include section numbers. |
| `IncludeTableOfContents` | Whether to include table of contents. |
| `IncludeTimestamp` | Whether to include timestamp. |
| `IncludeVisualizations` | Whether to include visualization data/references. |
| `IncludeWarnings` | Whether to include warnings (data issues, assumption violations). |
| `Language` | Language for the report. |
| `LatexDocumentClass` | LaTeX document class for LaTeX reports. |
| `LatexStandalone` | Whether to make LaTeX report standalone. |
| `MaxTableRows` | Maximum table rows before truncation. |
| `OutputEncoding` | Encoding for output file. |
| `OutputPath` | Output file path. |
| `SectionsToExclude` | Sections to exclude. |
| `SectionsToInclude` | Sections to include (if null, all applicable sections). |
| `Subtitle` | Subtitle or description. |
| `TimestampFormat` | Custom timestamp format. |
| `Title` | Title for the report. |
| `UsePercentages` | Whether to use percentage format where appropriate. |

