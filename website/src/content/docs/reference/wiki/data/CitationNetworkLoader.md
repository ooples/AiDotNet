---
title: "CitationNetworkLoader<T>"
description: "Loads citation network datasets (Cora, CiteSeer, PubMed) for node classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Graph`

Loads citation network datasets (Cora, CiteSeer, PubMed) for node classification.

## For Beginners

Citation networks are graphs of research papers.

**Structure:**

- **Nodes**: Research papers
- **Edges**: Citations (Paper A cites Paper B)
- **Node Features**: Bag-of-words representation of paper abstracts
- **Labels**: Research topic/category

**Datasets:**

**Cora:**

- 2,708 papers
- 5,429 citations
- 1,433 features (unique words)
- 7 classes (topics): Case_Based, Genetic_Algorithms, Neural_Networks,

Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory

- Task: Classify papers by topic

**CiteSeer:**

- 3,312 papers
- 4,732 citations
- 3,703 features
- 6 classes: Agents, AI, DB, IR, ML, HCI

**PubMed:**

- 19,717 papers (about diabetes)
- 44,338 citations
- 500 features
- 3 classes: Diabetes Mellitus Type 1, Type 2, Experimental

**Key Property: Homophily**
Papers tend to cite papers on similar topics. This makes GNNs effective:

- If neighbors are similar topics, aggregate their features
- GNN learns to propagate topic information through citation network
- Even unlabeled papers can be classified based on what they cite

## How It Works

Citation networks are classic benchmarks for graph neural networks. Each dataset represents
academic papers as nodes and citations as edges, with the task being to classify papers into
research topics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CitationNetworkLoader(CitationNetworkLoader<>.CitationDataset,String,Boolean)` | Initializes a new instance of the `CitationNetworkLoader` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |
| `NumClasses` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConvertEdgesToTensor(ValueTuple<Int32,Int32>[])` | Converts an array of edge tuples to a tensor of shape [num_edges, 2]. |
| `CreateGraphClassificationTask(Double,Double,Nullable<Int32>)` |  |
| `CreateLinkPredictionTask(Double,Double,Nullable<Int32>)` |  |
| `DownloadAndExtractDatasetAsync(String,CancellationToken)` | Downloads and extracts the dataset from the standard source. |
| `EnsureDataExistsAsync(String,CancellationToken)` | Ensures the dataset files exist locally, downloading if necessary. |
| `ExtractTarAsync(Stream,String,CancellationToken)` | Extracts a tar archive from a stream. |
| `ExtractTarGzAsync(String,String,CancellationToken)` | Extracts a .tar.gz archive to the specified directory. |
| `GetCitesFilePath(String)` | Gets the path to the cites file for the current dataset. |
| `GetContentFilePath(String)` | Gets the path to the content file for the current dataset. |
| `GetDefaultDataPath` | Gets the default data cache path. |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `ParseCitesFileAsync(String,Dictionary<String,Int32>,CancellationToken)` | Parses the cites file to extract edges. |
| `ParseContentFileAsync(String,CancellationToken)` | Parses the content file to extract node features and labels. |
| `ParseDatasetFilesAsync(String,CancellationToken)` | Parses the dataset files and builds the graph structure. |
| `UnloadDataCore` |  |

