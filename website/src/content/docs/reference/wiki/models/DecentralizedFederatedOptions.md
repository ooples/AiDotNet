---
title: "DecentralizedFederatedOptions"
description: "Configuration options for decentralized (serverless) federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for decentralized (serverless) federated learning.

## For Beginners

Standard FL uses a central server to aggregate models. Decentralized FL
removes this server — nodes communicate directly with each other using peer-to-peer protocols.
This eliminates single point of failure and can be more robust in edge/IoT deployments.

## Properties

| Property | Summary |
|:-----|:--------|
| `DFedAvgMMomentum` | Gets or sets the momentum coefficient for DFedAvgM. |
| `DFedBCANumBlocks` | Gets or sets the number of parameter blocks for DFedBCA. |
| `DFedBCASelectionStrategy` | Gets or sets the block selection strategy for DFedBCA. |
| `DeTAGLearningRate` | Gets or sets the learning rate for DeTAG gradient tracking. |
| `Enabled` | Gets or sets whether decentralized mode is enabled. |
| `GossipFanout` | Gets or sets the gossip fanout (number of random peers per round). |
| `MixingRoundsPerTrainingRound` | Gets or sets the number of mixing rounds per training round. |
| `SegmentedGossipNumSegments` | Gets or sets the number of model segments for SegmentedGossip. |
| `TimeVaryingSeed` | Gets or sets the random seed for time-varying topology generation. |
| `Topology` | Gets or sets the topology type. |

