# MARFIN Project - Financial Risk Assessment using Multimodal Analogical Reasoning

## Overview
The MARFIN project revolutionizes financial risk assessment by employing Multimodal Analogical Reasoning over knowledge graphs. This innovative approach integrates diverse financial data sources, including market trends, asset performance, and economic indicators, to uncover hidden relationships and patterns. By fusing visual and textual cues, researchers aim to enhance genre classification accuracy, while financial institutions seek to mitigate potential losses and optimize investment strategies using MARFIN's advanced risk identification techniques.

## Key Features
- **Multimodal Knowledge Graph Integration**: Combines various financial data types such as textual reports, numerical data, and multimedia elements like graphs or charts.
- **Financial Data Relationships**: Uncovers and represents complex relationships between entities such as companies, financial instruments, economic indicators, and market trends.
- **Advanced Risk Identification**: Utilizes multimodal data to enhance risk assessment and identification techniques.
- **IMDB Multimodal Dataset**: Leverages the IMDB Multimodal Vision & NLP Genre Classification dataset, offering a curated collection of movie posters and detailed plot summaries for genre classification.

## Components

### 1. Multimodal Knowledge Graph for Finance
In MARFIN, a multimodal knowledge graph integrates various types of financial data. For instance:
- **Nodes**: Represent entities such as companies, financial instruments, economic indicators, or market trends.
- **Edges**: Denote relationships between these entities (e.g., ownership, correlation, causality).

### 2. Knowledge Graph Construction
The `FinancialKnowledgeGraph` class constructs the knowledge graph. This example includes:
- **Entities**: "StockMarket," "MarketCrash," "CreditDefault."
- **Relationships**: "Affects" and "LeadsTo."

You can extend and modify this knowledge graph to represent more complex relationships in financial data.

### 3. Multimodal Embedding Generation
The `MultimodalEmbeddingGenerator` class generates embeddings for nodes in the knowledge graph based on multiple modalities (text, image, and numerical attributes). While the example uses a basic linear layer, real-world scenarios would likely use sophisticated models such as multimodal transformers or graph convolutional networks (GCNs) to capture complex relationships within and between modalities.

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt