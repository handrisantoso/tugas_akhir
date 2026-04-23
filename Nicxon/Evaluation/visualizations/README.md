# Embedding Visualization Summary

## Generated Visualizations

### 1. PCA 3D Visualization
- **File**: `embedding_visualization_pca.png` (static), `embedding_visualization_pca_interactive.html` (interactive)
- **Method**: Principal Component Analysis (PCA)
- **Purpose**: Shows linear relationships and major variance directions in the embedding space
- **Use in Report**: Demonstrates how RAG answers cluster relative to questions vs raw model answers

### 2. t-SNE 3D Visualization  
- **File**: `embedding_visualization_tsne.png` (static), `embedding_visualization_tsne_interactive.html` (interactive)
- **Method**: t-Distributed Stochastic Neighbor Embedding (t-SNE)
- **Purpose**: Reveals non-linear patterns and local neighborhood structures
- **Use in Report**: Shows semantic groupings and how similar answers cluster together

### 3. Similarity Heatmap
- **File**: `similarity_heatmap.png`
- **Purpose**: Quantitative overview of average cosine similarities between all model types
- **Use in Report**: Quick visual comparison of model performance and similarity patterns

## Interpretation Guide

### Color Coding
- 🔴 **Red**: Questions
- 🟦 **Teal**: RAG System Answers  
- 🔵 **Blue**: Raw Gemini Answers
- 🟢 **Green**: DeepSeek R1 Zero Answers

### What to Look For
1. **Clustering**: Do RAG answers cluster closer to questions than raw model answers?
2. **Separation**: Are there clear separations between different model types?
3. **Consistency**: Do similar question types produce similar answer patterns?
4. **Outliers**: Which questions/answers are most different from the rest?

### Project Report Usage
- Include both PCA and t-SNE visualizations to show different perspectives
- Use interactive plots for presentations/demos
- Reference specific clusters or patterns in your analysis
- Compare similarity scores between models using the heatmap

## Technical Details
- **Dimensionality**: Original 768D embeddings reduced to 3D
- **Models Compared**: RAG System vs Raw Gemini 2.0 vs DeepSeek R1 Zero
- **Embedding Model**: Google text-embedding-004
- **Questions**: 10 diverse library-related queries across multiple categories
