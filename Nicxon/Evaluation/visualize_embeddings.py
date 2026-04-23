import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any
import os

class EmbeddingVisualizer:
    def __init__(self):
        """Initialize the embedding visualizer"""
        # Set up color palette
        self.colors = {
            'question': '#FF6B6B',      # Red
            'RAG': '#4ECDC4',           # Teal
            'gemini': '#45B7D1',        # Blue  
            'deepseek': '#96CEB4'       # Green
        }
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        print("✅ Embedding visualizer initialized!")
    
    def load_embeddings(self, embeddings_file: str) -> Dict[str, Any]:
        """Load and prepare embeddings for visualization"""
        
        print("📖 Loading embeddings...")
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Prepare data for visualization
        all_embeddings = []
        labels = []
        question_ids = []
        categories = []
        
        for item in data:
            # Question embedding
            all_embeddings.append(item['question'])
            labels.append('Question')
            question_ids.append(f"Q{item['id']}")
            categories.append(item['metadata']['category'])
            
            # RAG answer embedding
            all_embeddings.append(item['RAG_gemini_flash_2.0'])
            labels.append('RAG')
            question_ids.append(f"Q{item['id']}")
            categories.append(item['metadata']['category'])
            
            # Gemini answer embedding
            all_embeddings.append(item['gemini_flash_2.0'])
            labels.append('Gemini')
            question_ids.append(f"Q{item['id']}")
            categories.append(item['metadata']['category'])
            
            # DeepSeek answer embedding
            all_embeddings.append(item['deepseek_r1_zero'])
            labels.append('DeepSeek')
            question_ids.append(f"Q{item['id']}")
            categories.append(item['metadata']['category'])
        
        return {
            'embeddings': np.array(all_embeddings),
            'labels': labels,
            'question_ids': question_ids,
            'categories': categories,
            'raw_data': data
        }
    
    def reduce_dimensions_pca(self, embeddings: np.ndarray, n_components: int = 3) -> np.ndarray:
        """Reduce embeddings to 3D using PCA"""
        print(f"🔄 Reducing dimensions to {n_components}D using PCA...")
        
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"   Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return reduced_embeddings
    
    def reduce_dimensions_tsne(self, embeddings: np.ndarray, n_components: int = 3) -> np.ndarray:
        """Reduce embeddings to 3D using t-SNE"""
        print(f"🔄 Reducing dimensions to {n_components}D using t-SNE...")
        
        # First reduce dimensions with PCA for computational efficiency
        # Use min of 50 or n_samples-1 to avoid the error
        if embeddings.shape[1] > 50:
            max_components = min(50, embeddings.shape[0] - 1)
            pca = PCA(n_components=max_components)
            embeddings = pca.fit_transform(embeddings)
            print(f"   Pre-reduced to {max_components}D using PCA before t-SNE")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(30, len(embeddings) // 4),
            random_state=42,
            n_iter=1000
        )
        reduced_embeddings = tsne.fit_transform(embeddings)
        
        return reduced_embeddings
    
    def create_matplotlib_3d_plot(self, data: Dict[str, Any], reduced_embeddings: np.ndarray, 
                                  method: str, output_file: str):
        """Create 3D matplotlib visualization"""
        
        print(f"📊 Creating 3D matplotlib plot with {method}...")
        
        fig = plt.figure(figsize=(15, 12))
        
        # Main 3D plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Plot points by model type
        unique_labels = list(set(data['labels']))
        
        for label in unique_labels:
            mask = np.array(data['labels']) == label
            points = reduced_embeddings[mask]
            
            ax1.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=self.colors.get(label.lower(), '#888888'),
                label=label,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax1.set_xlabel(f'{method} Component 1')
        ax1.set_ylabel(f'{method} Component 2')
        ax1.set_zlabel(f'{method} Component 3')
        ax1.set_title(f'3D Embedding Visualization ({method})\nQuestions vs Model Answers')
        ax1.legend()
        
        # 2D projections
        # XY projection
        ax2 = fig.add_subplot(222)
        for label in unique_labels:
            mask = np.array(data['labels']) == label
            points = reduced_embeddings[mask]
            ax2.scatter(points[:, 0], points[:, 1], 
                       c=self.colors.get(label.lower(), '#888888'), 
                       label=label, alpha=0.7, s=50)
        ax2.set_xlabel(f'{method} Component 1')
        ax2.set_ylabel(f'{method} Component 2')
        ax2.set_title('XY Projection')
        ax2.legend()
        
        # XZ projection
        ax3 = fig.add_subplot(223)
        for label in unique_labels:
            mask = np.array(data['labels']) == label
            points = reduced_embeddings[mask]
            ax3.scatter(points[:, 0], points[:, 2], 
                       c=self.colors.get(label.lower(), '#888888'), 
                       label=label, alpha=0.7, s=50)
        ax3.set_xlabel(f'{method} Component 1')
        ax3.set_ylabel(f'{method} Component 3')
        ax3.set_title('XZ Projection')
        ax3.legend()
        
        # YZ projection
        ax4 = fig.add_subplot(224)
        for label in unique_labels:
            mask = np.array(data['labels']) == label
            points = reduced_embeddings[mask]
            ax4.scatter(points[:, 1], points[:, 2], 
                       c=self.colors.get(label.lower(), '#888888'), 
                       label=label, alpha=0.7, s=50)
        ax4.set_xlabel(f'{method} Component 2')
        ax4.set_ylabel(f'{method} Component 3')
        ax4.set_title('YZ Projection')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to {output_file}")
    
    def create_interactive_plotly_3d(self, data: Dict[str, Any], reduced_embeddings: np.ndarray,
                                     method: str, output_file: str):
        """Create enhanced interactive 3D Plotly visualization with Question ID filtering"""
        
        print(f"📊 Creating interactive 3D Plotly visualization with {method}...")
        
        # Create DataFrame for easier handling
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1], 
            'z': reduced_embeddings[:, 2],
            'model': data['labels'],
            'question_id': data['question_ids'],
            'category': data['categories']
        })
        
        # Load questions for better hover info
        questions_file = 'test_questions.json'
        question_texts = {}
        if os.path.exists(questions_file):
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
                for i, q in enumerate(questions):
                    question_texts[f"Q{i+1}"] = q['question'][:80] + "..." if len(q['question']) > 80 else q['question']
        
        # Add question text to DataFrame
        df['question_text'] = df['question_id'].map(lambda qid: question_texts.get(qid, "Unknown"))
        
        # Create base figure
        fig = go.Figure()
        
        # Get unique question IDs and models
        unique_questions = sorted(df['question_id'].unique())
        unique_models = ['Question', 'RAG', 'Gemini', 'DeepSeek']
        
        # Add traces for each model and question combination
        for model in unique_models:
            for qid in unique_questions:
                model_question_data = df[(df['model'] == model) & (df['question_id'] == qid)]
                
                if len(model_question_data) > 0:
                    fig.add_trace(
                        go.Scatter3d(
                            x=model_question_data['x'],
                            y=model_question_data['y'],
                            z=model_question_data['z'],
                            mode='markers',
                            name=f"{model}",
                            legendgroup=model,
                            showlegend=(qid == unique_questions[0]),  # Only show legend for first question
                            marker=dict(
                                size=8,
                                color=self.colors.get(model.lower(), '#888888'),
                                line=dict(width=1, color='DarkSlateGrey'),
                                opacity=0.8
                            ),
                            text=[qid] * len(model_question_data),
                            customdata=np.column_stack((
                                model_question_data['question_id'],
                                model_question_data['category'],
                                model_question_data['question_text']
                            )),
                            hovertemplate=(
                                f"<b>{model}</b><br>" +
                                "Question ID: %{customdata[0]}<br>" +
                                "Category: %{customdata[1]}<br>" +
                                "Question: %{customdata[2]}<br>" +
                                f"{method} 1: %{{x:.3f}}<br>" +
                                f"{method} 2: %{{y:.3f}}<br>" +
                                f"{method} 3: %{{z:.3f}}<br>" +
                                "<extra></extra>"
                            ),
                            visible=True,
                            meta=qid  # Store question ID for filtering
                        )
                    )
        
        # Get axis ranges to keep consistent scaling
        all_x = df['x'].values
        all_y = df['y'].values  
        all_z = df['z'].values
        
        x_range = [all_x.min() * 1.1, all_x.max() * 1.1]
        y_range = [all_y.min() * 1.1, all_y.max() * 1.1]
        z_range = [all_z.min() * 1.1, all_z.max() * 1.1]
        
        # Create dropdown for Question ID filtering
        dropdown_buttons = [
            dict(
                label="Show All Questions",
                method="update",
                args=[
                    {"visible": [True] * len(fig.data)},
                    {
                        "title": f"Interactive 3D Embedding Visualization ({method})<br>All Questions vs Model Answers",
                        "scene.xaxis.range": x_range,
                        "scene.yaxis.range": y_range,
                        "scene.zaxis.range": z_range
                    }
                ]
            )
        ]
        
        # Add individual question filters
        for qid in unique_questions:
            visible_traces = []
            # Check each trace to see if it matches the selected question
            for trace in fig.data:
                if hasattr(trace, 'meta') and trace.meta == qid:
                    visible_traces.append(True)
                else:
                    visible_traces.append(False)
            
            dropdown_buttons.append(
                dict(
                    label=f"Show Only {qid}",
                    method="update", 
                    args=[
                        {"visible": visible_traces},
                        {
                            "title": f"Interactive 3D Embedding Visualization ({method})<br>{qid} vs Model Answers",
                            "scene.xaxis.range": x_range,
                            "scene.yaxis.range": y_range,
                            "scene.zaxis.range": z_range
                        }
                    ]
                )
            )
        
        # Update layout with enhanced controls
        fig.update_layout(
            title=dict(
                text=f'Interactive 3D Embedding Visualization ({method})<br>Questions vs Model Answers<br><sub>Use dropdown to filter by Question ID</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title=f'{method} Component 1',
                yaxis_title=f'{method} Component 2',
                zaxis_title=f'{method} Component 3',
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                zaxis=dict(range=z_range),
                bgcolor='rgba(0,0,0,0)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=0.80,
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=1,
                    font=dict(size=12)
                )
            ],
            font=dict(size=12),
            width=1200,
            height=900,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Save interactive plot
        fig.write_html(output_file)
        print(f"   ✅ Saved enhanced interactive plot to {output_file}")
        print(f"   🎯 Features: Question ID filtering, enhanced hover info")
        
        return fig
    
    def create_similarity_heatmap(self, similarities_file: str, output_file: str):
        """Create heatmap of cosine similarities"""
        
        print("📊 Creating similarity heatmap...")
        
        with open(similarities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Prepare similarity matrix
        models = ['Question', 'RAG', 'Gemini', 'DeepSeek']
        n_questions = len(data)
        
        # Create aggregate similarity matrix
        similarity_matrix = np.zeros((4, 4))
        
        # Fill the matrix with average similarities
        for item in data:
            # Question similarities (question vs answers)
            similarity_matrix[0, 1] += item['question_vs_rag']
            similarity_matrix[0, 2] += item['question_vs_gemini'] 
            similarity_matrix[0, 3] += item['question_vs_deepseek']
            
            # Answer similarities
            similarity_matrix[1, 2] += item['rag_vs_gemini']
            similarity_matrix[1, 3] += item['rag_vs_deepseek']
            similarity_matrix[2, 3] += item['gemini_vs_deepseek']
        
        # Average and make symmetric
        similarity_matrix /= n_questions
        similarity_matrix = similarity_matrix + similarity_matrix.T
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(similarity_matrix), k=1)
        
        sns.heatmap(
            similarity_matrix,
            mask=mask,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0.5,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .8},
            xticklabels=models,
            yticklabels=models
        )
        
        plt.title('Average Cosine Similarity Matrix\nBetween Questions and Model Answers', 
                 fontsize=16, pad=20)
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved similarity heatmap to {output_file}")
    
    def generate_all_visualizations(self, embeddings_file: str, similarities_file: str, 
                                  output_dir: str = "visualizations"):
        """Generate all visualizations for the project report"""
        
        print("🎨 Generating all visualizations for project report...")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        data = self.load_embeddings(embeddings_file)
        
        # PCA visualization
        print("\n🔄 Creating PCA visualizations...")
        pca_embeddings = self.reduce_dimensions_pca(data['embeddings'])
        
        self.create_matplotlib_3d_plot(
            data, pca_embeddings, 'PCA', 
            os.path.join(output_dir, 'embedding_visualization_pca.png')
        )
        
        self.create_interactive_plotly_3d(
            data, pca_embeddings, 'PCA',
            os.path.join(output_dir, 'embedding_visualization_pca_interactive.html')
        )
        
        # t-SNE visualization
        print("\n🔄 Creating t-SNE visualizations...")
        tsne_embeddings = self.reduce_dimensions_tsne(data['embeddings'])
        
        self.create_matplotlib_3d_plot(
            data, tsne_embeddings, 't-SNE',
            os.path.join(output_dir, 'embedding_visualization_tsne.png')
        )
        
        self.create_interactive_plotly_3d(
            data, tsne_embeddings, 't-SNE',
            os.path.join(output_dir, 'embedding_visualization_tsne_interactive.html')
        )
        
        # Similarity heatmap
        print("\n🔄 Creating similarity heatmap...")
        self.create_similarity_heatmap(
            similarities_file,
            os.path.join(output_dir, 'similarity_heatmap.png')
        )
        
        # Generate summary report
        self.generate_visualization_summary(output_dir)
        
        print("\n" + "=" * 60)
        print("🎉 All visualizations generated!")
        print(f"📁 Check the '{output_dir}' folder for:")
        print("   📊 PCA 3D plots (static & interactive)")
        print("   📊 t-SNE 3D plots (static & interactive)")  
        print("   🔥 Similarity heatmap")
        print("   📋 Visualization summary")
    
    def generate_visualization_summary(self, output_dir: str):
        """Generate a summary document for the visualizations"""
        
        summary = """# Embedding Visualization Summary

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
"""
        
        with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"   ✅ Saved visualization summary to {os.path.join(output_dir, 'README.md')}")

def main():
    """Main visualization runner"""
    
    print("🎨 Library Chatbot Embedding Visualization")
    print("=" * 50)
    
    # File paths
    embeddings_file = "model_embeddings.json"
    similarities_file = "cosine_similarities.json"
    
    # Check if files exist
    if not os.path.exists(embeddings_file):
        print(f"❌ Error: {embeddings_file} not found. Please run generate_embeddings.py first.")
        return
    
    if not os.path.exists(similarities_file):
        print(f"❌ Error: {similarities_file} not found. Please run generate_embeddings.py first.")
        return
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer()
    
    # Generate all visualizations
    visualizer.generate_all_visualizations(embeddings_file, similarities_file)

if __name__ == "__main__":
    main() 