#!/usr/bin/env python3
"""
Additional Visualization Generator for RAG System Evaluation Report

This script creates comprehensive visualizations from evaluation_report.json
to enhance the final project report with detailed performance analysis.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import os

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_evaluation_data(report_file: str = "evaluation_report.json") -> Dict[str, Any]:
    """Load evaluation report data"""
    with open(report_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_model_comparison_chart(data: Dict[str, Any], output_dir: str):
    """Create comprehensive model comparison charts"""
    
    # 1. Overall Performance Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Question Relevance Scores
    relevance_data = data['evaluation_summary']['average_relevance_scores']
    models = ['RAG System', 'Gemini 2.0', 'DeepSeek R1']
    scores = [
        relevance_data['RAG_to_question'],
        relevance_data['Gemini_to_question'], 
        relevance_data['DeepSeek_to_question']
    ]
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars1 = ax1.bar(models, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Average Question Relevance Scores', fontweight='bold', pad=20)
    ax1.set_ylabel('Cosine Similarity Score')
    ax1.set_ylim(0.75, 0.82)
    
    # Add value labels on bars
    for bar, score in zip(bars1, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Performance Range (Best vs Worst)
    best_scores = data['detailed_analysis']['best_question_relevance']
    worst_scores = data['detailed_analysis']['worst_question_relevance']
    
    x = np.arange(len(models))
    best_vals = [best_scores['RAG'], best_scores['Gemini'], best_scores['DeepSeek']]
    worst_vals = [worst_scores['RAG'], worst_scores['Gemini'], worst_scores['DeepSeek']]
    
    ax2.bar(x - 0.2, best_vals, 0.4, label='Best Performance', color=colors, alpha=0.8)
    ax2.bar(x + 0.2, worst_vals, 0.4, label='Worst Performance', color=colors, alpha=0.5)
    ax2.set_title('Performance Range (Best vs Worst)', fontweight='bold', pad=20)
    ax2.set_ylabel('Cosine Similarity Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    
    # Consistency Analysis (Lower std = more consistent)
    consistency_data = data['detailed_analysis']['consistency_analysis']
    consistency_scores = [
        consistency_data['RAG_consistency'],
        consistency_data['Gemini_consistency'],
        consistency_data['DeepSeek_consistency']
    ]
    
    bars3 = ax3.bar(models, consistency_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_title('Model Consistency (Lower = More Consistent)', fontweight='bold', pad=20)
    ax3.set_ylabel('Standard Deviation')
    
    for bar, score in zip(bars3, consistency_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Inter-Model Similarity
    inter_model_data = data['evaluation_summary']['average_inter_model_similarities']
    comparisons = ['RAG vs\nGemini', 'RAG vs\nDeepSeek', 'Gemini vs\nDeepSeek']
    similarities = [
        inter_model_data['RAG_vs_Gemini'],
        inter_model_data['RAG_vs_DeepSeek'],
        inter_model_data['Gemini_vs_DeepSeek']
    ]
    
    bars4 = ax4.bar(comparisons, similarities, color=['#FF6B6B', '#FFE66D', '#A8E6CF'], 
                   alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_title('Inter-Model Answer Similarity', fontweight='bold', pad=20)
    ax4.set_ylabel('Cosine Similarity Score')
    
    for bar, score in zip(bars4, similarities):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created model comparison overview chart")

def create_category_performance_chart(data: Dict[str, Any], output_dir: str):
    """Create category-wise performance analysis"""
    
    category_data = data['category_performance']
    categories = list(category_data.keys())
    
    # Prepare data for plotting
    rag_scores = [category_data[cat]['RAG_avg'] for cat in categories]
    gemini_scores = [category_data[cat]['Gemini_avg'] for cat in categories]
    deepseek_scores = [category_data[cat]['DeepSeek_avg'] for cat in categories]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Category': categories * 3,
        'Model': ['RAG System'] * len(categories) + ['Gemini 2.0'] * len(categories) + ['DeepSeek R1'] * len(categories),
        'Score': rag_scores + gemini_scores + deepseek_scores
    })
    
    # Create grouped bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Grouped bar chart
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax1.bar(x - width, rag_scores, width, label='RAG System', color='#4ECDC4', alpha=0.8)
    bars2 = ax1.bar(x, gemini_scores, width, label='Gemini 2.0', color='#45B7D1', alpha=0.8)
    bars3 = ax1.bar(x + width, deepseek_scores, width, label='DeepSeek R1', color='#96CEB4', alpha=0.8)
    
    ax1.set_title('Performance by Question Category', fontweight='bold', pad=20)
    ax1.set_ylabel('Average Cosine Similarity Score')
    ax1.set_xlabel('Question Categories')
    ax1.set_xticks(x)
    ax1.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Heatmap of category performance
    category_matrix = np.array([rag_scores, gemini_scores, deepseek_scores])
    
    im = ax2.imshow(category_matrix, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=0.92)
    ax2.set_title('Performance Heatmap by Category', fontweight='bold', pad=20)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['RAG System', 'Gemini 2.0', 'DeepSeek R1'])
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45, ha='right')
    
    # Add text annotations to heatmap
    for i in range(len(['RAG System', 'Gemini 2.0', 'DeepSeek R1'])):
        for j in range(len(categories)):
            ax2.text(j, i, f'{category_matrix[i, j]:.3f}', 
                    ha='center', va='center', fontweight='bold', color='white')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.1)
    cbar.set_label('Cosine Similarity Score')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created category performance analysis chart")

def create_difficulty_analysis_chart(data: Dict[str, Any], output_dir: str):
    """Create difficulty-based performance analysis"""
    
    individual_results = data['individual_results']
    
    # Group by difficulty
    difficulty_groups = {'easy': [], 'medium': [], 'hard': []}
    
    for result in individual_results:
        difficulty = result['difficulty']
        difficulty_groups[difficulty].append({
            'RAG': result['question_vs_rag'],
            'Gemini': result['question_vs_gemini'], 
            'DeepSeek': result['question_vs_deepseek']
        })
    
    # Calculate averages by difficulty
    difficulty_averages = {}
    for difficulty, results in difficulty_groups.items():
        if results:  # Only if we have results for this difficulty
            difficulty_averages[difficulty] = {
                'RAG': np.mean([r['RAG'] for r in results]),
                'Gemini': np.mean([r['Gemini'] for r in results]),
                'DeepSeek': np.mean([r['DeepSeek'] for r in results])
            }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart by difficulty
    difficulties = list(difficulty_averages.keys())
    rag_by_difficulty = [difficulty_averages[d]['RAG'] for d in difficulties]
    gemini_by_difficulty = [difficulty_averages[d]['Gemini'] for d in difficulties]
    deepseek_by_difficulty = [difficulty_averages[d]['DeepSeek'] for d in difficulties]
    
    x = np.arange(len(difficulties))
    width = 0.25
    
    ax1.bar(x - width, rag_by_difficulty, width, label='RAG System', color='#4ECDC4', alpha=0.8)
    ax1.bar(x, gemini_by_difficulty, width, label='Gemini 2.0', color='#45B7D1', alpha=0.8)
    ax1.bar(x + width, deepseek_by_difficulty, width, label='DeepSeek R1', color='#96CEB4', alpha=0.8)
    
    ax1.set_title('Performance by Question Difficulty', fontweight='bold', pad=20)
    ax1.set_ylabel('Average Cosine Similarity Score')
    ax1.set_xlabel('Question Difficulty')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.title() for d in difficulties])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Individual question performance scatter plot
    questions = [f"Q{r['id']}" for r in individual_results]
    rag_scores = [r['question_vs_rag'] for r in individual_results]
    gemini_scores = [r['question_vs_gemini'] for r in individual_results]
    deepseek_scores = [r['question_vs_deepseek'] for r in individual_results]
    
    x_pos = np.arange(len(questions))
    
    ax2.scatter(x_pos, rag_scores, color='#4ECDC4', label='RAG System', s=100, alpha=0.8, edgecolors='black')
    ax2.scatter(x_pos, gemini_scores, color='#45B7D1', label='Gemini 2.0', s=100, alpha=0.8, edgecolors='black')
    ax2.scatter(x_pos, deepseek_scores, color='#96CEB4', label='DeepSeek R1', s=100, alpha=0.8, edgecolors='black')
    
    # Connect points with lines for better visualization
    ax2.plot(x_pos, rag_scores, color='#4ECDC4', alpha=0.5, linewidth=2)
    ax2.plot(x_pos, gemini_scores, color='#45B7D1', alpha=0.5, linewidth=2)
    ax2.plot(x_pos, deepseek_scores, color='#96CEB4', alpha=0.5, linewidth=2)
    
    ax2.set_title('Individual Question Performance', fontweight='bold', pad=20)
    ax2.set_ylabel('Cosine Similarity Score')
    ax2.set_xlabel('Questions')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(questions, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/difficulty_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created difficulty analysis chart")

def create_rag_advantage_analysis(data: Dict[str, Any], output_dir: str):
    """Create analysis showing where RAG system provides advantage"""
    
    individual_results = data['individual_results']
    
    # Calculate RAG advantage over other models
    rag_vs_gemini_advantage = []
    rag_vs_deepseek_advantage = []
    categories = []
    question_ids = []
    
    for result in individual_results:
        rag_score = result['question_vs_rag']
        gemini_score = result['question_vs_gemini']
        deepseek_score = result['question_vs_deepseek']
        
        rag_vs_gemini_advantage.append(rag_score - gemini_score)
        rag_vs_deepseek_advantage.append(rag_score - deepseek_score)
        categories.append(result['category'])
        question_ids.append(f"Q{result['id']}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # RAG advantage bar chart
    x = np.arange(len(question_ids))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, rag_vs_gemini_advantage, width, 
                   label='RAG vs Gemini', alpha=0.8, 
                   color=['green' if x > 0 else 'red' for x in rag_vs_gemini_advantage])
    bars2 = ax1.bar(x + width/2, rag_vs_deepseek_advantage, width,
                   label='RAG vs DeepSeek', alpha=0.8,
                   color=['darkgreen' if x > 0 else 'darkred' for x in rag_vs_deepseek_advantage])
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_title('RAG System Performance Advantage/Disadvantage', fontweight='bold', pad=20)
    ax1.set_ylabel('Score Difference')
    ax1.set_xlabel('Questions')
    ax1.set_xticks(x)
    ax1.set_xticklabels(question_ids)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.005 if height > 0 else -0.015),
                    f'{height:.3f}', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Summary statistics
    avg_rag_vs_gemini = np.mean(rag_vs_gemini_advantage)
    avg_rag_vs_deepseek = np.mean(rag_vs_deepseek_advantage)
    
    wins_vs_gemini = sum(1 for x in rag_vs_gemini_advantage if x > 0)
    wins_vs_deepseek = sum(1 for x in rag_vs_deepseek_advantage if x > 0)
    
    # Create summary visualization
    summary_data = {
        'Metric': ['Avg Advantage\nvs Gemini', 'Avg Advantage\nvs DeepSeek', 
                  'Wins vs Gemini\n(out of 10)', 'Wins vs DeepSeek\n(out of 10)'],
        'Value': [avg_rag_vs_gemini, avg_rag_vs_deepseek, wins_vs_gemini, wins_vs_deepseek],
        'Colors': ['blue', 'green', 'orange', 'purple']
    }
    
    bars = ax2.bar(summary_data['Metric'], summary_data['Value'], 
                  color=summary_data['Colors'], alpha=0.7, edgecolor='black')
    ax2.set_title('RAG System Performance Summary', fontweight='bold', pad=20)
    ax2.set_ylabel('Score/Count')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, summary_data['Value']):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{value:.3f}' if abs(value) < 1 else f'{int(value)}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rag_advantage_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created RAG advantage analysis chart")

def create_correlation_analysis(data: Dict[str, Any], output_dir: str):
    """Create correlation analysis between different metrics"""
    
    individual_results = data['individual_results']
    
    # Prepare data for correlation analysis
    df = pd.DataFrame({
        'Question_ID': [r['id'] for r in individual_results],
        'RAG_Score': [r['question_vs_rag'] for r in individual_results],
        'Gemini_Score': [r['question_vs_gemini'] for r in individual_results],
        'DeepSeek_Score': [r['question_vs_deepseek'] for r in individual_results],
        'RAG_vs_Gemini_Similarity': [r['rag_vs_gemini'] for r in individual_results],
        'RAG_vs_DeepSeek_Similarity': [r['rag_vs_deepseek'] for r in individual_results],
        'Gemini_vs_DeepSeek_Similarity': [r['gemini_vs_deepseek'] for r in individual_results],
        'Category': [r['category'] for r in individual_results]
    })
    
    # Create correlation matrix
    corr_cols = ['RAG_Score', 'Gemini_Score', 'DeepSeek_Score', 
                'RAG_vs_Gemini_Similarity', 'RAG_vs_DeepSeek_Similarity', 
                'Gemini_vs_DeepSeek_Similarity']
    
    correlation_matrix = df[corr_cols].corr()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax1, cbar_kws={'label': 'Correlation Coefficient'})
    ax1.set_title('Correlation Matrix of Performance Metrics', fontweight='bold', pad=20)
    
    # Scatter plot: RAG vs baseline models
    ax2.scatter(df['Gemini_Score'], df['RAG_Score'], 
               color='#45B7D1', alpha=0.7, s=100, edgecolors='black', 
               label='RAG vs Gemini')
    ax2.scatter(df['DeepSeek_Score'], df['RAG_Score'], 
               color='#96CEB4', alpha=0.7, s=100, edgecolors='black',
               label='RAG vs DeepSeek')
    
    # Add diagonal line (y=x) for reference
    min_score = min(df['RAG_Score'].min(), df['Gemini_Score'].min(), df['DeepSeek_Score'].min())
    max_score = max(df['RAG_Score'].max(), df['Gemini_Score'].max(), df['DeepSeek_Score'].max())
    ax2.plot([min_score, max_score], [min_score, max_score], 
            'k--', alpha=0.5, label='Equal Performance Line')
    
    ax2.set_xlabel('Baseline Model Scores')
    ax2.set_ylabel('RAG System Scores')
    ax2.set_title('RAG vs Baseline Model Performance', fontweight='bold', pad=20)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add question labels to points
    for i, row in df.iterrows():
        ax2.annotate(f"Q{row['Question_ID']}", 
                    (row['Gemini_Score'], row['RAG_Score']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created correlation analysis chart")

def main():
    """Generate all additional visualizations"""
    
    print("🎨 Generating Additional Evaluation Visualizations")
    print("=" * 60)
    
    # Create output directory
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load evaluation data
    print("📖 Loading evaluation report data...")
    data = load_evaluation_data()
    
    # Generate all visualizations
    create_model_comparison_chart(data, output_dir)
    create_category_performance_chart(data, output_dir)
    create_difficulty_analysis_chart(data, output_dir)
    create_rag_advantage_analysis(data, output_dir)
    create_correlation_analysis(data, output_dir)
    
    print("\n🎉 All additional visualizations created successfully!")
    print(f"📁 Output directory: {output_dir}/")
    print("\nGenerated files:")
    print("  • model_comparison_overview.png")
    print("  • category_performance_analysis.png") 
    print("  • difficulty_analysis.png")
    print("  • rag_advantage_analysis.png")
    print("  • correlation_analysis.png")
    
    print("\n✨ These visualizations are ready for your final report!")

if __name__ == "__main__":
    main() 