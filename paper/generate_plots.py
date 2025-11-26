import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def parse_training_log(log_file):
    metrics = {
        'loss': [],
        'learning_rate': [],
        'epoch': [],
        'mean_token_accuracy': [],
        'entropy': []
    }

    with open(log_file, 'r') as f:
        for line in f:
            if "'loss':" in line:
                try:
                    data = eval(line.strip())
                    metrics['loss'].append(data.get('loss', None))
                    metrics['learning_rate'].append(data.get('learning_rate', None))
                    metrics['epoch'].append(data.get('epoch', None))
                    metrics['mean_token_accuracy'].append(data.get('mean_token_accuracy', None))
                    metrics['entropy'].append(data.get('entropy', None))
                except:
                    continue

    return {k: [x for x in v if x is not None] for k, v in metrics.items()}

def plot_training_curves():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    models = {
        'Llama-3.2-1B (SFT)': 'training_llama_1b.log',
        'SmolLM-360M (SFT)': 'training_smollm.log',
        'SmolLM-360M (DPO)': 'training_dpo_smollm.log',
        'Qwen2.5-0.5B (SFT)': 'training_qwen.log',
        'Qwen2.5-1.5B (SFT)': 'training_qwen_1.5b.log',
        'Phi-3-Medium (SFT)': 'training.log',
    }

    colors = {
        'Llama-3.2-1B (SFT)': '#1f77b4',
        'SmolLM-360M (SFT)': '#ff7f0e',
        'SmolLM-360M (DPO)': '#d62728',
        'Qwen2.5-0.5B (SFT)': '#2ca02c',
        'Qwen2.5-1.5B (SFT)': '#9467bd',
        'Phi-3-Medium (SFT)': '#8c564b'
    }

    for model_name, log_file in models.items():
        if Path(log_file).exists():
            metrics = parse_training_log(log_file)

            if metrics['epoch'] and metrics['loss']:
                axes[0, 0].plot(metrics['epoch'], metrics['loss'],
                               label=model_name, linewidth=1.5, color=colors[model_name])

            if metrics['epoch'] and metrics['mean_token_accuracy']:
                axes[0, 1].plot(metrics['epoch'], metrics['mean_token_accuracy'],
                               label=model_name, linewidth=1.5, color=colors[model_name])

            if metrics['epoch'] and metrics['learning_rate']:
                axes[1, 0].plot(metrics['epoch'], metrics['learning_rate'],
                               label=model_name, linewidth=1.5, color=colors[model_name])

            if metrics['epoch'] and metrics['entropy']:
                axes[1, 1].plot(metrics['epoch'], metrics['entropy'],
                               label=model_name, linewidth=1.5, color=colors[model_name])

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss vs Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Token Accuracy')
    axes[0, 1].set_title('Token Accuracy vs Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Entropy vs Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.pdf', bbox_inches='tight')
    plt.savefig('training_curves.png', bbox_inches='tight', dpi=300)
    print("Saved training_curves.pdf and training_curves.png")
    plt.close()

def plot_evaluation_results():
    results = {
        'Llama-3.2-1B (SFT)': {'EM': 0.82, 'F1': 0.87, 'Tool Select F1': 1.0, 'Hallucination': 0.99},
        'SmolLM-360M (SFT)': {'EM': 0.30, 'F1': 0.3747, 'Tool Select F1': 0.9367, 'Hallucination': 0.97},
        'SmolLM-360M (DPO)': {'EM': 0.08, 'F1': 0.1233, 'Tool Select F1': 0.7527, 'Hallucination': 0.86},
        'Qwen2.5-0.5B (SFT)': {'EM': 0.66, 'F1': 0.7367, 'Tool Select F1': 1.0, 'Hallucination': 0.99},
        'Phi-3-Medium (SFT)': {'EM': 0.24, 'F1': 0.328, 'Tool Select F1': 0.8933, 'Hallucination': 0.95},
        'Gemma-3-270M (SFT)': {'EM': 0.235, 'F1': 0.2863, 'Tool Select F1': 0.8691, 'Hallucination': 0.9695},
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    models = list(results.keys())
    x_pos = np.arange(len(models))

    em_scores = [results[m]['EM'] for m in models]
    f1_scores = [results[m]['F1'] for m in models]
    tool_f1_scores = [results[m]['Tool Select F1'] for m in models]
    halluc_scores = [results[m]['Hallucination'] for m in models]

    colors_list = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b']

    axes[0, 0].bar(x_pos, em_scores, color=colors_list)
    axes[0, 0].set_ylabel('Exact Match Rate')
    axes[0, 0].set_title('Exact Match Rate by Model')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.0])

    axes[0, 1].bar(x_pos, f1_scores, color=colors_list)
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score by Model')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1.0])

    axes[1, 0].bar(x_pos, tool_f1_scores, color=colors_list)
    axes[1, 0].set_ylabel('Tool Select F1')
    axes[1, 0].set_title('Tool Selection F1 Score by Model')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1.0])

    axes[1, 1].bar(x_pos, halluc_scores, color=colors_list)
    axes[1, 1].set_ylabel('Hallucination Rate')
    axes[1, 1].set_title('Hallucination Rate by Model')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig('evaluation_results.pdf', bbox_inches='tight')
    plt.savefig('evaluation_results.png', bbox_inches='tight', dpi=300)
    print("Saved evaluation_results.pdf and evaluation_results.png")
    plt.close()

def plot_model_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Llama-3.2-1B\n(SFT)', 'SmolLM-360M\n(SFT)', 'SmolLM-360M\n(DPO)',
              'Qwen2.5-0.5B\n(SFT)', 'Phi-3-Medium\n(SFT)', 'Gemma-3-270M\n(SFT)']

    metrics = {
        'Exact Match': [0.82, 0.30, 0.08, 0.66, 0.24, 0.235],
        'F1 Score': [0.87, 0.3747, 0.1233, 0.7367, 0.328, 0.2863],
        'Tool Select F1': [1.0, 0.9367, 0.7527, 1.0, 0.8933, 0.8691]
    }

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x - width, metrics['Exact Match'], width, label='Exact Match', color='#1f77b4')
    bars2 = ax.bar(x, metrics['F1 Score'], width, label='F1 Score', color='#ff7f0e')
    bars3 = ax.bar(x + width, metrics['Tool Select F1'], width, label='Tool Select F1', color='#2ca02c')

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison Across Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig('model_comparison.pdf', bbox_inches='tight')
    plt.savefig('model_comparison.png', bbox_inches='tight', dpi=300)
    print("Saved model_comparison.pdf and model_comparison.png")
    plt.close()

def plot_model_size_performance():
    fig, ax = plt.subplots(figsize=(8, 6))

    models_data = [
        ('Gemma-3-270M', 270, 0.235, 0.2863),
        ('SmolLM-360M\n(SFT)', 360, 0.30, 0.3747),
        ('SmolLM-360M\n(DPO)', 360, 0.08, 0.1233),
        ('Qwen2.5-0.5B', 500, 0.66, 0.7367),
        ('Llama-3.2-1B', 1000, 0.82, 0.87),
        ('Phi-3-Medium', 7000, 0.24, 0.328),
    ]

    sizes = [m[1] for m in models_data]
    f1_scores = [m[3] for m in models_data]
    labels = [m[0] for m in models_data]

    colors = ['#8c564b', '#ff7f0e', '#d62728', '#2ca02c', '#1f77b4', '#9467bd']

    scatter = ax.scatter(sizes, f1_scores, s=200, alpha=0.6, c=colors, edgecolors='black', linewidth=1.5)

    for i, label in enumerate(labels):
        ax.annotate(label, (sizes[i], f1_scores[i]),
                   textcoords="offset points", xytext=(0,10),
                   ha='center', fontsize=8)

    ax.set_xlabel('Model Size (Million Parameters)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Model Size vs Performance')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig('size_vs_performance.pdf', bbox_inches='tight')
    plt.savefig('size_vs_performance.png', bbox_inches='tight', dpi=300)
    print("Saved size_vs_performance.pdf and size_vs_performance.png")
    plt.close()

def parse_dpo_log(log_file):
    metrics = {
        'loss': [],
        'learning_rate': [],
        'epoch': [],
        'rewards_chosen': [],
        'rewards_rejected': [],
        'rewards_margins': [],
        'rewards_accuracies': []
    }

    with open(log_file, 'r') as f:
        for line in f:
            if "'loss':" in line and "rewards/chosen" in line:
                try:
                    data = eval(line.strip())
                    metrics['loss'].append(data.get('loss', None))
                    metrics['learning_rate'].append(data.get('learning_rate', None))
                    metrics['epoch'].append(data.get('epoch', None))
                    metrics['rewards_chosen'].append(data.get('rewards/chosen', None))
                    metrics['rewards_rejected'].append(data.get('rewards/rejected', None))
                    metrics['rewards_margins'].append(data.get('rewards/margins', None))
                    metrics['rewards_accuracies'].append(data.get('rewards/accuracies', None))
                except:
                    continue

    return {k: [x for x in v if x is not None] for k, v in metrics.items()}

def plot_dpo_training():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    dpo_logs = {
        'SmolLM-360M (DPO)': 'training_dpo_smollm.log',
        'DPO v1': 'training_dpo.log',
        'DPO v2': 'training_dpo_v2.log',
    }

    colors = {'SmolLM-360M (DPO)': '#d62728', 'DPO v1': '#9467bd', 'DPO v2': '#8c564b'}

    for model_name, log_file in dpo_logs.items():
        if Path(log_file).exists():
            metrics = parse_dpo_log(log_file)

            if metrics['epoch'] and metrics['loss']:
                axes[0, 0].plot(metrics['epoch'], metrics['loss'],
                               label=model_name, linewidth=1.5, color=colors[model_name])

            if metrics['epoch'] and metrics['rewards_margins']:
                axes[0, 1].plot(metrics['epoch'], metrics['rewards_margins'],
                               label=model_name, linewidth=1.5, color=colors[model_name])

            if metrics['epoch'] and metrics['rewards_accuracies']:
                axes[1, 0].plot(metrics['epoch'], metrics['rewards_accuracies'],
                               label=model_name, linewidth=1.5, color=colors[model_name])

            if metrics['epoch'] and metrics['rewards_chosen'] and metrics['rewards_rejected']:
                axes[1, 1].plot(metrics['epoch'], metrics['rewards_chosen'],
                               label=f'{model_name} (Chosen)', linewidth=1.5,
                               color=colors[model_name], linestyle='-')
                axes[1, 1].plot(metrics['epoch'], metrics['rewards_rejected'],
                               label=f'{model_name} (Rejected)', linewidth=1.5,
                               color=colors[model_name], linestyle='--', alpha=0.7)

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('DPO Loss')
    axes[0, 0].set_title('DPO Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reward Margin')
    axes[0, 1].set_title('Reward Margins (Chosen - Rejected)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Preference Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].set_title('Chosen vs Rejected Rewards')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dpo_training.pdf', bbox_inches='tight')
    plt.savefig('dpo_training.png', bbox_inches='tight', dpi=300)
    print("Saved dpo_training.pdf and dpo_training.png")
    plt.close()

if __name__ == '__main__':
    print("Generating plots...")
    plot_training_curves()
    plot_dpo_training()
    plot_evaluation_results()
    plot_model_comparison()
    plot_model_size_performance()
    print("All plots generated successfully!")
