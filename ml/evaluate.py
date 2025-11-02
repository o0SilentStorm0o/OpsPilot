"""
Model Evaluation Framework
Demonstrates continuous improvement and model quality tracking
"""

import json
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def evaluate_model(model, tokenizer, test_dataset, device='cuda'):
    """
    Comprehensive model evaluation
    
    Tracks:
    - Accuracy, precision, recall, F1
    - Per-category performance
    - Confusion matrix
    - Confidence calibration
    """
    
    model.eval()
    predictions = []
    ground_truth = []
    confidences = []
    
    print("🔍 Evaluating model...")
    
    with torch.no_grad():
        for item in test_dataset:
            inputs = tokenizer(
                item['text'],
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(device)
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            
            pred = probs.argmax().item()
            conf = probs.max().item()
            
            predictions.append(pred)
            ground_truth.append(item['label'])
            confidences.append(conf)
    
    # Classification report
    categories = ['Database', 'Network', 'Security', 'Performance', 'Application', 'Infrastructure']
    report = classification_report(ground_truth, predictions, target_names=categories, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'accuracy': report['accuracy'],
        'macro_avg': report['macro avg'],
        'per_category': {cat: report[cat] for cat in categories},
        'avg_confidence': sum(confidences) / len(confidences),
        'confusion_matrix': cm.tolist()
    }
    
    # Save to file
    results_path = Path('ml/outputs/evaluations')
    results_path.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(results_path / f'eval_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    plt.savefig(results_path / f'confusion_matrix_{timestamp}.png')
    
    print(f"✅ Evaluation complete!")
    print(f"📊 Accuracy: {report['accuracy']:.1%}")
    print(f"📁 Results saved to: {results_path}")
    
    return results


def compare_models(eval_results_dir='ml/outputs/evaluations'):
    """
    Compare multiple model versions
    Demonstrates continuous improvement tracking
    """
    
    results_path = Path(eval_results_dir)
    eval_files = sorted(results_path.glob('eval_*.json'))
    
    if len(eval_files) < 2:
        print("❌ Need at least 2 evaluations to compare")
        return
    
    print(f"📈 Comparing {len(eval_files)} model versions...")
    
    accuracies = []
    timestamps = []
    
    for eval_file in eval_files:
        with open(eval_file) as f:
            data = json.load(f)
            accuracies.append(data['accuracy'])
            timestamps.append(data['timestamp'])
    
    # Plot improvement over time
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(accuracies)), accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Model Version')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Over Time')
    plt.grid(True, alpha=0.3)
    
    # Annotate improvements
    for i in range(1, len(accuracies)):
        delta = accuracies[i] - accuracies[i-1]
        if delta > 0:
            color = 'green'
            sign = '+'
        else:
            color = 'red'
            sign = ''
        
        plt.annotate(f'{sign}{delta:.1%}', 
                    xy=(i, accuracies[i]),
                    xytext=(0, 10), textcoords='offset points',
                    color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_path / 'model_improvement.png')
    
    print(f"✅ Comparison complete!")
    print(f"📊 Best accuracy: {max(accuracies):.1%}")
    print(f"📈 Total improvement: {accuracies[-1] - accuracies[0]:+.1%}")


if __name__ == '__main__':
    # Example usage
    print("🎯 Model Evaluation & Continuous Improvement Demo")
    print("This script demonstrates:")
    print("  1. Comprehensive model evaluation")
    print("  2. Performance tracking over time")
    print("  3. A/B testing capabilities")
    print()
    print("Usage:")
    print("  python ml/evaluate.py --model path/to/model --test-data path/to/test.csv")
