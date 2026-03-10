from StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation
import pandas as pd
import matplotlib.pyplot as plt
from ProcessHeatMapResult import *
from additional_function import *
from torch import nn
import json
import os
if __name__ == '__main__':
    dataset_path = "/kaggle/input/datasets/nguyyentri/coad-aligned/COAD_Aligned/dataset"
    
    model = StratifiedKFoldCrossValidation(
        model_name="EfficientNetB1Classifier",
        dataset_path=dataset_path,
        k_folds=5,
        num_epochs=50,
        freeze_epochs=0,
        batch_size=32,
        lr=0.0002,
        weight_decay=1e-3,
        dropout_rate=0.6,
        focal_gamma=2.0,
        label_smoothing=0.1,
        use_class_aware_aug= False,
        use_weighted_sampling=False,
        use_temperature_scaling=False,
        calculate_cluster_metrics_flag=False,
        random_seed=42,
        lambda1=0.8
    )

    fold_results, fold_models, ensemble_metrics, class_names, calibrators, all_heatmaps, fold_test_results = model.run()

    print(f'/n{"="*60}')
    print('FINAL SUMMARY')
    print(f'{"="*60}')
    print(f'Classes: {class_names}')
    print(f'Number of folds: {len(fold_results)}')

    # Visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    for i, result in enumerate(fold_results):
        history = result['history']

        axes[0, 0].plot(history['train_loss'], label=f"Fold {i+1}", alpha=0.7)
        axes[0, 1].plot(history['val_loss'], label=f"Fold {i+1}", alpha=0.7)
        axes[0, 2].plot(history['train_f1'], label=f"Fold {i+1}", alpha=0.7)
        axes[1, 0].plot(history['train_acc'], label=f"Fold {i+1}", alpha=0.7)
        axes[1, 1].plot(history['val_acc'], label=f"Fold {i+1}", alpha=0.7)
        axes[1, 2].plot(history['val_f1'], label=f"Fold {i+1}", alpha=0.7)
        axes[2, 0].plot(history['train_auc'], label=f"Fold {i+1}", alpha=0.7)
        axes[2, 1].plot(history['val_auc'], label=f"Fold {i+1}", alpha=0.7)
        axes[2, 2].plot(history['val_sens'], label=f"Fold {i+1}", alpha=0.7)

    axes[0, 0].set_title('Training Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 2].set_title('Training F1-Score')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 2].set_title('Validation F1-Score')
    axes[2, 0].set_title('Training AUC')
    axes[2, 1].set_title('Validation AUC')
    axes[2, 2].set_title('Validation Sensitivity')
    
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('imbalanced_kfold_comprehensive_results.png', dpi=300, bbox_inches='tight')

    with open("/fold_test_results.txt", "w", encoding="utf-8") as f:
        fold_test_results = convert_numpy(fold_test_results)
        json.dump(fold_test_results, f, indent=4)

    print("/n✓ Visualization saved as 'imbalanced_kfold_comprehensive_results.png'")
    
    # Compute mean heatmap per class
    mean_heatmaps = calculate_mean(all_heatmaps)
    
    out = "Explanation/"
    os.mkdir(out)

    for cls, mean_heatmap in mean_heatmaps.items():

        # 1. Visualize
        visualize_mean_heatmap(mean_heatmap, f"{out}mean_heatmap_class_{cls}.png")

        # 2. Threshold (top 10%)
        threshold = np.percentile(mean_heatmap, 90)

        # 3. Critical pixels
        pixels = find_critical_pixel(mean_heatmap, threshold)
        save_to_csv(pixels, f"{out}critical_pixels_class_{cls}.csv")

        # 4. Map to genes
        genes_cnv = find_critical_gene(pixels,"/kaggle/input/datasets/nguyyentri/coad-aligned/COAD_Aligned/gene_coordinates_CNV.csv")
        save_to_csv(genes_cnv, f"{out}cnv_class_{cls}.csv")

        methyl = find_critical_gene(pixels,"/kaggle/input/datasets/nguyyentri/coad-aligned/COAD_Aligned/gene_coordinates_Methylation.csv")
        save_to_csv(methyl, f"{out}methyl_class_{cls}.csv")

        mrna = find_critical_gene(pixels,"/kaggle/input/datasets/nguyyentri/coad-aligned/COAD_Aligned/gene_coordinates_mRNA.csv")
        save_to_csv(mrna, f"{out}mRNA_class_{cls}.csv")
