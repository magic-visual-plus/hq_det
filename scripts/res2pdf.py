import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec


class TrainingVisualizer:
    def __init__(self, input_file='output/results.csv', output_file='output/training_report.pdf'):
        """Initialize the training visualizer with input and output file paths."""
        self.input_file = input_file
        self.output_file = output_file
        self.df = None
        
        # Ensure output directory exists
        if self.output_file and os.path.dirname(self.output_file):
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        # Set style for better visualization
        sns.set_style("whitegrid")
    
    def load_data(self):
        """Load data from CSV file."""
        try:
            self.df = pd.read_csv(self.input_file)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_performance_metrics_plot(self, fig, position):
        """Create plot for mAP, precision, recall, f1_score trend"""
        ax = fig.add_subplot(position)
        ax.plot(self.df.index, self.df['mAP'], marker='o', linestyle='-', label='mAP', markersize=4)
        ax.plot(self.df.index, self.df['precision'], marker='s', linestyle='-', label='Precision', markersize=4)
        ax.plot(self.df.index, self.df['recall'], marker='^', linestyle='-', label='Recall', markersize=4)
        ax.plot(self.df.index, self.df['f1_score'], marker='d', linestyle='-', label='F1 Score', markersize=4)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Metrics')
        ax.set_title('Performance Metrics Over Training')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        return ax
    
    def create_loss_plot(self, fig, position, loss_type, title):
        """Create plot for training or validation losses"""
        ax = fig.add_subplot(position)
        prefix = 'train/' if loss_type == 'train' else 'val/'
        ax.plot(self.df.index, self.df[f'{prefix}box_loss'], marker='o', linestyle='-', label='Box Loss', markersize=4)
        ax.plot(self.df.index, self.df[f'{prefix}cls_loss'], marker='s', linestyle='-', label='Class Loss', markersize=4)
        ax.plot(self.df.index, self.df[f'{prefix}giou_loss'], marker='^', linestyle='-', label='GIoU Loss', markersize=4)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        return ax
    
    def create_all_losses_plot(self, fig, position):
        """Create plot for all losses in one figure"""
        ax = fig.add_subplot(position)
        # Training losses
        ax.plot(self.df.index, self.df['train/box_loss'], marker='o', linestyle='-', label='Train Box Loss', markersize=3, alpha=0.7)
        ax.plot(self.df.index, self.df['train/cls_loss'], marker='s', linestyle='-', label='Train Class Loss', markersize=3, alpha=0.7)
        ax.plot(self.df.index, self.df['train/giou_loss'], marker='^', linestyle='-', label='Train GIoU Loss', markersize=3, alpha=0.7)
        # Validation losses
        ax.plot(self.df.index, self.df['val/box_loss'], marker='o', linestyle='--', label='Val Box Loss', markersize=3, alpha=0.7)
        ax.plot(self.df.index, self.df['val/cls_loss'], marker='s', linestyle='--', label='Val Class Loss', markersize=3, alpha=0.7)
        ax.plot(self.df.index, self.df['val/giou_loss'], marker='^', linestyle='--', label='Val GIoU Loss', markersize=3, alpha=0.7)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('All Losses Over Training')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        return ax
    
    def create_confidence_fnr_plot(self, fig, position):
        """Create plot for confidence and FNR"""
        ax = fig.add_subplot(position)
        ax.plot(self.df.index, self.df['confidence'], marker='o', linestyle='-', label='Confidence', markersize=4)
        ax.plot(self.df.index, self.df['fnr'], marker='s', linestyle='-', label='False Negative Rate', markersize=4)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Value')
        ax.set_title('Confidence and FNR Over Training')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        return ax
    
    def create_summary_page(self, pdf):
        """Create a summary page with key metrics"""
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.95, 'Training Results Summary', fontsize=16, ha='center')
        plt.text(0.5, 0.9, f'Data source: {self.input_file}', fontsize=12, ha='center')
        plt.text(0.1, 0.85, f'Final mAP: {self.df["mAP"].iloc[-1]:.4f}', fontsize=12)
        plt.text(0.1, 0.82, f'Final F1 Score: {self.df["f1_score"].iloc[-1]:.4f}', fontsize=12)
        plt.text(0.1, 0.79, f'Final Precision: {self.df["precision"].iloc[-1]:.4f}', fontsize=12)
        plt.text(0.1, 0.76, f'Final Recall: {self.df["recall"].iloc[-1]:.4f}', fontsize=12)
        plt.text(0.1, 0.73, f'Total Epochs: {len(self.df)}', fontsize=12)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    def create_metrics_and_losses_page(self, pdf):
        """Create a page with performance metrics and losses"""
        fig = plt.figure(figsize=(11, 8.5))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        self.create_performance_metrics_plot(fig, gs[0, 0])
        self.create_loss_plot(fig, gs[0, 1], 'train', 'Training Losses')
        self.create_loss_plot(fig, gs[1, 0], 'val', 'Validation Losses')
        self.create_confidence_fnr_plot(fig, gs[1, 1])
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    def create_all_losses_page(self, pdf):
        """Create a page with all losses combined"""
        fig = plt.figure(figsize=(11, 8.5))
        self.create_all_losses_plot(fig, 111)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    def generate_report(self):
        """Generate the complete PDF report"""
        if not self.load_data():
            return False
        
        try:
            with PdfPages(self.output_file) as pdf:
                self.create_summary_page(pdf)
                self.create_metrics_and_losses_page(pdf)
                self.create_all_losses_page(pdf)
            
            print(f"Successfully generated PDF report: {self.output_file}")
            print(f"Final mAP: {self.df['mAP'].iloc[-1]:.4f}")
            print(f"Final F1 Score: {self.df['f1_score'].iloc[-1]:.4f}")
            return True
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize training results from CSV file')
    parser.add_argument('-i', '--input', type=str, default='output/results.csv', 
                        help='Path to the results CSV file')
    parser.add_argument('-o', '--output', type=str, default='output/training_report.pdf', 
                        help='Path to save output PDF report')
    args = parser.parse_args()
    
    # Create visualizer and generate report
    visualizer = TrainingVisualizer(args.input, args.output)
    visualizer.generate_report()


if __name__ == "__main__":
    main()
