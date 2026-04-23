from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def print_training_arguments(args):
    console = Console(force_terminal=True, width=None, soft_wrap=False)
    args_table = Table(title="Training Arguments", show_header=True, expand=True)
    args_table.add_column("Parameter", style="cyan", no_wrap=False, width=30)
    args_table.add_column("Value", style="green", no_wrap=False, width=70)
    
    def format_value(value, max_length=500):
        if isinstance(value, (dict, list, tuple, set)):
            container_str = str(value)
            if len(container_str) > max_length:
                truncated = container_str[:max_length] + "..."
                return truncated
            else:
                return container_str
        else:
            formatted_value = str(value)
            if len(formatted_value) > max_length:
                lines = []
                for i in range(0, len(formatted_value), max_length):
                    lines.append(formatted_value[i:i+max_length])
                return '\n'.join(lines)
            return formatted_value
    for key, value in vars(args).items():
        formatted_value = format_value(value)
        args_table.add_row(key, formatted_value)
    
    console.print(args_table)
    
    for key, value in vars(args).items():
        if isinstance(value, (dict, list, tuple, set)):
            console.print(f"\n[cyan]{key}:[/cyan]")
            if isinstance(value, dict):
                for k, v in value.items():
                    console.print(f"  {k}: {format_value(v)}")
            else:
                for i, item in enumerate(value):
                    console.print(f"  [{i}]: {format_value(item)}")

def print_model_summary(model):
    console = Console(force_terminal=True, width=None, soft_wrap=False)
    
    model_structure_panel = Panel(
        Text(str(model), style="green"),
        title="Model Architecture",
        border_style="blue",
        expand=True
    )
    console.print(model_structure_panel)
    
    # Calculate parameters
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_parameters = total_parameters - trainable_parameters
    
    frozen_ratio = frozen_parameters / total_parameters * 100
    model_params_table = Table(title="Model Parameters Summary", show_header=True, expand=True)
    model_params_table.add_column("Parameter Type", style="cyan", no_wrap=False, width=20)
    model_params_table.add_column("Count", style="green", justify="right", no_wrap=False, width=30)
    
    model_params_table.add_row("Total", f"{total_parameters:,}")
    model_params_table.add_row("Trainable", f"{trainable_parameters:,}")
    model_params_table.add_row("Frozen", f"{frozen_parameters:,}")
    
    console.print(model_params_table)
    console.print(f"Frozen ratio: {frozen_ratio:.2f}%")

def print_dataset_summary(dataset_train, dataset_val):
    console = Console(force_terminal=True, width=None, soft_wrap=False)
    
    dataset_table = Table(title="Dataset Information", show_header=True, expand=True)
    dataset_table.add_column("Dataset", style="cyan", no_wrap=False, width=30)
    dataset_table.add_column("Size", style="magenta", justify="right", no_wrap=False, width=20)
    dataset_table.add_row("Training", f"{len(dataset_train):,}")
    dataset_table.add_row("Validation", f"{len(dataset_val):,}")
    console.print(dataset_table)
    
    class_table = Table(title=f"Class Mapping (Total: {len(dataset_train.class_id2names)})", show_header=True, expand=True)
    class_table.add_column("Classes", style="green", justify="left", no_wrap=False, width=50)
    
    for class_id, class_name in dataset_train.class_id2names.items():
        class_table.add_row(f"{class_id}: {class_name}")
    
    console.print(class_table)

def print_augmentation_steps(transforms_train, transforms_val):
    console = Console(force_terminal=True, width=None, soft_wrap=False)
    
    # Training augmentation steps table
    train_aug_table = Table(title="Training Data Augmentation Steps", show_header=True, expand=True)
    train_aug_table.add_column("No.", style="cyan", justify="right", no_wrap=False, width=10)
    train_aug_table.add_column("Transform Name", style="green", no_wrap=False, width=30)
    train_aug_table.add_column("Parameters", style="yellow", no_wrap=False, width=60)
    for i, transform in enumerate(transforms_train.transforms, 1):
        params = str(transform.__dict__) if hasattr(transform, '__dict__') else '{}'
        # if parameters are too long, wrap to next line
        if len(params) > 500:
            lines = []
            for j in range(0, len(params), 500):
                lines.append(params[j:j+500])
            params = '\n'.join(lines)
        train_aug_table.add_row(str(i), transform.__class__.__name__, params)
    console.print(train_aug_table)
    
    # Validation augmentation steps table
    val_aug_table = Table(title="Validation Data Augmentation Steps", show_header=True, expand=True)
    val_aug_table.add_column("No.", style="cyan", justify="right", no_wrap=False, width=10)
    val_aug_table.add_column("Transform Name", style="green", no_wrap=False, width=30)
    val_aug_table.add_column("Parameters", style="yellow", no_wrap=False, width=60)
    
    for i, transform in enumerate(transforms_val.transforms, 1):
        params = str(transform.__dict__) if hasattr(transform, '__dict__') else '{}'
        # if parameters are too long, wrap to next line
        if len(params) > 500:
            lines = []
            for j in range(0, len(params), 500):
                lines.append(params[j:j+500])
            params = '\n'.join(lines)
        val_aug_table.add_row(str(i), transform.__class__.__name__, params)
    console.print(val_aug_table)