"""Framework-owned interfaces for composable detection training behavior."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class OptimizerContext:
    """Information required to reproduce an optimizer recipe."""

    num_classes: int
    dataset_size: int
    epochs: int
    global_batch_size: int
    nominal_batch_size: int


@dataclass
class OptimizerBuildResult:
    """Optimizer plus the derived values used by the training loop."""

    optimizer: Any
    accumulate: int
    scaled_weight_decay: float
    iterations: int
    name: str
    learning_rate: float
    momentum: float


@dataclass
class SchedulerBuildResult:
    """Scheduler and its epoch multiplier function."""

    scheduler: Any
    lr_lambda: Callable[[int], float]


class AugmentationStrategy(ABC):
    """Build train/validation transforms without coupling a dataset to a vendor."""

    @abstractmethod
    def build(self, dataset: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def close_mosaic(self, dataset: Any, hyp: Any = None) -> Any:
        raise NotImplementedError


class DataLoaderStrategy(ABC):
    """Build deterministic loaders for a recipe."""

    @abstractmethod
    def build(self, dataset: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class OptimizerStrategy(ABC):
    """Build an optimizer and expose all derived batch-dependent settings."""

    @abstractmethod
    def build(self, model: Any, hyp: Any, context: OptimizerContext) -> OptimizerBuildResult:
        raise NotImplementedError


class SchedulerStrategy(ABC):
    """Build and warm up a learning-rate schedule."""

    @abstractmethod
    def build(self, optimizer: Any, hyp: Any, epochs: int) -> SchedulerBuildResult:
        raise NotImplementedError

    @abstractmethod
    def apply_warmup(
        self,
        optimizer: Any,
        hyp: Any,
        lr_lambda: Callable[[int], float],
        iteration: int,
        epoch: int,
        batches_per_epoch: int,
        base_accumulate: int,
        global_batch_size: int,
    ) -> int:
        raise NotImplementedError


class EMAStrategy(ABC):
    """Create the exponential moving average implementation for a recipe."""

    @abstractmethod
    def build(self, model: Any, decay: float, tau: float) -> Any:
        raise NotImplementedError


class PrecisionStrategy(ABC):
    """Resolve AMP support and create its gradient scaler."""

    @abstractmethod
    def resolve_amp(self, model: Any, device: Any, requested: bool) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_scaler(self, enabled: bool) -> Any:
        raise NotImplementedError


class RuntimeStrategy(ABC):
    """Own reproducibility and model-preparation behavior."""

    @abstractmethod
    def initialize_seed(self, seed: int, deterministic: bool, rank: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_model(self, model: Any, freeze: Any = None) -> None:
        raise NotImplementedError


class LossStrategy(ABC):
    """Bridge a model loss into the framework lifecycle."""

    @abstractmethod
    def compute(self, model: Any, batch_data: Any, forward_result: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def on_epoch_end(self, model: Any) -> None:
        raise NotImplementedError


class PostprocessStrategy(ABC):
    """Convert model outputs to framework prediction records."""

    @abstractmethod
    def process(
        self,
        model: Any,
        batch_data: Any,
        forward_result: Any,
        confidence: float = 0.0,
    ) -> Any:
        raise NotImplementedError


class DetectionTrainingRecipe(ABC):
    """Composite contract selected explicitly by a model trainer."""

    version: str
    augmentation: AugmentationStrategy
    dataloader: DataLoaderStrategy
    optimizer: OptimizerStrategy
    scheduler: SchedulerStrategy
    ema: EMAStrategy
    precision: PrecisionStrategy
    runtime: RuntimeStrategy
    loss: LossStrategy
    postprocess: PostprocessStrategy

    @abstractmethod
    def build_config(self, args: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def validate_arguments(self, args: Any) -> None:
        raise NotImplementedError
