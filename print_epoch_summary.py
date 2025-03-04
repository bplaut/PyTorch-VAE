import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class EpochSummary(Callback):
    """
    Callback to print training and validation loss after each epoch
    """
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
            
        metrics = trainer.callback_metrics
        print("\n" + "-" * 50)
        print(f"Epoch {trainer.current_epoch}:")
        if 'loss' in metrics:
            print(f"Training Loss: {metrics['loss']:.5f}")
        for key, value in metrics.items():
            if key == 'val_loss':
                print(f"{key}: {value:.5f}")
        print("-" * 50 + "\n")
