from tqdm.notebook import tqdm


class Callback:
    """
    An interface for contract of model callback
    """
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self, *args):
        pass


class BatchCallback(Callback):
    def __init__(self):
        self.progress_bar = None

    def on_batch_start(self, *args):
        pass

    def on_batch_end(self, *args):
        loss, acc, _, _, phase = args
        if phase == "train":
            self.progress_bar.set_postfix(train_loss=loss, train_acc=acc)
            self.progress_bar.colour = "#4CAF50"
        else:
            self.progress_bar.set_postfix(val_loss=loss, val_acc=acc)
            self.progress_bar.colour = "#FF00FF"
        self.progress_bar.update()

    def on_epoch_begin(self, description, total_batches):
        # self.progress_bar.reset(total_batches)
        self.progress_bar = tqdm(total=total_batches, unit=' batches')
        self.progress_bar.set_description(f"{description}")

    def on_epoch_end(self, *args):
        self.progress_bar.close()
from datetime import datetime

now = datetime.now()


class ShowPredCallBack(BatchCallback):
    """
    Show Prediction result every epoch
    """
    def __init__(self, trainer, checkpoint_path=None, prefix="", scheduler=None, train_writer=None, val_writer=None):
        self.preds = {}
        self.preds[0] = 0
        self.preds[1] = 0
        self.trainer = trainer
        self.best_val = trainer.best_val_loss
        self.checkpoint_path = checkpoint_path or 'chkpoint.state'
        self.prefix = prefix
        self.model_best_state = None
        self.scheduler = scheduler
        self.train_writer = train_writer
        self.val_writer = val_writer

    def on_batch_end(self, *args):
        super().on_batch_end(*args)
        loss, acc, preds, labels, phase = args

        for x in preds.cpu().numpy():
            self.preds[x] += 1

        if phase == "val":
            pred_vs_label = list(zip(preds.tolist(), labels.tolist()))
            print(f"{loss:07.4f}> {[f'{x}{y}' for (x, y) in pred_vs_label]}")

    def on_epoch_end(self, *args):
        super().on_epoch_end(*args)

        epoch, loss, acc, val_loss, val_acc = args
        train_loss = loss.avg
        train_acc = acc.avg
        val_loss = val_loss.avg
        val_acc = val_acc.avg

        if self.train_writer:
            self.train_writer.add_scalar(f"Loss", train_loss, epoch)
            self.train_writer.add_scalar(f"Accuracy", train_acc, epoch)
        if self.val_writer:
            self.val_writer.add_scalar(f"Loss", val_loss, epoch)
            self.val_writer.add_scalar(f"Accuracy", val_acc, epoch)

        print(
            f"Prediction: Benign=>{self.preds[0]}, Malignant=>{self.preds[1]}, Train:[loss={train_loss:.4f}, acc={train_acc * 100:5.2f}%], Val:[loss={val_loss:.4f}, acc={val_acc * 100:5.2f}%]")

        self.trainer.save_model(self.checkpoint_path, val_loss, epoch=epoch)
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.model_best_state = f'{self.prefix}_ep{str(epoch).zfill(3)}_{val_loss:.4f}_{now.strftime("%Y-%m-%d_%H:%M")}_{self.trainer.model._get_name()}.chkp'
            print(f"\033[95m*Best Validation loss: {val_loss:.4f} saved to {self.model_best_state}\x1b[0m")
            self.trainer.save_model(f"{self.model_best_state}", val_loss, epoch=epoch)
        self.preds[0] = 0
        self.preds[1] = 0

        self.scheduler and self.scheduler.step()
