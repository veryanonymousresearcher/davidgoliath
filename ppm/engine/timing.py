import time

class TimingMeter:
    """Utility to measure batch, epoch, and total training timings."""
    def __init__(self):
        self.reset_epoch()
        self.total_train_time = 0.0
        self.total_epochs = 0

    def reset_epoch(self):
        self.epoch_batch_times = []
        self.epoch_compute_times = []
        self.epoch_start = time.time()

    def record_batch(self, batch_time, compute_time):
        self.epoch_batch_times.append(float(batch_time))
        self.epoch_compute_times.append(float(compute_time))

    def epoch_summary(self):
        epoch_end = time.time()
        epoch_time = epoch_end - self.epoch_start
        num_batches = len(self.epoch_batch_times)

        avg_batch = (
            sum(self.epoch_batch_times) / num_batches if num_batches else 0.0
        )
        avg_compute = (
            sum(self.epoch_compute_times) / num_batches if num_batches else 0.0
        )
        compute_pct = (
            100 * avg_compute / avg_batch if avg_batch > 0 else 0.0
        )

        self.total_train_time += epoch_time
        self.total_epochs += 1

        return {
            "epoch_time": epoch_time,
            "avg_batch_time": avg_batch,
            "avg_compute_time": avg_compute,
            "avg_compute_pct": compute_pct,
            "num_batches": num_batches,
        }

    def print_epoch_summary(self, epoch_idx):
        stats = self.epoch_summary()
        print(
            f"Epoch {epoch_idx}: "
            f"avg_total={stats['avg_batch_time']:.3f}s, "
            f"avg_compute={stats['avg_compute_time']:.3f}s "
            f"({stats['avg_compute_pct']:.1f}%), "
            f"epoch_time={stats['epoch_time']:.2f}s over {stats['num_batches']} batches"
        )

    def print_total_summary(self):
        print(
            f"\nTotal training time over {self.total_epochs} epochs: "
            f"{self.total_train_time:.2f}s\n"
        )
