import torch

class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_items = next(self.loader)
        except StopIteration:
            self.next_items = None
            return

        with torch.cuda.stream(self.stream):
            self.next_x_cat, self.next_x_num, self.next_y_cat, self.next_y_num = (
                self.next_items[0].to(self.device, non_blocking=True),
                self.next_items[1].to(self.device, non_blocking=True),
                self.next_items[2].to(self.device, non_blocking=True),
                self.next_items[3].to(self.device, non_blocking=True),
            )

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_items is None:
            return None

        x_cat = self.next_x_cat
        x_num = self.next_x_num
        y_cat = self.next_y_cat
        y_num = self.next_y_num

        # Preload next batch asynchronously
        self.preload()

        return x_cat, x_num, y_cat, y_num
