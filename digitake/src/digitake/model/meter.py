## Credit, from pytorch imagenet example
## https://github.com/pytorch/examples/blob/master/imagenet/main.py


class Metric(object):
    pass


class AverageMeter(Metric):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':.4f'):
        super(AverageMeter, self).__init__()
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data_points = []

    def update(self, val, n=1):
        self.data_points.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self, batch_score, sample_num=1):
        self.update(batch_score, sample_num)
        return self.avg

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(Metric):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def __str__(self):
        entries = [self.prefix + self.batch_fmtstr.format(self.meters.batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)
