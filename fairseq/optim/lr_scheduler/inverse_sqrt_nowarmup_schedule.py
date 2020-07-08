# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('inverse_sqrt_nowarmup')
class InverseSquareRootNoWarmupSchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number, without warmup.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

      decay_factor =
      lr = args.lr * sqrt(args.decay_period) / sqrt(args.decay_period + update_num)
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        self.init_lr = args.lr[0]
        self.decay_period = args.decay_period

        # initial learning rate
        self.lr = self.init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--decay-period', default=4000, type=int, metavar='N',
                            help='decay period, after which the lr decays by 1/sqrt(2)')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""

        self.lr = self.init_lr * (self.decay_period/(self.decay_period + num_updates))**0.5
        self.optimizer.set_lr(self.lr)
        return self.lr
