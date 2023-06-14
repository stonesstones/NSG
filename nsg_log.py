from torch.utils.tensorboard import SummaryWriter
import os

class Logger:
    def __init__(self, log_dir, global_step=0):
        self._log_dir = log_dir
        self._writer = SummaryWriter(log_dir=self._log_dir)
        self.global_step = global_step
        print('########################')
        print('logging outputs to ', log_dir)
        print('global step start at: ', self.global_step)
        print('########################')
    
    def add_scalar(self, scalar_dict):
        for key, value in scalar_dict.items():
                print('{} : {}'.format(key, value))
                self._writer.add_scalar(f"{key}", value, global_step=self.global_step)

    def add_histogram(self, scalar_dict):
        for key, value in scalar_dict.items():
                print('{} : {}'.format(key, value))
                self._writer.add_histogram(f"{key}", value, global_step=self.global_step)

    def add_global_step(self):
        self.global_step += 1

    def flush(self):
        self._writer.flush()
    
    def should_record(self, step) -> bool:
        return (self.global_step % step) == 0

    def save_weights(self, mpdel, i):
        pass
        # path = os.path.join(self._log_dir, '{}_{:06d}.npy'.format(prefix, i))
        # np.save(path, weights)
        # print('saved weights at', path)
        # if args.latent_size > 0:
        #     for k in latent_encodings:
        #         save_weights(latent_encodings[k].numpy(), k, i)