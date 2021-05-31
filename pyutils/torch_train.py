##########################
#       torch model      #
##########################
import csv
import logging
import os
import random
import time
import traceback
from collections import OrderedDict

import numpy as np
import torch
from scipy import interpolate
from torchsummary import summary

from .general import ensure_dir

# disable_tf_warning()
# # Turn off some unnecessary messages
# os.environ['KMP_WARNINGS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.get_logger().setLevel(logging.ERROR)
# tf.compat.v1.logging.propagate = False


__all__ = ["set_torch_deterministic", "set_torch_stochastic", "get_random_state", "summary_model", "save_model", "BestKModelSaver", "load_model", "count_parameters", "check_converge",
           "ThresholdScheduler", "ThresholdScheduler_tf", "ValueRegister", "ValueTracer", "EMA", "export_traces_to_csv", "set_learning_rate", "get_learning_rate", "apply_weight_decay"]


def set_torch_deterministic(random_state: int = 0) -> None:
    random_state = int(random_state) % (2**32)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if(torch.cuda.is_available()):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)


def set_torch_stochastic():
    seed = int(time.time() * 1000) % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if(torch.cuda.is_available()):
        torch.backends.cudnn.deterministic = False
        torch.cuda.manual_seed_all(seed)


def get_random_state():
    return np.random.get_state()[1][0]


def summary_model(model, input):
    summary(model, input)


def save_model(model, path="./checkpoint/model.pt", print_msg=True):
    """Save PyTorch model in path

    Args:
        model (PyTorch model): PyTorch model
        path (str, optional): Full path of PyTorch model. Defaults to "./checkpoint/model.pt".
        print_msg (bool, optional): Control of message print. Defaults to True.
    """
    dir = os.path.dirname(path)
    if(not os.path.exists(dir)):
        os.mkdir(dir)
    try:
        torch.save(model.state_dict(), path)
        if(print_msg):
            print(f"[I] Model saved to {path}")
    except Exception as e:
        if(print_msg):
            print(f"[E] Model failed to be saved to {path}")
        traceback.print_exc(e)


class BestKModelSaver(object):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.model_cache = OrderedDict()

    def __insert_model_record(self, acc, dir, checkpoint_name, epoch=None):
        acc = round(acc * 100) / 100
        if(len(self.model_cache) < self.k):
            new_checkpoint_name = f"{checkpoint_name}_acc-{acc:.2f}{'' if epoch is None else '_epoch-'+str(epoch)}"
            path = os.path.join(dir, new_checkpoint_name+".pt")
            self.model_cache[path] = (acc, epoch)
            return path, None
        else:
            min_acc, min_epoch = sorted(
                list(self.model_cache.values()), key=lambda x: x[0])[0]
            if(acc >= min_acc + 0.01):
                del_checkpoint_name = f"{checkpoint_name}_acc-{min_acc:.2f}{'' if epoch is None else '_epoch-'+str(min_epoch)}"
                del_path = os.path.join(dir, del_checkpoint_name+".pt")
                try:
                    del self.model_cache[del_path]
                except:
                    print(
                        "[W] Cannot remove checkpoint: {} from cache".format(del_path), flush=True)
                new_checkpoint_name = f"{checkpoint_name}_acc-{acc:.2f}{'' if epoch is None else '_epoch-'+str(epoch)}"
                path = os.path.join(dir, new_checkpoint_name+".pt")
                self.model_cache[path] = (acc, epoch)
                return path, del_path
            # elif(acc == min_acc):
            #     new_checkpoint_name = f"{checkpoint_name}_acc-{acc:.2f}{'' if epoch is None else '_epoch-'+str(epoch)}"
            #     path = os.path.join(dir, new_checkpoint_name+".pt")
            #     self.model_cache[path] = (acc, epoch)
            #     return path, None
            else:
                return None, None

    def save_model(self, model, acc, epoch=None, path="./checkpoint/model.pt", other_params=None, save_model=False, print_msg=True):
        """Save PyTorch model in path

        Args:
            model (PyTorch model): PyTorch model
            acc (scalar): accuracy
            epoch (scalar, optional): epoch. Defaults to None
            path (str, optional): Full path of PyTorch model. Defaults to "./checkpoint/model.pt".
            other_params (dict, optional): Other saved params. Defaults to None
            save_model (bool, optional): whether save source code of nn.Module. Defaults to False
            print_msg (bool, optional): Control of message print. Defaults to True.
        """
        dir = os.path.dirname(path)
        ensure_dir(dir)
        checkpoint_name = os.path.splitext(os.path.basename(path))[0]
        if(isinstance(acc, torch.Tensor)):
            acc = acc.data.item()
        new_path, del_path = self.__insert_model_record(
            acc, dir, checkpoint_name, epoch)

        if(del_path is not None):
            try:
                os.remove(del_path)
                print(f"[I] Model {del_path} is removed", flush=True)
            except Exception as e:
                if(print_msg):
                    print(f"[E] Model {del_path} failed to be removed", flush=True)
                traceback.print_exc(e)

        if(new_path is None):
            if(print_msg):
                print(
                    f"[I] Not best {self.k}: {list(reversed(sorted(list(self.model_cache.values()))))}, skip this model ({acc:.2f}): {path}", flush=True)
        else:
            try:
                # torch.save(model.state_dict(), new_path)
                if(other_params is not None):
                    saved_dict = other_params
                else:
                    saved_dict = {}
                if(save_model):
                    saved_dict.update(
                        {"model": model, "state_dict": model.state_dict()})
                    torch.save(saved_dict, new_path)
                else:
                    saved_dict.update(
                        {"model": None, "state_dict": model.state_dict()})
                    torch.save(saved_dict, new_path)
                if(print_msg):
                    print(f"[I] Model saved to {new_path}. Current best {self.k}: {list(reversed(sorted(list(self.model_cache.values()))))}", flush=True)
            except Exception as e:
                if(print_msg):
                    print(f"[E] Model failed to be saved to {new_path}", flush=True)
                traceback.print_exc(e)


def load_model(model, path="./checkpoint/model.pt", ignore_size_mismatch: bool = False, print_msg=True):
    """Load PyTorch model in path

    Args:
        model (PyTorch model): PyTorch model
        path (str, optional): Full path of PyTorch model. Defaults to "./checkpoint/model.pt".
        ignore_size_mismatch (bool, optional): Whether ignore tensor size mismatch. Defaults to False.
        print_msg (bool, optional): Control of message print. Defaults to True.
    """
    try:
        raw_data = torch.load(
            path,  map_location=lambda storage, location: storage)
        if(isinstance(raw_data, OrderedDict) and "state_dict" not in raw_data):
            ### state_dict: OrderedDict
            state_dict = raw_data
        else:
            ### {"state_dict": ..., "model": ...}
            state_dict = raw_data["state_dict"]
        load_keys = set(state_dict.keys())
        model_keys = set(model.state_dict().keys())
        common_dict = load_keys & model_keys
        diff_dict = load_keys ^ model_keys
        extra_keys = load_keys - model_keys
        lack_keys = model_keys - load_keys
        cur_state_dict = model.state_dict()
        if(ignore_size_mismatch):
            size_mismatch_dict = set(key for key in common_dict if model.state_dict()[key].size() != state_dict[key].size())
            print(f"[W] {size_mismatch_dict} are ignored due to size mismatch", flush=True)
            common_dict = common_dict - size_mismatch_dict

        cur_state_dict.update({key: state_dict[key] for key in common_dict})
        if(len(diff_dict) > 0):
            print(
                f"[W] Warning! Model is not the same as the checkpoint. not found keys {lack_keys}. extra unused keys {extra_keys}")

        model.load_state_dict(cur_state_dict)
        if(print_msg):
            print(f"[I] Model loaded from {path}")
    except Exception as e:
        traceback.print_exc(e)
        if(print_msg):
            print(f"[E] Model failed to be loaded from {path}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_converge(trace, epsilon=0.002):
    if(len(trace) <= 1):
        return False
    if(np.abs(trace[-1] - trace[-2]) / (np.abs(trace[-1]) + 1e-8) < epsilon):
        return True
    return False


class ThresholdScheduler(object):
    ''' Intepolation between begin point and end point. step must be within two endpoints
    '''

    def __init__(self, step_beg, step_end, thres_beg, thres_end, mode='tanh'):
        assert mode in {
            "linear", "tanh"}, "Threshold scheduler only supports linear and tanh modes"
        self.mode = mode
        self.step_beg = step_beg
        self.step_end = step_end
        self.thres_beg = thres_beg
        self.thres_end = thres_end
        self.func = self.createFunc()

    def normalize(self, step, factor=2):
        return (step - self.step_beg) / (self.step_end - self.step_beg) * factor

    def createFunc(self):
        if(self.mode == "linear"):
            return lambda x: (self.thres_end - self.thres_beg) * x + self.thres_beg
        elif(self.mode == "tanh"):
            x = self.normalize(
                np.arange(self.step_beg, self.step_end + 1).astype(np.float32))
            y = np.tanh(x) * (self.thres_end - self.thres_beg) + self.thres_beg
            return interpolate.interp1d(x, y)

    def __call__(self, x):
        return self.func(self.normalize(x)).tolist()


class ThresholdScheduler_tf(object):
    ''' smooth increasing threshold with tensorflow model pruning scheduler
    '''

    def __init__(self, step_beg, step_end, thres_beg, thres_end):
        import tensorflow as tf
        import tensorflow_model_optimization as tfmot
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.enable_eager_execution(config=config)
        self.step_beg = step_beg
        self.step_end = step_end
        self.thres_beg = thres_beg
        self.thres_end = thres_end
        if(thres_beg < thres_end):
            self.thres_min = thres_beg
            self.thres_range = (thres_end - thres_beg)
            self.descend = False

        else:
            self.thres_min = thres_end
            self.thres_range = (thres_beg - thres_end)
            self.descend = True

        self.pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0, final_sparsity=0.9999999,
            begin_step=self.step_beg, end_step=self.step_end)

    def __call__(self, x):
        if(x < self.step_beg):
            return self.thres_beg
        elif(x > self.step_end):
            return self.thres_end
        res_norm = self.pruning_schedule(x)[1].numpy()
        if(self.descend == False):
            res = res_norm * self.thres_range + self.thres_beg
        else:
            res = self.thres_beg - res_norm * self.thres_range

        if(np.abs(res - self.thres_end) <= 1e-6):
            res = self.thres_end
        return res


class ValueRegister(object):
    def __init__(self, operator, name="", show=True):
        self.op = operator
        self.cache = None
        self.show = show
        self.name = name if len(name) > 0 else "value"

    def register_value(self, x):
        self.cache = self.op(x, self.cache) if self.cache is not None else x
        if(self.show):
            print(f"Recorded {self.name} is {self.cache}")


class ValueTracer(object):
    def __init__(self, show=True):
        self.cache = {}
        self.show = show

    def add_value(self, name, value, step):
        if(name not in self.cache):
            self.cache[name] = {}
        self.cache[name][step] = value
        if(self.show):
            print(f"Recorded {name}: step = {step}, value = {value}")

    def get_trace_by_name(self, name):
        return self.cache.get(name, {})

    def get_all_traces(self):
        return self.cache

    def __len__(self):
        return len(self.cache)

    def get_num_trace(self):
        return len(self.cache)

    def get_len_trace_by_name(self, name):
        return len(self.cache.get(name, {}))

    def dump_trace_to_file(self, name, file):
        if(name not in self.cache):
            print(f"[W] Trace name '{name}' not found in tracer")
            return
        torch.save(self.cache[name], file)
        print(f"[I] Trace {name} saved to {file}")

    def dump_all_traces_to_file(self, file):
        torch.save(self.cache, file)
        print(f"[I] All traces saved to {file}")

    def load_all_traces_from_file(self, file):
        self.cache = torch.load(file)
        return self.cache


class EMA(object):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone().data

    def __call__(self, name, x, mask=None):
        if(name not in self.shadow):
            self.register(name, x)
            return x.data

        old_average = self.shadow[name]
        new_average = (1 - self.mu) * x + self.mu * old_average
        if(mask is not None):
            new_average[mask].copy_(old_average[mask])
        self.shadow[name] = new_average.clone()
        return new_average.data


def export_traces_to_csv(trace_file, csv_file, fieldnames=None):
    traces = torch.load(trace_file)

    with open(csv_file, 'w', newline='') as csvfile:
        if(fieldnames is None):
            fieldnames = list(traces.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        max_len = max([len(traces[field]) for field in fieldnames])

        for idx in range(max_len):
            row = {}
            for field in fieldnames:
                value = traces[field][idx] if idx < len(traces[field]) else ""
                row[field] = value.data.item() if isinstance(
                    value, torch.Tensor) else value
            writer.writerow(row)


def set_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def apply_weight_decay(W, decay_rate, learning_rate, mask=None):
    # in mask, 1 represents fixed variables, 0 represents trainable variables
    if(mask is not None):
        W[~mask] -= W[~mask] * decay_rate * learning_rate
    else:
        W -= W * decay_rate * learning_rate
