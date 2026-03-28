import os, sys, time
import numpy as np
import argparse
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_utils import get_data_loader
from utils.optimizer_utils import set_scheduler, set_optimizer
from utils.loss_utils import LossMSE
from utils.misc_utils import compute_grad_norm, vis_fields, l2_err
from utils.domains import DomainXY
from utils.sweeps import sweep_name_suffix
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from collections import OrderedDict

# models
import models.ffn
import models.fno

def print_mem():
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

def set_seed(params, world_size):
    seed = params.seed
    if seed is None:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if world_size > 0:
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

class Trainer():
    """ trainer class """
    def __init__(self, params, args):
        self.sweep_id = args.sweep_id
        self.root_dir = args.root_dir
        self.config = args.config 
        self.run_num = args.run_num
        self.world_size = 1
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        if self.world_size > 1:
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.benchmark = True
        
        self.log_to_screen = params.log_to_screen and self.world_rank==0
        self.log_to_wandb = params.log_to_wandb and self.world_rank==0
        params['name'] = args.config + '_' + args.run_num
        params['group'] = 'op_' + args.config
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        self.params = params
        self.params.device = self.device
        self.debug_runtime_logging = bool(getattr(self.params, 'debug_runtime_logging', False))
        self.runtime_debug_heartbeat_path = None
        self.checkpoint_selection_metric = str(
            getattr(self.params, 'checkpoint_selection_metric', 'val_loss')
        ).lower()
        if self.checkpoint_selection_metric not in ['val_loss', 'val_err']:
            raise ValueError(
                "checkpoint_selection_metric must be one of ['val_loss', 'val_err']"
            )

    def init_exp_dir(self, exp_dir):
        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
                os.makedirs(os.path.join(exp_dir, 'wandb/'))
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['checkpoint_path'] = os.path.join(exp_dir, 'checkpoints/ckpt.tar')
        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False
        self.runtime_debug_heartbeat_path = os.path.join(exp_dir, 'runtime_debug_heartbeat.txt')
        self.runtime_debug_event(
            'init_exp_dir',
            experiment_dir=self.params['experiment_dir'],
            checkpoint_path=self.params['checkpoint_path'],
            resuming=self.params['resuming'],
        )

    def runtime_debug_enabled(self):
        return self.debug_runtime_logging and self.world_rank == 0

    def _runtime_debug_storage_summary(self):
        if not self.runtime_debug_enabled():
            return {}

        summary = {}
        for label, path in (
            ('experiment_dir', getattr(self.params, 'experiment_dir', None)),
            ('tmpdir', os.environ.get('TMPDIR')),
        ):
            if not path:
                summary[label] = '<unset>'
                continue
            try:
                usage = shutil.disk_usage(path)
                summary[label] = (
                    f"path={path} free_gb={usage.free / (1024**3):.2f} "
                    f"used_gb={usage.used / (1024**3):.2f}"
                )
            except Exception as exc:
                summary[label] = f"path={path} error={exc}"
        return summary

    def _runtime_debug_path_state(self, path):
        if not path:
            return '<unset>'
        try:
            exists = os.path.exists(path)
            if exists and os.path.isfile(path):
                size_mb = os.path.getsize(path) / (1024**2)
                return f"path={path} exists=true size_mb={size_mb:.2f}"
            return f"path={path} exists={str(exists).lower()}"
        except Exception as exc:
            return f"path={path} error={exc}"

    def runtime_debug_event(self, phase, **details):
        if not self.runtime_debug_enabled():
            return

        payload = {
            'phase': phase,
            'epoch': getattr(self, 'epoch', None),
            'iters': getattr(self, 'iters', None),
        }
        payload.update(details)
        payload.update(self._runtime_debug_storage_summary())
        message = "Runtime debug: " + ", ".join(
            f"{key}={value}" for key, value in payload.items() if value is not None
        )
        logging.warning(message)

        if not self.runtime_debug_heartbeat_path:
            return
        try:
            with open(self.runtime_debug_heartbeat_path, 'w') as heartbeat:
                for key, value in payload.items():
                    heartbeat.write(f"{key}={value}\n")
        except Exception:
            logging.exception("Runtime debug: failed to update heartbeat file")

    def launch(self):

        if self.sweep_id:
            if self.world_rank==0:
                with wandb.init() as run:
                    hpo_config = wandb.config
                    self.params.update_params(hpo_config)
                    self.modify_bs_for_subsampling()
                    logging.info(self.params.name+'_'+sweep_name_suffix(self.params, self.sweep_id))
                    run.name = self.params.name+'_'+sweep_name_suffix(self.params, self.sweep_id)
                    self.name = run.name
                    self.params.name = self.name
                    exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep_id, self.name])
                    self.init_exp_dir(exp_dir)
                    logging.info('HPO sweep %s, trial cfg %s'%(self.sweep_id, self.name))
                    self.build_and_run()
            else:
                self.build_and_run()

        else:
            self.modify_bs_for_subsampling()
            exp_dir = os.path.join(*[self.root_dir, 'expts', self.config, self.run_num])
            self.init_exp_dir(exp_dir)
            if self.log_to_wandb:
                wandb.init(dir=os.path.join(exp_dir, "wandb"),
                           config=self.params.params, name=self.params.name, group=self.params.group, project=self.params.project, 
                           entity=self.params.entity, resume=self.params.resuming)
            self.build_and_run()



    def build_and_run(self):

        if self.sweep_id and dist.is_initialized():
            # Broadcast sweep config to other ranks
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            assert self.world_rank == rank
            if rank != 0:
                self.params = None
            self.params = comm.bcast(self.params, root=0)
            self.params.device = self.device # dont broadcast 0s device

        if self.world_rank == 0:
            logging.info(self.params.log())

        set_seed(self.params, self.world_size)
        self.runtime_debug_event(
            'post_seed_setup',
            seed=getattr(self.params, 'seed', None),
            world_size=self.world_size,
            device=self.device,
        )

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = int(self.params.valid_batch_size//self.world_size)

        # dump the yaml used
        if self.world_rank == 0:
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in self.params.params.items():
                hparams[str(key)] = str(value)
            with open(os.path.join(self.params['experiment_dir'], 'hyperparams.yaml'), 'w') as hpfile:
                yaml.dump(hparams,  hpfile )

        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, self.params.train_path, dist.is_initialized(), train=True, pack=self.params.pack_data)
        self.val_data_loader, self.val_dataset, self.valid_sampler = get_data_loader(self.params, self.params.val_path, dist.is_initialized(), train=False, pack=self.params.pack_data)

        # domain grid
        self.domain = DomainXY(self.params)

        
        if self.params.model == 'fno':
            self.model = models.fno.fno(self.params).to(self.device)
        else:
            assert(False), "Error, model arch invalid."

        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                device_ids=[self.local_rank],
                                                output_device=[self.local_rank])



        self.optimizer = set_optimizer(self.params, self.model)

        self.scheduler = set_scheduler(self.params, self.optimizer)

        if self.params.loss_func == "mse":
            self.loss_func = LossMSE(self.params, self.model)
        else:
            assert(False), "Error,  loss func invalid."

        self.iters = 0
        self.startEpoch = 0

        if hasattr(self.params, 'weights'):
            self.params.resuming = False
            logging.info("Loading IC weights %s"%self.params.weights)
            self.load_model(self.params.weights)

        if self.params.resuming:
            logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)

        self.epoch = self.startEpoch
        self.logs = {}
        self.train_loss = self.data_loss = self.bc_loss = self.pde_loss = self.grad = 0.0
        n_params = count_parameters(self.model)
        if self.log_to_screen:
            logging.info(self.model)
            logging.info('number of model parameters: {}'.format(n_params))
        self.runtime_debug_event(
            'training_launch',
            start_epoch=self.startEpoch,
            max_epochs=self.params.max_epochs,
            checkpoint_path=self.params.checkpoint_path,
        )

        # launch training
        self.train()

    def train(self):
        if self.log_to_screen:
            logging.info("Starting training loop...")
        best_selection_value = np.inf
        best_loss = np.inf

        best_epoch = 0
        best_err = 1
        self.logs['best_epoch'] = best_epoch
        plot_figs = self.params.plot_figs

        for epoch in range(self.startEpoch, self.params.max_epochs):
            self.epoch = epoch
            self.runtime_debug_event('epoch_start', epoch=epoch, max_epochs=self.params.max_epochs)
            if dist.is_initialized():
                # shuffles data before every epoch
                self.train_sampler.set_epoch(epoch)
            start = time.time()

            self.loss_func.set_epoch(self.epoch, self.params.max_epochs)
            self.loss_func.set_phase('train')
            # train
            self.runtime_debug_event('train_one_epoch_start')
            tr_time = self.train_one_epoch()
            self.runtime_debug_event('train_one_epoch_done', train_seconds=tr_time)
            self.loss_func.set_phase('eval')
            self.runtime_debug_event('val_one_epoch_start')
            val_time, fields = self.val_one_epoch()
            self.runtime_debug_event('val_one_epoch_done', val_seconds=val_time)
            self.logs['wt_norm'] = self.get_model_wt_norm(self.model)
            self.logs['pde_al_lambda'] = self.loss_func.get_aug_lagrange_multiplier()

            if self.params.scheduler == 'reducelr':
                self.scheduler.step(self.logs['train_loss'])
            elif self.params.scheduler == 'cosine':
                self.scheduler.step()

            selection_value = float(self.logs[self.checkpoint_selection_metric])
            if selection_value <= best_selection_value:
                is_best_loss = True
                best_selection_value = selection_value
                best_loss = self.logs['val_loss']
                best_err = self.logs['val_err']
            else:
                is_best_loss = False
            self.logs['best_val_loss'] = best_loss
            self.logs['best_val_err'] = best_err
            self.logs['best_checkpoint_selection_metric'] = best_selection_value
            self.logs['checkpoint_selection_metric'] = self.checkpoint_selection_metric

            best_epoch = self.epoch if is_best_loss else best_epoch
            self.logs['best_epoch'] = best_epoch

            if self.params.save_checkpoint:
                if self.world_rank == 0:
                    #checkpoint at the end of every epoch
                    if is_best_loss:
                        self.runtime_debug_event('save_logs_best_start')
                        self.save_logs(tag="_best")
                        self.runtime_debug_event('save_logs_best_done')
                    self.runtime_debug_event(
                        'checkpoint_save_start',
                        checkpoint_path=self.params.checkpoint_path,
                        best_checkpoint_path=self.params.checkpoint_path.replace('.tar', '_best.tar'),
                        is_best=is_best_loss,
                    )
                    try:
                        self.save_checkpoint(self.params.checkpoint_path, is_best=is_best_loss)
                    except Exception:
                        logging.exception(
                            "Runtime debug: checkpoint save failed at epoch %s",
                            self.epoch + 1,
                        )
                        raise
                    self.runtime_debug_event(
                        'checkpoint_save_done',
                        checkpoint_path=self._runtime_debug_path_state(self.params.checkpoint_path),
                        best_checkpoint_path=self._runtime_debug_path_state(
                            self.params.checkpoint_path.replace('.tar', '_best.tar')
                        ),
                        is_best=is_best_loss,
                    )

            if self.log_to_wandb:
                # log visualizations every epoch
                if plot_figs:
                    self.runtime_debug_event('wandb_image_start')
                    fig = None
                    try:
                        fig = vis_fields(fields, self.params, self.domain)
                        self.logs['vis'] = wandb.Image(fig)
                    except Exception:
                        logging.exception(
                            "Runtime debug: wandb image creation failed at epoch %s",
                            self.epoch + 1,
                        )
                        raise
                    finally:
                        if fig is not None:
                            plt.close(fig)
                    self.runtime_debug_event('wandb_image_done')
                self.logs['learning_rate'] = self.optimizer.param_groups[0]['lr']
                self.logs['time_per_epoch'] = tr_time
                self.runtime_debug_event('wandb_log_start', step=self.epoch + 1)
                try:
                    wandb.log(self.logs, step=self.epoch+1)
                except Exception:
                    logging.exception(
                        "Runtime debug: wandb.log failed at epoch %s",
                        self.epoch + 1,
                    )
                    raise
                self.runtime_debug_event('wandb_log_done', step=self.epoch + 1)

            if self.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec; with {}/{} in tr/val'.format(self.epoch+1, time.time()-start, tr_time, val_time))
                logging.info(
                    'Loss (total = data + bc + pde + zero) {} = {} + {} + {} + {}'.format(
                        self.logs['train_loss'], self.logs['data_loss'],
                        self.logs['bc_loss'], self.logs['pde_loss'],
                        self.logs['zero_mode_constraint_loss']
                    )
                )
                logging.info('Constraint metrics: tr_zero_loss={} tr_pde_res={} tr_zero_mode={} tr_bc_raw={} tr_bc_final={} val_zero_loss={} val_pde_res={} val_zero_mode={} val_bc_raw={} val_bc_final={} pde_al_lambda={}'.format(
                    self.logs['zero_mode_constraint_loss'],
                    self.logs['pde_residual_norm'], self.logs['zero_mode_violation'],
                    self.logs['bc_violation_raw'], self.logs['bc_violation_final'],
                    self.logs['val_zero_mode_constraint_loss'],
                    self.logs['val_pde_residual_norm'], self.logs['val_zero_mode_violation'],
                    self.logs['val_bc_violation_raw'], self.logs['val_bc_violation_final'],
                    self.logs['pde_al_lambda']
                ))
                logging.info('Validation errors: full={} interior={}'.format(
                    self.logs['val_err'], self.logs['val_err_interior']
                ))
                logging.info(
                    'Checkpoint selection metric: {} (best={})'.format(
                        self.checkpoint_selection_metric,
                        self.logs['best_checkpoint_selection_metric']
                    )
                )


        if self.log_to_wandb:
            self.runtime_debug_event('wandb_finish_start')
            try:
                wandb.finish()
            except Exception:
                logging.exception("Runtime debug: wandb.finish failed")
                raise
            self.runtime_debug_event('wandb_finish_done')

    
    def get_model_wt_norm(self, model):
        n = 0
        for p in model.parameters():
            p_norm = p.data.detach().norm(2)
            n += p_norm.item()**2
        n = n**0.5
        return n

    def save_logs(self, tag=""):
        with open(os.path.join(self.params.experiment_dir, "logs"+tag+".txt"), "w") as f:
            f.write("epoch,{}\n".format(self.epoch))
            for k, v in self.logs.items():
                f.write("{},{}\n".format(k,v))


    def train_one_epoch(self):
        tr_time = 0
        self.model.train()

        # buffers for logs
        logs_buff = torch.zeros((12), dtype=torch.float32, device=self.device)
        self.logs['train_loss'] = logs_buff[0].view(-1)
        self.logs['data_loss'] = logs_buff[1].view(-1)
        self.logs['bc_loss'] = logs_buff[2].view(-1)
        self.logs['pde_loss'] = logs_buff[3].view(-1)
        self.logs['zero_mode_constraint_loss'] = logs_buff[4].view(-1)
        self.logs['pde_residual_norm'] = logs_buff[5].view(-1)
        self.logs['zero_mode_violation'] = logs_buff[6].view(-1)
        self.logs['grad'] = logs_buff[7].view(-1)
        self.logs['tr_err'] = logs_buff[8].view(-1)
        self.logs['pde_al_lambda'] = logs_buff[9].view(-1)
        self.logs['bc_violation_raw'] = logs_buff[10].view(-1)
        self.logs['bc_violation_final'] = logs_buff[11].view(-1)


        for i, (inputs, targets) in enumerate(self.train_data_loader):
            self.iters += 1
            if not self.params.pack_data: # send to gpu if not already packed in the dataloader
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            tr_start = time.time()

            self.model.zero_grad()
            u_raw = self.model(inputs)
            u_final = self.loss_func.project_bc(inputs, u_raw)

            loss_data = self.loss_func.data(inputs, u_final, targets)
            loss_pde = self.loss_func.pde(inputs, u_final, targets)
            loss_bc = self.loss_func.bc(inputs, u_raw, targets)
            loss_zero = self.loss_func.zero_mode_constraint(inputs, u_final, targets)
            loss = loss_data + loss_bc + loss_pde + loss_zero

            loss.backward()
            self.optimizer.step()

            self.loss_func.update_bc_metrics(inputs, u_raw, u_final)
            pde_residual_norm = self.loss_func.get_last_pde_residual_norm()
            zero_mode_violation = self.loss_func.zero_mode_violation(inputs, u_final)
            bc_violation_raw = self.loss_func.get_last_bc_violation_raw()
            bc_violation_final = self.loss_func.get_last_bc_violation_final()
            grad_norm = compute_grad_norm(self.model.parameters())
            tr_err = l2_err(u_final.detach(), targets.detach())
    
            # add all the minibatch losses
            self.logs['train_loss'] += loss.detach()
            self.logs['data_loss'] += loss_data.detach()
            self.logs['bc_loss'] += loss_bc.detach()
            self.logs['pde_loss'] += loss_pde.detach()
            self.logs['zero_mode_constraint_loss'] += loss_zero.detach()
            self.logs['pde_residual_norm'] += pde_residual_norm.detach()
            self.logs['zero_mode_violation'] += zero_mode_violation.detach()
            self.logs['bc_violation_raw'] += bc_violation_raw.detach()
            self.logs['bc_violation_final'] += bc_violation_final.detach()
            self.logs['grad'] += grad_norm
            self.logs['tr_err'] += tr_err
            self.logs['pde_al_lambda'] += torch.tensor(
                self.loss_func.get_aug_lagrange_multiplier(), device=self.device, dtype=torch.float32
            )

            tr_time += time.time() - tr_start

        self.logs['train_loss'] /= len(self.train_data_loader)
        self.logs['data_loss'] /= len(self.train_data_loader)
        self.logs['bc_loss'] /= len(self.train_data_loader)
        self.logs['pde_loss'] /= len(self.train_data_loader)
        self.logs['zero_mode_constraint_loss'] /= len(self.train_data_loader)
        self.logs['pde_residual_norm'] /= len(self.train_data_loader)
        self.logs['zero_mode_violation'] /= len(self.train_data_loader)
        self.logs['bc_violation_raw'] /= len(self.train_data_loader)
        self.logs['bc_violation_final'] /= len(self.train_data_loader)
        self.logs['grad'] /= len(self.train_data_loader)
        self.logs['tr_err'] /= len(self.train_data_loader)
        self.logs['pde_al_lambda'] /= len(self.train_data_loader)

        logs_to_reduce = [
            'train_loss', 'data_loss', 'bc_loss', 'pde_loss',
            'zero_mode_constraint_loss', 'pde_residual_norm', 'zero_mode_violation',
            'bc_violation_raw', 'bc_violation_final',
            'grad', 'tr_err', 'pde_al_lambda'
        ]

        if dist.is_initialized():
            for key in logs_to_reduce:
                dist.all_reduce(self.logs[key].detach())
                # todo change loss to unscaled
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        return tr_time

    def val_one_epoch(self):
        self.model.eval() # need gradients
        #self.model.train() # need gradients
        val_start = time.time()

        logs_buff = torch.zeros((8), dtype=torch.float32, device=self.device)
        self.logs['val_err'] = logs_buff[0].view(-1)
        self.logs['val_loss'] = logs_buff[1].view(-1)
        self.logs['val_zero_mode_constraint_loss'] = logs_buff[2].view(-1)
        self.logs['val_pde_residual_norm'] = logs_buff[3].view(-1)
        self.logs['val_zero_mode_violation'] = logs_buff[4].view(-1)
        self.logs['val_bc_violation_raw'] = logs_buff[5].view(-1)
        self.logs['val_bc_violation_final'] = logs_buff[6].view(-1)
        self.logs['val_err_interior'] = logs_buff[7].view(-1)
        idx = np.random.randint(0, len(self.val_data_loader))
        img_idx = np.random.randint(0, self.params.local_valid_batch_size)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_data_loader):
                if not self.params.pack_data:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                u_raw = self.model(inputs)
                u_final = self.loss_func.project_bc(inputs, u_raw)
                loss_data = self.loss_func.data(inputs, u_final, targets)
                loss_pde = self.loss_func.pde(inputs, u_final, targets)
                loss_bc = self.loss_func.bc(inputs, u_raw, targets)
                loss_zero = self.loss_func.zero_mode_constraint(inputs, u_final, targets)
                loss = loss_data + loss_bc + loss_pde + loss_zero
                self.loss_func.update_bc_metrics(inputs, u_raw, u_final)
                pde_residual_norm = self.loss_func.get_last_pde_residual_norm()
                zero_mode_violation = self.loss_func.zero_mode_violation(inputs, u_final)
                bc_violation_raw = self.loss_func.get_last_bc_violation_raw()
                bc_violation_final = self.loss_func.get_last_bc_violation_final()
                self.logs['val_err'] += l2_err(u_final.detach(), targets.detach())
                self.logs['val_loss'] += loss.detach()
                self.logs['val_zero_mode_constraint_loss'] += loss_zero.detach()
                self.logs['val_pde_residual_norm'] += pde_residual_norm.detach()
                self.logs['val_zero_mode_violation'] += zero_mode_violation.detach()
                self.logs['val_bc_violation_raw'] += bc_violation_raw.detach()
                self.logs['val_bc_violation_final'] += bc_violation_final.detach()
                self.logs['val_err_interior'] += self.loss_func.interior_relative_l2(inputs, u_final, targets).detach()
                if i == idx: 
                    source = inputs[img_idx,0].detach().cpu().numpy() 
                    soln = targets[img_idx,0].detach().cpu().numpy()
                    pred = u_final[img_idx,0].detach().cpu().numpy()
                    pde_residual = self.loss_func.get_last_pde_residual_map()
                    if pde_residual is None:
                        pde_res = 0*pred
                    else:
                        pde_res = pde_residual[img_idx].detach().cpu().numpy()
                    temp = np.abs(pde_res)

        fields = [source, soln, pred, pde_res, temp]

        self.logs['val_loss'] /= len(self.val_data_loader)
        self.logs['val_err'] /= len(self.val_data_loader)
        self.logs['val_zero_mode_constraint_loss'] /= len(self.val_data_loader)
        self.logs['val_pde_residual_norm'] /= len(self.val_data_loader)
        self.logs['val_zero_mode_violation'] /= len(self.val_data_loader)
        self.logs['val_bc_violation_raw'] /= len(self.val_data_loader)
        self.logs['val_bc_violation_final'] /= len(self.val_data_loader)
        self.logs['val_err_interior'] /= len(self.val_data_loader)
        if dist.is_initialized():
            for key in [
                'val_loss', 'val_err', 'val_zero_mode_constraint_loss', 'val_pde_residual_norm',
                'val_zero_mode_violation', 'val_bc_violation_raw', 'val_bc_violation_final',
                'val_err_interior'
            ]:
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        val_time = time.time() - val_start

        return val_time, fields

    def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
        if not model:
            model = self.model
        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler is not None else None)}, checkpoint_path)
        if is_best:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if  self.scheduler is not None else None)}, checkpoint_path.replace('.tar', '_best.tar'))

    def _normalize_state_dict_keys(self, state_dict):
        model_state = self.model.state_dict()
        model_has_module = any(k.startswith('module.') for k in model_state.keys())
        ckpt_has_module = any(k.startswith('module.') for k in state_dict.keys())

        if model_has_module == ckpt_has_module:
            return OrderedDict(state_dict.items())

        if ckpt_has_module and not model_has_module:
            return OrderedDict((k[7:], v) if k.startswith('module.') else (k, v) for k, v in state_dict.items())

        return OrderedDict((f'module.{k}', v) if not k.startswith('module.') else (k, v) for k, v in state_dict.items())

    def _adapt_fc0_input_channels(self, state_dict):
        adapted = OrderedDict(state_dict.items())
        model_state = self.model.state_dict()
        for key, target_tensor in model_state.items():
            if not key.endswith('fc0.weight'):
                continue
            if key not in adapted:
                continue

            source_tensor = adapted[key]
            if source_tensor.shape == target_tensor.shape:
                continue

            if source_tensor.ndim != 2 or target_tensor.ndim != 2:
                continue
            if source_tensor.shape[0] != target_tensor.shape[0]:
                continue

            # Safe migration path for mixed pretraining checkpoints:
            # old in_dim=7 -> new in_dim=9 (BC channels appended).
            if source_tensor.shape[1] < target_tensor.shape[1]:
                expanded = torch.zeros_like(target_tensor)
                expanded[:, :source_tensor.shape[1]] = source_tensor
                adapted[key] = expanded
            else:
                adapted[key] = source_tensor[:, :target_tensor.shape[1]]
        return adapted

    def _load_model_state_with_migration(self, raw_state_dict):
        normalized = self._normalize_state_dict_keys(raw_state_dict)
        migrated = self._adapt_fc0_input_channels(normalized)
        self.model.load_state_dict(migrated, strict=True)

    def restore_checkpoint(self, checkpoint_path):
        if torch.cuda.is_available():
            map_location = 'cuda:{}'.format(self.local_rank)
        else:
            map_location = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=map_location) 
        self._load_model_state_with_migration(checkpoint['model_state'])

        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def load_model(self, checkpoint_path):
        if torch.cuda.is_available():
            map_location = 'cuda:{}'.format(self.local_rank)
        else:
            map_location = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=map_location) 
        self._load_model_state_with_migration(checkpoint['model_state'])
 
    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False


    def modify_bs_for_subsampling(self):
        '''Reduce batchsize for very small datasets'''
        sz = self.params.subsample
        if sz >= 512:
            fac = np.log2(sz) - 8
            self.params.batch_size = int(128/2**fac)
