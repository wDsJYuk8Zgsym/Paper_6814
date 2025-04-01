import argparse
import math
import os
import sys
import json
import deepspeed
import numpy as np
import torch
from deepspeed import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM
from transformers import get_scheduler
from transformers import SchedulerType
from torch.utils.tensorboard import SummaryWriter
from trl import AutoModelForCausalLMWithValueHead
from utils.tokenizer import *


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


from utils.utils import (
    get_print_func,
    _print_args,
    get_print_local_rank_0_func,
    get_print_gloabl_rank_0_func,
    show_multi_columns_data,
    get_gpu_memory_usage,
    print_rank_0,
    save_hf_format,
    set_random_seed,
    get_optimizer_grouped_parameters,
    save_zero_three_model,
    load_hf_tokenizer,
    _print_args
)
from utils.ds_utils import get_train_ds_config
from utils.module.lora import (
    convert_linear_layer_to_lora,
    convert_lora_to_linear_layer,
    only_optimize_lora_parameters,
    make_model_gradient_checkpointing_compatible,
)
from deepspeed.utils import logger
import os
from deepspeed.runtime.utils import empty_cache
from utils.model.model_utils import create_hf_model
from utils.utils import get_gpu_memory_usage
from data_utils import create_training_dataset,prepare_input_datas,DistributedStateWiseSampler

import warnings
warnings.filterwarnings("ignore")
 

class DataCollatorDAPO:
    def __call__(self, data):
        batch = {}
        batch["states"] = [f[0] for f in data]
        batch["states_ids"] =  [f[1] for f in data]
        batch["actions"] =  [f[2] for f in data]
        batch["actions_ids"] =  [f[3] for f in data]
        batch["advantages"] =  [f[4] for f in data]
        batch["idxs"] = [f[5] for f in data]
        return batch

def divide_list(data_list,n):
    new_list = [[] for i in range(n)]
    for i,item in enumerate(data_list):
        assign_node = i % n
        new_list[assign_node].append(item)
    return new_list
    
def Average(arr):
    if len(arr) > 0:
        return sum(arr)/len(arr) 
    else:
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description="PRM")
    parser.add_argument(
        "--data_path",
        type=str
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        help=(
            "Where to store the data-related files such as shuffle index."
            "This needs to be on a local storage of a node (not on a shared storage)"
        ),
        default=None

    )
    parser.add_argument(
        "--data_output_name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--ref_data_output_name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=[]
    )
    parser.add_argument(
        "--math_step_split_token",
        type=str,
        default="\n"
    )
    parser.add_argument(
        "--question_key",
        type=str,
        default="problem"
    )
    parser.add_argument(
        "--step_key",
        type=str,
        default="steps"
    )
    parser.add_argument(
        "--next_step_key",
        type=str,
        default="next_steps"
    )
    parser.add_argument(
        "--advantage_key",
        type=str,
        default="advantages"
    )
    parser.add_argument(
        "--tb_workpath",
        type=str,
        default=""
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to tokenizer",
        default = None,
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        help="Path to tokenizer",
        default = "AutoTokenizer",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        help="Path to tokenizer",
        default = "AutoTokenizer",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--run_dir", type=str, default=None, help="Where to run this job."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--do_not_use_baseline",
        action="store_true",
    )
    parser.add_argument(
        "--disable_DAPOpout",
        action="store_true",
        help="Disable the DAPOpout of the model.",
    )
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Training data type",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--save_every_n_epoch",
        type=float,
        default=-1,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="beta",
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=0,
        help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument(
        "--print_loss", action="store_true", help="Prints loss at each step."
    )
    parser.add_argument(
        "--state-wise-learning",
        action="store_true",
        dest="state_wise_learning"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="llama_7b_cot",
        dest="prompt_type"
    )
    parser.add_argument(
        "--start_epoch",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--end_epoch",
        type=float,
        default=None,
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    if WORLD_SIZE > 1:
        args.local_rank = LOCAL_RANK
    else:
        args.local_rank = -1
    return args

from typing import List
from trl.core import logprobs_from_logits

def get_log_trajectory_probs(model : torch.nn.Module, 
                             input_ids : torch.LongTensor, 
                             attention_mask : torch.LongTensor,
                             train_masks : torch.LongTensor):

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states = True,
        past_key_values = None
    )
    lm_logits = output.logits
    if lm_logits.dtype != torch.float32:
        lm_logits = lm_logits.float()
    logprobs = logprobs_from_logits(lm_logits[:, :-1, :], input_ids[:, 1:])
    log_trajectory_prob = (logprobs * (train_masks[:,:-1])).sum(axis=-1)
    return logprobs, log_trajectory_prob




def main():
    import os
    import time 

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    
    _print = get_print_local_rank_0_func(args.local_rank,args.global_rank)
    _print_global_rank_0 = get_print_gloabl_rank_0_func(args.local_rank,args.global_rank)
    _print_all_rank = get_print_func(args.local_rank,args.global_rank)

    args.rank = args.local_rank
    _print_args("DAPO",args)    
    world_size = torch.distributed.get_world_size()
    ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tb_workpath,
        tb_name="",
    )
    if args.global_rank == 0:
        writer = SummaryWriter(log_dir=args.tb_workpath)
    else:
        writer = None

    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    )
    set_random_seed(args.seed)
    tokenizer = build_tokenizer(args)
    if "qwen" in args.prompt_type and tokenizer.tokenizer.bos_token is None:
        bos_token = "<|im_start|>"
        tokenizer.tokenizer.bos_token =  bos_token
        tokenizer.tokenizer.bos_token_id = tokenizer.tokenize(bos_token)[-1]
    
    math_step_split_token_id =   tokenizer.tokenize(args.math_step_split_token)[-1]
    msg = f"math step split token {args.math_step_split_token} is tokenized to : {math_step_split_token_id}"

    white_space_token_id =  tokenizer.tokenize(" ")[-1]
    msg = f'white space token " " is tokenized to : {white_space_token_id}'

    setattr(tokenizer,"white_space_token_id", white_space_token_id) 
    setattr(tokenizer,"math_step_split_token_id", math_step_split_token_id) 
    setattr(tokenizer,"math_step_split_token", args.math_step_split_token) 

    if tokenizer.tokenizer.pad_token is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token
        tokenizer.tokenizer.pad_token_id = tokenizer.eos_token_id

    torch.distributed.barrier()

    train_dataset_fname = create_training_dataset(
        global_rank = args.global_rank,
        data_path = args.data_path,
        seed = args.seed,
        tokenizer = tokenizer,
        question_key = args.question_key,
        prompt_type = args.prompt_type,
        step_key = args.step_key,
        next_step_key = args.next_step_key,
        advantage_key = args.advantage_key,
        reload=False,
        data_save_name =  args.data_output_name,
        max_seq_len = args.max_seq_len
    )
    train_dataset = torch.load(train_dataset_fname)
    tmp_dir = os.path.join(args.run_dir,"tmp")
    os.makedirs(tmp_dir,exist_ok=True)

    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.displot(train_dataset.advantages, kde=True)
    plt.title('Distribution of advantage')
    plt.savefig(os.path.join(tmp_dir,"distribution_of_advantages.png"))


    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    elif args.state_wise_learning:
        train_sampler = DistributedStateWiseSampler(train_dataset, DAPOp_last=True)
    else:
        train_sampler = DistributedSampler(train_dataset, DAPOp_last=True)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=DataCollatorDAPO(),
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        DAPOp_last=True
    )    

    #--------------------- prepare ref data ---------------------
    cache_found = os.path.isfile(args.ref_data_output_name)
    reload = False
    if reload or not cache_found:
        ref_ds_config = get_train_ds_config(
            offload=args.offload,
            dtype=args.dtype,
            stage=3  if args.zero_stage == 3 else 0
        )
        ref_model = create_hf_model(
            model_class = AutoModelForCausalLM,
            model_name_or_path = args.model_name_or_path,
            tokenizer = tokenizer,
            ds_config = ds_config,
            rlhf_training = False
        )
        ref_ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ref_ds_config["train_batch_size"] = (
            args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
        )
        ref_model, *_ = deepspeed.initialize(model=ref_model, config=ref_ds_config)
        ref_model.eval()
        torch.distributed.barrier()

        idx_2_ref_log_prob = dict()
        st = time.time()
        with torch.no_grad():
            for step, batch in enumerate(train_dataloader):
                input_ids, attention_mask, train_masks, input_id_lens = \
                    prepare_input_datas(
                        states_ids = batch["states_ids"],
                        actions_ids = batch["actions_ids"],
                        pad_token_id = tokenizer.tokenizer.pad_token_id,
                        device = ref_model.device,
                        max_seq_len = args.max_seq_len
                    )
                logprobs, log_trajectory_probs = get_log_trajectory_probs(
                             model = ref_model,  
                             input_ids =input_ids,
                             attention_mask  = attention_mask,
                             train_masks = train_masks)

                for batch_idx in range(len(batch["idxs"])):
                    sample_idx = batch["idxs"][batch_idx]
                    idx_2_ref_log_prob[sample_idx] = log_trajectory_probs[batch_idx].cpu().item()

                if step == 0 and args.global_rank <= 4:
                    _, f = show_multi_columns_data(
                        column_datas = \
                            [input_ids[0].cpu().numpy().tolist()[:-1],
                            attention_mask[0].cpu().float().numpy().tolist()[:-1],
                            train_masks[0].cpu().float().numpy().tolist()[:-1],
                            logprobs[0].cpu().float().numpy().tolist(),
                            ],
                        column_names = \
                            ["s_t","Non pad token","train pi(\cdot|s_t)","log pi(a_t|s_t)"],
                        column_str_format_method = \
                            [
                            lambda token_id : '%.5d' % token_id,
                            lambda attention_mask : '%.5s' % attention_mask,
                            lambda attention_mask : '%.5s' % attention_mask,
                            lambda log_prob : '%.3f' % log_prob                    
                            ],
                        token_id_col_idx = 0,
                        tokenizer = tokenizer,
                        output_dir = os.path.join(tmp_dir,f"ref_data_input_example_{args.global_rank + 1}.txt"),
                        return_file_handler = True
                    )
                    f.write(f"log trajectory prob : {log_trajectory_probs[0].cpu().item()}\n")
                    f.write(f"idx_2_ref_log_prob : {idx_2_ref_log_prob}")
                    f.close()

        all_idx_2_ref_log_prob = [None for _rank in range(world_size)]
        torch.distributed.all_gather_object(all_idx_2_ref_log_prob, idx_2_ref_log_prob)
        if args.global_rank == 0:
            for _rank, other_rank_idx_2_ref_log_prob in enumerate(all_idx_2_ref_log_prob):
                if _rank > 0:
                    idx_2_ref_log_prob.update(other_rank_idx_2_ref_log_prob)
            torch.save(idx_2_ref_log_prob, args.ref_data_output_name)
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()

        del ref_model
        empty_cache()
    else:
        idx_2_ref_log_prob = torch.load(args.ref_data_output_name)
 
    torch.distributed.barrier()
    model = create_hf_model(
        model_class = AutoModelForCausalLM,
        model_name_or_path = args.model_name_or_path,
        tokenizer = tokenizer,
        ds_config = ds_config,
        rlhf_training = False
    )


    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, lora_lr = None,
    )

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    torch.distributed.barrier()
    total_micro_step = 0
    total_iter = 0
    import time
    total_epochs = args.num_train_epochs
    total_ckpt_num = int(total_epochs // args.save_every_n_epoch)
    save_ckpt_micro_steps = [int((ckpt_idx + 1) * args.save_every_n_epoch * len(train_dataloader)) for ckpt_idx in range(total_ckpt_num)]

    start_micro_steps = int(args.start_epoch * len(train_dataloader))
    if args.end_epoch is not None:
        end_micro_steps = int(args.end_epoch * len(train_dataloader))
    else:
        end_micro_steps = int(args.num_train_epochs * len(train_dataloader)) + 2

    total_valid_micro_steps = end_micro_steps - start_micro_steps

    for epoch in range(total_epochs):
        train_res_key = ["loss","acc","positive_log_ratio","negative_log_ratio","states"]
        train_res_dict = dict([key,[]] for key in train_res_key)
        model.train()

        for step, batch in enumerate(train_dataloader):

            if epoch * len(train_dataloader) + step < start_micro_steps:
                continue

            if total_micro_step >= total_valid_micro_steps:
                break
            
            print_this = epoch * len(train_dataloader) + step == start_micro_steps and epoch == int(args.start_epoch)

            start = time.time()
            input_ids, attention_mask, train_masks, input_id_lens = \
                prepare_input_datas(
                    states_ids = batch["states_ids"],
                    actions_ids = batch["actions_ids"],
                    pad_token_id = tokenizer.tokenizer.pad_token_id,
                    device = model.device,
                    max_seq_len = args.max_seq_len
                )
            _, log_trajectory_probs = get_log_trajectory_probs(
                            model = model,  
                            input_ids =input_ids,
                            attention_mask  = attention_mask,
                            train_masks = train_masks)

            ref_log_trajectory_probs = [idx_2_ref_log_prob[sample_idx] for sample_idx in batch["idxs"]]
            ref_log_trajectory_probs = torch.tensor(ref_log_trajectory_probs).to(log_trajectory_probs.dtype)

            advantages = torch.tensor(batch["advantages"]).to(log_trajectory_probs.dtype).to(model.device)
            target = advantages/args.beta
            log_ratio = log_trajectory_probs - ref_log_trajectory_probs.to(model.device)
            loss = ((log_ratio - target) * (log_ratio - target)) * 0.5
            loss = loss.mean()
            model.backward(loss)
            model.step()
            _loss = loss.detach().cpu().item()
            _log_ratios = log_ratio.detach().cpu()
            target = target.cpu()
            
            pred_sign = (_log_ratios > 0)
            true_sign = (target > 0)
            acc = (pred_sign == true_sign).sum().item() / len(_log_ratios)
            positive_ratios = []
            negative_ratios = []
            for i in  range(len(_log_ratios)):
                _target = target[i].item()
                _log_ratio = _log_ratios[i].item()
                if _target >= 0:
                    positive_ratios.append(_log_ratio)
                else:
                    negative_ratios.append(_log_ratio)
            
            train_res_dict["loss"].append(_loss)
            train_res_dict["acc"].append(acc)
            train_res_dict["positive_log_ratio"] += positive_ratios       
            train_res_dict["negative_log_ratio"] += negative_ratios         
            train_res_dict["states"] += batch["states"]         

            end = time.time()
            total_micro_step += 1

            if total_micro_step % args.gradient_accumulation_steps == 0:
                total_iter += 1
                iter_data = dict([key,[]] for key in train_res_key)
                all_rank_train_res_dict = [None for _rank in range(world_size)]
                torch.distributed.all_gather_object(all_rank_train_res_dict, train_res_dict)

                if writer is not None:
                    for _rank in range(world_size):
                        _rank_train_res_dict = all_rank_train_res_dict[_rank]
                        for _key in train_res_key:
                            iter_data[_key] += _rank_train_res_dict[_key]
                    
                    mini_batch_num = len(iter_data["loss"])
                    batch_size = mini_batch_num * args.per_device_train_batch_size * world_size
                    writer.add_scalar("train/loss" ,Average(iter_data["loss"]),total_iter)
                    writer.add_scalar("train/accuracy",Average(iter_data["acc"]),total_iter)
                    writer.add_scalar("train/log ratio of positive samples",Average(iter_data["positive_log_ratio"]),total_iter)
                    writer.add_scalar("train/log ratio of negative samples",Average(iter_data["negative_log_ratio"]),total_iter)
                    writer.add_scalar("train/log ratio diff",Average(iter_data["positive_log_ratio"]) - Average(iter_data["negative_log_ratio"]),total_iter)
                    writer.add_scalar("train/global batch size",len(iter_data["states"]),total_iter)
                    writer.add_scalar("train/unique states in a global batch",len(set(iter_data["states"])),total_iter)

    
                train_res_dict = dict([key,[]] for key in train_res_key)

            if total_micro_step in save_ckpt_micro_steps:
                epoch_progress = (save_ckpt_micro_steps.index(total_micro_step) + 1) * args.save_every_n_epoch
                sub_folder = "actor_model_epoch_%.2f" % epoch_progress
                output_model_dir = os.path.join(args.output_dir, sub_folder)
                os.makedirs(output_model_dir, exist_ok=TabError())
                model = convert_lora_to_linear_layer(model)
                if args.global_rank == 0:
                    save_hf_format(model, tokenizer.tokenizer, args, sub_folder=sub_folder)
                if args.zero_stage == 3:
                    save_zero_three_model(
                        model,
                        args.global_rank,
                        output_model_dir,
                        zero_stage=args.zero_stage,
                    )


        model.tput_timer.update_epoch_count()




if __name__ == "__main__":
    main()
