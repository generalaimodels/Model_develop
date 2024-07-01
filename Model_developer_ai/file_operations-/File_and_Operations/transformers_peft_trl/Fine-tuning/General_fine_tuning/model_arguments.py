from dataclasses import dataclass
from typing import Optional, Union, Dict, List
from typing_extensions import Literal
import argparse
import yaml
from transformers import TrainingArguments
IntervalStrategy = Literal["no", "steps", "epoch"]
SchedulerType = Literal["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
OptimizerNames = Literal["adamw", "adam", "sgd", "adamax", "adagrad", "adafactor", "fused_lamb", "fused_sgd", "lamb"]
HubStrategy = Literal["every_save", "checkpoint"]
FSDPOption = Literal["auto_wrap", "auto_dp_3d", "deepspeed"]


@dataclass
class ModelArguments:
    output_dir: str
    overwrite_output_dir: bool = False
    do_train: bool = False
    do_eval: bool = False
    do_predict: bool = False
    evaluation_strategy: Union[IntervalStrategy, str] = "no"
    prediction_loss_only: bool = False
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    per_gpu_train_batch_size: Optional[int] = None
    per_gpu_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: Optional[int] = None
    eval_delay: Optional[float] = None
    learning_rate: float = 0.00005
    weight_decay: float = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1
    num_train_epochs: float = 3
    max_steps: int = -1
    lr_scheduler_type: Union[SchedulerType, str] = "linear"
    lr_scheduler_kwargs: Optional[Dict] = None
    warmup_ratio: float = 0
    warmup_steps: int = 0
    log_level: Optional[str] = "passive"
    log_level_replica: Optional[str] = "warning"
    log_on_each_node: bool = True
    logging_dir: Optional[str] = None
    logging_strategy: Union[IntervalStrategy, str] = "steps"
    logging_first_step: bool = False
    logging_steps: float = 500
    logging_nan_inf_filter: bool = True
    save_strategy: Union[IntervalStrategy, str] = "steps"
    save_steps: float = 500
    save_total_limit: Optional[int] = None
    save_safetensors: Union[bool, None] = True
    save_on_each_node: bool = False
    save_only_model: bool = False
    no_cuda: bool = False
    use_cpu: bool = False
    use_mps_device: bool = False
    seed: int = 42
    data_seed: Optional[int] = None
    jit_mode_eval: bool = False
    use_ipex: bool = False
    bf16: bool = False
    fp16: bool = False
    fp16_opt_level: str = "O1"
    half_precision_backend: str = "auto"
    bf16_full_eval: bool = False
    fp16_full_eval: bool = False
    tf32: Optional[bool] = None
    local_rank: int = -1
    ddp_backend: Optional[str] = None
    tpu_num_cores: Optional[int] = None
    tpu_metrics_debug: bool = False
    debug: Union[str, List[str]] = ""
    dataloader_drop_last: bool = False
    eval_steps: Optional[float] = None
    dataloader_num_workers: int = 0
    past_index: int = -1
    run_name: Optional[str] = None
    disable_tqdm: Optional[bool] = None
    remove_unused_columns: Optional[bool] = True
    label_names: Optional[List[str]] = None
    load_best_model_at_end: Union[bool, None] = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: Union[bool, None] = None
    ignore_data_skip: bool = False
    fsdp: Union[List[FSDPOption], str, None] = ""
    fsdp_min_num_params: int = 0
    fsdp_config: Optional[str] = None
    fsdp_transformer_layer_cls_to_wrap: Optional[str] = None
    deepspeed: Optional[str] = None
    label_smoothing_factor: float = 0
    optim: Union[OptimizerNames, str] = "default_optim"
    optim_args: Optional[str] = None
    adafactor: bool = False
    group_by_length: bool = False
    length_column_name: Optional[str] = "length"
    report_to: Optional[List[str]] = None
    ddp_find_unused_parameters: Optional[bool] = None
    ddp_bucket_cap_mb: Optional[int] = None
    ddp_broadcast_buffers: Optional[bool] = None
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    skip_memory_metrics: bool = True
    use_legacy_prediction_loop: bool = False
    push_to_hub: bool = False
    resume_from_checkpoint: Optional[str] = None
    hub_model_id: Optional[str] = None
    hub_strategy: Union[HubStrategy, str] = "every_save"
    hub_token: Optional[str] = None
    hub_private_repo: bool = False
    hub_always_push: bool = False
    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: Optional[dict] = None
    include_inputs_for_metrics: bool = False
    fp16_backend: str = "auto"
    push_to_hub_model_id: Optional[str] = None
    push_to_hub_organization: Optional[str] = None
    push_to_hub_token: Optional[str] = None
    mp_parameters: str = ""
    auto_find_batch_size: bool = False
    full_determinism: bool = False
    torchdynamo: Optional[str] = None
    ray_scope: Optional[str] = "last"
    ddp_timeout: Optional[int] = 1800
    torch_compile: bool = False
    torch_compile_backend: Optional[str] = None
    torch_compile_mode: Optional[str] = None
    dispatch_batches: Optional[bool] = None
    split_batches: Optional[bool] = False
    include_tokens_per_second: Optional[bool] = False
    include_num_input_tokens_seen: Optional[bool] = False
    neftune_noise_alpha: Optional[float] = None




def model_parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for Model Arguments")
    
    parser.add_argument("--output_dir", type=str, default='', help="Output directory")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite output directory if it exists")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction")
    parser.add_argument("--evaluation_strategy", type=str, choices=["no", "steps", "epoch"], default="no", help="Evaluation strategy")
    parser.add_argument("--prediction_loss_only", action="store_true", help="Compute and log only the prediction loss without training loss")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per GPU/TPU core/CPU for evaluation")
    parser.add_argument("--per_gpu_train_batch_size", type=int, help="Deprecated, use per_device_train_batch_size instead")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, help="Deprecated, use per_device_eval_batch_size instead")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--eval_accumulation_steps", type=int, help="Number of steps to accumulate for evaluation")
    parser.add_argument("--eval_delay", type=float, help="Delay evaluation until after a certain number of steps")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Initial learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay to apply")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon hyperparameter for Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1, help="Max gradient norm")
    parser.add_argument("--num_train_epochs", type=float, default=3, help="Total number of training epochs to perform")
    parser.add_argument("--max_steps", type=int, default=-1, help="Total number of training steps to perform")
    parser.add_argument("--lr_scheduler_type", type=str, choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], default="linear", help="Learning rate scheduler type")
    parser.add_argument("--lr_scheduler_kwargs", type=str, help="Additional arguments for learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps for learning rate scheduler")
    parser.add_argument("--log_level", type=str, choices=["passive", "active", "debug"], default="passive", help="Log level")
    parser.add_argument("--log_level_replica", type=str, choices=["warning", "info", "error"], default="warning", help="Log level for replica")
    parser.add_argument("--log_on_each_node", action="store_true", help="Whether to log on each node")
    parser.add_argument("--logging_dir", type=str, help="Logging directory")
    parser.add_argument("--logging_strategy", type=str, choices=["no", "steps", "epoch"], default="steps", help="Logging strategy")
    parser.add_argument("--logging_first_step", action="store_true", help="Log on the first step")
    parser.add_argument("--logging_steps", type=float, default=500, help="Log every N steps")
    parser.add_argument("--logging_nan_inf_filter", action="store_true", help="Filter out NaN and Inf in logs")
    parser.add_argument("--save_strategy", type=str, choices=["no", "steps", "epoch"], default="steps", help="Saving strategy")
    parser.add_argument("--save_steps", type=float, default=500, help="Save every N steps")
    parser.add_argument("--save_total_limit", type=int, help="Limit total amount of saved checkpoints")
    parser.add_argument("--save_safetensors", type=bool, help="Whether to save safe tensors")
    parser.add_argument("--save_on_each_node", action="store_true", help="Whether to save on each node")
    parser.add_argument("--save_only_model", action="store_true", help="Whether to save only model")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU")
    parser.add_argument("--use_mps_device", action="store_true", help="Use MPS device")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators")
    parser.add_argument("--data_seed", type=int, help="Seed for data loaders")
    parser.add_argument("--jit_mode_eval", action="store_true", help="JIT mode for evaluation")
    parser.add_argument("--use_ipex", action="store_true", help="Use IPex")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Use fp16")
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="FP16 optimization level")
    parser.add_argument("--half_precision_backend", type=str, default="auto", help="Half precision backend")
    parser.add_argument("--bf16_full_eval", action="store_true", help="Use bfloat16 for full evaluation")
    parser.add_argument("--fp16_full_eval", action="store_true", help="Use fp16 for full evaluation")
    parser.add_argument("--tf32", type=bool, help="Use TF32")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--ddp_backend", type=str, help="Distributed data parallel backend")
    parser.add_argument("--tpu_num_cores", type=int, help="Number of TPU cores")
    parser.add_argument("--tpu_metrics_debug", action="store_true", help="TPU metrics debug")
    parser.add_argument("--debug", type=str, nargs="+", help="Debug options")
    parser.add_argument("--dataloader_drop_last", action="store_true", help="Drop last batch in DataLoader")
    parser.add_argument("--eval_steps", type=float, help="Evaluate every N steps")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--past_index", type=int, default=-1, help="Past index")
    parser.add_argument("--run_name", type=str, help="Run name")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--remove_unused_columns", type=bool, help="Remove unused columns")
    parser.add_argument("--label_names", type=str, nargs="+", help="Label names")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model at end of training")
    parser.add_argument("--metric_for_best_model", type=str, help="Metric for best model")
    parser.add_argument("--greater_is_better", type=bool, help="Greater is better for the metric")
    parser.add_argument("--ignore_data_skip", action="store_true", help="Ignore data skip")
    parser.add_argument("--fsdp", type=str, nargs="+", choices=["auto_wrap", "auto_dp_3d", "deepspeed"], help="FSDP options")
    parser.add_argument("--fsdp_min_num_params", type=int, default=0, help="FSDP minimum number of parameters")
    parser.add_argument("--fsdp_config", type=str, help="FSDP configuration")
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, help="FSDP transformer layer class to wrap")
    parser.add_argument("--deepspeed", type=str, help="DeepSpeed configuration")
    parser.add_argument("--label_smoothing_factor", type=float, default=0, help="Label smoothing factor")
    parser.add_argument("--optim", type=str, choices=["adamw", "adam", "sgd", "adamax", "adagrad", "adafactor", "fused_lamb", "fused_sgd", "lamb"], default="default_optim", help="Optimizer name")
    parser.add_argument("--optim_args", type=str, help="Optimizer arguments")
    parser.add_argument("--adafactor", action="store_true", help="Use AdaFactor optimizer")
    parser.add_argument("--group_by_length", action="store_true", help="Group by length for DataLoader")
    parser.add_argument("--length_column_name", type=str, default="length", help="Length column name")
    parser.add_argument("--report_to", type=str, nargs="+", help="Report to")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", help="Find unused parameters in distributed data parallel")
    parser.add_argument("--ddp_bucket_cap_mb", type=int, help="Distributed data parallel bucket capacity in MB")
    parser.add_argument("--ddp_broadcast_buffers", action="store_true", help="Broadcast buffers in distributed data parallel")
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="Pin memory for DataLoader")
    parser.add_argument("--dataloader_persistent_workers", action="store_true", help="Persistent workers for DataLoader")
    parser.add_argument("--skip_memory_metrics", action="store_true", help="Skip memory metrics")
    parser.add_argument("--use_legacy_prediction_loop", action="store_true", help="Use legacy prediction loop")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hub")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--hub_model_id", type=str, help="Hub model ID")
    parser.add_argument("--hub_strategy", type=str, choices=["every_save", "checkpoint"], default="every_save", help="Hub strategy")
    parser.add_argument("--hub_token", type=str, help="Hub token")
    parser.add_argument("--hub_private_repo", action="store_true", help="Hub private repo")
    parser.add_argument("--hub_always_push", action="store_true", help="Always push to Hub")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Gradient checkpointing")
    parser.add_argument("--gradient_checkpointing_kwargs", type=str, help="Gradient checkpointing kwargs")
    parser.add_argument("--include_inputs_for_metrics", action="store_true", help="Include inputs for metrics")
    parser.add_argument("--fp16_backend", type=str, default="auto", help="FP16 backend")
    parser.add_argument("--push_to_hub_model_id", type=str, help="Push to Hub model ID")
    parser.add_argument("--push_to_hub_organization", type=str, help="Push to Hub organization")
    parser.add_argument("--push_to_hub_token", type=str, help="Push to Hub token")
    parser.add_argument("--mp_parameters", type=str, help="MP parameters")
    parser.add_argument("--auto_find_batch_size", action="store_true", help="Auto find batch size")
    parser.add_argument("--full_determinism", action="store_true", help="Full determinism")
    parser.add_argument("--torchdynamo", type=str, help="TorchDynamo")
    parser.add_argument("--ray_scope", type=str, default="last", help="Ray scope")
    parser.add_argument("--ddp_timeout", type=int, default=1800, help="Distributed data parallel timeout")
    parser.add_argument("--torch_compile", action="store_true", help="Torch compile")
    parser.add_argument("--torch_compile_backend", type=str, help="Torch compile backend")
    parser.add_argument("--torch_compile_mode", type=str, help="Torch compile mode")
    parser.add_argument("--dispatch_batches", action="store_true", help="Dispatch batches")
    parser.add_argument("--split_batches", action="store_true", help="Split batches")
    parser.add_argument("--include_tokens_per_second", action="store_true", help="Include tokens per second")
    parser.add_argument("--include_num_input_tokens_seen", action="store_true", help="Include number of input tokens seen")
    parser.add_argument("--neftune_noise_alpha", type=float, help="Neftune noise alpha")
    
    args = parser.parse_args()
    return args

    


def load_arguments_from_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        try:
            arguments = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            arguments = {}
    return arguments


args=model_parse_arguments()
config_args = load_arguments_from_yaml(yaml_file_path='training_config.yaml')
for key, value in config_args.items():
    setattr(args, key, value)

training_args=TrainingArguments(
output_dir=args.output_dir,
overwrite_output_dir=args.overwrite_output_dir,
do_train=args.do_train,
do_eval=args.do_eval,
do_predict=args.do_predict,
evaluation_strategy=args.evaluation_strategy,
prediction_loss_only=args.prediction_loss_only,
per_device_train_batch_size=args.per_device_train_batch_size,
per_device_eval_batch_size=args.per_device_eval_batch_size,
per_gpu_train_batch_size=args.per_gpu_train_batch_size,
per_gpu_eval_batch_size=args.per_gpu_eval_batch_size,
gradient_accumulation_steps=args.gradient_accumulation_steps,
eval_accumulation_steps=args.eval_accumulation_steps,
eval_delay=args.eval_delay,
learning_rate=args.learning_rate,
weight_decay=args.weight_decay,
adam_beta1=args.adam_beta1,
adam_beta2=args.adam_beta2,
adam_epsilon=args.adam_epsilon,
max_grad_norm=args.max_grad_norm,
num_train_epochs=args.num_train_epochs,
max_steps=args.max_steps,
lr_scheduler_type=args.lr_scheduler_type,
lr_scheduler_kwargs=args.lr_scheduler_kwargs,
warmup_ratio=args.warmup_ratio,
warmup_steps=args.warmup_steps,
log_level=args.log_level,
log_level_replica=args.log_level_replica,
log_on_each_node=args.log_on_each_node,
logging_dir=args.logging_dir,
logging_strategy=args.logging_strategy,
logging_first_step=args.logging_first_step,
logging_steps=args.logging_steps,
logging_nan_inf_filter=args.logging_nan_inf_filter,
save_strategy=args.save_strategy,
save_steps=args.save_steps,
save_total_limit=args.save_total_limit,
save_safetensors=args.save_safetensors,
save_on_each_node=args.save_on_each_node,
save_only_model=args.save_only_model,
no_cuda=args.no_cuda,
use_cpu=args.use_cpu,
use_mps_device=args.use_mps_device,
seed=args.seed,
data_seed=args.data_seed,
jit_mode_eval=args.jit_mode_eval,
use_ipex=args.use_ipex,
bf16=args.bf16,
fp16=args.fp16,
fp16_opt_level=args.fp16_opt_level,
half_precision_backend=args.half_precision_backend,
bf16_full_eval=args.bf16_full_eval,
fp16_full_eval=args.fp16_full_eval,
tf32=args.tf32,
local_rank=args.local_rank,
ddp_backend=args.ddp_backend,
tpu_num_cores=args.tpu_num_cores,
tpu_metrics_debug=args.tpu_metrics_debug,
debug=args.debug,
dataloader_drop_last=args.dataloader_drop_last,
eval_steps=args.eval_steps,
dataloader_num_workers=args.dataloader_num_workers,
past_index=args.past_index,
run_name=args.run_name,
disable_tqdm=args.disable_tqdm,
remove_unused_columns=args.remove_unused_columns,
label_names=args.label_names,
load_best_model_at_end=args.load_best_model_at_end,
metric_for_best_model=args.metric_for_best_model,
greater_is_better=args.greater_is_better,
ignore_data_skip=args.ignore_data_skip,
fsdp=args.fsdp,
fsdp_min_num_params=args.fsdp_min_num_params,
fsdp_config=args.fsdp_config,
fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
deepspeed=args.deepspeed,
label_smoothing_factor=args.label_smoothing_factor,
optim=args.optim,
optim_args=args.optim_args,
adafactor=args.adafactor,
group_by_length=args.group_by_length,
length_column_name=args.length_column_name,
report_to=args.report_to,
ddp_find_unused_parameters=args.ddp_find_unused_parameters,
ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
ddp_broadcast_buffers= args.ddp_broadcast_buffers,
dataloader_pin_memory=args.dataloader_pin_memory,
dataloader_persistent_workers=args.dataloader_persistent_workers,
skip_memory_metrics=args.skip_memory_metrics,
use_legacy_prediction_loop=args.use_legacy_prediction_loop,
push_to_hub=args.push_to_hub,
resume_from_checkpoint=args.resume_from_checkpoint,
hub_model_id=args.hub_model_id,
hub_strategy=args.hub_strategy,
hub_token=args.hub_token,
hub_private_repo=args.hub_private_repo,
hub_always_push=args.hub_always_push,
gradient_checkpointing=args.gradient_checkpointing,
gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
include_inputs_for_metrics=args.include_inputs_for_metrics,
fp16_backend=args.fp16_backend,
push_to_hub_model_id=args.push_to_hub_model_id,
push_to_hub_organization=args.push_to_hub_organization,
push_to_hub_token=args.push_to_hub_token,
mp_parameters=args.mp_parameters,
auto_find_batch_size=args.auto_find_batch_size,
full_determinism=args.full_determinism,
torchdynamo=args.torchdynamo,
ray_scope=args.ray_scope,
ddp_timeout=args.ddp_timeout,
torch_compile=args.torch_compile,
torch_compile_backend=args.torch_compile_backend,
torch_compile_mode=args.torch_compile_mode,
dispatch_batches=args.dispatch_batches,
split_batches=args.split_batches,
include_tokens_per_second=args.include_tokens_per_second,
include_num_input_tokens_seen=args.include_num_input_tokens_seen,
neftune_noise_alpha=args.neftune_noise_alpha,
)

