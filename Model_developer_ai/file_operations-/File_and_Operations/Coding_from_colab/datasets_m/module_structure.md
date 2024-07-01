# Structure of the `trl` Module

```plaintext
├── __pycache__/
├── environment/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── base_environment.py
│   │   │   Class: StoppingCriteria
│   │   │   Class: StoppingCriteriaList
│   │   │   Class: StringStoppingCriteria
│   │   │   Class: Text
│   │   │   Class: TextEnvironment
│   │   │   Class: TextHistory
│   │   │   Function: extract_model_from_parallel
│   │   │   Function: is_rich_available
│   │   │   Function: print
├── extras/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── best_of_n_sampler.py
│   │   │   Class: Any
│   │   │   Class: BestOfNSampler
│   │   │   Class: GenerationConfig
│   │   │   Class: PreTrainedModelWrapper
│   │   │   Class: PreTrainedTokenizer
│   │   │   Class: PreTrainedTokenizerFast
│   │   │   Function: set_seed
├── models/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── modeling_base.py
│   │   │   Class: EntryNotFoundError
│   │   │   Class: HFValidationError
│   │   │   Class: LocalEntryNotFoundError
│   │   │   Class: PartialState
│   │   │   Class: PeftConfig
│   │   │   Class: PeftModel
│   │   │   Class: PeftModelForCausalLM
│   │   │   Class: PeftModelForSeq2SeqLM
│   │   │   Class: PreTrainedModel
│   │   │   Class: PreTrainedModelWrapper
│   │   │   Class: PromptLearningConfig
│   │   │   Class: RepositoryNotFoundError
│   │   │   Function: create_reference_model
│   │   │   Function: deepcopy
│   │   │   Function: get_peft_model
│   │   │   Function: hf_hub_download
│   │   │   Function: is_deepspeed_zero3_enabled
│   │   │   Function: is_npu_available
│   │   │   Function: is_peft_available
│   │   │   Function: is_transformers_greater_than
│   │   │   Function: is_xpu_available
│   │   │   Function: prepare_model_for_kbit_training
│   │   │   Function: safe_load_file
│   ├── modeling_sd_base.py
│   │   │   (Error: No module named 'diffusers')
│   ├── modeling_value_head.py
│   │   │   Class: AutoModelForCausalLM
│   │   │   Class: AutoModelForCausalLMWithValueHead
│   │   │   Class: AutoModelForSeq2SeqLM
│   │   │   Class: AutoModelForSeq2SeqLMWithValueHead
│   │   │   Class: PreTrainedModelWrapper
│   │   │   Class: ValueHead
├── trainer/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── base.py
│   │   │   Class: BaseTrainer
│   │   │   Class: PyTorchModelHubMixin
│   ├── ddpo_config.py
│   │   │   Class: DDPOConfig
│   │   │   Function: dataclass
│   │   │   Function: field
│   │   │   Function: flatten_dict
│   │   │   Function: is_bitsandbytes_available
│   │   │   Function: is_torchvision_available
│   ├── ddpo_trainer.py
│   │   │   (Error: cannot import name 'DDPOStableDiffusionPipeline' from 'trl.models' (c:\Users\heman\AppData\Local\Programs\Python\Python311\Lib\site-packages\trl\models\__init__.py))
│   ├── dpo_trainer.py
│   │   │   Class: Any
│   │   │   Class: AutoModelForCausalLM
│   │   │   Class: DPODataCollatorWithPadding
│   │   │   Class: DPOTrainer
│   │   │   Class: DataLoader
│   │   │   Class: Dataset
│   │   │   Class: EvalLoopOutput
│   │   │   Class: PeftModel
│   │   │   Class: PreTrainedModel
│   │   │   Class: PreTrainedModelWrapper
│   │   │   Class: PreTrainedTokenizerBase
│   │   │   Class: Trainer
│   │   │   Class: TrainerCallback
│   │   │   Class: TrainingArguments
│   │   │   Function: create_reference_model
│   │   │   Function: deepcopy
│   │   │   Class: defaultdict
│   │   │   Function: disable_dropout_in_model
│   │   │   Function: get_peft_model
│   │   │   Function: is_deepspeed_available
│   │   │   Function: is_peft_available
│   │   │   Function: is_wandb_available
│   │   │   Class: nullcontext
│   │   │   Function: pad_to_length
│   │   │   Function: prepare_model_for_kbit_training
│   │   │   Function: tqdm
│   │   │   Function: trl_sanitze_kwargs_for_tagging
│   │   │   Function: wraps
│   ├── iterative_sft_trainer.py
│   │   │   Class: DataCollatorForLanguageModeling
│   │   │   Class: DataCollatorForSeq2Seq
│   │   │   Class: DataLoader
│   │   │   Class: Dataset
│   │   │   Class: EvalLoopOutput
│   │   │   Class: IterativeSFTTrainer
│   │   │   Class: PPODecorators
│   │   │   Class: PeftModel
│   │   │   Class: PreTrainedModel
│   │   │   Class: PreTrainedTokenizerBase
│   │   │   Class: Trainer
│   │   │   Class: TrainingArguments
│   │   │   Function: is_peft_available
│   ├── ppo_config.py
│   │   │   Class: Annotated
│   │   │   Class: PPOConfig
│   │   │   Function: dataclass
│   │   │   Function: exact_div
│   │   │   Function: field
│   │   │   Function: flatten_dict
│   │   │   Function: is_wandb_available
│   ├── ppo_trainer.py
│   │   │   Class: Accelerator
│   │   │   Class: Adam
│   │   │   Class: AdaptiveKLController
│   │   │   Class: BaseTrainer
│   │   │   Class: DataCollatorForLanguageModeling
│   │   │   Class: Dataset
│   │   │   Class: FixedKLController
│   │   │   Class: PPOConfig
│   │   │   Class: PPODecorators
│   │   │   Class: PPOTrainer
│   │   │   Class: PreTrainedModelWrapper
│   │   │   Class: PreTrainedTokenizer
│   │   │   Class: PreTrainedTokenizerBase
│   │   │   Class: PreTrainedTokenizerFast
│   │   │   Class: ProjectConfiguration
│   │   │   Class: RunningMoments
│   │   │   Function: clip_by_value
│   │   │   Function: convert_to_scalar
│   │   │   Function: create_reference_model
│   │   │   Function: entropy_from_logits
│   │   │   Function: flatten_dict
│   │   │   Function: gather_object
│   │   │   Function: is_deepspeed_available
│   │   │   Function: is_npu_available
│   │   │   Function: is_torch_greater_2_0
│   │   │   Function: is_xpu_available
│   │   │   Function: logprobs_from_logits
│   │   │   Function: masked_mean
│   │   │   Function: masked_var
│   │   │   Function: masked_whiten
│   │   │   Class: nullcontext
│   │   │   Function: set_seed
│   │   │   Function: stack_dicts
│   │   │   Function: stats_to_np
│   ├── reward_trainer.py
│   │   │   Class: Any
│   │   │   Class: Dataset
│   │   │   Class: EvalPrediction
│   │   │   Class: FrozenInstanceError
│   │   │   Class: PeftModel
│   │   │   Class: PreTrainedModel
│   │   │   Class: PreTrainedTokenizerBase
│   │   │   Class: RewardConfig
│   │   │   Class: RewardDataCollatorWithPadding
│   │   │   Class: RewardTrainer
│   │   │   Class: Trainer
│   │   │   Class: TrainerCallback
│   │   │   Class: TrainingArguments
│   │   │   Function: compute_accuracy
│   │   │   Function: get_peft_model
│   │   │   Function: is_peft_available
│   │   │   Function: nested_detach
│   │   │   Function: prepare_model_for_kbit_training
│   │   │   Function: replace
│   ├── sft_trainer.py
│   │   │   Class: AutoModelForCausalLM
│   │   │   Class: AutoTokenizer
│   │   │   Class: ConstantLengthDataset
│   │   │   Class: DataCollatorForCompletionOnlyLM
│   │   │   Class: DataCollatorForLanguageModeling
│   │   │   Class: Dataset
│   │   │   Class: DatasetGenerationError
│   │   │   Class: EvalPrediction
│   │   │   Class: PeftConfig
│   │   │   Class: PeftModel
│   │   │   Class: PreTrainedModel
│   │   │   Class: PreTrainedTokenizerBase
│   │   │   Class: SFTTrainer
│   │   │   Class: SchemaInferenceError
│   │   │   Class: Trainer
│   │   │   Class: TrainerCallback
│   │   │   Class: TrainingArguments
│   │   │   Function: get_peft_model
│   │   │   Function: is_peft_available
│   │   │   Function: neftune_post_forward_hook
│   │   │   Function: peft_module_casting_to_bf16
│   │   │   Function: prepare_model_for_kbit_training
│   │   │   Function: trl_sanitze_kwargs_for_tagging
│   │   │   Function: unwrap_model
│   │   │   Function: wraps
│   ├── training_configs.py
│   │   │   Class: RewardConfig
│   │   │   Class: TrainingArguments
│   │   │   Function: dataclass
│   ├── utils.py
│   │   │   Class: AdaptiveKLController
│   │   │   Class: Any
│   │   │   Class: ConstantLengthDataset
│   │   │   Class: DPODataCollatorWithPadding
│   │   │   Class: DataCollatorForCompletionOnlyLM
│   │   │   Class: DataCollatorForLanguageModeling
│   │   │   Class: FixedKLController
│   │   │   Class: IterableDataset
│   │   │   Class: PerPromptStatTracker
│   │   │   Class: PreTrainedTokenizerBase
│   │   │   Class: RewardDataCollatorWithPadding
│   │   │   Class: RunningMoments
│   │   │   Function: compute_accuracy
│   │   │   Function: dataclass
│   │   │   Class: deque
│   │   │   Function: disable_dropout_in_model
│   │   │   Function: exact_div
│   │   │   Function: get_global_statistics
│   │   │   Function: neftune_post_forward_hook
│   │   │   Function: pad_sequence
│   │   │   Function: pad_to_length
│   │   │   Function: peft_module_casting_to_bf16
│   │   │   Function: trl_sanitze_kwargs_for_tagging
├── __init__.py
├── core.py
│   │   Class: LengthSampler
│   │   Class: Mapping
│   │   Class: PPODecorators
│   │   Function: add_suffix
│   │   Function: average_torch_dicts
│   │   Function: clip_by_value
│   │   Function: contextmanager
│   │   Function: convert_to_scalar
│   │   Function: entropy_from_logits
│   │   Function: flatten_dict
│   │   Function: is_npu_available
│   │   Function: is_xpu_available
│   │   Function: logprobs_from_logits
│   │   Function: masked_mean
│   │   Function: masked_var
│   │   Function: masked_whiten
│   │   Function: pad_sequence
│   │   Function: pad_to_size
│   │   Function: randn_tensor
│   │   Function: respond_to_batch
│   │   Function: set_seed
│   │   Function: stack_dicts
│   │   Function: stats_to_np
│   │   Function: top_k_top_p_filtering
│   │   Function: whiten
├── import_utils.py
│   │   Function: is_accelerate_greater_20_0
│   │   Function: is_bitsandbytes_available
│   │   Function: is_diffusers_available
│   │   Function: is_npu_available
│   │   Function: is_peft_available
│   │   Function: is_rich_available
│   │   Function: is_torch_greater_2_0
│   │   Function: is_torchvision_available
│   │   Function: is_transformers_greater_than
│   │   Function: is_wandb_available
│   │   Function: is_xpu_available
```
