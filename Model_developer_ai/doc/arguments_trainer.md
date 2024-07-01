- output_dir   str  The output directory where the model predictions and checkpoints will be written.

-  overwrite_output_dir (`bool`, *optional*, defaults to `False`):
            If `True`, overwrite the content of the output directory. Use this to continue training if `output_dir`
            points to a checkpoint directory.
- do_train (`bool`, *optional*, defaults to `False`):
            Whether to run training or not. This argument is not directly used by [`Trainer`], it's intended to be used
            by your training/evaluation scripts instead.
-         do_eval (`bool`, *optional*):
            Whether to run evaluation on the validation set or not. Will be set to `True` if `eval_strategy` is
            different from `"no"`. This argument is not directly used by [`Trainer`], it's intended to be used by your
            training/evaluation scripts instead. See the
- do_predict (`bool`, *optional*, defaults to `False`):
            Whether to run predictions on the test set or not. This argument is not directly used by [`Trainer`], it's
            intended to be used by your training/evaluation scripts instead.

- eval_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`): The evaluation strategy to adopt during training. Possible values are:- `"no"`: No evaluation is done during training.- `"steps"`: Evaluation is done (and logged) every `eval_steps`.- `"epoch"`: Evaluation is done at the end of each epoch.