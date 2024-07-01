import inspect
from transformers import TrainingArguments

# Retrieve the full argument specification for the TrainingArguments initializer
args_spec = inspect.getfullargspec(TrainingArguments.__init__)

print("Arguments for TrainingArguments:")
# Exclude 'self' from the arguments list as it's not an actual parameter
for arg in args_spec.args[1:]:  # args[0] is 'self', which we skip
    print(arg)

# If you want to include default values and other metadata, you could do:
if args_spec.defaults:
    # The last 'len(defaults)' arguments have default values
    defaults_offset = len(args_spec.args) - len(args_spec.defaults)
    for idx, arg in enumerate(args_spec.args[1:]):
        if idx >= defaults_offset:
            default = args_spec.defaults[idx - defaults_offset]
            print(f"{arg} (default: {default})")
        else:
            print(arg)
else:
    for arg in args_spec.args[1:]:
        print(arg)