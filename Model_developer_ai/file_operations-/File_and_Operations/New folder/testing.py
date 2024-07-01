# import inspect
# from transformers import TrainingArguments

# # Retrieve the full argument specification for the TrainingArguments initializer
# args_spec = inspect.getfullargspec(TrainingArguments.__init__)

# print("Arguments for TrainingArguments:")
# # Exclude 'self' from the arguments list as it's not an actual parameter
# for arg in args_spec.args[:]:  # args[0] is 'self', which we skip
#     print(arg)
