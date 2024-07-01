import torch 
funcity=dir(torch)
func =[f for f in funcity if not  f.startswith('-')]

print(func)