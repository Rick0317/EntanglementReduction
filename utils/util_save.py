import csv
import inspect

def printx(params,iteration):
    #global iteration
    iteration += 1
    with open('params.txt', 'a') as f:
        f.write(f'Step {iteration}: params = {params}\n')
    print(f'{params}')
    
    return
    
    
def save_parameters(cost_fns,h_variables,maxit,parameters,hamil_name):

    cost_list =[]
    cost_list = ['...']
    
    for i, fn in enumerate(cost_fns):
        source = inspect.getsource(fn).strip()  # Get the source code of the lambda function
        expression = source.split(":")[1].strip()  # Split at ':' and take the part after it
        cost_list.append(expression)
    
    with open("parameters.csv", "a", newline="") as file:
        writer = csv.writer(file)
               
        writer.writerow(['']) 
        writer.writerow([''] + ["h_variables:"] + [h_variables] + [''] + ["maxit:", maxit] + [''] + ["hamil:", hamil_name])
        writer.writerow([''])
        writer.writerow([''] + ['I_xy'] + ['I_yz'] + ['I_xz'] + ['MPS 1'] + ['MPS 2'] + ['CP 1'] + ['CP 2'] + ['Cov_xy'] + ['Cov_yz'] + ['Cov_xz'])
        # Write rows for parameters and corresponding costs
        for param, cost in zip(parameters, cost_list):
            writer.writerow([cost] + param)
            
        writer.writerow([''])   
        
    return