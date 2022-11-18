import numpy as np
import random
max_neurons=200
max_conv_kernel=3
def select_breed_mutate(pop,scores,x):
    print("length of pop")
    print(len(pop))
    print(scores)
    idx = np.argsort(scores)
    idx = np.flip(idx)
    top_x = idx[:3]
    
    pop2 = [pop[i] for i in top_x]
    new_pop=[]
    #start mutation
    for i in range(5):
            new_child={}
            #breed
            #####################
            fc_parent = random.choice(pop2)
            #conv_parent=random.choice(pop2)
            conv_parent=fc_parent
            #epoch_parent=random.choice(pop2)
            epoch_parent=fc_parent
            n_epochs=epoch_parent['n_epochs']
            n_dense_layers=fc_parent['n_dense_layers']
            n_conv_layers=conv_parent['n_conv_layers']
            neurons_dense=fc_parent['neurons_dense']
            conv_output_channels=conv_parent['conv_output_channels']
            conv_kernel_sizes=conv_parent['conv_kernel_sizes']
            ###################
            #mutate
            n_dense_layers2=(np.random.randn(1)[0]*5+n_dense_layers).astype(int)
            n_conv_layers2=(np.random.randn(1)[0]*5+n_conv_layers).astype(int)
            n_epochs=(np.random.randn(1)[0]*2*n_epochs+n_epochs).astype(int)
            if n_epochs<1:
                n_epochs=2
            if n_epochs>12:
                n_epochs=5
            if n_dense_layers2<1:
                n_dense_layers2=1
            if n_dense_layers2>12:
                n_dense_layers2=5
            if n_conv_layers2<1:
                n_conv_layers2=1
            if n_conv_layers2>11:
                n_conv_layers2=5            
            neurons_dense=(np.random.randn(n_dense_layers)*15+neurons_dense).astype(int)
            neurons_dense[neurons_dense<1]=10
            neurons_dense[neurons_dense>400]=150
            neurons_dense2=np.random.randint(20,max_neurons,n_dense_layers2)
            neurons_dense2[:np.min([n_dense_layers,n_dense_layers2])]=neurons_dense[:np.min([n_dense_layers,n_dense_layers2])]
            neurons_dense=neurons_dense2
            n_dense_layers=n_dense_layers2

            

            conv_output_channels=(np.random.randn(n_conv_layers)*1+conv_output_channels).astype(int)
            conv_output_channels[conv_output_channels<1]=1
            conv_output_channels[conv_output_channels>8]=5
            conv_output_channels2=np.random.randint(1,7,n_conv_layers2)
            conv_output_channels2[:np.min([n_conv_layers,n_conv_layers2])]=conv_output_channels[:np.min([n_conv_layers,n_conv_layers2])]
            conv_output_channels=conv_output_channels2
                    
            
              
            conv_kernel_sizes=(np.random.randn(n_conv_layers)*2+conv_kernel_sizes).astype(int)
            conv_kernel_sizes[conv_kernel_sizes<1]=1
            conv_kernel_sizes[conv_kernel_sizes>10]=5
            conv_kernel_sizes2=np.random.randint(1,10,n_conv_layers2)
            conv_kernel_sizes2[:np.min([n_conv_layers,n_conv_layers2])]=conv_kernel_sizes[:np.min([n_conv_layers,n_conv_layers2])]
            conv_kernel_sizes=conv_kernel_sizes2
            n_conv_layers=n_conv_layers2
            #################### end mutation
            #create new child based on breeding and mutation
            new_child['n_epochs']=n_epochs
            new_child['n_dense_layers']=n_dense_layers
            new_child['n_conv_layers']=n_conv_layers
            new_child['neurons_dense']=neurons_dense
            new_child['conv_output_channels']=conv_output_channels
            new_child['conv_kernel_sizes']=conv_kernel_sizes
            
            new_pop.append(new_child)
    for i in pop2: new_pop.append(i) 
        
    return new_pop
            
            
            
            
            

