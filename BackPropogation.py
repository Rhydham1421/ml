import numpy as np 

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])

X =X/np.amax(X,axis=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x*(1-x)

epoch = 7000
lr = 0.1
inputlayer_neuron = 2
hiddenlayer_neuron = 1
output_neuron = 1

wh = np.random.uniform(size=(inputlayer_neuron,hiddenlayer_neuron))
bh = np.random.uniform(size = (1,hiddenlayer_neuron))
wout = np.random.uniform(size=(hiddenlayer_neuron,output_neuron))
bout = np.random.uniform(size = (1,output_neuron))

for i in range(epoch):
    hinp1 = np.dot(X,wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1 = np.dot(hlayer_act,wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)
    
E0 = y-output
outgrad = derivatives_sigmoid(output)
d_output = E0 * outgrad
EH = d_output.dot(wout.T)
hiddengrad = derivatives_sigmoid(hlayer_act)
d_hiddenlayer = EH * hiddengrad
wout += hlayer_act.T.dot(d_output)
wh += X.T.dot(d_hiddenlayer)

print("Input:\n"+str(X))
print("Actual Output:\n"+str(X))
print("Predicted Output:\n",output)