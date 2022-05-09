import numpy as np 
lr=0.01 
# initialise the weight and expected outputs as per truth table. 
# RED='0', BLUE='1' 
inputs=np.array([[3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1]]) 
expected_output=np.array([[0],[1],[0],[1],[0],[1],[0],[1]]) 
# initialise weightb asn bias with random values. 
inputLayerNeurons,hiddenLayerNeurons,outputLayerNeurons=2,2,1 
hidden_weights=np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons)) 
hidden_bias=np.random.uniform(size=(1,hiddenLayerNeurons)) 
output_weights=np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons)) 
output_bias=np.random.uniform(size=(1,outputLayerNeurons)) 
# the fwd pass involves compute the predicted output, which is a function of the # weighted sum of input given to neurons. 
def sigmoid(x): 
    return 1/(1+np.exp(-x)) 
def sigmoid_derivative(x): 
    return x*(1-x)
for i in range (1000): 
    hidden_layer_activation = np.dot(inputs,hidden_weights) 
    hidden_layer_activation += hidden_bias 
    hidden_layer_output = sigmoid(hidden_layer_activation) 
    
# print('hidden layer output: ',hidden_layer_output) 
    output_layer_activation = np.dot(hidden_layer_output,output_weights) 
    output_layer_activation += output_bias 
    predicted_output = sigmoid(output_layer_activation) 
    
    # print('\n') 
    # print('predicted output: ',predicted_output) 
    # apply back propogation. 
    # backpropogation 
    
    error=expected_output-predicted_output 
    d_predicted_output=error*sigmoid_derivative(predicted_output) 
    error_hidden_layer=d_predicted_output.dot(output_weights.T) 
    d_hidden_layer=error_hidden_layer*sigmoid_derivative(hidden_layer_output) 
    
    # updating weights and bias 
    output_weights+=hidden_layer_output.T.dot(d_predicted_output)*lr 
    output_bias+=np.sum(d_predicted_output,axis=0,keepdims=True)*lr 
    hidden_weights+=inputs.T.dot(d_hidden_layer)*lr 
    hidden_bias+=np.sum(d_hidden_layer,axis=0,keepdims=True)*lr 
    
    # visualise the result. print('Final Hidden Weights: ',end='') 
print(*hidden_weights) 
print('Final Hidden Bias: ', end='') 
print(*hidden_bias) 
print('Final Output Weights: ', end='') 
print(*output_weights) 
print('Final Output Bias: ', end='') 
print(*output_bias) 
print('\nOutput from neural networks after 10000 epoch: ', end='') 

print(*predicted_output) 
print('\n') 

hidden_layer_activation = np.dot([[4.5,1]],hidden_weights) 
hidden_layer_activation += hidden_bias 
hidden_layer_output = sigmoid(hidden_layer_activation) 
print('Hidden Layer Output: ',hidden_layer_output)
    
output_layer_activation = np.dot(hidden_layer_output,output_weights) 
output_layer_activation += output_bias 
predicted_output = sigmoid(output_layer_activation) 
print('Predicted Output: ',predicted_output)