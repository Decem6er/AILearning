<<<<<<< HEAD
import numpy
#定义sigmoid Pay Attantion
def sigmoid(x):
    return 1/(1+numpy.exp(-x))
class neuralnetwork:
    #initialise the neural network
    def __init__ (self,inputnodes,hiddennodes,outputnodes,learningrate):
        #set number of nodes in each input, hidden, output layer
        #which is also the Matrix that can be caculate
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        #learning rate
        self.lr=learningrate

        #wih means input layer to hidden layer
        #who means hidden layer to output layer
        #==>i->h->o
        #link weight matrics, wih and who
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

    
    #train the neural network
    def train(self,input_list,targets_list):
        #convert inouts list to 2d arraay
        inputs=numpy.array(input_list,ndmin=2).T
        targets=numpy.array(targets_list.ndmin-2).T

        #caculate signal into hidden lau\yer
        hidden_inputs=numpy.dot(self.wih,inputs)
        #caculate the signals emerging from hidden layers
        hidden_outputs=sigmoid(hidden_inputs)

        #sigmoid->final output layers
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #signals emerging from final output layer
        final_outputs=sigmoid(final_inputs)
        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors)

        #Update the weight for the links between the hidden and output layers
        self.who+=self.lr*numpy.dot((output_errors*hidden_outputs*(1.0-final_outputs)),numpy.T(inputs))

        
        

    #query the neural network
    def query(self,input_list):
        #convert inouts list to 2d array
        #.T 是啥？
        inputs=numpy.array(input_list,ndmin=2).T 
        #csculate signalds into hidden layer
        hidden_inputs=numpy.dot(self.wih,inputs)
        #caculate the singnals emerging from hidden layer 
        hidden_outputs=sigmoid(hidden_inputs)
        #caculate signal into final outputs layer
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #caculate the signals energing fromfinal outputs layer
        final_outputs=sigmoid(final_inputs)
        return final_outputs
        

input_nodes=3
hidden_nodes=3
output_nodes=3
learning_rate=0.3

#create instance of neural network
n=neuralnetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

print(n.query([1.0,0.5,-1.5]))


=======
import numpy
#定义sigmoid Pay Attantion
def sigmoid(x):
    return 1/(1+numpy.exp(-x))
class neuralnetwork:
    #initialise the neural network
    def __init__ (self,inputnodes,hiddennodes,outputnodes,learningrate):
        #set number of nodes in each input, hidden, output layer
        #which is also the Matrix that can be caculate
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        #learning rate
        self.lr=learningrate

        #wih means input layer to hidden layer
        #who means hidden layer to output layer
        #==>i->h->o
        #link weight matrics, wih and who
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

    
    #train the neural network
    def train(self,input_list,targets_list):
        #convert inouts list to 2d arraay
        inputs=numpy.array(input_list,ndmin=2).T
        targets=numpy.array(targets_list.ndmin-2).T

        #caculate signal into hidden lau\yer
        hidden_inputs=numpy.dot(self.wih,inputs)
        #caculate the signals emerging from hidden layers
        hidden_outputs=sigmoid(hidden_inputs)

        #sigmoid->final output layers
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #signals emerging from final output layer
        final_outputs=sigmoid(final_inputs)
        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors)

        #Update the weight for the links between the hidden and output layers
        self.who+=self.lr*numpy.dot((output_errors*hidden_outputs*(1.0-final_outputs)),numpy.T(inputs))

        
        

    #query the neural network
    def query(self,input_list):
        #convert inouts list to 2d array
        #.T 是啥？
        inputs=numpy.array(input_list,ndmin=2).T 
        #csculate signalds into hidden layer
        hidden_inputs=numpy.dot(self.wih,inputs)
        #caculate the singnals emerging from hidden layer 
        hidden_outputs=sigmoid(hidden_inputs)
        #caculate signal into final outputs layer
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #caculate the signals energing fromfinal outputs layer
        final_outputs=sigmoid(final_inputs)
        return final_outputs
        

input_nodes=3
hidden_nodes=3
output_nodes=3
learning_rate=0.3

#create instance of neural network
n=neuralnetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

print(n.query([1.0,0.5,-1.5]))


>>>>>>> 0
