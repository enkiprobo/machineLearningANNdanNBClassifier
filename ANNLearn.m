function [MSE MSEList W1 W2 B1 B2 mT] = ANNLearn(Xtrain,ytrain)
    %this function will produce the model of ANN
    %with Multi Layer Perceptron I-H-O
    
    %initiate the ANN
    nClass = max(ytrain); %number of class or number of perceptron in output layer
    nAttributes = length(Xtrain(1,:)); %number of attributes in one data
    nHP = nClass*2; %number of perceptron in hidden layer
    
    lr = 0.003; %learning rate
    nEpoch = 5000; %number of epoch
    maxMSE = 0.001; %target MSE

    W1 = rand(nAttributes,nHP)*1-0.5;
    B1 = rand(1,nHP)*1-0.5;
    W2 = rand(nHP,nClass)*1-0.5;
    B2 = rand(1,nClass)*1-0.5;
    
    mT = eye(nClass); %matrix identity for output matrix
  
    %start learning
    nEL = 0;
    MSE = 1;
    MSEList = [];
    while (nEL<nEpoch && MSE>maxMSE)
        [MSE W1 W2 B1 B2] = epoch(Xtrain,ytrain,W1,W2,B1,B2,mT,lr);
        nEL = nEL +1;
        MSEList = [MSEList; MSE];
    end
    %end learning
    plot(MSEList');
    MSE
end