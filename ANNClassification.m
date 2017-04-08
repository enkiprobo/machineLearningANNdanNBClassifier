function predict = ANNClassification(Xtrain,ytrain,xtest)
    %this function will claasify the data test
    %with Multi Layer Perceptron I-H-O
    
    %initiate the ANN
    nClass = max(ytrain); %number of class or number of perceptron in output layer
    nAttributes = length(Xtrain(1,:)); %number of attributes in one data
    nHP = 3; %number of perceptron in hidden layer
    
    lr = 2; %learning rate
    nEpoch = 100; %number of epoch

    W1 = randn(nAttributes,nHP)*2-1;
    B1 = randn(1,nHP)*2-1;
    W2 = randn(nHP,nClass)*2-1;
    B2 = randn(1,nClass)*2-1;
    
    mT = eye(nClass); %matrix identity for output matrix
  
    %start learning
    for i=1:nEpoch
        [E MSE W1 W2 B1 B2] = epoch(Xtrain,ytrain,W1,W2,B1,B2,mT,lr);
    end
    %end learning
    
    %start perdicting
    %searching the value of A1
    %A1 is the output of first hidden layer perceptrons
    V1 = (xtest * W1)+B1;
    A1 = zeros(1,length(W1(1,:)));
    for i=1:length(W1(1,:))
       ex = exp(-V1(i));
       A1(i) = 1 / (1+ex);
    end
    %searching the value of A2
    %A2 is the output of output layer perceptrons
    V2 = (A1 * W2)+B2;
    A2 = zeros(1,length(W2(1,:)));
    for i=1:length(W2(1,:))
       ex = exp(-V2(i));
       A2(i) = 1 / (1+ex);
    end

    [M I] = max(A2);
    A2 = A2*0;
    A2(I) = 1;

    
    i = 1;
    while (prod(mT(i,:)==A2)~=1)
       i = i+1;
    end
    
    predict = i;
end