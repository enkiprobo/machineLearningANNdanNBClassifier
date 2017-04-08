function [e NW1 NW2 NB1 NB2] = backpropagation(x,y,W1,W2,B1,B2,mT,lr)
   %this function will implement backpropagation
   %algorithm to the expected MLP model
   
   %searching the value of A1
   %A1 is the output of first hidden layer perceptrons
   V1 = x*W1+B1;
   A1 = zeros(1,length(W1(1,:)));
   for i=1:length(W1(1,:))
       ex = exp(-V1(i));
       A1(i) = 1/(1+ex);
   end
   %searching the value of A2
   %A2 is the output of output layer perceptrons
   V2 = A1*W2+B2;
   A2 = zeros(1,length(W2(1,:)));
   for i=1:length(W2(1,:))
       ex = exp(-V2(i));
       A2(i) = 1/(1+ex);
   end
   
   %calculate the error
   E = mT(y,:)-A2;
   e = norm(E);
   
   %doing the back propagation algorithm
   D2 = A2.*(1-A2).*E;
   D1 = A1.*(1-A1).*(D2*W2');

   dW2 = lr*((D2'*A1)');
   dB2 = lr*D2;
   dW1 = lr*((D1'*x)');
   dB1 = lr*D1;
   
   NW1 = W1+dW1;
   NB1 = B1+dB1;
   NW2 = W2+dW2;
   NB2 = B2+dB2;
end