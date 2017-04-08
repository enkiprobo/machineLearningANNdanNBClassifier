function [MSE NW1 NW2 NB1 NB2] = epoch(X,Y,W1,W2,B1,B2,mT,lr)
    
    e = zeros(length(Y),1);
    for i=1:length(X(:,1))
       [e(i) W1 W2 B1 B2] = backpropagation(X(i,:),Y(i),W1,W2,B1,B2,mT,lr);
    end
    
    MSE = sum(e.^2)/length(Y);
    NW1 = W1;
    NW2 = W2;
    NB1 = B1;
    NB2 = B2;
end