function predict = ANNPredict(W1,W2,B1,B2,mT,xtest)
  
    %start perdicting
    
    V1 = (xtest*W1)+B1;
    A1 = zeros(1,length(W1(1,:)));
    for i=1:length(W1(1,:))
       ex = exp(-V1(i));
       A1(i) = 1/(1+ex);
    end
    %searching the value of A2
    %A2 is the output of output layer perceptrons
    V2 = (A1*W2)+B2;
    A2 = zeros(1,length(W2(1,:)));
    for i=1:length(W2(1,:))
       ex = exp(-V2(i));
       A2(i) = 1/(1+ex);
    end
%    for i=1:length(A2)
%       if(A2(i)<0.5)
%           A2(i) = 0;
%       else
%           A2(i) = 1;
%       end
%    end
    %make A2 as discrete value
    [M I] = max(A2); %find the maximum and maximum index value
    A2 = A2*0; %zero all A2
    A2(I) = 1; %make the maximum value as 1

    i = 1;
    while (prod(mT(i,:)==A2)~=1)
       i = i+1;
    end
    
    predict = i;
  
end