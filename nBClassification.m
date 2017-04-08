function predict = nBClassification(Xtrain,ytrain,xtest)
  %this will classify the data test with naive bayes classification
  
  nClass = max(ytrain);
  postProbList = zeros(nClass,1);
  
  for i=1:nClass,
    postProbList(i) = postProb(Xtrain,ytrain,xtest,i);
  end
  
  [M,I] = max(postProbList);
  
  predict = I(1);
end