function predicts = nBPredictAll(X,y)
  %predict all data from datasets
  
  predicts = [];
 
  for i=1:length(y),
    Xtrain = X;
    ytrain = y;
    Xtest = Xtrain(i,:);
    ytest = ytrain(i,1);
    Xtrain(i,:) = [];
    ytrain(i,:) = [];
    
    predicts = [predicts; nBClassification(Xtrain,ytrain,Xtest)];
  end
  
end