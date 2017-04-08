function predicts = ANNPredictAll(W1,W2,B1,B2,mT,Xtest)
  %predict all data from datasets
  
  predicts = [];
  
  for i=1:length(Xtest(:,1)),
    predicts = [predicts; ANNPredict(W1,W2,B1,B2,mT,Xtest(i,:))];
  end
 
end