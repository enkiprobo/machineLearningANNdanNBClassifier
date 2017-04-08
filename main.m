[X y] = loadDatasets('D31.csv');

scatterPlot(X,y);

[MSE MSEList W1 W2 B1 B2 mT] = ANNLearn(X,y);
predicts = ANNPredictAll(W1,W2,B1,B2,mT,X);
score = pScore(y, predicts, 3);

ANNdecbound2D(X,W1,W2,B1,B2,mT,predicts);