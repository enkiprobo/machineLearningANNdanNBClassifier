function sPlot = scatterPlot(X,y)
  %plotting the dataset from poin X and target label y

  sPlot = gscatter(X(:,1),X(:,2),y);
end