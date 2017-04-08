function prior = getPrior(y, target)
  %get the prior of target class from dataset
  
  prior = length(find(y==target))/length(y);
end